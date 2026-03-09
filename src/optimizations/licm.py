"""Вынос инвариантного кода из циклов (Loop Invariant Code Motion — LICM).

Находит выражения в теле цикла, не зависящие от переменных, изменяемых
в этом цикле, и выносит их вычисление в новые присваивания перед циклом.

Пример (bench_licm.f):
    A = 7.5 : B = 3.2 : C = 2.1
    DO I = 1, 1000
        S = S + (A * B + C) * FLOAT(I)    ! A*B+C — инвариант
    ENDDO

    →

    _licm_1 = A * B + C
    DO I = 1, 1000
        S = S + _licm_1 * FLOAT(I)
    ENDDO

Алгоритм:
    1. Определить множество modified_vars — переменных, которым присваивается
       значение в теле цикла (включая переменную самого цикла).
    2. Рекурсивно обойти выражения в теле. Если подвыражение:
       - не содержит переменных из modified_vars
       - не является простым литералом или переменной (избегаем тривиальных выносов)
       - достаточно «дорогое» (BinaryOp с операцией * / ** или FunctionCall)
       то вынести его в новую временную переменную _licm_N.
    3. Поместить присваивания _licm_N = expr перед циклом.

Рекурсия: обрабатывает вложенные циклы снизу вверх.
"""
from __future__ import annotations
from typing import List, Set, Tuple, Dict
from dataclasses import replace as dc_replace

from src.core import (
    Program, Statement, Expression,
    Assignment, DoLoop, LabeledDoLoop, DoWhile, LabeledDoWhile,
    IfStatement, SimpleIfStatement,
    Variable, BinaryOp, UnaryOp, FunctionCall, ArrayRef,
    IntegerLiteral, RealLiteral, LogicalLiteral, StringLiteral,
    PrintStatement, WriteStatement, ReadStatement, CallStatement,
)
from src.optimizations.base import ASTOptimizationPass


_HOIST_OPS = {'*', '/', '**', '+', '-'}
_EXPENSIVE_OPS = {'*', '/', '**'}


def _collect_modified(stmts: List[Statement], result: Set[str]) -> None:
    """Собрать переменные, которым присваивается значение в stmts."""
    for stmt in stmts:
        if isinstance(stmt, Assignment):
            result.add(stmt.target)
        elif isinstance(stmt, (DoLoop, LabeledDoLoop)):
            result.add(stmt.var)
            _collect_modified(stmt.body, result)
        elif isinstance(stmt, (DoWhile, LabeledDoWhile)):
            _collect_modified(stmt.body, result)
        elif isinstance(stmt, IfStatement):
            _collect_modified(stmt.then_body, result)
            for _, b in stmt.elif_parts:
                _collect_modified(b, result)
            if stmt.else_body:
                _collect_modified(stmt.else_body, result)


def _uses_modified(expr: Expression, modified: Set[str]) -> bool:
    """Проверить, использует ли выражение хотя бы одну переменную из modified."""
    if isinstance(expr, Variable):
        return expr.name in modified
    if isinstance(expr, ArrayRef):
        return expr.name in modified or any(_uses_modified(i, modified) for i in expr.indices)
    if isinstance(expr, BinaryOp):
        return _uses_modified(expr.left, modified) or _uses_modified(expr.right, modified)
    if isinstance(expr, UnaryOp):
        return _uses_modified(expr.operand, modified)
    if isinstance(expr, FunctionCall):
        return any(_uses_modified(a, modified) for a in expr.args)
    return False


def _is_trivial(expr: Expression) -> bool:
    """Проверить, что выражение слишком простое для выноса."""
    return isinstance(expr, (IntegerLiteral, RealLiteral, LogicalLiteral,
                             StringLiteral, Variable))


def _contains_real(expr: Expression) -> bool:
    """Проверить, содержит ли выражение REAL-значение."""
    if isinstance(expr, RealLiteral):
        return True
    if isinstance(expr, IntegerLiteral):
        return False
    if isinstance(expr, BinaryOp):
        return _contains_real(expr.left) or _contains_real(expr.right)
    if isinstance(expr, UnaryOp):
        return _contains_real(expr.operand)
    if isinstance(expr, FunctionCall):
        # Математические функции возвращают REAL
        real_funcs = {'FLOAT', 'REAL', 'SQRT', 'SIN', 'COS', 'TAN',
                      'EXP', 'LOG', 'ABS', 'ASIN', 'ACOS', 'ATAN', 'LOG10'}
        return expr.name.upper() in real_funcs
    # Variable — считаем INTEGER по умолчанию (безопаснее для LICM)
    # REAL-выражения будут идентифицированы через RealLiteral или FunctionCall
    return False


def _is_worth_hoisting(expr: Expression) -> bool:
    """Стоит ли выносить это выражение? Только нетривиальные REAL BinaryOp и FunctionCall.

    Целочисленные выражения (индексы массивов) не выносим — они нужны как i32.
    """
    if isinstance(expr, BinaryOp):
        if expr.op not in _HOIST_OPS:
            return False
        # Выносить только если выражение содержит хотя бы одно REAL-значение
        # Чисто INTEGER-выражения (I+1, I-1) не трогаем
        if not _contains_real(expr):
            return False
        return True
    if isinstance(expr, FunctionCall):
        return True
    if isinstance(expr, UnaryOp):
        return not _is_trivial(expr.operand) and _contains_real(expr.operand)
    return False


class _ExprHoister:
    """Вспомогательный класс: обходит тело цикла и выносит инварианты."""

    def __init__(self, modified: Set[str], counter_ref: List[int]):
        self.modified = modified
        self.counter_ref = counter_ref
        self.hoisted: List[Tuple[str, Expression]] = []  # (tmp_name, expr)
        self.cache: Dict[str, str] = {}  # repr(expr) -> tmp_name

    def hoist_expr(self, expr: Expression) -> Expression:
        """Рекурсивно обойти выражение и вынести инварианты."""
        if _is_trivial(expr):
            return expr

        if _uses_modified(expr, self.modified):
            # Спуститься вглубь
            if isinstance(expr, BinaryOp):
                new_left = self.hoist_expr(expr.left)
                new_right = self.hoist_expr(expr.right)
                if new_left is not expr.left or new_right is not expr.right:
                    expr = dc_replace(expr, left=new_left, right=new_right)
            elif isinstance(expr, UnaryOp):
                new_op = self.hoist_expr(expr.operand)
                if new_op is not expr.operand:
                    expr = dc_replace(expr, operand=new_op)
            elif isinstance(expr, FunctionCall):
                new_args = [self.hoist_expr(a) for a in expr.args]
                if any(na is not a for na, a in zip(new_args, expr.args)):
                    expr = dc_replace(expr, args=new_args)
            elif isinstance(expr, ArrayRef):
                new_indices = [self.hoist_expr(i) for i in expr.indices]
                if any(ni is not i for ni, i in zip(new_indices, expr.indices)):
                    expr = dc_replace(expr, indices=new_indices)
            return expr

        # Выражение является инвариантом
        if not _is_worth_hoisting(expr):
            return expr

        key = _expr_repr(expr)
        if key in self.cache:
            return Variable(name=self.cache[key], line=expr.line, col=expr.col)

        self.counter_ref[0] += 1
        tmp_name = f"_licm_{self.counter_ref[0]}"
        self.hoisted.append((tmp_name, expr))
        self.cache[key] = tmp_name
        return Variable(name=tmp_name, line=expr.line, col=expr.col)

    def hoist_stmt(self, stmt: Statement) -> Statement:
        if isinstance(stmt, Assignment):
            new_val = self.hoist_expr(stmt.value)
            new_idx = [self.hoist_expr(i) for i in stmt.indices]
            return dc_replace(stmt, value=new_val, indices=new_idx)
        elif isinstance(stmt, (DoLoop, LabeledDoLoop)):
            new_start = self.hoist_expr(stmt.start)
            new_end = self.hoist_expr(stmt.end)
            new_step = self.hoist_expr(stmt.step) if stmt.step else stmt.step
            new_body = [self.hoist_stmt(s) for s in stmt.body]
            return dc_replace(stmt, start=new_start, end=new_end,
                               step=new_step, body=new_body)
        elif isinstance(stmt, (DoWhile, LabeledDoWhile)):
            new_cond = self.hoist_expr(stmt.condition)
            new_body = [self.hoist_stmt(s) for s in stmt.body]
            return dc_replace(stmt, condition=new_cond, body=new_body)
        elif isinstance(stmt, IfStatement):
            new_cond = self.hoist_expr(stmt.condition)
            new_then = [self.hoist_stmt(s) for s in stmt.then_body]
            new_elif = [(self.hoist_expr(c), [self.hoist_stmt(s) for s in b])
                        for c, b in stmt.elif_parts]
            new_else = ([self.hoist_stmt(s) for s in stmt.else_body]
                        if stmt.else_body else stmt.else_body)
            return dc_replace(stmt, condition=new_cond, then_body=new_then,
                               elif_parts=new_elif, else_body=new_else)
        elif isinstance(stmt, SimpleIfStatement):
            new_cond = self.hoist_expr(stmt.condition)
            new_s = self.hoist_stmt(stmt.statement)
            return dc_replace(stmt, condition=new_cond, statement=new_s)
        elif isinstance(stmt, PrintStatement):
            new_items = [self.hoist_expr(i) for i in stmt.items]
            return dc_replace(stmt, items=new_items)
        elif isinstance(stmt, WriteStatement):
            new_items = [self.hoist_expr(i) for i in stmt.items]
            return dc_replace(stmt, items=new_items)
        elif isinstance(stmt, CallStatement):
            new_args = [self.hoist_expr(a) for a in stmt.args]
            return dc_replace(stmt, args=new_args)
        return stmt


def _expr_repr(expr: Expression) -> str:
    """Каноническое строковое представление выражения для дедупликации."""
    if isinstance(expr, IntegerLiteral):
        return f"int:{expr.value}"
    if isinstance(expr, RealLiteral):
        return f"real:{expr.value}"
    if isinstance(expr, Variable):
        return f"var:{expr.name}"
    if isinstance(expr, BinaryOp):
        return f"({_expr_repr(expr.left)}{expr.op}{_expr_repr(expr.right)})"
    if isinstance(expr, UnaryOp):
        return f"({expr.op}{_expr_repr(expr.operand)})"
    if isinstance(expr, FunctionCall):
        args = ",".join(_expr_repr(a) for a in expr.args)
        return f"{expr.name}({args})"
    if isinstance(expr, ArrayRef):
        idx = ",".join(_expr_repr(i) for i in expr.indices)
        return f"{expr.name}[{idx}]"
    return repr(expr)


def _process_loop(loop: Statement, counter: List[int]) -> Tuple[List[Statement], Statement]:
    """Обработать один цикл: вынести инварианты, вернуть (preheader_stmts, new_loop)."""
    if isinstance(loop, (DoLoop, LabeledDoLoop)):
        modified: Set[str] = {loop.var}
        _collect_modified(loop.body, modified)
    elif isinstance(loop, (DoWhile, LabeledDoWhile)):
        modified = set()
        _collect_modified(loop.body, modified)
    else:
        return [], loop

    hoister = _ExprHoister(modified, counter)
    new_body = [hoister.hoist_stmt(s) for s in loop.body]
    new_loop = dc_replace(loop, body=new_body)

    preheader = []
    for tmp_name, expr in hoister.hoisted:
        preheader.append(Assignment(
            target=tmp_name,
            value=expr,
            indices=[],
            stmt_label=None,
            line=loop.line,
            col=loop.col,
        ))
    return preheader, new_loop


def _process_stmts(stmts: List[Statement], counter: List[int]) -> List[Statement]:
    """Рекурсивно обработать список операторов, выполняя LICM для всех циклов."""
    result = []
    for stmt in stmts:
        if isinstance(stmt, (DoLoop, LabeledDoLoop, DoWhile, LabeledDoWhile)):
            # Сначала обработать тело (снизу вверх)
            if isinstance(stmt, (DoLoop, LabeledDoLoop)):
                new_body = _process_stmts(stmt.body, counter)
                stmt = dc_replace(stmt, body=new_body)
            else:
                new_body = _process_stmts(stmt.body, counter)
                stmt = dc_replace(stmt, body=new_body)
            # Затем вынести из текущего цикла
            preheader, new_loop = _process_loop(stmt, counter)
            result.extend(preheader)
            result.append(new_loop)
        elif isinstance(stmt, IfStatement):
            new_then = _process_stmts(stmt.then_body, counter)
            new_elif = [(c, _process_stmts(b, counter)) for c, b in stmt.elif_parts]
            new_else = _process_stmts(stmt.else_body, counter) if stmt.else_body else stmt.else_body
            result.append(dc_replace(stmt, then_body=new_then,
                                      elif_parts=new_elif, else_body=new_else))
        else:
            result.append(stmt)
    return result


_GLOBAL_LICM_COUNTER = [0]


class LoopInvariantCodeMotion(ASTOptimizationPass):
    """Выносит инвариантные вычисления из тел циклов."""

    name = "LoopInvariantCodeMotion"

    def run(self, program: Program) -> Program:
        counter = _GLOBAL_LICM_COUNTER
        before = counter[0]

        new_stmts = _process_stmts(program.statements, counter)
        new_subs = [
            dc_replace(s, statements=_process_stmts(s.statements, counter))
            for s in program.subroutines
        ]
        new_funcs = [
            dc_replace(f, statements=_process_stmts(f.statements, counter))
            for f in program.functions
        ]

        self.stats = {"hoisted": counter[0] - before}
        return dc_replace(program, statements=new_stmts,
                          subroutines=new_subs, functions=new_funcs)
