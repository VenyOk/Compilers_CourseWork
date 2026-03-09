"""Устранение общих подвыражений (Common Subexpression Elimination — CSE).

Находит повторяющиеся вычисления в последовательном блоке кода и заменяет
их ссылкой на временную переменную.

Пример (bench_cse.f):
    S1 = S1 + (A * B + C)
    S2 = S2 + (A * B + C) * 2.0    ! A*B+C вычислен снова
    S3 = S3 + (A * B + C) / 3.0    ! и снова

    →

    _cse_1 = A * B + C
    S1 = S1 + _cse_1
    S2 = S2 + _cse_1 * 2.0
    S3 = S3 + _cse_1 / 3.0

Ограничения:
    - Работает только внутри «прямого» блока (без условных переходов).
    - При любом присваивании переменной X сбрасывает все CSE-записи,
      содержащие X.
    - Не выносит за пределы цикла — это задача LICM.
"""
from __future__ import annotations
from typing import List, Dict, Set, Tuple
from dataclasses import replace as dc_replace

from src.core import (
    Program, Statement, Expression,
    Assignment, DoLoop, LabeledDoLoop, DoWhile, LabeledDoWhile,
    IfStatement, SimpleIfStatement,
    Variable, BinaryOp, UnaryOp, FunctionCall, ArrayRef,
    IntegerLiteral, RealLiteral,
    PrintStatement, WriteStatement, CallStatement,
)
from src.optimizations.base import ASTOptimizationPass


def _expr_key(expr: Expression) -> str:
    """Каноническое представление выражения для сравнения."""
    if isinstance(expr, IntegerLiteral):
        return f"i{expr.value}"
    if isinstance(expr, RealLiteral):
        return f"r{expr.value}"
    if isinstance(expr, Variable):
        return f"v:{expr.name}"
    if isinstance(expr, BinaryOp):
        # Коммутативные операции: нормализуем порядок операндов
        lk = _expr_key(expr.left)
        rk = _expr_key(expr.right)
        if expr.op in ('+', '*') and lk > rk:
            lk, rk = rk, lk
        return f"({lk}{expr.op}{rk})"
    if isinstance(expr, UnaryOp):
        return f"({expr.op}{_expr_key(expr.operand)})"
    if isinstance(expr, FunctionCall):
        args = ",".join(_expr_key(a) for a in expr.args)
        return f"{expr.name}({args})"
    if isinstance(expr, ArrayRef):
        idx = ",".join(_expr_key(i) for i in expr.indices)
        return f"{expr.name}[{idx}]"
    return repr(expr)


def _is_pure(expr: Expression) -> bool:
    """Выражение чистое (без side-эффектов и массивов для записи)."""
    if isinstance(expr, (IntegerLiteral, RealLiteral, Variable)):
        return True
    if isinstance(expr, ArrayRef):
        return all(_is_pure(i) for i in expr.indices)
    if isinstance(expr, BinaryOp):
        return _is_pure(expr.left) and _is_pure(expr.right)
    if isinstance(expr, UnaryOp):
        return _is_pure(expr.operand)
    if isinstance(expr, FunctionCall):
        # Считаем математические функции чистыми
        pure_funcs = {'SQRT', 'SIN', 'COS', 'TAN', 'LOG', 'EXP', 'ABS',
                      'INT', 'FLOAT', 'REAL', 'MOD', 'MIN', 'MAX', 'ASIN',
                      'ACOS', 'ATAN', 'LOG10', 'SIGN', 'POW'}
        return expr.name.upper() in pure_funcs and all(_is_pure(a) for a in expr.args)
    return False


def _is_trivial(expr: Expression) -> bool:
    return isinstance(expr, (IntegerLiteral, RealLiteral, Variable))


def _vars_in_expr(expr: Expression, result: Set[str]) -> None:
    if isinstance(expr, Variable):
        result.add(expr.name)
    elif isinstance(expr, ArrayRef):
        result.add(expr.name)
        for i in expr.indices:
            _vars_in_expr(i, result)
    elif isinstance(expr, BinaryOp):
        _vars_in_expr(expr.left, result)
        _vars_in_expr(expr.right, result)
    elif isinstance(expr, UnaryOp):
        _vars_in_expr(expr.operand, result)
    elif isinstance(expr, FunctionCall):
        for a in expr.args:
            _vars_in_expr(a, result)


class _CSEBlock:
    """Обрабатывает линейный блок операторов, накапливая CSE-кэш."""

    def __init__(self, counter: List[int]):
        self.counter = counter
        # key → (tmp_name, expr)
        self.cache: Dict[str, Tuple[str, Expression]] = {}
        # tmp_name → set of vars used in its expression
        self.tmp_vars: Dict[str, Set[str]] = {}
        self.result: List[Statement] = []
        self.new_assigns: List[Assignment] = []
        self.eliminated = 0

    def _invalidate(self, var_name: str) -> None:
        """Сбросить все CSE-записи, зависящие от переменной var_name."""
        keys_to_del = []
        for k, (tmp, expr) in self.cache.items():
            deps = set()
            _vars_in_expr(expr, deps)
            if var_name in deps:
                keys_to_del.append(k)
        for k in keys_to_del:
            del self.cache[k]

    def _subst(self, expr: Expression) -> Expression:
        """Подставить CSE в выражение рекурсивно."""
        if _is_trivial(expr) or not _is_pure(expr):
            return expr

        # Сначала рекурсивно обойти операнды
        if isinstance(expr, BinaryOp):
            new_left = self._subst(expr.left)
            new_right = self._subst(expr.right)
            if new_left is not expr.left or new_right is not expr.right:
                expr = dc_replace(expr, left=new_left, right=new_right)
        elif isinstance(expr, UnaryOp):
            new_op = self._subst(expr.operand)
            if new_op is not expr.operand:
                expr = dc_replace(expr, operand=new_op)
        elif isinstance(expr, FunctionCall):
            new_args = [self._subst(a) for a in expr.args]
            if any(na is not a for na, a in zip(new_args, expr.args)):
                expr = dc_replace(expr, args=new_args)

        # Проверить кэш
        if _is_trivial(expr):
            return expr
        key = _expr_key(expr)
        if key in self.cache:
            tmp_name, _ = self.cache[key]
            self.eliminated += 1
            return Variable(name=tmp_name, line=expr.line, col=expr.col)

        # Занести в кэш (если выражение достаточно «дорогое»)
        if isinstance(expr, (BinaryOp, FunctionCall)):
            self.counter[0] += 1
            tmp_name = f"_cse_{self.counter[0]}"
            deps: Set[str] = set()
            _vars_in_expr(expr, deps)
            self.cache[key] = (tmp_name, expr)
            self.tmp_vars[tmp_name] = deps
            # Добавить присваивание tmp перед текущей позицией
            assign = Assignment(
                target=tmp_name,
                value=expr,
                indices=[],
                stmt_label=None,
                line=expr.line,
                col=expr.col,
            )
            self.new_assigns.append(assign)
            return Variable(name=tmp_name, line=expr.line, col=expr.col)

        return expr

    def process(self, stmt: Statement) -> List[Statement]:
        """Обработать один оператор, вернуть список (с возможными новыми присваиваниями)."""
        self.new_assigns = []

        if isinstance(stmt, Assignment):
            if not stmt.indices:
                new_val = self._subst(stmt.value)
                new_stmt = dc_replace(stmt, value=new_val)
                self._invalidate(stmt.target)
            else:
                new_val = self._subst(stmt.value)
                new_idx = [self._subst(i) for i in stmt.indices]
                new_stmt = dc_replace(stmt, value=new_val, indices=new_idx)
                self._invalidate(stmt.target)
        elif isinstance(stmt, PrintStatement):
            new_items = [self._subst(i) for i in stmt.items]
            new_stmt = dc_replace(stmt, items=new_items)
        elif isinstance(stmt, WriteStatement):
            new_items = [self._subst(i) for i in stmt.items]
            new_stmt = dc_replace(stmt, items=new_items)
        elif isinstance(stmt, CallStatement):
            new_args = [self._subst(a) for a in stmt.args]
            new_stmt = dc_replace(stmt, args=new_args)
            # Вызов подпрограммы может изменить любую переменную — сброс кэша
            self.cache.clear()
        else:
            # Циклы, условия — сбрасываем кэш (консервативно)
            self.cache.clear()
            new_stmt = stmt

        return self.new_assigns + [new_stmt]


def _apply_cse_to_stmts(stmts: List[Statement], counter: List[int]) -> List[Statement]:
    """Применить CSE к плоскому списку операторов."""
    block = _CSEBlock(counter)
    result = []
    for stmt in stmts:
        # Для циклов и условий — рекурсия внутрь, но сбрасываем кэш
        if isinstance(stmt, (DoLoop, LabeledDoLoop)):
            block.cache.clear()
            new_body = _apply_cse_to_stmts(stmt.body, counter)
            stmt = dc_replace(stmt, body=new_body)
            result.append(stmt)
        elif isinstance(stmt, (DoWhile, LabeledDoWhile)):
            block.cache.clear()
            new_body = _apply_cse_to_stmts(stmt.body, counter)
            stmt = dc_replace(stmt, body=new_body)
            result.append(stmt)
        elif isinstance(stmt, IfStatement):
            block.cache.clear()
            new_then = _apply_cse_to_stmts(stmt.then_body, counter)
            new_elif = [(c, _apply_cse_to_stmts(b, counter)) for c, b in stmt.elif_parts]
            new_else = (_apply_cse_to_stmts(stmt.else_body, counter)
                        if stmt.else_body else stmt.else_body)
            stmt = dc_replace(stmt, then_body=new_then, elif_parts=new_elif, else_body=new_else)
            result.append(stmt)
        else:
            result.extend(block.process(stmt))
    return result


_GLOBAL_CSE_COUNTER = [0]


class CommonSubexpressionElimination(ASTOptimizationPass):
    """Устраняет повторяющиеся вычисления одинаковых выражений."""

    name = "CommonSubexpressionElimination"

    def run(self, program: Program) -> Program:
        counter = _GLOBAL_CSE_COUNTER
        before = counter[0]

        new_stmts = _apply_cse_to_stmts(program.statements, counter)
        new_subs = [
            dc_replace(s, statements=_apply_cse_to_stmts(s.statements, counter))
            for s in program.subroutines
        ]
        new_funcs = [
            dc_replace(f, statements=_apply_cse_to_stmts(f.statements, counter))
            for f in program.functions
        ]
        self.stats = {"cse_vars": counter[0] - before}
        return dc_replace(program, statements=new_stmts,
                          subroutines=new_subs, functions=new_funcs)
