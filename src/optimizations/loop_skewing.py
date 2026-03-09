"""Скашивание циклов (Loop Skewing).

Применяется к гнёздам циклов с отрицательными зависимостями (например,
стенсилы Гаусса-Зейделя), где тайлинг напрямую недопустим.

Идея (Метелица 2024, стр. 70–73; Wolf & Lam 1991):
    Заменить переменную внутреннего цикла J на J' = J + s*I (скашивание).
    При достаточно большом s все расстояния зависимостей становятся
    неотрицательными, что разрешает последующий тайлинг.

Трансформация для 2D-гнезда со скашиванием s по J:

    ! До:
    DO I = 1, N
        DO J = 1, N
            u(I,J) = u(I-1,J) + u(I,J-1) + ...   ! обратная зависимость по I
        ENDDO
    ENDDO

    ! После (J' = J + s*I):
    DO I = 1, N
        DO Jp = I*s + 1, I*s + N    ! Jp = J + s*I
            u(I, Jp - s*I) = u(I-1, Jp-1 - s*(I-1)) + ...
        ENDDO
    ENDDO

Примечание: трансформация расширяет границы внутреннего цикла. Для
последующего тайлинга используется метод гиперплоскостей (Метелица §4).

После скашивания проход LoopTiling должен быть применён повторно.
"""
from __future__ import annotations
from typing import List, Optional, Dict
from dataclasses import replace as dc_replace

from src.core import (
    Program, Statement, Expression,
    DoLoop, LabeledDoLoop,
    Variable, IntegerLiteral, RealLiteral, BinaryOp,
    Assignment, ArrayRef, FunctionCall, UnaryOp,
)
from src.optimizations.base import ASTOptimizationPass
from src.optimizations.loop_analysis import (
    LoopNest, _build_nest, needs_skewing, get_skew_factors,
)


# ---------------------------------------------------------------------------
# Подстановка скашивания в выражения
# ---------------------------------------------------------------------------

class _SkewSubstituter:
    """Подставляет замену переменных: J → J' - s*I (обратная замена в теле)."""

    def __init__(self, skew_var: str, outer_var: str, skew_factor: int):
        """
        skew_var    — переменная, которую заменяем (J)
        outer_var   — внешняя переменная (I)
        skew_factor — коэффициент скашивания s
        """
        self.skew_var = skew_var
        self.outer_var = outer_var
        self.skew_factor = skew_factor

    def subst(self, expr: Expression) -> Expression:
        """Заменить skew_var на (skew_var' - skew_factor * outer_var)."""
        if isinstance(expr, Variable):
            if expr.name == self.skew_var:
                # J → Jp - s*I
                outer = Variable(name=self.outer_var, line=expr.line, col=expr.col)
                jp = Variable(name=self.skew_var + 'p', line=expr.line, col=expr.col)
                if self.skew_factor == 0:
                    return jp
                s_expr = IntegerLiteral(value=self.skew_factor, line=expr.line, col=expr.col)
                s_i = BinaryOp(left=s_expr, op='*', right=outer, line=expr.line, col=expr.col)
                return BinaryOp(left=jp, op='-', right=s_i, line=expr.line, col=expr.col)
            return expr

        if isinstance(expr, BinaryOp):
            return dc_replace(expr,
                               left=self.subst(expr.left),
                               right=self.subst(expr.right))
        if isinstance(expr, UnaryOp):
            return dc_replace(expr, operand=self.subst(expr.operand))
        if isinstance(expr, FunctionCall):
            return dc_replace(expr, args=[self.subst(a) for a in expr.args])
        if isinstance(expr, ArrayRef):
            return dc_replace(expr, indices=[self.subst(i) for i in expr.indices])
        return expr

    def subst_stmt(self, stmt: Statement) -> Statement:
        if isinstance(stmt, Assignment):
            new_val = self.subst(stmt.value)
            new_idx = [self.subst(i) for i in stmt.indices]
            return dc_replace(stmt, value=new_val, indices=new_idx)
        if isinstance(stmt, (DoLoop, LabeledDoLoop)):
            new_start = self.subst(stmt.start)
            new_end = self.subst(stmt.end)
            new_step = self.subst(stmt.step) if stmt.step else stmt.step
            new_body = [self.subst_stmt(s) for s in stmt.body]
            return dc_replace(stmt, start=new_start, end=new_end,
                               step=new_step, body=new_body)
        return stmt


def _make_expr(outer_var: str, outer_mult: int, const: int,
               line: int, col: int) -> Expression:
    """Построить выражение outer_mult * outer_var + const."""
    if outer_mult == 0:
        return IntegerLiteral(value=const, line=line, col=col)
    outer = Variable(name=outer_var, line=line, col=col)
    if outer_mult == 1:
        term = outer
    else:
        s_expr = IntegerLiteral(value=outer_mult, line=line, col=col)
        term = BinaryOp(left=s_expr, op='*', right=outer, line=line, col=col)
    if const == 0:
        return term
    c_expr = IntegerLiteral(value=abs(const), line=line, col=col)
    op = '+' if const > 0 else '-'
    return BinaryOp(left=term, op=op, right=c_expr, line=line, col=col)


def _skew_nest(nest: LoopNest, factors: List[int]) -> Statement:
    """Применить скашивание к совершенному гнезду глубиной 2.

    Поддерживает только 2D-гнездо (outer I, inner J).
    factors[0] всегда 0 (внешний цикл не скашивается).
    factors[1] = s (коэффициент скашивания J' = J + s*I).
    """
    assert nest.depth >= 2
    outer_li = nest.loops[0]
    inner_li = nest.loops[1]
    body = nest.body
    line, col = outer_li.node.line, outer_li.node.col
    s = factors[1]  # скашивающий коэффициент

    if s == 0:
        return outer_li.node  # ничего не изменилось

    # Имя новой переменной
    jp_name = inner_li.var + 'p'

    # Применить замену J → Jp - s*I в теле цикла
    subst = _SkewSubstituter(inner_li.var, outer_li.var, s)
    new_body = [subst.subst_stmt(stmt) for stmt in body]

    # Новые границы внутреннего цикла Jp:
    # Jp_start = s*I + J_start  =  s*I + 1  (если J начинается с 1)
    # Jp_end   = s*I + J_end    =  s*I + N

    # Извлечь константу из old start/end (если IntegerLiteral)
    def _extract_const(e: Expression, default: int) -> int:
        if isinstance(e, IntegerLiteral):
            return e.value
        return default

    j_start_const = _extract_const(inner_li.start, 1)
    j_end_const   = _extract_const(inner_li.end,   None)

    new_inner_start = _make_expr(outer_li.var, s, j_start_const, line, col)
    if j_end_const is not None:
        new_inner_end = _make_expr(outer_li.var, s, j_end_const, line, col)
    else:
        # Используем s*I + N (где N = end переменной)
        outer_v = Variable(name=outer_li.var, line=line, col=col)
        s_expr = IntegerLiteral(value=s, line=line, col=col)
        s_i = BinaryOp(left=s_expr, op='*', right=outer_v, line=line, col=col)
        new_inner_end = BinaryOp(left=s_i, op='+', right=inner_li.end, line=line, col=col)

    new_inner = DoLoop(
        var=jp_name,
        start=new_inner_start,
        end=new_inner_end,
        step=inner_li.step,
        body=new_body,
        stmt_label=None,
        line=line,
        col=col,
    )

    new_outer = dc_replace(outer_li.node, body=[new_inner])
    return new_outer


def _try_skew(loop: Statement, counter: List[int]) -> Statement:
    """Попытаться применить скашивание к циклу."""
    if not isinstance(loop, (DoLoop, LabeledDoLoop)):
        return loop

    nest = _build_nest(loop)

    if nest.depth >= 2 and needs_skewing(nest):
        factors = get_skew_factors(nest)
        if any(f > 0 for f in factors):
            skewed = _skew_nest(nest, factors)
            counter[0] += 1
            return skewed

    # Рекурсивно обработать тело
    new_body = [_try_skew(s, counter) for s in loop.body]
    return dc_replace(loop, body=new_body)


def _process_stmts(stmts: List[Statement], counter: List[int]) -> List[Statement]:
    result = []
    for stmt in stmts:
        if isinstance(stmt, (DoLoop, LabeledDoLoop)):
            result.append(_try_skew(stmt, counter))
        else:
            result.append(stmt)
    return result


class LoopSkewing(ASTOptimizationPass):
    """Применяет скашивание к гнёздам с отрицательными зависимостями.

    Подготавливает гнёзда типа Гаусса-Зейделя к последующему тайлингу
    (LoopTiling должен быть применён после LoopSkewing).
    """

    name = "LoopSkewing"

    def run(self, program: Program) -> Program:
        from src.core import Declaration
        counter = [0]
        new_stmts = _process_stmts(program.statements, counter)
        new_subs = [
            dc_replace(s, statements=_process_stmts(s.statements, counter))
            for s in program.subroutines
        ]
        new_funcs = [
            dc_replace(f, statements=_process_stmts(f.statements, counter))
            for f in program.functions
        ]

        # Добавить объявления для скошенных переменных (Jp, Kp, ...)
        skew_vars: set = set()
        self._collect_skew_vars(new_stmts, skew_vars)
        existing: set = {
            n for d in program.declarations
            if isinstance(d, Declaration)
            for n, _ in d.names
        }
        new_skew = [v for v in sorted(skew_vars) if v not in existing]
        new_decls = list(program.declarations)
        if new_skew:
            skew_decl = Declaration(
                type='INTEGER',
                names=[(v, None) for v in new_skew],
                line=0, col=0,
            )
            new_decls = [skew_decl] + new_decls

        self.stats = {"skewed": counter[0]}
        return dc_replace(program, statements=new_stmts,
                          declarations=new_decls,
                          subroutines=new_subs, functions=new_funcs)

    def _collect_skew_vars(self, stmts: List[Statement], result: set) -> None:
        """Собрать имена скошенных переменных (заканчивающихся на p)."""
        for stmt in stmts:
            if isinstance(stmt, (DoLoop, LabeledDoLoop)):
                if stmt.var.endswith('p') and len(stmt.var) >= 2:
                    result.add(stmt.var)
                self._collect_skew_vars(stmt.body, result)
