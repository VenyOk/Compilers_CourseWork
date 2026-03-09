"""Свёртка констант (Constant Folding).

Вычисляет значения бинарных и унарных выражений, операнды которых —
константы, на этапе компиляции. Заменяет такие выражения литералами.

Примеры:
    (3 + 4) * (10 - 2)  →  56
    2.0 ** 3             →  8.0
    .NOT. .TRUE.         →  .FALSE.
"""
from __future__ import annotations
import math
from dataclasses import replace
from typing import Optional
from src.core import (
    Program, Expression, BinaryOp, UnaryOp,
    IntegerLiteral, RealLiteral, LogicalLiteral,
)
from src.optimizations.base import ASTOptimizationPass


def _as_number(expr: Expression) -> Optional[float | int]:
    """Вернуть числовое значение, если выражение — числовая константа."""
    if isinstance(expr, IntegerLiteral):
        return expr.value
    if isinstance(expr, RealLiteral):
        return expr.value
    return None


def _is_const(expr: Expression) -> bool:
    return isinstance(expr, (IntegerLiteral, RealLiteral, LogicalLiteral))


def _fold_binop(expr: BinaryOp) -> Optional[Expression]:
    """Попытаться свернуть BinaryOp в константу. Возвращает None при неудаче."""
    op = expr.op
    left, right = expr.left, expr.right

    # Логические операции
    if isinstance(left, LogicalLiteral) and isinstance(right, LogicalLiteral):
        lv, rv = left.value, right.value
        if op == '.AND.':
            return LogicalLiteral(value=lv and rv, line=left.line, col=left.col)
        if op == '.OR.':
            return LogicalLiteral(value=lv or rv, line=left.line, col=left.col)
        if op in ('.EQV.', '.EQV'):
            return LogicalLiteral(value=lv == rv, line=left.line, col=left.col)
        if op in ('.NEQV.', '.NEQV'):
            return LogicalLiteral(value=lv != rv, line=left.line, col=left.col)

    lv = _as_number(left)
    rv = _as_number(right)
    if lv is None or rv is None:
        return None

    line, col = left.line, left.col
    use_float = isinstance(left, RealLiteral) or isinstance(right, RealLiteral)

    def _int(v):
        return IntegerLiteral(value=int(v), line=line, col=col)

    def _real(v):
        return RealLiteral(value=float(v), line=line, col=col)

    def _logical(v):
        return LogicalLiteral(value=bool(v), line=line, col=col)

    def _num(v):
        return _real(v) if use_float else _int(int(v))

    try:
        if op == '+':
            return _num(lv + rv)
        if op == '-':
            return _num(lv - rv)
        if op == '*':
            return _num(lv * rv)
        if op == '/':
            if rv == 0:
                return None
            if use_float:
                return _real(lv / rv)
            return _int(int(lv) // int(rv))
        if op == '**':
            result = lv ** rv
            if use_float or isinstance(rv, float) or (isinstance(rv, int) and rv < 0):
                return _real(float(result))
            return _int(int(result))
        if op in ('.EQ.', '=='):
            return _logical(lv == rv)
        if op in ('.NE.', '/='):
            return _logical(lv != rv)
        if op in ('.LT.', '<'):
            return _logical(lv < rv)
        if op in ('.LE.', '<='):
            return _logical(lv <= rv)
        if op in ('.GT.', '>'):
            return _logical(lv > rv)
        if op in ('.GE.', '>='):
            return _logical(lv >= rv)
    except (ArithmeticError, ValueError, OverflowError):
        pass
    return None


def _fold_unaryop(expr: UnaryOp) -> Optional[Expression]:
    """Попытаться свернуть UnaryOp."""
    op = expr.op
    operand = expr.operand
    line, col = operand.line, operand.col

    if op in ('.NOT.', 'NOT') and isinstance(operand, LogicalLiteral):
        return LogicalLiteral(value=not operand.value, line=line, col=col)

    v = _as_number(operand)
    if v is None:
        return None

    if op == '-':
        if isinstance(operand, RealLiteral):
            return RealLiteral(value=-v, line=line, col=col)
        return IntegerLiteral(value=-int(v), line=line, col=col)
    if op == '+':
        return operand
    return None


class ConstantFolding(ASTOptimizationPass):
    """Вычисляет значения константных выражений на этапе компиляции."""

    name = "ConstantFolding"

    def run(self, program: Program) -> Program:
        self._folded = 0
        from dataclasses import replace as dc_replace

        new_stmts = self.transform_stmts(program.statements)
        new_subs = [
            dc_replace(s, statements=self.transform_stmts(s.statements))
            for s in program.subroutines
        ]
        new_funcs = [
            dc_replace(f, statements=self.transform_stmts(f.statements))
            for f in program.functions
        ]
        self.stats = {"folded": self._folded}
        return dc_replace(program, statements=new_stmts,
                          subroutines=new_subs, functions=new_funcs)

    def transform_expr(self, expr: Expression) -> Expression:
        # Сначала рекурсивно обойти дочерние узлы
        expr = super().transform_expr(expr)

        if isinstance(expr, BinaryOp):
            result = _fold_binop(expr)
            if result is not None:
                self._folded += 1
                return result

        elif isinstance(expr, UnaryOp):
            result = _fold_unaryop(expr)
            if result is not None:
                self._folded += 1
                return result

        return expr
