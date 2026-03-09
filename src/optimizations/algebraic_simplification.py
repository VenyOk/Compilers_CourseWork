"""Алгебраические упрощения (Algebraic Simplification).

Применяет тождества:
    x + 0  →  x       0 + x  →  x
    x - 0  →  x       x - x  →  0
    x * 1  →  x       1 * x  →  x
    x * 0  →  0       0 * x  →  0
    x / 1  →  x
    x ** 1 →  x       x ** 0  →  1
    x ** 2 →  уже в strength_reduction (здесь не трогаем)
    .NOT. .NOT. x → x
"""
from __future__ import annotations
from dataclasses import replace as dc_replace
from src.core import (
    Program, Expression, BinaryOp, UnaryOp,
    IntegerLiteral, RealLiteral, LogicalLiteral,
)
from src.optimizations.base import ASTOptimizationPass


def _is_zero(e: Expression) -> bool:
    return (isinstance(e, IntegerLiteral) and e.value == 0) or \
           (isinstance(e, RealLiteral) and e.value == 0.0)


def _is_one(e: Expression) -> bool:
    return (isinstance(e, IntegerLiteral) and e.value == 1) or \
           (isinstance(e, RealLiteral) and e.value == 1.0)


def _zero_like(e: Expression) -> Expression:
    if isinstance(e, RealLiteral):
        return RealLiteral(value=0.0, line=e.line, col=e.col)
    return IntegerLiteral(value=0, line=e.line, col=e.col)


def _one_like(e: Expression) -> Expression:
    if isinstance(e, RealLiteral):
        return RealLiteral(value=1.0, line=e.line, col=e.col)
    return IntegerLiteral(value=1, line=e.line, col=e.col)


def _same_var(a: Expression, b: Expression) -> bool:
    from src.core import Variable
    return (isinstance(a, type(b)) and isinstance(a, (IntegerLiteral, RealLiteral))
            and a.value == b.value) or \
           (type(a) is type(b) and hasattr(a, 'name') and a.name == b.name)


class AlgebraicSimplification(ASTOptimizationPass):
    """Упрощает выражения с помощью алгебраических тождеств."""

    name = "AlgebraicSimplification"

    def run(self, program: Program) -> Program:
        self._simplified = 0
        new_stmts = self.transform_stmts(program.statements)
        new_subs = [
            dc_replace(s, statements=self.transform_stmts(s.statements))
            for s in program.subroutines
        ]
        new_funcs = [
            dc_replace(f, statements=self.transform_stmts(f.statements))
            for f in program.functions
        ]
        self.stats = {"simplified": self._simplified}
        return dc_replace(program, statements=new_stmts,
                          subroutines=new_subs, functions=new_funcs)

    def transform_expr(self, expr: Expression) -> Expression:
        expr = super().transform_expr(expr)  # рекурсивный спуск

        if not isinstance(expr, BinaryOp):
            if isinstance(expr, UnaryOp) and expr.op in ('.NOT.', 'NOT'):
                if isinstance(expr.operand, UnaryOp) and expr.operand.op in ('.NOT.', 'NOT'):
                    self._simplified += 1
                    return expr.operand.operand
            return expr

        op = expr.op
        left, right = expr.left, expr.right

        if op in ('+',):
            if _is_zero(right):
                self._simplified += 1
                return left
            if _is_zero(left):
                self._simplified += 1
                return right

        elif op in ('-',):
            if _is_zero(right):
                self._simplified += 1
                return left
            if _same_var(left, right):
                self._simplified += 1
                return _zero_like(left)

        elif op in ('*',):
            if _is_one(right):
                self._simplified += 1
                return left
            if _is_one(left):
                self._simplified += 1
                return right
            if _is_zero(right):
                self._simplified += 1
                return _zero_like(left)
            if _is_zero(left):
                self._simplified += 1
                return _zero_like(right)

        elif op in ('/',):
            if _is_one(right):
                self._simplified += 1
                return left

        elif op == '**':
            if _is_one(right):
                self._simplified += 1
                return left
            if _is_zero(right):
                self._simplified += 1
                return _one_like(left)

        return expr
