"""Снижение стоимости операций (Strength Reduction).

Заменяет дорогие операции на эквивалентные более дешёвые:
    x ** 2  →  x * x
    x ** 3  →  x * x * x
    x * 2   →  x + x
    x * 2.0 →  x + x
"""
from __future__ import annotations
from dataclasses import replace as dc_replace
from src.core import (
    Program, Expression, BinaryOp,
    IntegerLiteral, RealLiteral,
)
from src.optimizations.base import ASTOptimizationPass


class StrengthReduction(ASTOptimizationPass):
    """Заменяет x**2→x*x, x**3→x*x*x, x*2→x+x."""

    name = "StrengthReduction"

    def run(self, program: Program) -> Program:
        self._reduced = 0
        new_stmts = self.transform_stmts(program.statements)
        new_subs = [
            dc_replace(s, statements=self.transform_stmts(s.statements))
            for s in program.subroutines
        ]
        new_funcs = [
            dc_replace(f, statements=self.transform_stmts(f.statements))
            for f in program.functions
        ]
        self.stats = {"reduced": self._reduced}
        return dc_replace(program, statements=new_stmts,
                          subroutines=new_subs, functions=new_funcs)

    def transform_expr(self, expr: Expression) -> Expression:
        expr = super().transform_expr(expr)

        if not isinstance(expr, BinaryOp):
            return expr

        op = expr.op
        left, right = expr.left, expr.right
        line, col = left.line, left.col

        # x ** 2  →  x * x
        if op == '**' and isinstance(right, (IntegerLiteral, RealLiteral)):
            if right.value == 2:
                self._reduced += 1
                return BinaryOp(left=left, op='*', right=left, line=line, col=col)
            # x ** 3  →  x * x * x
            if right.value == 3:
                self._reduced += 1
                sq = BinaryOp(left=left, op='*', right=left, line=line, col=col)
                return BinaryOp(left=sq, op='*', right=left, line=line, col=col)

        # x * 2  →  x + x  (только для целых)
        if op == '*':
            if isinstance(right, IntegerLiteral) and right.value == 2:
                self._reduced += 1
                return BinaryOp(left=left, op='+', right=left, line=line, col=col)
            if isinstance(left, IntegerLiteral) and left.value == 2:
                self._reduced += 1
                return BinaryOp(left=right, op='+', right=right, line=line, col=col)

        return expr
