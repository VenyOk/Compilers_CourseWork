"""Распространение констант (Constant Propagation).

Отслеживает переменные, которым присвоены константные значения,
и заменяет их использования соответствующими литералами.

Пример:
    SCALE = 2.5
    X = FLOAT(I) * SCALE    →    X = FLOAT(I) * 2.5
    N = 10
    DO I = 1, N              →    DO I = 1, 10
"""
from __future__ import annotations
from typing import Dict, Optional
from dataclasses import replace as dc_replace

from src.core import (
    Program, Statement, Expression,
    Assignment, DoLoop, LabeledDoLoop, DoWhile, LabeledDoWhile,
    IfStatement, SimpleIfStatement, Variable,
    IntegerLiteral, RealLiteral, LogicalLiteral,
)
from src.optimizations.base import ASTOptimizationPass

Literal = (IntegerLiteral, RealLiteral, LogicalLiteral)


class ConstantPropagation(ASTOptimizationPass):
    """Заменяет переменные-константы их значениями в выражениях."""

    name = "ConstantPropagation"

    def run(self, program: Program) -> Program:
        self._propagated = 0
        new_stmts = self._propagate_stmts(program.statements, {})
        new_subs = [
            dc_replace(s, statements=self._propagate_stmts(s.statements, {}))
            for s in program.subroutines
        ]
        new_funcs = [
            dc_replace(f, statements=self._propagate_stmts(f.statements, {}))
            for f in program.functions
        ]
        self.stats = {"propagated": self._propagated}
        return dc_replace(program, statements=new_stmts,
                          subroutines=new_subs, functions=new_funcs)

    def _propagate_stmts(self, stmts: list, env: Dict[str, Expression]) -> list:
        """Обойти список операторов, отслеживая константы в env."""
        result = []
        for stmt in stmts:
            new_stmt, env = self._propagate_stmt(stmt, env)
            result.append(new_stmt)
        return result

    def _propagate_stmt(self, stmt: Statement,
                        env: Dict[str, Expression]):
        """Обработать один оператор, вернуть (новый_stmt, обновлённый_env)."""
        if isinstance(stmt, Assignment) and not stmt.indices:
            # Сначала подставляем константы в правую часть
            new_val = self._subst_expr(stmt.value, env)
            new_stmt = dc_replace(stmt, value=new_val)
            # Если RHS стал константой — записываем в env
            if isinstance(new_val, Literal):
                env = {**env, stmt.target: new_val}
            else:
                # Если переменной присвоено не-константное значение — сбрасываем
                env = {k: v for k, v in env.items() if k != stmt.target}
            return new_stmt, env

        elif isinstance(stmt, Assignment) and stmt.indices:
            new_val = self._subst_expr(stmt.value, env)
            new_idx = [self._subst_expr(i, env) for i in stmt.indices]
            return dc_replace(stmt, value=new_val, indices=new_idx), env

        elif isinstance(stmt, (DoLoop, LabeledDoLoop)):
            new_start = self._subst_expr(stmt.start, env)
            new_end = self._subst_expr(stmt.end, env)
            new_step = self._subst_expr(stmt.step, env) if stmt.step else stmt.step
            # В теле цикла переменная цикла может меняться — убираем из env
            loop_env = {k: v for k, v in env.items() if k != stmt.var}
            new_body = self._propagate_stmts(stmt.body, loop_env)
            return dc_replace(stmt, start=new_start, end=new_end,
                               step=new_step, body=new_body), env

        elif isinstance(stmt, (DoWhile, LabeledDoWhile)):
            new_cond = self._subst_expr(stmt.condition, env)
            new_body = self._propagate_stmts(stmt.body, env)
            return dc_replace(stmt, condition=new_cond, body=new_body), env

        elif isinstance(stmt, IfStatement):
            new_cond = self._subst_expr(stmt.condition, env)
            new_then = self._propagate_stmts(stmt.then_body, dict(env))
            new_elif = [
                (self._subst_expr(c, env), self._propagate_stmts(b, dict(env)))
                for c, b in stmt.elif_parts
            ]
            new_else = (self._propagate_stmts(stmt.else_body, dict(env))
                        if stmt.else_body else stmt.else_body)
            # После if — сбрасываем константы, изменённые в любой ветке
            written = _vars_written_in(stmt)
            env = {k: v for k, v in env.items() if k not in written}
            return dc_replace(stmt, condition=new_cond, then_body=new_then,
                               elif_parts=new_elif, else_body=new_else), env

        elif isinstance(stmt, SimpleIfStatement):
            new_cond = self._subst_expr(stmt.condition, env)
            new_s, env = self._propagate_stmt(stmt.statement, env)
            return dc_replace(stmt, condition=new_cond, statement=new_s), env

        else:
            # Другие операторы (print, call, goto, ...) — только подставляем
            new_stmt = self.transform_stmt(stmt)
            return new_stmt, env

    def _subst_expr(self, expr: Expression, env: Dict[str, Expression]) -> Expression:
        """Подставить константы из env в выражение."""
        if isinstance(expr, Variable) and expr.name in env:
            self._propagated += 1
            return env[expr.name]
        return self.transform_expr_with_env(expr, env)

    def transform_expr_with_env(self, expr: Expression,
                                 env: Dict[str, Expression]) -> Expression:
        """Рекурсивный обход выражения с подстановкой констант."""
        from src.core import BinaryOp, UnaryOp, FunctionCall, ArrayRef
        from dataclasses import replace

        if isinstance(expr, Variable) and expr.name in env:
            self._propagated += 1
            return env[expr.name]

        if isinstance(expr, BinaryOp):
            new_left = self.transform_expr_with_env(expr.left, env)
            new_right = self.transform_expr_with_env(expr.right, env)
            if new_left is not expr.left or new_right is not expr.right:
                expr = replace(expr, left=new_left, right=new_right)
        elif isinstance(expr, UnaryOp):
            new_op = self.transform_expr_with_env(expr.operand, env)
            if new_op is not expr.operand:
                expr = replace(expr, operand=new_op)
        elif isinstance(expr, FunctionCall):
            new_args = [self.transform_expr_with_env(a, env) for a in expr.args]
            if any(na is not a for na, a in zip(new_args, expr.args)):
                expr = replace(expr, args=new_args)
        elif isinstance(expr, ArrayRef):
            new_indices = [self.transform_expr_with_env(i, env) for i in expr.indices]
            if any(ni is not i for ni, i in zip(new_indices, expr.indices)):
                expr = replace(expr, indices=new_indices)
        return expr


def _vars_written_in(stmt: Statement) -> set:
    """Вернуть множество переменных, записываемых в операторе (без рекурсии в петли)."""
    result = set()
    if isinstance(stmt, Assignment):
        result.add(stmt.target)
    elif isinstance(stmt, IfStatement):
        for s in stmt.then_body:
            result |= _vars_written_in(s)
        for _, body in stmt.elif_parts:
            for s in body:
                result |= _vars_written_in(s)
        if stmt.else_body:
            for s in stmt.else_body:
                result |= _vars_written_in(s)
    elif isinstance(stmt, (DoLoop, LabeledDoLoop)):
        result.add(stmt.var)
        for s in stmt.body:
            result |= _vars_written_in(s)
    elif isinstance(stmt, (DoWhile, LabeledDoWhile)):
        for s in stmt.body:
            result |= _vars_written_in(s)
    return result
