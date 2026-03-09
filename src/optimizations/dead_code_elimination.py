"""Удаление мёртвого кода (Dead Code Elimination).

Удаляет присваивания скалярным переменным, значения которых никогда
не используются после присваивания.

Алгоритм: два прохода —
1. Сбор множества «используемых» переменных (читается хотя бы один раз
   после присваивания, или используется в print/call/условии/индексе).
2. Удаление присваиваний, чья LHS-переменная не входит в «используемые».

Примечание: массивы и переменные циклов никогда не удаляются.
"""
from __future__ import annotations
from typing import List, Set
from dataclasses import replace as dc_replace

from src.core import (
    Program, Statement, Expression,
    Assignment, DoLoop, LabeledDoLoop, DoWhile, LabeledDoWhile,
    IfStatement, SimpleIfStatement,
    PrintStatement, WriteStatement, ReadStatement, CallStatement,
    Variable, ArrayRef, FunctionCall, BinaryOp, UnaryOp,
    ReturnStatement, StopStatement, GotoStatement,
)
from src.optimizations.base import ASTOptimizationPass


def _collect_read_vars(expr: Expression, result: Set[str]) -> None:
    """Собрать все переменные, читаемые в выражении."""
    if isinstance(expr, Variable):
        result.add(expr.name)
    elif isinstance(expr, ArrayRef):
        result.add(expr.name)
        for i in expr.indices:
            _collect_read_vars(i, result)
    elif isinstance(expr, BinaryOp):
        _collect_read_vars(expr.left, result)
        _collect_read_vars(expr.right, result)
    elif isinstance(expr, UnaryOp):
        _collect_read_vars(expr.operand, result)
    elif isinstance(expr, FunctionCall):
        for a in expr.args:
            _collect_read_vars(a, result)


def _collect_used_in_stmts(stmts: List[Statement]) -> Set[str]:
    """Вернуть все переменные, читаемые в списке операторов."""
    used: Set[str] = set()
    for stmt in stmts:
        _collect_used_in_stmt(stmt, used)
    return used


def _collect_used_in_stmt(stmt: Statement, used: Set[str]) -> None:
    if isinstance(stmt, Assignment):
        _collect_read_vars(stmt.value, used)
        for i in stmt.indices:
            _collect_read_vars(i, used)
        if stmt.indices:
            used.add(stmt.target)
    elif isinstance(stmt, (DoLoop, LabeledDoLoop)):
        _collect_read_vars(stmt.start, used)
        _collect_read_vars(stmt.end, used)
        if stmt.step:
            _collect_read_vars(stmt.step, used)
        for s in stmt.body:
            _collect_used_in_stmt(s, used)
    elif isinstance(stmt, (DoWhile, LabeledDoWhile)):
        _collect_read_vars(stmt.condition, used)
        for s in stmt.body:
            _collect_used_in_stmt(s, used)
    elif isinstance(stmt, IfStatement):
        _collect_read_vars(stmt.condition, used)
        for s in stmt.then_body:
            _collect_used_in_stmt(s, used)
        for c, b in stmt.elif_parts:
            _collect_read_vars(c, used)
            for s in b:
                _collect_used_in_stmt(s, used)
        if stmt.else_body:
            for s in stmt.else_body:
                _collect_used_in_stmt(s, used)
    elif isinstance(stmt, SimpleIfStatement):
        _collect_read_vars(stmt.condition, used)
        _collect_used_in_stmt(stmt.statement, used)
    elif isinstance(stmt, PrintStatement):
        for item in stmt.items:
            _collect_read_vars(item, used)
    elif isinstance(stmt, WriteStatement):
        for item in stmt.items:
            _collect_read_vars(item, used)
    elif isinstance(stmt, ReadStatement):
        for name in stmt.items:
            if isinstance(name, str):
                used.add(name)
            elif isinstance(name, Expression):
                _collect_read_vars(name, used)
    elif isinstance(stmt, CallStatement):
        for a in stmt.args:
            _collect_read_vars(a, used)
    elif isinstance(stmt, ReturnStatement):
        pass


def _elim_stmts(stmts: List[Statement], used: Set[str]) -> List[Statement]:
    """Удалить мёртвые присваивания скалярам из списка."""
    result = []
    for stmt in stmts:
        new_stmt = _elim_stmt(stmt, used)
        if new_stmt is not None:
            result.append(new_stmt)
    return result


def _elim_stmt(stmt: Statement, used: Set[str]):
    if isinstance(stmt, Assignment) and not stmt.indices:
        if stmt.target not in used:
            return None  # мёртвое присваивание
        return stmt
    elif isinstance(stmt, (DoLoop, LabeledDoLoop)):
        new_body = _elim_stmts(stmt.body, used)
        return dc_replace(stmt, body=new_body)
    elif isinstance(stmt, (DoWhile, LabeledDoWhile)):
        new_body = _elim_stmts(stmt.body, used)
        return dc_replace(stmt, body=new_body)
    elif isinstance(stmt, IfStatement):
        new_then = _elim_stmts(stmt.then_body, used)
        new_elif = [(c, _elim_stmts(b, used)) for c, b in stmt.elif_parts]
        new_else = _elim_stmts(stmt.else_body, used) if stmt.else_body else stmt.else_body
        return dc_replace(stmt, then_body=new_then, elif_parts=new_elif, else_body=new_else)
    elif isinstance(stmt, SimpleIfStatement):
        new_s = _elim_stmt(stmt.statement, used)
        if new_s is None:
            return None
        return dc_replace(stmt, statement=new_s)
    return stmt


class DeadCodeElimination(ASTOptimizationPass):
    """Удаляет присваивания скалярам, результаты которых не используются."""

    name = "DeadCodeElimination"

    def run(self, program: Program) -> Program:
        self._eliminated = 0

        def process(stmts):
            used = _collect_used_in_stmts(stmts)
            result = _elim_stmts(stmts, used)
            self._eliminated += len(stmts) - len(result)
            return result

        new_stmts = process(program.statements)
        new_subs = [
            dc_replace(s, statements=process(s.statements))
            for s in program.subroutines
        ]
        new_funcs = [
            dc_replace(f, statements=process(f.statements))
            for f in program.functions
        ]
        self.stats = {"eliminated": self._eliminated}
        return dc_replace(program, statements=new_stmts,
                          subroutines=new_subs, functions=new_funcs)
