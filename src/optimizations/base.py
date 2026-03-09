from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List


class ASTOptimizationPass(ABC):
    name: str = "UnnamedPass"

    def __init__(self):
        self.stats: dict = {}

    @abstractmethod
    def run(self, program) -> object:
        ...

    def transformExpr(self, expr):
        from src.core import BinaryOp, UnaryOp, FunctionCall, ArrayRef
        from dataclasses import replace
        if isinstance(expr, BinaryOp):
            nl = self.transformExpr(expr.left)
            nr = self.transformExpr(expr.right)
            if nl is not expr.left or nr is not expr.right:
                expr = replace(expr, left=nl, right=nr)
        elif isinstance(expr, UnaryOp):
            no = self.transformExpr(expr.operand)
            if no is not expr.operand:
                expr = replace(expr, operand=no)
        elif isinstance(expr, FunctionCall):
            na = [self.transformExpr(a) for a in expr.args]
            if any(x is not y for x, y in zip(na, expr.args)):
                expr = replace(expr, args=na)
        elif isinstance(expr, ArrayRef):
            ni = [self.transformExpr(i) for i in expr.indices]
            if any(x is not y for x, y in zip(ni, expr.indices)):
                expr = replace(expr, indices=ni)
        return expr

    def transformStmt(self, stmt):
        from src.core import (
            Assignment, DoLoop, LabeledDoLoop, DoWhile, LabeledDoWhile,
            IfStatement, SimpleIfStatement,
        )
        from dataclasses import replace
        if isinstance(stmt, Assignment):
            nv = self.transformExpr(stmt.value)
            ni = [self.transformExpr(i) for i in stmt.indices]
            if nv is not stmt.value or any(x is not y for x, y in zip(ni, stmt.indices)):
                stmt = replace(stmt, value=nv, indices=ni)
        elif isinstance(stmt, (DoLoop, LabeledDoLoop)):
            ns = self.transformExpr(stmt.start)
            ne = self.transformExpr(stmt.end)
            nst = self.transformExpr(stmt.step) if stmt.step else stmt.step
            nb = self.transformStmts(stmt.body)
            if ns is not stmt.start or ne is not stmt.end or nst is not stmt.step or nb is not stmt.body:
                stmt = replace(stmt, start=ns, end=ne, step=nst, body=nb)
        elif isinstance(stmt, (DoWhile, LabeledDoWhile)):
            nc = self.transformExpr(stmt.condition)
            nb = self.transformStmts(stmt.body)
            if nc is not stmt.condition or nb is not stmt.body:
                stmt = replace(stmt, condition=nc, body=nb)
        elif isinstance(stmt, IfStatement):
            nc = self.transformExpr(stmt.condition)
            nt = self.transformStmts(stmt.then_body)
            nel = [(self.transformExpr(c), self.transformStmts(b)) for c, b in stmt.elif_parts]
            ne = self.transformStmts(stmt.else_body) if stmt.else_body else stmt.else_body
            stmt = replace(stmt, condition=nc, then_body=nt, elif_parts=nel, else_body=ne)
        elif isinstance(stmt, SimpleIfStatement):
            nc = self.transformExpr(stmt.condition)
            ns = self.transformStmt(stmt.statement)
            if nc is not stmt.condition or ns is not stmt.statement:
                stmt = replace(stmt, condition=nc, statement=ns)
        return stmt

    def transformStmts(self, stmts: List) -> List:
        result = []
        changed = False
        for s in stmts:
            ns = self.transformStmt(s)
            result.append(ns)
            if ns is not s:
                changed = True
        return result if changed else stmts
