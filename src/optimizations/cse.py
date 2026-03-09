from __future__ import annotations
from typing import List, Dict, Set, Tuple
from dataclasses import replace as dcReplace

from src.core import (
    Program, Statement, Expression,
    Assignment, DoLoop, LabeledDoLoop, DoWhile, LabeledDoWhile,
    IfStatement, SimpleIfStatement,
    Variable, BinaryOp, UnaryOp, FunctionCall, ArrayRef,
    IntegerLiteral, RealLiteral,
    PrintStatement, WriteStatement, CallStatement,
)
from src.optimizations.base import ASTOptimizationPass


GLOBAL_CSE_COUNTER = [0]


def exprKey(expr: Expression) -> str:
    if isinstance(expr, IntegerLiteral):
        return f"i{expr.value}"
    if isinstance(expr, RealLiteral):
        return f"r{expr.value}"
    if isinstance(expr, Variable):
        return f"v:{expr.name}"
    if isinstance(expr, BinaryOp):
        lk = exprKey(expr.left)
        rk = exprKey(expr.right)
        if expr.op in ('+', '*'):
            lk, rk = min(lk, rk), max(lk, rk)
        return f"({lk}{expr.op}{rk})"
    if isinstance(expr, UnaryOp):
        return f"({expr.op}{exprKey(expr.operand)})"
    if isinstance(expr, FunctionCall):
        args = ",".join(exprKey(a) for a in expr.args)
        return f"{expr.name.upper()}({args})"
    if isinstance(expr, ArrayRef):
        idx = ",".join(exprKey(i) for i in expr.indices)
        return f"{expr.name}[{idx}]"
    return repr(expr)


def isPure(expr: Expression) -> bool:
    if isinstance(expr, (IntegerLiteral, RealLiteral, Variable)):
        return True
    if isinstance(expr, ArrayRef):
        return all(isPure(i) for i in expr.indices)
    if isinstance(expr, BinaryOp):
        return isPure(expr.left) and isPure(expr.right)
    if isinstance(expr, UnaryOp):
        return isPure(expr.operand)
    if isinstance(expr, FunctionCall):
        pureFuncs = {'SQRT', 'SIN', 'COS', 'TAN', 'LOG', 'EXP', 'ABS',
                     'INT', 'FLOAT', 'REAL', 'MOD', 'MIN', 'MAX', 'ASIN',
                     'ACOS', 'ATAN', 'LOG10', 'SIGN', 'POW'}
        return expr.name.upper() in pureFuncs and all(isPure(a) for a in expr.args)
    return False


def isTrivial(expr: Expression) -> bool:
    return isinstance(expr, (IntegerLiteral, RealLiteral, Variable))


def varsInExpr(expr: Expression, result: Set[str]) -> None:
    if isinstance(expr, Variable):
        result.add(expr.name)
    elif isinstance(expr, ArrayRef):
        result.add(expr.name)
        for i in expr.indices:
            varsInExpr(i, result)
    elif isinstance(expr, BinaryOp):
        varsInExpr(expr.left, result)
        varsInExpr(expr.right, result)
    elif isinstance(expr, UnaryOp):
        varsInExpr(expr.operand, result)
    elif isinstance(expr, FunctionCall):
        for a in expr.args:
            varsInExpr(a, result)


class CSEBlock:
    def __init__(self, counter: List[int]):
        self.counter = counter
        self.cache: Dict[str, Tuple[str, Expression]] = {}
        self.tmpVars: Dict[str, Set[str]] = {}
        self.result: List[Statement] = []
        self.newAssigns: List[Assignment] = []
        self.eliminated = 0

    def invalidate(self, varName: str) -> None:
        toDelete = []
        for k, (tmp, expr) in self.cache.items():
            deps = set()
            varsInExpr(expr, deps)
            if varName in deps:
                toDelete.append(k)
        for k in toDelete:
            del self.cache[k]

    def subst(self, expr: Expression) -> Expression:
        if isTrivial(expr) or not isPure(expr):
            return expr
        if isinstance(expr, BinaryOp):
            nl = self.subst(expr.left)
            nr = self.subst(expr.right)
            if nl is not expr.left or nr is not expr.right:
                expr = dcReplace(expr, left=nl, right=nr)
        elif isinstance(expr, UnaryOp):
            no = self.subst(expr.operand)
            if no is not expr.operand:
                expr = dcReplace(expr, operand=no)
        elif isinstance(expr, FunctionCall):
            na = [self.subst(a) for a in expr.args]
            if any(x is not y for x, y in zip(na, expr.args)):
                expr = dcReplace(expr, args=na)
        if isTrivial(expr):
            return expr
        key = exprKey(expr)
        if key in self.cache:
            tmpName, _ = self.cache[key]
            self.eliminated += 1
            return Variable(name=tmpName, line=expr.line, col=expr.col)
        if isinstance(expr, (BinaryOp, FunctionCall)):
            self.counter[0] += 1
            tmpName = f"cse{self.counter[0]}"
            deps: Set[str] = set()
            varsInExpr(expr, deps)
            self.cache[key] = (tmpName, expr)
            self.tmpVars[tmpName] = deps
            assign = Assignment(
                target=tmpName,
                value=expr,
                indices=[],
                stmt_label=None,
                line=expr.line,
                col=expr.col,
            )
            self.newAssigns.append(assign)
            return Variable(name=tmpName, line=expr.line, col=expr.col)
        return expr

    def process(self, stmt: Statement) -> List[Statement]:
        self.newAssigns = []
        if isinstance(stmt, Assignment):
            if not stmt.indices:
                nv = self.subst(stmt.value)
                newStmt = dcReplace(stmt, value=nv)
                self.invalidate(stmt.target)
            else:
                nv = self.subst(stmt.value)
                ni = [self.subst(i) for i in stmt.indices]
                newStmt = dcReplace(stmt, value=nv, indices=ni)
                self.invalidate(stmt.target)
        elif isinstance(stmt, PrintStatement):
            ni = [self.subst(i) for i in stmt.items]
            newStmt = dcReplace(stmt, items=ni)
        elif isinstance(stmt, WriteStatement):
            ni = [self.subst(i) for i in stmt.items]
            newStmt = dcReplace(stmt, items=ni)
        elif isinstance(stmt, CallStatement):
            na = [self.subst(a) for a in stmt.args]
            newStmt = dcReplace(stmt, args=na)
            self.cache.clear()
        else:
            self.cache.clear()
            newStmt = stmt
        return self.newAssigns + [newStmt]


def applyCseToStmts(stmts: List[Statement], counter: List[int]) -> List[Statement]:
    block = CSEBlock(counter)
    result = []
    for stmt in stmts:
        if isinstance(stmt, (DoLoop, LabeledDoLoop)):
            block.cache.clear()
            nb = applyCseToStmts(stmt.body, counter)
            stmt = dcReplace(stmt, body=nb)
            result.append(stmt)
        elif isinstance(stmt, (DoWhile, LabeledDoWhile)):
            block.cache.clear()
            nb = applyCseToStmts(stmt.body, counter)
            stmt = dcReplace(stmt, body=nb)
            result.append(stmt)
        elif isinstance(stmt, IfStatement):
            block.cache.clear()
            result.append(stmt)
        else:
            result.extend(block.process(stmt))
    return result


class CommonSubexpressionElimination(ASTOptimizationPass):
    name = "CommonSubexpressionElimination"

    def run(self, program: Program) -> Program:
        counter = GLOBAL_CSE_COUNTER
        before = counter[0]
        newStmts = applyCseToStmts(program.statements, counter)
        newSubs = [
            dcReplace(s, statements=applyCseToStmts(s.statements, counter))
            for s in program.subroutines
        ]
        newFuncs = [
            dcReplace(f, statements=applyCseToStmts(f.statements, counter))
            for f in program.functions
        ]
        self.stats = {"cse_vars": counter[0] - before}
        return dcReplace(program, statements=newStmts,
                         subroutines=newSubs, functions=newFuncs)
