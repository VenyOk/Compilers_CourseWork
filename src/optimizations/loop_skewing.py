from __future__ import annotations
from typing import List
from dataclasses import replace as dcReplace

from src.core import (
    Program, Statement, Expression,
    DoLoop, LabeledDoLoop,
    Variable, IntegerLiteral, BinaryOp,
    Assignment, ArrayRef, FunctionCall, UnaryOp,
)
from src.optimizations.base import ASTOptimizationPass
from src.optimizations.loop_analysis import buildNest, needsSkewing, getSkewFactors


class SkewSubstituter:
    def __init__(self, skewVar: str, outerVar: str, skewFactor: int):
        self.skewVar = skewVar
        self.outerVar = outerVar
        self.skewFactor = skewFactor

    def subst(self, expr: Expression) -> Expression:
        if isinstance(expr, Variable):
            if expr.name == self.skewVar:
                outer = Variable(name=self.outerVar, line=expr.line, col=expr.col)
                jp = Variable(name=self.skewVar + 'p', line=expr.line, col=expr.col)
                if self.skewFactor == 0:
                    return jp
                sExpr = IntegerLiteral(value=self.skewFactor, line=expr.line, col=expr.col)
                sI = BinaryOp(left=sExpr, op='*', right=outer, line=expr.line, col=expr.col)
                return BinaryOp(left=jp, op='-', right=sI, line=expr.line, col=expr.col)
            return expr
        if isinstance(expr, BinaryOp):
            return dcReplace(expr, left=self.subst(expr.left), right=self.subst(expr.right))
        if isinstance(expr, UnaryOp):
            return dcReplace(expr, operand=self.subst(expr.operand))
        if isinstance(expr, FunctionCall):
            return dcReplace(expr, args=[self.subst(a) for a in expr.args])
        if isinstance(expr, ArrayRef):
            return dcReplace(expr, indices=[self.subst(i) for i in expr.indices])
        return expr

    def substStmt(self, stmt: Statement) -> Statement:
        if isinstance(stmt, Assignment):
            nv = self.subst(stmt.value)
            ni = [self.subst(i) for i in stmt.indices]
            return dcReplace(stmt, value=nv, indices=ni)
        if isinstance(stmt, (DoLoop, LabeledDoLoop)):
            ns = self.subst(stmt.start)
            ne = self.subst(stmt.end)
            nst = self.subst(stmt.step) if stmt.step else stmt.step
            nb = [self.substStmt(s) for s in stmt.body]
            return dcReplace(stmt, start=ns, end=ne, step=nst, body=nb)
        return stmt


def makeExpr(outerVar: str, outerMult: int, const: int, line: int, col: int) -> Expression:
    if outerMult == 0:
        return IntegerLiteral(value=const, line=line, col=col)
    outer = Variable(name=outerVar, line=line, col=col)
    if outerMult == 1:
        term = outer
    else:
        sExpr = IntegerLiteral(value=outerMult, line=line, col=col)
        term = BinaryOp(left=sExpr, op='*', right=outer, line=line, col=col)
    if const == 0:
        return term
    cExpr = IntegerLiteral(value=abs(const), line=line, col=col)
    op = '+' if const > 0 else '-'
    return BinaryOp(left=term, op=op, right=cExpr, line=line, col=col)


def extractConst(e: Expression, default) -> int:
    if isinstance(e, IntegerLiteral):
        return e.value
    return default


def skewNest(nest, factors: List[int]) -> Statement:
    outerLi = nest.loops[0]
    innerLi = nest.loops[1]
    body = nest.body
    line, col = outerLi.node.line, outerLi.node.col
    s = factors[1]
    if s == 0:
        return outerLi.node
    jpName = innerLi.var + 'p'
    sub = SkewSubstituter(innerLi.var, outerLi.var, s)
    newBody = [sub.substStmt(stmt) for stmt in body]
    jStartConst = extractConst(innerLi.start, 1)
    jEndConst = extractConst(innerLi.end, None)
    newInnerStart = makeExpr(outerLi.var, s, jStartConst, line, col)
    if jEndConst is not None:
        newInnerEnd = makeExpr(outerLi.var, s, jEndConst, line, col)
    else:
        outerV = Variable(name=outerLi.var, line=line, col=col)
        sExpr = IntegerLiteral(value=s, line=line, col=col)
        sI = BinaryOp(left=sExpr, op='*', right=outerV, line=line, col=col)
        newInnerEnd = BinaryOp(left=sI, op='+', right=innerLi.end, line=line, col=col)
    newInner = DoLoop(
        var=jpName, start=newInnerStart, end=newInnerEnd, step=innerLi.step,
        body=newBody, stmt_label=None, line=line, col=col,
    )
    newOuter = dcReplace(outerLi.node, body=[newInner])
    return newOuter


def trySkew(loop: Statement, counter: List[int]) -> Statement:
    if not isinstance(loop, (DoLoop, LabeledDoLoop)):
        return loop
    nest = buildNest(loop)
    if nest.depth >= 2 and needsSkewing(nest):
        factors = getSkewFactors(nest)
        if any(f > 0 for f in factors):
            skewed = skewNest(nest, factors)
            counter[0] += 1
            return skewed
    newBody = [trySkew(s, counter) for s in loop.body]
    return dcReplace(loop, body=newBody)


def processStmts(stmts: List[Statement], counter: List[int]) -> List[Statement]:
    result = []
    for stmt in stmts:
        if isinstance(stmt, (DoLoop, LabeledDoLoop)):
            result.append(trySkew(stmt, counter))
        else:
            result.append(stmt)
    return result


def collectSkewVars(stmts: List[Statement], result: set) -> None:
    for stmt in stmts:
        if isinstance(stmt, (DoLoop, LabeledDoLoop)):
            if stmt.var.endswith('p') and len(stmt.var) >= 2:
                result.add(stmt.var)
            collectSkewVars(stmt.body, result)


class LoopSkewing(ASTOptimizationPass):
    name = "LoopSkewing"

    def run(self, program: Program) -> Program:
        from src.core import Declaration
        counter = [0]
        newStmts = processStmts(program.statements, counter)
        newSubs = [
            dcReplace(s, statements=processStmts(s.statements, counter))
            for s in program.subroutines
        ]
        newFuncs = [
            dcReplace(f, statements=processStmts(f.statements, counter))
            for f in program.functions
        ]
        skewVars: set = set()
        collectSkewVars(newStmts, skewVars)
        existing: set = {
            n for d in program.declarations
            if isinstance(d, Declaration)
            for n, _ in d.names
        }
        newSkew = [v for v in sorted(skewVars) if v not in existing]
        newDecls = list(program.declarations)
        if newSkew:
            skewDecl = Declaration(
                type='INTEGER',
                names=[(v, None) for v in newSkew],
                line=0, col=0,
            )
            newDecls = [skewDecl] + newDecls
        self.stats = {"skewed": counter[0]}
        return dcReplace(program, statements=newStmts,
                         declarations=newDecls,
                         subroutines=newSubs, functions=newFuncs)
