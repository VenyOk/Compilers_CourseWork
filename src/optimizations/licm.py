from __future__ import annotations
from typing import List, Set, Tuple, Dict
from dataclasses import replace as dcReplace

from src.core import (
    Program, Statement, Expression,
    Assignment, DoLoop, LabeledDoLoop, DoWhile, LabeledDoWhile,
    IfStatement, SimpleIfStatement,
    Variable, BinaryOp, UnaryOp, FunctionCall, ArrayRef,
    IntegerLiteral, RealLiteral, LogicalLiteral, StringLiteral,
    PrintStatement, WriteStatement, ReadStatement, CallStatement,
)
from src.optimizations.base import ASTOptimizationPass


HOIST_OPS = {'*', '/', '**', '+', '-'}
GLOBAL_LICM_COUNTER = [0]


def collectModified(stmts: List[Statement], result: Set[str]) -> None:
    for stmt in stmts:
        if isinstance(stmt, Assignment):
            result.add(stmt.target)
        elif isinstance(stmt, (DoLoop, LabeledDoLoop)):
            result.add(stmt.var)
            collectModified(stmt.body, result)
        elif isinstance(stmt, (DoWhile, LabeledDoWhile)):
            collectModified(stmt.body, result)
        elif isinstance(stmt, IfStatement):
            collectModified(stmt.then_body, result)
            for _, b in stmt.elif_parts:
                collectModified(b, result)
            if stmt.else_body:
                collectModified(stmt.else_body, result)


def usesModified(expr: Expression, modified: Set[str]) -> bool:
    if isinstance(expr, Variable):
        return expr.name in modified
    if isinstance(expr, ArrayRef):
        return expr.name in modified or any(usesModified(i, modified) for i in expr.indices)
    if isinstance(expr, BinaryOp):
        return usesModified(expr.left, modified) or usesModified(expr.right, modified)
    if isinstance(expr, UnaryOp):
        return usesModified(expr.operand, modified)
    if isinstance(expr, FunctionCall):
        return any(usesModified(a, modified) for a in expr.args)
    return False


def isTrivialExpr(expr: Expression) -> bool:
    return isinstance(expr, (IntegerLiteral, RealLiteral, LogicalLiteral, StringLiteral, Variable))


def containsReal(expr: Expression) -> bool:
    if isinstance(expr, RealLiteral):
        return True
    if isinstance(expr, IntegerLiteral):
        return False
    if isinstance(expr, BinaryOp):
        return containsReal(expr.left) or containsReal(expr.right)
    if isinstance(expr, UnaryOp):
        return containsReal(expr.operand)
    if isinstance(expr, FunctionCall):
        realFuncs = {'FLOAT', 'REAL', 'SQRT', 'SIN', 'COS', 'TAN',
                     'EXP', 'LOG', 'ABS', 'ASIN', 'ACOS', 'ATAN', 'LOG10'}
        return expr.name.upper() in realFuncs
    return False


def worthHoisting(expr: Expression) -> bool:
    if isinstance(expr, BinaryOp):
        if expr.op not in HOIST_OPS:
            return False
        if not containsReal(expr):
            return False
        return True
    if isinstance(expr, FunctionCall):
        return True
    if isinstance(expr, UnaryOp):
        return not isTrivialExpr(expr.operand) and containsReal(expr.operand)
    return False


def exprRepr(expr: Expression) -> str:
    if isinstance(expr, IntegerLiteral):
        return f"int:{expr.value}"
    if isinstance(expr, RealLiteral):
        return f"real:{expr.value}"
    if isinstance(expr, Variable):
        return f"var:{expr.name}"
    if isinstance(expr, BinaryOp):
        return f"({exprRepr(expr.left)}{expr.op}{exprRepr(expr.right)})"
    if isinstance(expr, UnaryOp):
        return f"({expr.op}{exprRepr(expr.operand)})"
    if isinstance(expr, FunctionCall):
        args = ",".join(exprRepr(a) for a in expr.args)
        return f"{expr.name}({args})"
    if isinstance(expr, ArrayRef):
        idx = ",".join(exprRepr(i) for i in expr.indices)
        return f"{expr.name}[{idx}]"
    return repr(expr)


class ExprHoister:
    def __init__(self, modified: Set[str], counterRef: List[int]):
        self.modified = modified
        self.counterRef = counterRef
        self.hoisted: List[Tuple[str, Expression]] = []
        self.cache: Dict[str, str] = {}

    def hoistExpr(self, expr: Expression) -> Expression:
        if isTrivialExpr(expr):
            return expr
        if usesModified(expr, self.modified):
            if isinstance(expr, BinaryOp):
                nl = self.hoistExpr(expr.left)
                nr = self.hoistExpr(expr.right)
                if nl is not expr.left or nr is not expr.right:
                    expr = dcReplace(expr, left=nl, right=nr)
            elif isinstance(expr, UnaryOp):
                no = self.hoistExpr(expr.operand)
                if no is not expr.operand:
                    expr = dcReplace(expr, operand=no)
            elif isinstance(expr, FunctionCall):
                na = [self.hoistExpr(a) for a in expr.args]
                if any(x is not y for x, y in zip(na, expr.args)):
                    expr = dcReplace(expr, args=na)
            elif isinstance(expr, ArrayRef):
                ni = [self.hoistExpr(i) for i in expr.indices]
                if any(x is not y for x, y in zip(ni, expr.indices)):
                    expr = dcReplace(expr, indices=ni)
            return expr
        if not worthHoisting(expr):
            return expr
        key = exprRepr(expr)
        if key in self.cache:
            return Variable(name=self.cache[key], line=expr.line, col=expr.col)
        self.counterRef[0] += 1
        tmpName = f"lcm{self.counterRef[0]}"
        self.hoisted.append((tmpName, expr))
        self.cache[key] = tmpName
        return Variable(name=tmpName, line=expr.line, col=expr.col)

    def hoistStmt(self, stmt: Statement) -> Statement:
        if isinstance(stmt, Assignment):
            nv = self.hoistExpr(stmt.value)
            ni = [self.hoistExpr(i) for i in stmt.indices]
            return dcReplace(stmt, value=nv, indices=ni)
        elif isinstance(stmt, (DoLoop, LabeledDoLoop)):
            ns = self.hoistExpr(stmt.start)
            ne = self.hoistExpr(stmt.end)
            nst = self.hoistExpr(stmt.step) if stmt.step else stmt.step
            nb = [self.hoistStmt(s) for s in stmt.body]
            return dcReplace(stmt, start=ns, end=ne, step=nst, body=nb)
        elif isinstance(stmt, (DoWhile, LabeledDoWhile)):
            nc = self.hoistExpr(stmt.condition)
            nb = [self.hoistStmt(s) for s in stmt.body]
            return dcReplace(stmt, condition=nc, body=nb)
        elif isinstance(stmt, IfStatement):
            nc = self.hoistExpr(stmt.condition)
            nt = [self.hoistStmt(s) for s in stmt.then_body]
            nel = [(self.hoistExpr(c), [self.hoistStmt(s) for s in b]) for c, b in stmt.elif_parts]
            ne = ([self.hoistStmt(s) for s in stmt.else_body] if stmt.else_body else stmt.else_body)
            return dcReplace(stmt, condition=nc, then_body=nt, elif_parts=nel, else_body=ne)
        elif isinstance(stmt, SimpleIfStatement):
            nc = self.hoistExpr(stmt.condition)
            ns = self.hoistStmt(stmt.statement)
            return dcReplace(stmt, condition=nc, statement=ns)
        elif isinstance(stmt, PrintStatement):
            ni = [self.hoistExpr(i) for i in stmt.items]
            return dcReplace(stmt, items=ni)
        elif isinstance(stmt, WriteStatement):
            ni = [self.hoistExpr(i) for i in stmt.items]
            return dcReplace(stmt, items=ni)
        elif isinstance(stmt, CallStatement):
            na = [self.hoistExpr(a) for a in stmt.args]
            return dcReplace(stmt, args=na)
        return stmt


def processLoop(loop: Statement, counter: List[int]) -> Tuple[List[Statement], Statement]:
    if isinstance(loop, (DoLoop, LabeledDoLoop)):
        modified: Set[str] = {loop.var}
        collectModified(loop.body, modified)
    elif isinstance(loop, (DoWhile, LabeledDoWhile)):
        modified = set()
        collectModified(loop.body, modified)
    else:
        return [], loop
    hoister = ExprHoister(modified, counter)
    newBody = [hoister.hoistStmt(s) for s in loop.body]
    newLoop = dcReplace(loop, body=newBody)
    preheader = []
    for tmpName, expr in hoister.hoisted:
        preheader.append(Assignment(
            target=tmpName,
            value=expr,
            indices=[],
            stmt_label=None,
            line=loop.line,
            col=loop.col,
        ))
    return preheader, newLoop


def processStmts(stmts: List[Statement], counter: List[int]) -> List[Statement]:
    result = []
    for stmt in stmts:
        if isinstance(stmt, (DoLoop, LabeledDoLoop, DoWhile, LabeledDoWhile)):
            newBody = processStmts(stmt.body, counter)
            stmt = dcReplace(stmt, body=newBody)
            preheader, newLoop = processLoop(stmt, counter)
            result.extend(preheader)
            result.append(newLoop)
        elif isinstance(stmt, IfStatement):
            nt = processStmts(stmt.then_body, counter)
            nel = [(c, processStmts(b, counter)) for c, b in stmt.elif_parts]
            ne = processStmts(stmt.else_body, counter) if stmt.else_body else stmt.else_body
            result.append(dcReplace(stmt, then_body=nt, elif_parts=nel, else_body=ne))
        else:
            result.append(stmt)
    return result


class LoopInvariantCodeMotion(ASTOptimizationPass):
    name = "LoopInvariantCodeMotion"

    def run(self, program: Program) -> Program:
        counter = GLOBAL_LICM_COUNTER
        before = counter[0]
        newStmts = processStmts(program.statements, counter)
        newSubs = [
            dcReplace(s, statements=processStmts(s.statements, counter))
            for s in program.subroutines
        ]
        newFuncs = [
            dcReplace(f, statements=processStmts(f.statements, counter))
            for f in program.functions
        ]
        self.stats = {"hoisted": counter[0] - before}
        return dcReplace(program, statements=newStmts,
                         subroutines=newSubs, functions=newFuncs)
