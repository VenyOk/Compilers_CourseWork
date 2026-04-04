from __future__ import annotations

from dataclasses import replace as dcReplace
from typing import List, Set

from src.core import (
    ArrayRef,
    Assignment,
    BinaryOp,
    CallStatement,
    DoLoop,
    DoWhile,
    Expression,
    FunctionCall,
    IfStatement,
    IntegerLiteral,
    LabeledDoLoop,
    LabeledDoWhile,
    PrintStatement,
    Program,
    ReadStatement,
    SimpleIfStatement,
    Statement,
    UnaryOp,
    Variable,
    WriteStatement,
)
from src.optimizations.base import ASTOptimizationPass


def gatherExpr(expr: Expression, out: Set[str]) -> None:
    if isinstance(expr, Variable):
        out.add(expr.name)
    elif isinstance(expr, BinaryOp):
        gatherExpr(expr.left, out)
        gatherExpr(expr.right, out)
    elif isinstance(expr, UnaryOp):
        gatherExpr(expr.operand, out)
    elif isinstance(expr, FunctionCall):
        for a in expr.args:
            gatherExpr(a, out)
    elif isinstance(expr, ArrayRef):
        for i in expr.indices:
            gatherExpr(i, out)


def gatherStmts(stmts: List[Statement], out: Set[str]) -> None:
    for stmt in stmts:
        if isinstance(stmt, Assignment):
            gatherExpr(stmt.value, out)
            for i in stmt.indices:
                gatherExpr(i, out)
        elif isinstance(stmt, (DoLoop, LabeledDoLoop)):
            gatherExpr(stmt.start, out)
            gatherExpr(stmt.end, out)
            if stmt.step:
                gatherExpr(stmt.step, out)
            gatherStmts(stmt.body, out)
        elif isinstance(stmt, (DoWhile, LabeledDoWhile)):
            gatherExpr(stmt.condition, out)
            gatherStmts(stmt.body, out)
        elif isinstance(stmt, IfStatement):
            gatherExpr(stmt.condition, out)
            gatherStmts(stmt.then_body, out)
            for cond, body in stmt.elif_parts:
                gatherExpr(cond, out)
                gatherStmts(body, out)
            if stmt.else_body:
                gatherStmts(stmt.else_body, out)
        elif isinstance(stmt, SimpleIfStatement):
            gatherExpr(stmt.condition, out)
            gatherStmts([stmt.body], out)
        elif isinstance(stmt, (PrintStatement, WriteStatement)):
            for a in stmt.items:
                gatherExpr(a, out)
        elif isinstance(stmt, CallStatement):
            for a in stmt.args:
                gatherExpr(a, out)


def isDeadGenerated(stmt: Statement, live: Set[str]) -> bool:
    if not isinstance(stmt, Assignment) or stmt.indices:
        return False
    name = stmt.target
    return (name.startswith("cse_tmp_") or name.startswith("licm_tmp_")) and name not in live


def filterStmts(stmts: List[Statement], live: Set[str], counter: List[int]) -> List[Statement]:
    result = []
    for stmt in stmts:
        if isDeadGenerated(stmt, live):
            counter[0] += 1
            continue
        if isinstance(stmt, (DoLoop, LabeledDoLoop)):
            stmt = dcReplace(stmt, body=filterStmts(stmt.body, live, counter))
        elif isinstance(stmt, (DoWhile, LabeledDoWhile)):
            stmt = dcReplace(stmt, body=filterStmts(stmt.body, live, counter))
        elif isinstance(stmt, IfStatement):
            stmt = dcReplace(
                stmt,
                then_body=filterStmts(stmt.then_body, live, counter),
                elif_parts=[(c, filterStmts(b, live, counter)) for c, b in stmt.elif_parts],
                else_body=filterStmts(stmt.else_body, live, counter) if stmt.else_body else None,
            )
        result.append(stmt)
    return result


def processUnit(stmts: List[Statement], counter: List[int]) -> List[Statement]:
    live: Set[str] = set()
    gatherStmts(stmts, live)
    return filterStmts(stmts, live, counter)


class DeadCodeElimination(ASTOptimizationPass):
    name = "DeadCodeElimination"

    def run(self, program: Program) -> Program:
        counter = [0]
        new_statements = processUnit(program.statements, counter)
        new_subroutines = [
            dcReplace(sub, statements=processUnit(sub.statements, counter))
            for sub in program.subroutines
        ]
        new_functions = [
            dcReplace(fn, statements=processUnit(fn.statements, counter))
            for fn in program.functions
        ]
        self.stats = {"eliminated": counter[0]}
        return dcReplace(
            program,
            statements=new_statements,
            subroutines=new_subroutines,
            functions=new_functions,
        )
