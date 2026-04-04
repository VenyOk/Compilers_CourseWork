from __future__ import annotations

from dataclasses import replace as dcReplace
from typing import List

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


def isSmallPosInt(expr: Expression) -> bool:
    return isinstance(expr, IntegerLiteral) and 2 <= expr.value <= 4


def expandPower(base: Expression, n: int, line: int, col: int) -> Expression:
    result = base
    for _ in range(n - 1):
        result = BinaryOp(left=result, op="*", right=base, line=line, col=col)
    return result


def reduceExpr(expr: Expression, counter: List[int]) -> Expression:
    if isinstance(expr, BinaryOp):
        left = reduceExpr(expr.left, counter)
        right = reduceExpr(expr.right, counter)
        if expr.op == "**" and isSmallPosInt(right):
            counter[0] += 1
            return expandPower(left, right.value, expr.line, expr.col)
        return dcReplace(expr, left=left, right=right)
    if isinstance(expr, UnaryOp):
        return dcReplace(expr, operand=reduceExpr(expr.operand, counter))
    if isinstance(expr, FunctionCall):
        return dcReplace(expr, args=[reduceExpr(a, counter) for a in expr.args])
    if isinstance(expr, ArrayRef):
        return dcReplace(expr, indices=[reduceExpr(i, counter) for i in expr.indices])
    return expr


def reduceStmt(stmt: Statement, counter: List[int]) -> Statement:
    if isinstance(stmt, Assignment):
        return dcReplace(
            stmt,
            value=reduceExpr(stmt.value, counter),
            indices=[reduceExpr(i, counter) for i in stmt.indices],
        )
    if isinstance(stmt, (DoLoop, LabeledDoLoop)):
        return dcReplace(
            stmt,
            start=reduceExpr(stmt.start, counter),
            end=reduceExpr(stmt.end, counter),
            body=[reduceStmt(s, counter) for s in stmt.body],
        )
    if isinstance(stmt, (DoWhile, LabeledDoWhile)):
        return dcReplace(
            stmt,
            condition=reduceExpr(stmt.condition, counter),
            body=[reduceStmt(s, counter) for s in stmt.body],
        )
    if isinstance(stmt, IfStatement):
        return dcReplace(
            stmt,
            condition=reduceExpr(stmt.condition, counter),
            then_body=[reduceStmt(s, counter) for s in stmt.then_body],
            elif_parts=[
                (reduceExpr(c, counter), [reduceStmt(s, counter) for s in b])
                for c, b in stmt.elif_parts
            ],
            else_body=[reduceStmt(s, counter) for s in stmt.else_body] if stmt.else_body else None,
        )
    if isinstance(stmt, SimpleIfStatement):
        return dcReplace(
            stmt,
            condition=reduceExpr(stmt.condition, counter),
            body=reduceStmt(stmt.body, counter),
        )
    if isinstance(stmt, PrintStatement):
        return dcReplace(stmt, items=[reduceExpr(a, counter) for a in stmt.items])
    if isinstance(stmt, WriteStatement):
        return dcReplace(stmt, items=[reduceExpr(a, counter) for a in stmt.items])
    if isinstance(stmt, CallStatement):
        return dcReplace(stmt, args=[reduceExpr(a, counter) for a in stmt.args])
    return stmt


def processStmts(stmts: List[Statement], counter: List[int]) -> List[Statement]:
    return [reduceStmt(s, counter) for s in stmts]


class StrengthReduction(ASTOptimizationPass):
    name = "StrengthReduction"

    def run(self, program: Program) -> Program:
        counter = [0]
        new_statements = processStmts(program.statements, counter)
        new_subroutines = [
            dcReplace(sub, statements=processStmts(sub.statements, counter))
            for sub in program.subroutines
        ]
        new_functions = [
            dcReplace(fn, statements=processStmts(fn.statements, counter))
            for fn in program.functions
        ]
        self.stats = {"reduced": counter[0]}
        return dcReplace(
            program,
            statements=new_statements,
            subroutines=new_subroutines,
            functions=new_functions,
        )
