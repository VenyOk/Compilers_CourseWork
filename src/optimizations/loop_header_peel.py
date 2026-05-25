from __future__ import annotations

from dataclasses import replace as dcReplace
from typing import List, Optional, Tuple

from src.frontend.ast import (
    BinaryOp,
    DoLoop,
    Expression,
    FunctionCall,
    IntegerLiteral,
    LabeledDoLoop,
    Program,
    Statement,
    Variable,
)
from src.optimizations.base import ASTOptimizationPass

def intValue(expr: Expression) -> Optional[int]:
    if isinstance(expr, IntegerLiteral):
        return expr.value
    return None

def intExpr(value: int, template: Expression) -> IntegerLiteral:
    return IntegerLiteral(value=value, line=template.line, col=template.col)

def addInt(expr: Expression, value: int) -> Expression:
    if value == 0:
        return expr
    return BinaryOp(left=expr, op="+", right=intExpr(value, expr), line=expr.line, col=expr.col)

def tileSpan(inner: DoLoop, tile_var: str) -> Optional[int]:
    if not isinstance(inner.end, FunctionCall) or inner.end.name.upper() != "MIN" or len(inner.end.args) != 2:
        return None
    left, right = inner.end.args
    if not isinstance(left, BinaryOp) or left.op != "+":
        return None
    if not isinstance(left.left, Variable) or left.left.name != tile_var:
        return None
    span = intValue(left.right)
    step = intValue(inner.step)
    if span is None or step is None or step <= 0:
        return None
    if span % step != 0:
        return None
    return span

def literalLower(inner: DoLoop, tile_var: str) -> Optional[int]:
    if isinstance(inner.start, IntegerLiteral):
        return inner.start.value
    if isinstance(inner.start, FunctionCall) and inner.start.name.upper() == "MAX" and len(inner.start.args) == 2:
        tile_arg, bound = inner.start.args
        if isinstance(tile_arg, Variable) and tile_arg.name == tile_var:
            return intValue(bound)
    if isinstance(inner.start, Variable) and inner.start.name == tile_var:
        return intValue(inner.start)
    return None

def literalUpper(inner: DoLoop, tile_var: str) -> Optional[int]:
    if not isinstance(inner.end, FunctionCall) or inner.end.name.upper() != "MIN" or len(inner.end.args) != 2:
        return None
    return intValue(inner.end.args[1])

def firstDirectPointLoop(body: List[Statement]) -> Optional[DoLoop]:
    for stmt in body:
        if isinstance(stmt, DoLoop) and not stmt.var.startswith("tile_"):
            return stmt
    return None

def peelTilePointPair(outer: DoLoop, inner: DoLoop) -> Optional[List[Statement]]:
    tile_var = outer.var
    if not tile_var.startswith("tile_"):
        return None
    lower = literalLower(inner, tile_var)
    upper = literalUpper(inner, tile_var)
    span = tileSpan(inner, tile_var)
    outer_start = intValue(outer.start)
    outer_end = intValue(outer.end)
    outer_step = intValue(outer.step)
    inner_step = intValue(inner.step)
    if None in (lower, upper, span, outer_start, outer_end, outer_step, inner_step):
        return None
    if inner_step <= 0 or outer_step <= 0:
        return None
    tile_points = span // inner_step + 1
    if tile_points <= 0 or outer_step != inner_step * tile_points:
        return None
    total = upper - lower + 1
    if total <= 0:
        return None
    full = (total // tile_points) * tile_points
    if full <= 0:
        return None
    line = inner.line
    col = inner.col
    tile_expr = Variable(name=tile_var, line=line, col=col)
    steady_end = lower + full - outer_step
    steady_inner_end = addInt(tile_expr, span)
    steady_outer = dcReplace(
        outer,
        end=intExpr(steady_end, outer.end),
        body=[
            dcReplace(
                inner,
                start=tile_expr,
                end=steady_inner_end,
                body=inner.body,
            )
        ],
    )
    if full >= total:
        return [steady_outer]
    epilog_start = lower + full
    epilog = dcReplace(
        inner,
        start=intExpr(epilog_start, inner.start),
        end=intExpr(upper, inner.end),
        body=inner.body,
    )
    return [steady_outer, epilog]

def peelStmt(stmt: Statement, counter: List[int]) -> Statement:
    if isinstance(stmt, (DoLoop, LabeledDoLoop)):
        new_body = peelStmts(stmt.body, counter)
        stmt = dcReplace(stmt, body=new_body)
        if stmt.var.startswith("tile_"):
            inner = firstDirectPointLoop(stmt.body)
            if inner is not None:
                peeled = peelTilePointPair(stmt, inner)
                if peeled is not None:
                    counter[0] += 1
                    return peeled
        return stmt
    return stmt

def peelStmts(stmts: List[Statement], counter: List[int]) -> List[Statement]:
    result: List[Statement] = []
    for stmt in stmts:
        peeled = peelStmt(stmt, counter)
        if isinstance(peeled, list):
            result.extend(peeled)
        else:
            result.append(peeled)
    return result

class LoopHeaderPeel(ASTOptimizationPass):
    name = "LoopHeaderPeel"

    def run(self, program: Program) -> Program:
        counter = [0]
        new_statements = peelStmts(program.statements, counter)
        new_subroutines = [
            dcReplace(subroutine, statements=peelStmts(subroutine.statements, counter))
            for subroutine in program.subroutines
        ]
        new_functions = [
            dcReplace(function, statements=peelStmts(function.statements, counter))
            for function in program.functions
        ]
        self.stats = {"peeled": counter[0]}
        return dcReplace(
            program,
            statements=new_statements,
            subroutines=new_subroutines,
            functions=new_functions,
        )
