from __future__ import annotations

from dataclasses import replace as dcReplace
from typing import Dict, List

from src.core import Assignment, BinaryOp, DoLoop, IfStatement, IntegerLiteral, LabeledDoLoop, Program, Statement, Variable
from src.optimizations.base import ASTOptimizationPass
from src.optimizations.loop_analysis import buildNest, constantInt, shouldWavefrontNest
from src.optimizations.loop_skewing import isSkewVar
from src.optimizations.loop_tiling import addExpr, intExpr, isTileVar, subExpr, substituteExpr


def wavefrontVarName(index: int) -> str:
    return f"wf_h{index}"


def isWavefrontVar(var: str) -> bool:
    return var.startswith("wf_")


def tileDepth(nest) -> int:
    depth = 0
    for loop_info in nest.loops:
        if isTileVar(loop_info.var):
            depth += 1
        else:
            break
    return depth


def hasSkewedPointLoops(nest, depth: int) -> bool:
    return any(isSkewVar(loop_info.var) for loop_info in nest.loops[depth:])


def stepsMatch(tile_loops) -> bool:
    if not tile_loops:
        return False
    first = constantInt(tile_loops[0].step)
    if first is None or first <= 0:
        return False
    for loop_info in tile_loops[1:]:
        step = constantInt(loop_info.step)
        if step != first:
            return False
    return True


def isWavefrontCandidate(nest) -> bool:
    depth = tileDepth(nest)
    if depth < 2:
        return False
    if any(isWavefrontVar(loop_info.var) for loop_info in nest.loops[:depth]):
        return False
    if not hasSkewedPointLoops(nest, depth):
        return False
    if not stepsMatch(nest.loops[:depth]):
        return False
    return shouldWavefrontNest(nest)


def comparison(left, op: str, right, line: int, col: int) -> BinaryOp:
    return BinaryOp(left=left, op=op, right=right, line=line, col=col)


def sumExprs(exprs: List, line: int, col: int):
    total = intExpr(0, line, col)
    for expr in exprs:
        total = addExpr(total, expr, line, col)
    return total


def buildHyperBounds(tile_loops) -> tuple:
    line = tile_loops[0].node.line
    col = tile_loops[0].node.col
    lower_env: Dict[str, object] = {}
    upper_env: Dict[str, object] = {}
    lower_parts = []
    upper_parts = []
    for loop_info in tile_loops:
        lower = substituteExpr(loop_info.start, lower_env)
        upper = substituteExpr(loop_info.end, upper_env)
        lower_env[loop_info.var] = lower
        upper_env[loop_info.var] = upper
        lower_parts.append(lower)
        upper_parts.append(upper)
    return sumExprs(lower_parts, line, col), sumExprs(upper_parts, line, col)


def derivedLastTileValue(hyper_expr, tile_loops, line: int, col: int):
    prefix = [Variable(name=loop_info.var, line=line, col=col) for loop_info in tile_loops[:-1]]
    if not prefix:
        return hyper_expr
    return subExpr(hyper_expr, sumExprs(prefix, line, col), line, col)


def wavefrontNest(loop: Statement, counter: List[int]) -> Statement:
    nest = buildNest(loop)
    depth = tileDepth(nest)
    tile_loops = nest.loops[:depth]
    if depth < 2:
        return loop
    line = tile_loops[0].node.line
    col = tile_loops[0].node.col
    hyper_var = wavefrontVarName(counter[0])
    hyper_expr = Variable(name=hyper_var, line=line, col=col)
    hyper_start, hyper_end = buildHyperBounds(tile_loops)
    last_loop = tile_loops[-1]
    derived_last = derivedLastTileValue(hyper_expr, tile_loops, line, col)
    current_env = {
        loop_info.var: Variable(name=loop_info.var, line=line, col=col)
        for loop_info in tile_loops[:-1]
    }
    lower_bound = substituteExpr(last_loop.start, current_env)
    upper_bound = substituteExpr(last_loop.end, current_env)
    condition = comparison(
        comparison(derived_last, ".GE.", lower_bound, line, col),
        ".AND.",
        comparison(derived_last, ".LE.", upper_bound, line, col),
        line,
        col,
    )
    guarded_body = [
        Assignment(
            target=last_loop.var,
            value=derived_last,
            indices=[],
            stmt_label=None,
            line=line,
            col=col,
        )
    ] + last_loop.node.body
    body = [IfStatement(condition=condition, then_body=guarded_body, elif_parts=[], else_body=None, line=line, col=col)]
    for loop_info in reversed(tile_loops[:-1]):
        body = [DoLoop(
            var=loop_info.var,
            start=loop_info.start,
            end=loop_info.end,
            step=loop_info.step,
            body=body,
            stmt_label=None,
            line=loop_info.node.line,
            col=loop_info.node.col,
        )]
    return DoLoop(
        var=hyper_var,
        start=hyper_start,
        end=hyper_end,
        step=tile_loops[0].step,
        body=body,
        stmt_label=None,
        line=line,
        col=col,
    )


def tryWavefront(loop: Statement, counter: List[int]) -> Statement:
    if not isinstance(loop, (DoLoop, LabeledDoLoop)):
        return loop
    nest = buildNest(loop)
    if isWavefrontCandidate(nest):
        transformed = wavefrontNest(loop, counter)
        if transformed is not loop:
            counter[0] += 1
            return transformed
    return dcReplace(loop, body=[tryWavefront(stmt, counter) for stmt in loop.body])


def processStmts(stmts: List[Statement], counter: List[int]) -> List[Statement]:
    result = []
    for stmt in stmts:
        if isinstance(stmt, (DoLoop, LabeledDoLoop)):
            result.append(tryWavefront(stmt, counter))
        else:
            result.append(stmt)
    return result


class LoopWavefront(ASTOptimizationPass):
    name = "LoopWavefront"

    def run(self, program: Program) -> Program:
        counter = [0]
        new_statements = processStmts(program.statements, counter)
        new_subroutines = [
            dcReplace(subroutine, statements=processStmts(subroutine.statements, counter))
            for subroutine in program.subroutines
        ]
        new_functions = [
            dcReplace(function, statements=processStmts(function.statements, counter))
            for function in program.functions
        ]
        self.stats = {"wavefronted": counter[0]}
        return dcReplace(
            program,
            statements=new_statements,
            subroutines=new_subroutines,
            functions=new_functions,
        )
