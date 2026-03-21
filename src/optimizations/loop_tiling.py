from __future__ import annotations

import math
from dataclasses import replace as dcReplace
from typing import Dict, List, Optional

from src.core import (
    ArrayRef,
    BinaryOp,
    DoLoop,
    Expression,
    FunctionCall,
    IntegerLiteral,
    LabeledDoLoop,
    Program,
    Statement,
    UnaryOp,
    Variable,
)
from src.optimizations.base import ASTOptimizationPass
from src.optimizations.loop_analysis import LoopNest, buildNest, constantInt, estimateTripCount, shouldTileNest


def tileVarName(var: str) -> str:
    return f"tile_{var}"


def isTileVar(var: str) -> bool:
    return var.startswith("tile_")


def optimalTileSize(depth: int, l1Bytes: int = 32 * 1024, elemSize: int = 8) -> int:
    nElems = max(l1Bytes // elemSize, 16)
    size = int(round(nElems ** (1.0 / max(depth, 1))))
    return max(size, 2)


def intExpr(value: int, line: int, col: int) -> IntegerLiteral:
    return IntegerLiteral(value=value, line=line, col=col)


def isZeroExpr(expr: Expression) -> bool:
    return isinstance(expr, IntegerLiteral) and expr.value == 0


def addExpr(left: Expression, right: Expression, line: int, col: int) -> Expression:
    if isZeroExpr(left):
        return right
    if isZeroExpr(right):
        return left
    return BinaryOp(left=left, op="+", right=right, line=line, col=col)


def subExpr(left: Expression, right: Expression, line: int, col: int) -> Expression:
    if isZeroExpr(right):
        return left
    return BinaryOp(left=left, op="-", right=right, line=line, col=col)


def negExpr(expr: Expression, line: int, col: int) -> Expression:
    if isZeroExpr(expr):
        return expr
    if isinstance(expr, IntegerLiteral):
        return intExpr(-expr.value, line, col)
    return UnaryOp(op="-", operand=expr, line=line, col=col)


def mulExprByInt(expr: Expression, factor: int, line: int, col: int) -> Expression:
    if factor == 0:
        return intExpr(0, line, col)
    if factor == 1:
        return expr
    if factor == -1:
        return negExpr(expr, line, col)
    return BinaryOp(left=intExpr(factor, line, col), op="*", right=expr, line=line, col=col)


def addInt(expr: Expression, value: int) -> Expression:
    if value == 0:
        return expr
    return addExpr(expr, intExpr(value, expr.line, expr.col), expr.line, expr.col)


def minExpr(left: Expression, right: Expression) -> Expression:
    return FunctionCall(name="MIN", args=[left, right], line=left.line, col=left.col)


def maxExpr(left: Expression, right: Expression) -> Expression:
    return FunctionCall(name="MAX", args=[left, right], line=left.line, col=left.col)


def substituteExpr(expr: Expression, substitutions: Dict[str, Expression]) -> Expression:
    if isinstance(expr, Variable) and expr.name in substitutions:
        return substitutions[expr.name]
    if isinstance(expr, BinaryOp):
        return dcReplace(
            expr,
            left=substituteExpr(expr.left, substitutions),
            right=substituteExpr(expr.right, substitutions),
        )
    if isinstance(expr, UnaryOp):
        return dcReplace(expr, operand=substituteExpr(expr.operand, substitutions))
    if isinstance(expr, FunctionCall):
        return dcReplace(expr, args=[substituteExpr(arg, substitutions) for arg in expr.args])
    if isinstance(expr, ArrayRef):
        return dcReplace(expr, indices=[substituteExpr(index, substitutions) for index in expr.indices])
    return expr


def tileSizesForNest(nest: LoopNest, baseTileSize: Optional[int], l1Bytes: int) -> List[int]:
    side = baseTileSize if baseTileSize is not None else optimalTileSize(nest.depth, l1Bytes=l1Bytes)
    trip_counts = [estimateTripCount(loop_info) for loop_info in nest.loops]
    known_counts = [count for count in trip_counts if count is not None and count > 0]
    reference = max(known_counts) if known_counts else None

    def chooseAxis(count: Optional[int]) -> int:
        effective = count if count is not None else reference
        if effective is None:
            return max(side, 2)
        if effective < 16:
            return max(2, effective)
        for candidate in [64, 48, 32, 24, 16, 12, 8, 6, 4, 2]:
            if candidate <= side and candidate < effective and effective // candidate >= 2:
                return candidate
        return max(2, min(side, effective))

    return [chooseAxis(count) for count in trip_counts]


def buildBounds(nest: LoopNest, tileSizes: List[int]):
    line = nest.loops[0].node.line
    col = nest.loops[0].node.col
    lower_env: Dict[str, Expression] = {}
    upper_env: Dict[str, Expression] = {}
    point_env: Dict[str, Expression] = {}
    tile_infos = []
    point_infos = []
    for index, loop_info in enumerate(nest.loops):
        tile_var = tileVarName(loop_info.var)
        tile_expr = Variable(name=tile_var, line=line, col=col)
        start_bound = substituteExpr(loop_info.start, lower_env)
        end_bound = substituteExpr(loop_info.end, upper_env)
        point_lower = substituteExpr(loop_info.start, point_env)
        point_upper = substituteExpr(loop_info.end, point_env)
        step_value = constantInt(loop_info.step)
        if step_value is None or step_value <= 0:
            return None
        point_start = maxExpr(tile_expr, point_lower)
        point_end = minExpr(addInt(tile_expr, step_value * (tileSizes[index] - 1)), point_upper)
        lower_env[loop_info.var] = point_start
        upper_env[loop_info.var] = point_end
        point_env[loop_info.var] = Variable(name=loop_info.var, line=line, col=col)
        tile_infos.append((tile_var, start_bound, end_bound, step_value * tileSizes[index]))
        point_infos.append((loop_info.var, point_start, point_end, loop_info.step))
    return tile_infos, point_infos


def tileAffineNest(nest: LoopNest, tileSizes: List[int]) -> Optional[Statement]:
    bounds = buildBounds(nest, tileSizes)
    if bounds is None:
        return None
    tile_infos, point_infos = bounds
    line = nest.loops[0].node.line
    col = nest.loops[0].node.col
    body = nest.body
    for loop_info, point_info in zip(reversed(nest.loops), reversed(point_infos)):
        _, point_start, point_end, point_step = point_info
        body = [DoLoop(
            var=loop_info.var,
            start=point_start,
            end=point_end,
            step=point_step,
            body=body,
            stmt_label=None,
            line=line,
            col=col,
        )]
    for tile_var, tile_start, tile_end, tile_step in reversed(tile_infos):
        body = [DoLoop(
            var=tile_var,
            start=tile_start,
            end=tile_end,
            step=intExpr(tile_step, line, col),
            body=body,
            stmt_label=None,
            line=line,
            col=col,
        )]
    return body[0]


def tryTile(loop: Statement, baseTileSize: Optional[int], minDepth: int, l1Bytes: int, counter: List[int]) -> Statement:
    if not isinstance(loop, (DoLoop, LabeledDoLoop)):
        return loop
    nest = buildNest(loop)
    if any(isTileVar(var) for var in nest.vars):
        return dcReplace(loop, body=[tryTile(stmt, baseTileSize, minDepth, l1Bytes, counter) for stmt in loop.body])
    tile_probe = baseTileSize if baseTileSize is not None else optimalTileSize(max(nest.depth, 1), l1Bytes=l1Bytes)
    if not shouldTileNest(nest, tile_probe, minDepth):
        return dcReplace(loop, body=[tryTile(stmt, baseTileSize, minDepth, l1Bytes, counter) for stmt in loop.body])
    tile_sizes = tileSizesForNest(nest, baseTileSize, l1Bytes)
    transformed = tileAffineNest(nest, tile_sizes)
    if transformed is not None:
        counter[0] += 1
        return transformed
    return dcReplace(loop, body=[tryTile(stmt, baseTileSize, minDepth, l1Bytes, counter) for stmt in loop.body])


def processStmts(stmts: List[Statement], baseTileSize: Optional[int], minDepth: int, l1Bytes: int, counter: List[int]) -> List[Statement]:
    result = []
    for stmt in stmts:
        if isinstance(stmt, (DoLoop, LabeledDoLoop)):
            result.append(tryTile(stmt, baseTileSize, minDepth, l1Bytes, counter))
        else:
            result.append(stmt)
    return result


class LoopTiling(ASTOptimizationPass):
    name = "LoopTiling"

    def __init__(self, tileSize: Optional[int] = None, minDepth: int = 2, l1Bytes: int = 32 * 1024):
        super().__init__()
        self.baseTileSize = tileSize
        self.minDepth = minDepth
        self.l1Bytes = l1Bytes

    def run(self, program: Program) -> Program:
        counter = [0]
        new_statements = processStmts(program.statements, self.baseTileSize, self.minDepth, self.l1Bytes, counter)
        new_subroutines = [
            dcReplace(subroutine, statements=processStmts(subroutine.statements, self.baseTileSize, self.minDepth, self.l1Bytes, counter))
            for subroutine in program.subroutines
        ]
        new_functions = [
            dcReplace(function, statements=processStmts(function.statements, self.baseTileSize, self.minDepth, self.l1Bytes, counter))
            for function in program.functions
        ]
        self.stats = {
            "tiled": counter[0],
            "tile_size": self.baseTileSize if self.baseTileSize is not None else optimalTileSize(2, l1Bytes=self.l1Bytes),
        }
        return dcReplace(
            program,
            statements=new_statements,
            subroutines=new_subroutines,
            functions=new_functions,
        )
