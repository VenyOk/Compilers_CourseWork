from __future__ import annotations

import math
import os
from copy import deepcopy
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
from src.optimizations.loop_analysis import (
    LoopNest,
    buildNest,
    chooseIntraTileLoopOrder,
    constantInt,
    countArrayAccesses,
    effectiveStateCarriedPrefixDepth,
    estimateTripCount,
    isStencilLikeNest,
    stencilFamily,
    shouldTileNest,
)


def tileVarName(var: str) -> str:
    return f"tile_{var}"


def isTileVar(var: str) -> bool:
    return var.startswith("tile_")


def optimalTileSize(depth: int, l1Bytes: int = 32 * 1024, elemSize: int = 8) -> int:
    nElems = max(l1Bytes // elemSize, 16)
    d = max(depth, 2)
    size = int(round(nElems ** (1.0 / d))) - 2
    return max(size, 2)


def tileSizesFromEnv() -> Optional[List[int]]:
    raw = os.environ.get("FORTRAN_TILE_SIZES", "").strip()
    if not raw:
        return None
    try:
        values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    except Exception:
        return None
    if not values:
        return None
    return [max(value, 1) for value in values]


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
        replacement = deepcopy(substitutions[expr.name])
        if isinstance(replacement, Variable) and replacement.name == expr.name:
            return replacement
        return substituteExpr(replacement, substitutions)
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


def tileSizesForNest(nest: LoopNest, baseTileSize: Optional[int], l1Bytes: int, override: Optional[List[int]] = None) -> List[int]:
    if override:
        result = list(override[:nest.depth])
        while len(result) < nest.depth:
            result.append(result[-1])
        return result
    side = baseTileSize if baseTileSize is not None else optimalTileSize(nest.depth, l1Bytes=l1Bytes)
    trip_counts = [estimateTripCount(loop_info) for loop_info in nest.loops]
    known_counts = [count for count in trip_counts if count is not None and count > 0]
    reference = max(known_counts) if known_counts else None
    prefix_depth = effectiveStateCarriedPrefixDepth(nest)
    stencil_like = isStencilLikeNest(nest)
    access_count = countArrayAccesses(nest)
    family = stencilFamily(nest)

    def chooseCandidate(count: Optional[int], candidates: List[int], fallback: int) -> int:
        effective = count if count is not None else reference
        if effective is None:
            return max(fallback, 2)
        if effective <= fallback:
            return max(2, effective)
        for candidate in candidates:
            if candidate < effective and effective // candidate >= 1:
                return candidate
        return max(2, min(fallback, effective))

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

    if family in {"dirichlet_gs", "coefficient_stencil"} and prefix_depth > 0:
        sizes = []
        for index, count in enumerate(trip_counts):
            if index < prefix_depth:
                sizes.append(chooseCandidate(count, [64, 48, 40, 32, 24, 16, 8], 32))
            else:
                sizes.append(chooseCandidate(count, [50, 64, 40, 32, 25, 20, 16, 12, 8], 32))
        return sizes

    if family == "gauss_seidel":
        sizes = []
        for index, count in enumerate(trip_counts):
            if index < prefix_depth:
                sizes.append(chooseCandidate(count, [48, 64, 32, 24, 16, 8], 32))
            else:
                sizes.append(chooseCandidate(count, [40, 50, 64, 32, 25, 20, 16, 12, 8], 32))
        return sizes

    if family == "dirichlet_gs":
        sizes = []
        for index, count in enumerate(trip_counts):
            if index < prefix_depth:
                sizes.append(chooseCandidate(count, [24, 16, 12, 8, 4], 16))
            else:
                sizes.append(chooseCandidate(count, [32, 24, 20, 16, 12, 8], 24))
        return sizes

    if family == "gauss_seidel":
        sizes = []
        for index, count in enumerate(trip_counts):
            if index < prefix_depth:
                sizes.append(chooseCandidate(count, [24, 16, 12, 8, 4], 16))
            else:
                sizes.append(chooseCandidate(count, [24, 20, 16, 12, 8], 16))
        return sizes

    if stencil_like and prefix_depth > 0 and access_count >= 5:
        sizes: List[int] = []
        for index, count in enumerate(trip_counts):
            if index < prefix_depth:
                sizes.append(chooseCandidate(count, [64, 48, 40, 32, 24, 16, 8], 32))
            else:
                sizes.append(chooseCandidate(count, [50, 64, 40, 32, 25, 20, 16, 12, 8], 32))
        return sizes

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
    ordered_points = [(nest.loops[index], point_infos[index]) for index in range(nest.depth)]
    for loop_info, point_info in reversed(ordered_points):
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


def tileDiagnostic(nest: LoopNest, tile_sizes: List[int]) -> Dict[str, object]:
    return {
        "vars": list(nest.vars),
        "family": stencilFamily(nest),
        "tile_sizes": list(tile_sizes),
        "point_order": list(nest.vars),
        "state_prefix_depth": effectiveStateCarriedPrefixDepth(nest),
        "accesses": countArrayAccesses(nest),
    }


def tryTile(loop: Statement, baseTileSize: Optional[int], minDepth: int, l1Bytes: int, counter: List[int], diagnostics: List[Dict[str, object]]) -> Statement:
    if not isinstance(loop, (DoLoop, LabeledDoLoop)):
        return loop
    nest = buildNest(loop)
    if any(isTileVar(var) for var in nest.vars):
        return dcReplace(loop, body=[tryTile(stmt, baseTileSize, minDepth, l1Bytes, counter, diagnostics) for stmt in loop.body])
    override = tileSizesFromEnv()
    tile_probe = override[0] if override else (baseTileSize if baseTileSize is not None else optimalTileSize(max(nest.depth, 1), l1Bytes=l1Bytes))
    should_tile = shouldTileNest(nest, tile_probe, minDepth)
    family = stencilFamily(nest)
    article_like_skewed_stencil = family in {"dirichlet_gs", "coefficient_stencil"} and any(var.startswith("skew_") for var in nest.vars)
    if not should_tile and (article_like_skewed_stencil or family == "gauss_seidel") and nest.depth >= minDepth and isStencilLikeNest(nest) and countArrayAccesses(nest) >= 5:
        should_tile = True
    if not should_tile:
        return dcReplace(loop, body=[tryTile(stmt, baseTileSize, minDepth, l1Bytes, counter, diagnostics) for stmt in loop.body])
    tile_sizes = tileSizesForNest(nest, baseTileSize, l1Bytes, override=override)
    transformed = tileAffineNest(nest, tile_sizes)
    if transformed is not None:
        counter[0] += 1
        diagnostics.append(tileDiagnostic(nest, tile_sizes))
        return transformed
    return dcReplace(loop, body=[tryTile(stmt, baseTileSize, minDepth, l1Bytes, counter, diagnostics) for stmt in loop.body])


def processStmts(stmts: List[Statement], baseTileSize: Optional[int], minDepth: int, l1Bytes: int, counter: List[int], diagnostics: List[Dict[str, object]]) -> List[Statement]:
    result = []
    for stmt in stmts:
        if isinstance(stmt, (DoLoop, LabeledDoLoop)):
            result.append(tryTile(stmt, baseTileSize, minDepth, l1Bytes, counter, diagnostics))
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
        diagnostics: List[Dict[str, object]] = []
        new_statements = processStmts(program.statements, self.baseTileSize, self.minDepth, self.l1Bytes, counter, diagnostics)
        new_subroutines = [
            dcReplace(subroutine, statements=processStmts(subroutine.statements, self.baseTileSize, self.minDepth, self.l1Bytes, counter, diagnostics))
            for subroutine in program.subroutines
        ]
        new_functions = [
            dcReplace(function, statements=processStmts(function.statements, self.baseTileSize, self.minDepth, self.l1Bytes, counter, diagnostics))
            for function in program.functions
        ]
        self.stats = {
            "tiled": counter[0],
            "tile_size": self.baseTileSize if self.baseTileSize is not None else optimalTileSize(2, l1Bytes=self.l1Bytes),
            "tile_override": ",".join(str(x) for x in tileSizesFromEnv()) if tileSizesFromEnv() else "",
            "diagnostics": diagnostics,
        }
        return dcReplace(
            program,
            statements=new_statements,
            subroutines=new_subroutines,
            functions=new_functions,
        )
