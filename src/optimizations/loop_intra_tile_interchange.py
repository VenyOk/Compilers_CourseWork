from __future__ import annotations

from dataclasses import replace as dcReplace
from typing import List

from src.frontend.ast import DoLoop, LabeledDoLoop, Program, Statement
from src.optimizations.base import ASTOptimizationPass
from src.optimizations.loop_analysis import (
    LoopInfo,
    LoopNest,
    buildNest,
    canInterchangePointOrder,
    chooseIntraTileLoopOrder,
    pointBandSpan,
    sideSliceBandStart,
    tileBandSpan,
)
from src.optimizations.side_slice import (
    applySideSliceToPointInfos,
    buildPointLoops,
    desiredPointVarOrder,
    sideSliceOrderApplied,
)

def wrapLoop(loop_info: LoopInfo, body: List[Statement]) -> Statement:
    if isinstance(loop_info.node, LabeledDoLoop):
        return dcReplace(
            loop_info.node,
            var=loop_info.var,
            start=loop_info.start,
            end=loop_info.end,
            step=loop_info.step,
            body=body,
        )
    return DoLoop(
        var=loop_info.var,
        start=loop_info.start,
        end=loop_info.end,
        step=loop_info.step,
        body=body,
        stmt_label=None,
        line=loop_info.node.line,
        col=loop_info.node.col,
    )

def pointSubNest(nest: LoopNest, point_start: int, point_depth: int) -> LoopNest:
    loops = [
        LoopInfo(
            var=loop_info.var,
            start=loop_info.start,
            end=loop_info.end,
            step=loop_info.step,
            node=loop_info.node,
        )
        for loop_info in nest.loops[point_start:point_start + point_depth]
    ]
    return LoopNest(loops=loops, body=nest.body)

def rebuildNest(nest: LoopNest, point_start: int, point_order: List[int]) -> Statement:
    point_depth = len(point_order)
    tile_start, tile_count = tileBandSpan(nest)
    line = nest.loops[point_start].node.line
    col = nest.loops[point_start].node.col
    point_infos = [
        (
            loop_info.var,
            loop_info.start,
            loop_info.end,
            loop_info.step,
        )
        for loop_info in nest.loops[point_start:point_start + point_depth]
    ]
    remapped, side_sliced = applySideSliceToPointInfos(
        point_infos,
        point_order,
        set(nest.vars),
        line,
        col,
    )
    body = nest.body
    if side_sliced:
        body = buildPointLoops(remapped, body, line, col)
    else:
        for point_index in reversed(point_order):
            loop_info = nest.loops[point_start + point_index]
            body = [wrapLoop(loop_info, body)]
    if tile_count > 0:
        for tile_index in reversed(range(tile_count)):
            loop_info = nest.loops[tile_start + tile_index]
            body = [wrapLoop(loop_info, body)]
        prefix_end = tile_start
    else:
        prefix_end = point_start
    for prefix_index in reversed(range(prefix_end)):
        loop_info = nest.loops[prefix_index]
        body = [wrapLoop(loop_info, body)]
    return body[0]

def tryInterchange(loop: Statement, counter: List[int], diagnostics: List[dict]) -> Statement:
    if not isinstance(loop, (DoLoop, LabeledDoLoop)):
        return loop
    nest = buildNest(loop)
    point_start, point_depth = pointBandSpan(nest)
    _, tile_count = tileBandSpan(nest)
    slice_start = sideSliceBandStart(nest)
    if point_depth > 1 and (tile_count > 0 or point_start > 0):
        if sideSliceOrderApplied(nest, slice_start, nest.depth - slice_start):
            return dcReplace(loop, body=[tryInterchange(stmt, counter, diagnostics) for stmt in loop.body])
        point_nest = pointSubNest(nest, slice_start, nest.depth - slice_start)
        order = chooseIntraTileLoopOrder(point_nest)
        if order != list(range(point_nest.depth)) and canInterchangePointOrder(nest, slice_start, order):
            counter[0] += 1
            tile_start, _ = tileBandSpan(nest)
            diagnostics.append({
                "tile_vars": [loop_info.var for loop_info in nest.loops[tile_start:tile_start + tile_count]],
                "point_vars": [loop_info.var for loop_info in nest.loops[slice_start:nest.depth]],
                "point_order": desiredPointVarOrder(nest, slice_start, nest.depth - slice_start),
            })
            return rebuildNest(nest, slice_start, order)
    new_body = [tryInterchange(stmt, counter, diagnostics) for stmt in loop.body]
    if new_body != loop.body:
        return dcReplace(loop, body=new_body)
    return loop

def processStatements(statements: List[Statement], counter: List[int], diagnostics: List[dict]) -> List[Statement]:
    result = []
    for stmt in statements:
        if isinstance(stmt, (DoLoop, LabeledDoLoop)):
            result.append(tryInterchange(stmt, counter, diagnostics))
        else:
            result.append(stmt)
    return result

class IntraTileLoopInterchange(ASTOptimizationPass):
    name = "IntraTileLoopInterchange"

    def run(self, program: Program) -> Program:
        counter = [0]
        diagnostics: List[dict] = []
        new_statements = processStatements(program.statements, counter, diagnostics)
        new_subroutines = [
            dcReplace(subroutine, statements=processStatements(subroutine.statements, counter, diagnostics))
            for subroutine in program.subroutines
        ]
        new_functions = [
            dcReplace(function, statements=processStatements(function.statements, counter, diagnostics))
            for function in program.functions
        ]
        self.stats = {"interchanged": counter[0], "diagnostics": diagnostics}
        return dcReplace(
            program,
            statements=new_statements,
            subroutines=new_subroutines,
            functions=new_functions,
        )
