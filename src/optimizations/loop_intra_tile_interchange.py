from __future__ import annotations

from dataclasses import replace as dcReplace
from typing import List

from src.core import DoLoop, LabeledDoLoop, Program, Statement
from src.optimizations.base import ASTOptimizationPass
from src.optimizations.loop_analysis import LoopNest, LoopInfo, buildNest, chooseIntraTileLoopOrder, prefixLoopDepth


def rebuildNest(nest, tile_depth: int, point_order: List[int]) -> Statement:
    body = nest.body
    for point_index in reversed(point_order):
        loop_info = nest.loops[tile_depth + point_index]
        body = [dcReplace(loop_info.node, body=body)]
    for tile_index in reversed(range(tile_depth)):
        loop_info = nest.loops[tile_index]
        body = [dcReplace(loop_info.node, body=body)]
    return body[0]


def pointSubNest(nest, tile_depth: int) -> LoopNest:
    loops = [
        LoopInfo(
            var=loop_info.var,
            start=loop_info.start,
            end=loop_info.end,
            step=loop_info.step,
            node=loop_info.node,
        )
        for loop_info in nest.loops[tile_depth:]
    ]
    return LoopNest(loops=loops, body=nest.body)


def tryInterchange(loop: Statement, counter: List[int], diagnostics: List[dict]) -> Statement:
    if not isinstance(loop, (DoLoop, LabeledDoLoop)):
        return loop
    nest = buildNest(loop)
    tile_depth = prefixLoopDepth(nest, "tile_")
    point_depth = nest.depth - tile_depth
    if tile_depth > 0 and point_depth > 1:
        point_nest = pointSubNest(nest, tile_depth)
        order = chooseIntraTileLoopOrder(point_nest)
        if order != list(range(point_depth)):
            counter[0] += 1
            diagnostics.append({
                "tile_vars": [loop_info.var for loop_info in nest.loops[:tile_depth]],
                "point_vars": [loop_info.var for loop_info in nest.loops[tile_depth:]],
                "point_order": [point_nest.loops[index].var for index in order],
            })
            return rebuildNest(nest, tile_depth, order)
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
