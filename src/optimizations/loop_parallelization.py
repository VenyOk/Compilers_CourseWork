from __future__ import annotations

from dataclasses import replace as dcReplace
from typing import List

from src.core import DoLoop, IfStatement, LabeledDoLoop, ParallelDoLoop, Program, SimpleIfStatement, Statement
from src.optimizations.base import ASTOptimizationPass
from src.optimizations.loop_analysis import (
    buildNest,
    chooseParallelGrain,
    collectPrivatizableScalars,
    collectRegionLoopVars,
    shouldParallelizeIndependentNest,
    shouldParallelizeTiledBand,
    shouldParallelizeWavefrontBand,
)


def isWavefrontVar(var: str) -> bool:
    return var.startswith("wf_")


def isTileVar(var: str) -> bool:
    return var.startswith("tile_")


def parallelizeStmt(stmt: Statement, counter: List[int], insideWavefront: bool = False) -> Statement:
    if isinstance(stmt, ParallelDoLoop):
        return stmt
    if isinstance(stmt, DoLoop):
        nest = buildNest(stmt)
        if insideWavefront and isTileVar(stmt.var) and shouldParallelizeWavefrontBand(nest):
            private_base = collectRegionLoopVars(stmt.body) | {stmt.var}
            private_vars = sorted(collectPrivatizableScalars(stmt.body, private_base))
            counter[0] += 1
            return ParallelDoLoop(
                var=stmt.var,
                start=stmt.start,
                end=stmt.end,
                step=stmt.step,
                body=stmt.body,
                stmt_label=stmt.stmt_label,
                line=stmt.line,
                col=stmt.col,
                grain=chooseParallelGrain(nest),
                threads_hint=0,
                strategy="wavefront",
                private_vars=private_vars,
            )
        if not insideWavefront and isTileVar(stmt.var) and shouldParallelizeTiledBand(nest):
            private_base = collectRegionLoopVars(stmt.body) | {stmt.var}
            private_vars = sorted(collectPrivatizableScalars(stmt.body, private_base))
            counter[0] += 1
            return ParallelDoLoop(
                var=stmt.var,
                start=stmt.start,
                end=stmt.end,
                step=stmt.step,
                body=stmt.body,
                stmt_label=stmt.stmt_label,
                line=stmt.line,
                col=stmt.col,
                grain=chooseParallelGrain(nest),
                threads_hint=0,
                strategy="tiled",
                private_vars=private_vars,
            )
        if not insideWavefront and shouldParallelizeIndependentNest(nest):
            private_base = collectRegionLoopVars(stmt.body) | {stmt.var}
            private_vars = sorted(collectPrivatizableScalars(stmt.body, private_base))
            counter[0] += 1
            return ParallelDoLoop(
                var=stmt.var,
                start=stmt.start,
                end=stmt.end,
                step=stmt.step,
                body=stmt.body,
                stmt_label=stmt.stmt_label,
                line=stmt.line,
                col=stmt.col,
                grain=chooseParallelGrain(nest),
                threads_hint=0,
                strategy="independent",
                private_vars=private_vars,
            )
        next_inside = insideWavefront or isWavefrontVar(stmt.var)
        new_body = [parallelizeStmt(inner, counter, next_inside) for inner in stmt.body]
        if new_body != stmt.body:
            return dcReplace(stmt, body=new_body)
        return stmt
    if isinstance(stmt, LabeledDoLoop):
        next_inside = insideWavefront or isWavefrontVar(stmt.var)
        new_body = [parallelizeStmt(inner, counter, next_inside) for inner in stmt.body]
        if new_body != stmt.body:
            return dcReplace(stmt, body=new_body)
        return stmt
    if isinstance(stmt, IfStatement):
        new_then = [parallelizeStmt(inner, counter, insideWavefront) for inner in stmt.then_body]
        new_elif = [
            (cond, [parallelizeStmt(inner, counter, insideWavefront) for inner in body])
            for cond, body in stmt.elif_parts
        ]
        new_else = [parallelizeStmt(inner, counter, insideWavefront) for inner in stmt.else_body] if stmt.else_body else stmt.else_body
        if new_then != stmt.then_body or new_elif != stmt.elif_parts or new_else != stmt.else_body:
            return dcReplace(stmt, then_body=new_then, elif_parts=new_elif, else_body=new_else)
        return stmt
    if isinstance(stmt, SimpleIfStatement):
        new_stmt = parallelizeStmt(stmt.statement, counter, insideWavefront)
        if new_stmt is not stmt.statement:
            return dcReplace(stmt, statement=new_stmt)
        return stmt
    return stmt


def processStatements(stmts: List[Statement], counter: List[int]) -> List[Statement]:
    return [parallelizeStmt(stmt, counter, False) for stmt in stmts]


class LoopParallelization(ASTOptimizationPass):
    name = "LoopParallelization"

    def run(self, program: Program) -> Program:
        counter = [0]
        new_statements = processStatements(program.statements, counter)
        new_subroutines = [
            dcReplace(subroutine, statements=processStatements(subroutine.statements, counter))
            for subroutine in program.subroutines
        ]
        new_functions = [
            dcReplace(function, statements=processStatements(function.statements, counter))
            for function in program.functions
        ]
        self.stats = {"parallelized": counter[0]}
        return dcReplace(
            program,
            statements=new_statements,
            subroutines=new_subroutines,
            functions=new_functions,
        )
