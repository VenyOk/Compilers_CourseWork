from __future__ import annotations
from typing import List
from dataclasses import replace as dcReplace

from src.core import Program, Statement, DoLoop, LabeledDoLoop
from src.optimizations.base import ASTOptimizationPass
from src.optimizations.loop_analysis import canInterchange, buildNest, preferInterchange


def interchangeNest(outer: Statement, inner: Statement) -> Statement:
    if not isinstance(inner, (DoLoop, LabeledDoLoop)):
        return outer
    if not isinstance(outer, (DoLoop, LabeledDoLoop)):
        return outer
    newInner = dcReplace(outer, start=outer.start, end=outer.end,
                         step=outer.step, body=inner.body)
    newOuter = dcReplace(inner, start=inner.start, end=inner.end,
                         step=inner.step, body=[newInner])
    return newOuter


def tryInterchange(loop: Statement) -> Statement:
    if not isinstance(loop, (DoLoop, LabeledDoLoop)):
        return loop
    body = loop.body
    innerLoops = [s for s in body if isinstance(s, (DoLoop, LabeledDoLoop))]
    nonLoops = [s for s in body if not isinstance(s, (DoLoop, LabeledDoLoop))]
    if len(innerLoops) != 1 or len(nonLoops) != 0:
        newBody = [tryInterchange(s) for s in body]
        if newBody != body:
            return dcReplace(loop, body=newBody)
        return loop
    inner = innerLoops[0]
    nest = buildNest(loop)
    if nest.depth >= 2 and canInterchange(nest) and preferInterchange(nest):
        interchanged = interchangeNest(loop, inner)
        if isinstance(interchanged, (DoLoop, LabeledDoLoop)):
            newBody = [tryInterchange(s) for s in interchanged.body]
            return dcReplace(interchanged, body=newBody)
        return interchanged
    newBody = [tryInterchange(s) for s in body]
    if any(nb is not b for nb, b in zip(newBody, body)):
        return dcReplace(loop, body=newBody)
    return loop


def processStmts(stmts: List[Statement], counter: List[int]) -> List[Statement]:
    result = []
    changed = False
    for stmt in stmts:
        if isinstance(stmt, (DoLoop, LabeledDoLoop)):
            newStmt = tryInterchange(stmt)
            if newStmt is not stmt:
                counter[0] += 1
                changed = True
            result.append(newStmt)
        else:
            result.append(stmt)
    return result if changed else stmts


class LoopInterchange(ASTOptimizationPass):
    name = "LoopInterchange"

    def run(self, program: Program) -> Program:
        counter = [0]
        newStmts = processStmts(program.statements, counter)
        newSubs = [
            dcReplace(s, statements=processStmts(s.statements, counter))
            for s in program.subroutines
        ]
        newFuncs = [
            dcReplace(f, statements=processStmts(f.statements, counter))
            for f in program.functions
        ]
        self.stats = {"interchanged": counter[0]}
        return dcReplace(program, statements=newStmts,
                         subroutines=newSubs, functions=newFuncs)
