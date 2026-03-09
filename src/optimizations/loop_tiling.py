from __future__ import annotations
import math
from typing import List, Optional
from dataclasses import replace as dcReplace

from src.core import (
    Program, Statement, Expression,
    DoLoop, LabeledDoLoop,
    Variable, IntegerLiteral, FunctionCall, BinaryOp,
)
from src.optimizations.base import ASTOptimizationPass
from src.optimizations.loop_analysis import LoopNest, buildNest


def optimalTileSize(l1Bytes: int = 32 * 1024, elemSize: int = 8) -> int:
    nElems = l1Bytes // elemSize
    t = int(math.sqrt(nElems + 4)) - 2
    return max(t, 2)


def minExpr(a: Expression, b: Expression) -> Expression:
    return FunctionCall(name='MIN', args=[a, b], line=a.line, col=a.col)


def addExpr(a: Expression, b: int) -> Expression:
    if b == 0:
        return a
    return BinaryOp(left=a, op='+', right=IntegerLiteral(value=b, line=a.line, col=a.col),
                    line=a.line, col=a.col)


def tilePerfectNest(nest: LoopNest, tileSize: int) -> Statement:
    loops = nest.loops
    body = nest.body
    n = len(loops)
    line = loops[0].node.line
    col = loops[0].node.col
    T = IntegerLiteral(value=tileSize, line=line, col=col)
    pointBody = body
    for k in range(n - 1, -1, -1):
        li = loops[k]
        tileVar = f"{li.var}T"
        tileVarExpr = Variable(name=tileVar, line=line, col=col)
        upper = minExpr(addExpr(tileVarExpr, tileSize - 1), li.end)
        innerLoop = DoLoop(
            var=li.var, start=tileVarExpr, end=upper, step=li.step,
            body=pointBody, stmt_label=None, line=line, col=col,
        )
        pointBody = [innerLoop]
    tileBody = pointBody
    for k in range(n - 1, -1, -1):
        li = loops[k]
        tileVar = f"{li.var}T"
        outerLoop = DoLoop(
            var=tileVar, start=li.start, end=li.end, step=T,
            body=tileBody, stmt_label=None, line=line, col=col,
        )
        tileBody = [outerLoop]
    return tileBody[0]


def tryTile(loop: Statement, tileSize: int, minDepth: int, counter: List[int]) -> Statement:
    if not isinstance(loop, (DoLoop, LabeledDoLoop)):
        return loop
    nest = buildNest(loop)
    if nest.depth >= minDepth:
        tiled = tilePerfectNest(nest, tileSize)
        counter[0] += 1
        return tiled
    newBody = [tryTile(s, tileSize, minDepth, counter) for s in loop.body]
    return dcReplace(loop, body=newBody)


def collectTileVars(stmts: List[Statement], result: set) -> None:
    for stmt in stmts:
        if isinstance(stmt, (DoLoop, LabeledDoLoop)):
            if stmt.var.endswith('T') and len(stmt.var) >= 2:
                result.add(stmt.var)
            collectTileVars(stmt.body, result)


def processStmts(stmts: List[Statement], tileSize: int, minDepth: int, counter: List[int]) -> List[Statement]:
    result = []
    for stmt in stmts:
        if isinstance(stmt, (DoLoop, LabeledDoLoop)):
            result.append(tryTile(stmt, tileSize, minDepth, counter))
        else:
            result.append(stmt)
    return result


class LoopTiling(ASTOptimizationPass):
    name = "LoopTiling"

    def __init__(self, tileSize: Optional[int] = None, minDepth: int = 2, l1Bytes: int = 32 * 1024):
        super().__init__()
        self.tileSize = tileSize if tileSize is not None else optimalTileSize(l1Bytes)
        self.minDepth = minDepth

    def run(self, program: Program) -> Program:
        from src.core import Declaration
        counter = [0]
        newStmts = processStmts(program.statements, self.tileSize, self.minDepth, counter)
        newSubs = [
            dcReplace(s, statements=processStmts(s.statements, self.tileSize, self.minDepth, counter))
            for s in program.subroutines
        ]
        newFuncs = [
            dcReplace(f, statements=processStmts(f.statements, self.tileSize, self.minDepth, counter))
            for f in program.functions
        ]
        tileVars: set = set()
        collectTileVars(newStmts, tileVars)
        existingNames: set = {
            name for decl in program.declarations
            if isinstance(decl, Declaration)
            for name, _ in decl.names
        }
        newTileVars = [v for v in sorted(tileVars) if v not in existingNames]
        newDecls = list(program.declarations)
        if newTileVars:
            tileDecl = Declaration(
                type='INTEGER',
                names=[(v, None) for v in newTileVars],
                line=0, col=0,
            )
            newDecls = [tileDecl] + newDecls
        self.stats = {"tiled": counter[0], "tile_size": self.tileSize}
        return dcReplace(program, statements=newStmts,
                         declarations=newDecls,
                         subroutines=newSubs, functions=newFuncs)
