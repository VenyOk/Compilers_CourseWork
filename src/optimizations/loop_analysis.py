from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Set
from src.core import (
    Statement, Expression, Assignment,
    DoLoop, LabeledDoLoop,
    Variable, BinaryOp, UnaryOp, IntegerLiteral, ArrayRef,
)


@dataclass
class LoopInfo:
    var: str
    start: Expression
    end: Expression
    step: Expression
    node: Statement


@dataclass
class LoopNest:
    loops: List[LoopInfo]
    body: List[Statement]
    depth: int = field(init=False)

    def __post_init__(self):
        self.depth = len(self.loops)

    @property
    def vars(self) -> List[str]:
        return [li.var for li in self.loops]


@dataclass
class LinearIndex:
    var: Optional[str]
    coeff: int
    const: int


@dataclass
class ArrayAccess:
    array_name: str
    indices: List[LinearIndex]
    is_write: bool


@dataclass
class DependenceVector:
    distances: List[Optional[int]]
    source: ArrayAccess
    sink: ArrayAccess
    is_loop_independent: bool = False

    def hasNegative(self) -> bool:
        return any(d is not None and d < 0 for d in self.distances)

    def allNonNegative(self) -> bool:
        return all(d is None or d >= 0 for d in self.distances)


def parseLinear(expr: Expression, loopVars: Set[str]) -> Optional[LinearIndex]:
    if isinstance(expr, IntegerLiteral):
        return LinearIndex(var=None, coeff=0, const=expr.value)
    if isinstance(expr, Variable):
        if expr.name in loopVars:
            return LinearIndex(var=expr.name, coeff=1, const=0)
        return None
    if isinstance(expr, BinaryOp):
        op = expr.op
        if op == '+':
            left = parseLinear(expr.left, loopVars)
            right = parseLinear(expr.right, loopVars)
            if left is not None and right is not None:
                if left.var is None and right.var is None:
                    return LinearIndex(var=None, coeff=0, const=left.const + right.const)
                if left.var is not None and right.var is None:
                    return LinearIndex(var=left.var, coeff=left.coeff, const=left.const + right.const)
                if left.var is None and right.var is not None:
                    return LinearIndex(var=right.var, coeff=right.coeff, const=left.const + right.const)
        if op == '-':
            left = parseLinear(expr.left, loopVars)
            right = parseLinear(expr.right, loopVars)
            if left is not None and right is not None:
                if left.var is None and right.var is None:
                    return LinearIndex(var=None, coeff=0, const=left.const - right.const)
                if right.var is None:
                    return LinearIndex(var=left.var, coeff=left.coeff, const=left.const - right.const)
                if left.var is None:
                    return LinearIndex(var=right.var, coeff=-right.coeff, const=left.const - right.const)
        if op == '*':
            left = parseLinear(expr.left, loopVars)
            right = parseLinear(expr.right, loopVars)
            if left is not None and right is not None:
                if left.var is None and right.var is None:
                    return LinearIndex(var=None, coeff=0, const=left.const * right.const)
                if left.var is None and right.var is not None:
                    return LinearIndex(var=right.var, coeff=left.const * right.coeff, const=0)
                if left.var is not None and right.var is None:
                    return LinearIndex(var=left.var, coeff=left.coeff * right.const, const=0)
    if isinstance(expr, UnaryOp) and expr.op == '-':
        inner = parseLinear(expr.operand, loopVars)
        if inner is not None:
            return LinearIndex(var=inner.var, coeff=-inner.coeff, const=-inner.const)
    return None


def collectInExpr(expr: Expression, loopVars: Set[str], accesses: List[ArrayAccess], isWrite: bool = False) -> None:
    if isinstance(expr, ArrayRef):
        parsed = [parseLinear(i, loopVars) for i in expr.indices]
        if all(p is not None for p in parsed):
            accesses.append(ArrayAccess(array_name=expr.name, indices=parsed, is_write=isWrite))
        for i in expr.indices:
            collectInExpr(i, loopVars, accesses, False)
    elif isinstance(expr, BinaryOp):
        collectInExpr(expr.left, loopVars, accesses)
        collectInExpr(expr.right, loopVars, accesses)
    elif isinstance(expr, UnaryOp):
        collectInExpr(expr.operand, loopVars, accesses)


def collectInStmt(stmt: Statement, loopVars: Set[str], accesses: List[ArrayAccess]) -> None:
    if isinstance(stmt, Assignment):
        if stmt.indices:
            parsed = [parseLinear(i, loopVars) for i in stmt.indices]
            if all(p is not None for p in parsed):
                accesses.append(ArrayAccess(array_name=stmt.target, indices=parsed, is_write=True))
        collectInExpr(stmt.value, loopVars, accesses, False)
        for i in stmt.indices:
            collectInExpr(i, loopVars, accesses, False)
    elif isinstance(stmt, (DoLoop, LabeledDoLoop)):
        newVars = loopVars | {stmt.var}
        for s in stmt.body:
            collectInStmt(s, newVars, accesses)


def collectAccesses(stmts: List[Statement], loopVars: Set[str]) -> List[ArrayAccess]:
    accesses: List[ArrayAccess] = []
    for stmt in stmts:
        collectInStmt(stmt, loopVars, accesses)
    return accesses


def computeDistances(src: ArrayAccess, snk: ArrayAccess, loopVars: List[str]) -> Optional[List[Optional[int]]]:
    nDims = len(src.indices)
    if nDims != len(snk.indices):
        return None
    varToIdx = {v: i for i, v in enumerate(loopVars)}
    distances = [None] * len(loopVars)
    for dim in range(nDims):
        liSrc = src.indices[dim]
        liSnk = snk.indices[dim]
        if liSrc.var is None and liSnk.var is None:
            continue
        if (liSrc.var is not None and liSrc.var == liSnk.var
                and liSrc.coeff == liSnk.coeff and liSrc.coeff != 0):
            d = liSnk.const - liSrc.const
            varIdx = varToIdx.get(liSrc.var)
            if varIdx is not None:
                if distances[varIdx] is None:
                    distances[varIdx] = d
                elif distances[varIdx] != d:
                    return None
    return distances


def computeDependenceVectors(nest: LoopNest) -> List[DependenceVector]:
    loopVars = set(nest.vars)
    accesses = collectAccesses(nest.body, loopVars)
    deps: List[DependenceVector] = []
    for i, src in enumerate(accesses):
        for j, snk in enumerate(accesses):
            if i == j:
                continue
            if not src.is_write and not snk.is_write:
                continue
            if src.array_name != snk.array_name:
                continue
            if len(src.indices) != len(snk.indices):
                continue
            distances = computeDistances(src, snk, nest.vars)
            if distances is not None:
                isIndep = all(d == 0 for d in distances)
                deps.append(DependenceVector(
                    distances=distances,
                    source=src,
                    sink=snk,
                    is_loop_independent=isIndep,
                ))
    return deps


def buildNest(loop: Statement) -> LoopNest:
    loops: List[LoopInfo] = []
    current = loop
    while isinstance(current, (DoLoop, LabeledDoLoop)):
        li = LoopInfo(
            var=current.var,
            start=current.start,
            end=current.end,
            step=current.step,
            node=current,
        )
        loops.append(li)
        body = current.body
        innerLoops = [s for s in body if isinstance(s, (DoLoop, LabeledDoLoop))]
        nonLoop = [s for s in body if not isinstance(s, (DoLoop, LabeledDoLoop))]
        if len(innerLoops) == 1 and len(nonLoop) == 0:
            current = innerLoops[0]
        else:
            return LoopNest(loops=loops, body=body)
    return LoopNest(loops=loops, body=[])


def extractLoopNests(stmts: List[Statement]) -> List[Tuple[List[Statement], LoopNest, int]]:
    results = []
    for i, stmt in enumerate(stmts):
        if isinstance(stmt, (DoLoop, LabeledDoLoop)):
            nest = buildNest(stmt)
            if nest.depth >= 1:
                results.append((stmts[:i], nest, i))
    return results


def canInterchange(nest: LoopNest) -> bool:
    if nest.depth < 2:
        return False
    deps = computeDependenceVectors(nest)
    for dep in deps:
        if dep.hasNegative():
            return False
    return True


def needsSkewing(nest: LoopNest) -> bool:
    deps = computeDependenceVectors(nest)
    return any(dep.hasNegative() for dep in deps)


def getSkewFactors(nest: LoopNest) -> List[int]:
    deps = computeDependenceVectors(nest)
    n = nest.depth
    factors = [0] * n
    for dep in deps:
        for k in range(1, n):
            dk = dep.distances[k]
            if dk is not None and dk < 0:
                dPrev = dep.distances[k - 1]
                if dPrev is not None and dPrev > 0:
                    s = (-dk + dPrev - 1) // dPrev
                else:
                    s = abs(dk) + 1
                factors[k] = max(factors[k], s)
    return factors
