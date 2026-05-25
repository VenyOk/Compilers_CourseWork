from __future__ import annotations

import math
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Dict, List, Optional, Set, Tuple

from src.frontend.ast import (
    ArrayRef,
    Assignment,
    BinaryOp,
    DoLoop,
    Expression,
    FunctionCall,
    IfStatement,
    IntegerLiteral,
    LabeledDoLoop,
    LabeledDoWhile,
    SimpleIfStatement,
    Statement,
    UnaryOp,
    Variable,
    WriteStatement,
    ReadStatement,
    PrintStatement,
    CallStatement,
    ReturnStatement,
    StopStatement,
    DoWhile,
    ContinueStatement,
    GotoStatement,
    ArithmeticIfStatement,
    ExitStatement,
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
        return [loop_info.var for loop_info in self.loops]

@dataclass
class AffineExpr:
    coeffs: Dict[str, int]
    const: int

    def coeff(self, var: str) -> int:
        return self.coeffs.get(var, 0)

@dataclass
class ArrayAccess:
    array_name: str
    indices: List[AffineExpr]
    is_write: bool
    order: int

def articleModeEnabled() -> bool:
    return True

@dataclass
class DependenceVector:
    distances: List[Optional[int]]
    source: ArrayAccess
    sink: ArrayAccess
    carrier: Optional[int] = field(init=False)
    is_loop_independent: bool = field(init=False)
    carriers: List[int] = field(init=False)

    def __post_init__(self):
        self.carrier = None
        for index, distance in enumerate(self.distances):
            if distance not in (None, 0):
                self.carrier = index
                break
        self.is_loop_independent = self.carrier is None and all(
            distance == 0 for distance in self.distances if distance is not None
        )
        last_negative = None
        for index, distance in enumerate(self.distances):
            if distance is not None and distance < 0:
                last_negative = index
        if last_negative is not None:
            self.carriers = list(range(last_negative))
        elif self.carrier is not None:
            self.carriers = [self.carrier]
        else:
            self.carriers = []

    def hasNegative(self) -> bool:
        return any(distance is not None and distance < 0 for distance in self.distances)

    def allNonNegative(self) -> bool:
        return all(distance is None or distance >= 0 for distance in self.distances)

def constantInt(expr: Expression) -> Optional[int]:
    if isinstance(expr, IntegerLiteral):
        return expr.value
    if isinstance(expr, UnaryOp) and expr.op == "-":
        inner = constantInt(expr.operand)
        if inner is not None:
            return -inner
    return None

def mergeCoeffs(left: Dict[str, int], right: Dict[str, int], sign: int = 1) -> Dict[str, int]:
    result = dict(left)
    for var, coeff in right.items():
        updated = result.get(var, 0) + sign * coeff
        if updated == 0:
            result.pop(var, None)
        else:
            result[var] = updated
    return result

def parseAffine(expr: Expression, loop_vars: Set[str], allow_symbols: bool = False) -> Optional[AffineExpr]:
    if isinstance(expr, IntegerLiteral):
        return AffineExpr(coeffs={}, const=expr.value)
    if isinstance(expr, Variable):
        if expr.name in loop_vars:
            return AffineExpr(coeffs={expr.name: 1}, const=0)
        if allow_symbols:
            return AffineExpr(coeffs={}, const=0)
        return None
    if isinstance(expr, UnaryOp) and expr.op == "-":
        inner = parseAffine(expr.operand, loop_vars, allow_symbols=allow_symbols)
        if inner is None:
            return None
        return AffineExpr(
            coeffs={var: -coeff for var, coeff in inner.coeffs.items()},
            const=-inner.const,
        )
    if isinstance(expr, BinaryOp):
        if expr.op == "+":
            left = parseAffine(expr.left, loop_vars, allow_symbols=allow_symbols)
            right = parseAffine(expr.right, loop_vars, allow_symbols=allow_symbols)
            if left is None or right is None:
                return None
            return AffineExpr(
                coeffs=mergeCoeffs(left.coeffs, right.coeffs),
                const=left.const + right.const,
            )
        if expr.op == "-":
            left = parseAffine(expr.left, loop_vars, allow_symbols=allow_symbols)
            right = parseAffine(expr.right, loop_vars, allow_symbols=allow_symbols)
            if left is None or right is None:
                return None
            return AffineExpr(
                coeffs=mergeCoeffs(left.coeffs, right.coeffs, sign=-1),
                const=left.const - right.const,
            )
        if expr.op == "*":
            left_const = constantInt(expr.left)
            right_const = constantInt(expr.right)
            if left_const is not None:
                right = parseAffine(expr.right, loop_vars, allow_symbols=allow_symbols)
                if right is None:
                    return None
                return AffineExpr(
                    coeffs={var: left_const * coeff for var, coeff in right.coeffs.items()},
                    const=left_const * right.const,
                )
            if right_const is not None:
                left = parseAffine(expr.left, loop_vars, allow_symbols=allow_symbols)
                if left is None:
                    return None
                return AffineExpr(
                    coeffs={var: right_const * coeff for var, coeff in left.coeffs.items()},
                    const=right_const * left.const,
                )
    return None

def isAffineNest(nest: LoopNest) -> bool:
    if nest.depth == 0:
        return False
    loop_vars = set(nest.vars)
    for loop_info in nest.loops:
        if parseAffine(loop_info.start, loop_vars, allow_symbols=True) is None:
            return False
        if parseAffine(loop_info.end, loop_vars, allow_symbols=True) is None:
            return False
        step = constantInt(loop_info.step)
        if step is None or step <= 0:
            return False
    return True

def collectInExpr(
    expr: Expression,
    loop_vars: Set[str],
    accesses: List[ArrayAccess],
    order_ref: List[int],
) -> None:
    if isinstance(expr, ArrayRef):
        for index in expr.indices:
            collectInExpr(index, loop_vars, accesses, order_ref)
        parsed = [parseAffine(index, loop_vars) for index in expr.indices]
        if all(item is not None for item in parsed):
            accesses.append(ArrayAccess(
                array_name=expr.name,
                indices=parsed,
                is_write=False,
                order=order_ref[0],
            ))
            order_ref[0] += 1
        return
    if isinstance(expr, BinaryOp):
        collectInExpr(expr.left, loop_vars, accesses, order_ref)
        collectInExpr(expr.right, loop_vars, accesses, order_ref)
        return
    if isinstance(expr, UnaryOp):
        collectInExpr(expr.operand, loop_vars, accesses, order_ref)
        return
    if isinstance(expr, FunctionCall):
        for arg in expr.args:
            collectInExpr(arg, loop_vars, accesses, order_ref)

def collectInStmt(
    stmt: Statement,
    loop_vars: Set[str],
    accesses: List[ArrayAccess],
    order_ref: List[int],
) -> None:
    if isinstance(stmt, Assignment):
        for index in stmt.indices:
            collectInExpr(index, loop_vars, accesses, order_ref)
        collectInExpr(stmt.value, loop_vars, accesses, order_ref)
        if stmt.indices:
            parsed = [parseAffine(index, loop_vars) for index in stmt.indices]
            if all(item is not None for item in parsed):
                accesses.append(ArrayAccess(
                    array_name=stmt.target,
                    indices=parsed,
                    is_write=True,
                    order=order_ref[0],
                ))
                order_ref[0] += 1
        return
    if isinstance(stmt, (DoLoop, LabeledDoLoop)):
        nested_vars = loop_vars | {stmt.var}
        for inner in stmt.body:
            collectInStmt(inner, nested_vars, accesses, order_ref)
        return
    if isinstance(stmt, (DoWhile, LabeledDoWhile)):
        for inner in stmt.body:
            collectInStmt(inner, loop_vars, accesses, order_ref)
        return
    if isinstance(stmt, IfStatement):
        collectInExpr(stmt.condition, loop_vars, accesses, order_ref)
        for inner in stmt.then_body:
            collectInStmt(inner, loop_vars, accesses, order_ref)
        for condition, body in stmt.elif_parts:
            collectInExpr(condition, loop_vars, accesses, order_ref)
            for inner in body:
                collectInStmt(inner, loop_vars, accesses, order_ref)
        if stmt.else_body:
            for inner in stmt.else_body:
                collectInStmt(inner, loop_vars, accesses, order_ref)
        return
    if isinstance(stmt, SimpleIfStatement):
        collectInExpr(stmt.condition, loop_vars, accesses, order_ref)
        collectInStmt(stmt.statement, loop_vars, accesses, order_ref)
        return
    if isinstance(stmt, PrintStatement):
        for item in stmt.items:
            collectInExpr(item, loop_vars, accesses, order_ref)
        return
    if isinstance(stmt, WriteStatement):
        for item in stmt.items:
            collectInExpr(item, loop_vars, accesses, order_ref)
        return
    if isinstance(stmt, ReadStatement):
        return
    if isinstance(stmt, CallStatement):
        for arg in stmt.args:
            collectInExpr(arg, loop_vars, accesses, order_ref)
        return
    if isinstance(stmt, (ReturnStatement, StopStatement, ContinueStatement)):
        return

def collectAccesses(stmts: List[Statement], loop_vars: Set[str]) -> List[ArrayAccess]:
    accesses: List[ArrayAccess] = []
    order_ref = [0]
    for stmt in stmts:
        collectInStmt(stmt, loop_vars, accesses, order_ref)
    return accesses

def exprAffineStatus(expr: Expression, loop_vars: Set[str]) -> Tuple[bool, bool]:
    if isinstance(expr, ArrayRef):
        nested_access = False
        nested_affine = True
        for index in expr.indices:
            has_access, is_affine = exprAffineStatus(index, loop_vars)
            nested_access = nested_access or has_access
            nested_affine = nested_affine and is_affine
        parsed = [parseAffine(index, loop_vars) for index in expr.indices]
        return True, nested_affine and all(item is not None for item in parsed)
    if isinstance(expr, BinaryOp):
        left_access, left_affine = exprAffineStatus(expr.left, loop_vars)
        right_access, right_affine = exprAffineStatus(expr.right, loop_vars)
        return left_access or right_access, left_affine and right_affine
    if isinstance(expr, UnaryOp):
        return exprAffineStatus(expr.operand, loop_vars)
    if isinstance(expr, FunctionCall):
        has_access = False
        is_affine = True
        for arg in expr.args:
            arg_access, arg_affine = exprAffineStatus(arg, loop_vars)
            has_access = has_access or arg_access
            is_affine = is_affine and arg_affine
        return has_access, is_affine
    return False, True

def stmtAffineStatus(stmt: Statement, loop_vars: Set[str]) -> Tuple[bool, bool]:
    if isinstance(stmt, Assignment):
        has_access = False
        is_affine = True
        for index in stmt.indices:
            index_access, index_affine = exprAffineStatus(index, loop_vars)
            has_access = has_access or index_access
            is_affine = is_affine and index_affine
        value_access, value_affine = exprAffineStatus(stmt.value, loop_vars)
        has_access = has_access or value_access or bool(stmt.indices)
        is_affine = is_affine and value_affine
        if stmt.indices:
            parsed = [parseAffine(index, loop_vars) for index in stmt.indices]
            is_affine = is_affine and all(item is not None for item in parsed)
        return has_access, is_affine
    if isinstance(stmt, (DoLoop, LabeledDoLoop)):
        nested_vars = loop_vars | {stmt.var}
        has_access = False
        is_affine = True
        for inner in stmt.body:
            inner_access, inner_affine = stmtAffineStatus(inner, nested_vars)
            has_access = has_access or inner_access
            is_affine = is_affine and inner_affine
        return has_access, is_affine
    if isinstance(stmt, (DoWhile, LabeledDoWhile)):
        has_access = False
        is_affine = True
        for inner in stmt.body:
            inner_access, inner_affine = stmtAffineStatus(inner, loop_vars)
            has_access = has_access or inner_access
            is_affine = is_affine and inner_affine
        return has_access, is_affine
    if isinstance(stmt, IfStatement):
        has_access, is_affine = exprAffineStatus(stmt.condition, loop_vars)
        for inner in stmt.then_body:
            inner_access, inner_affine = stmtAffineStatus(inner, loop_vars)
            has_access = has_access or inner_access
            is_affine = is_affine and inner_affine
        for condition, body in stmt.elif_parts:
            cond_access, cond_affine = exprAffineStatus(condition, loop_vars)
            has_access = has_access or cond_access
            is_affine = is_affine and cond_affine
            for inner in body:
                inner_access, inner_affine = stmtAffineStatus(inner, loop_vars)
                has_access = has_access or inner_access
                is_affine = is_affine and inner_affine
        if stmt.else_body:
            for inner in stmt.else_body:
                inner_access, inner_affine = stmtAffineStatus(inner, loop_vars)
                has_access = has_access or inner_access
                is_affine = is_affine and inner_affine
        return has_access, is_affine
    if isinstance(stmt, SimpleIfStatement):
        cond_access, cond_affine = exprAffineStatus(stmt.condition, loop_vars)
        stmt_access, stmt_affine = stmtAffineStatus(stmt.statement, loop_vars)
        return cond_access or stmt_access, cond_affine and stmt_affine
    if isinstance(stmt, PrintStatement):
        has_access = False
        is_affine = True
        for item in stmt.items:
            item_access, item_affine = exprAffineStatus(item, loop_vars)
            has_access = has_access or item_access
            is_affine = is_affine and item_affine
        return has_access, is_affine
    if isinstance(stmt, WriteStatement):
        has_access = False
        is_affine = True
        for item in stmt.items:
            item_access, item_affine = exprAffineStatus(item, loop_vars)
            has_access = has_access or item_access
            is_affine = is_affine and item_affine
        return has_access, is_affine
    if isinstance(stmt, CallStatement):
        has_access = False
        is_affine = True
        for arg in stmt.args:
            arg_access, arg_affine = exprAffineStatus(arg, loop_vars)
            has_access = has_access or arg_access
            is_affine = is_affine and arg_affine
        return has_access, is_affine
    return False, True

def accessStatus(stmts: List[Statement], loop_vars: Set[str]) -> Tuple[bool, bool]:
    has_access = False
    all_affine = True
    for stmt in stmts:
        stmt_access, stmt_affine = stmtAffineStatus(stmt, loop_vars)
        has_access = has_access or stmt_access
        all_affine = all_affine and stmt_affine
    return has_access, all_affine

def hasObservableSideEffects(stmts: List[Statement]) -> bool:
    for stmt in stmts:
        if isinstance(stmt, (PrintStatement, WriteStatement, ReadStatement, CallStatement, ReturnStatement, StopStatement, GotoStatement, ArithmeticIfStatement)):
            return True
        if isinstance(stmt, (DoLoop, LabeledDoLoop, DoWhile, LabeledDoWhile)):
            if hasObservableSideEffects(stmt.body):
                return True
        elif isinstance(stmt, IfStatement):
            if hasObservableSideEffects(stmt.then_body):
                return True
            for _, body in stmt.elif_parts:
                if hasObservableSideEffects(body):
                    return True
            if stmt.else_body and hasObservableSideEffects(stmt.else_body):
                return True
        elif isinstance(stmt, SimpleIfStatement):
            if hasObservableSideEffects([stmt.statement]):
                return True
    return False

def coefficientMatrix(access: ArrayAccess, loop_vars: List[str]) -> List[List[int]]:
    return [[index.coeff(var) for var in loop_vars] for index in access.indices]

def solveIntegerSystem(matrix: List[List[int]], rhs: List[int], nvars: int) -> Optional[List[Optional[int]]]:
    active_vars = [
        var_index
        for var_index in range(nvars)
        if any(row[var_index] != 0 for row in matrix)
    ]
    if not active_vars:
        if all(value == 0 for value in rhs):
            return [None] * nvars
        return None
    reduced = [
        [Fraction(row[var_index]) for var_index in active_vars] + [Fraction(value)]
        for row, value in zip(matrix, rhs)
    ]
    rows = len(reduced)
    cols = len(active_vars)
    pivot_cols: List[int] = []
    pivot_row = 0
    for col in range(cols):
        pivot = None
        for row in range(pivot_row, rows):
            if reduced[row][col] != 0:
                pivot = row
                break
        if pivot is None:
            continue
        reduced[pivot_row], reduced[pivot] = reduced[pivot], reduced[pivot_row]
        divisor = reduced[pivot_row][col]
        for index in range(col, cols + 1):
            reduced[pivot_row][index] /= divisor
        for row in range(rows):
            if row == pivot_row or reduced[row][col] == 0:
                continue
            factor = reduced[row][col]
            for index in range(col, cols + 1):
                reduced[row][index] -= factor * reduced[pivot_row][index]
        pivot_cols.append(col)
        pivot_row += 1
        if pivot_row == rows:
            break
    for row in range(rows):
        if all(reduced[row][col] == 0 for col in range(cols)) and reduced[row][cols] != 0:
            return None
    if len(pivot_cols) != cols:
        return None
    reduced_solution = [Fraction(0) for _ in range(cols)]
    for row, col in enumerate(pivot_cols):
        reduced_solution[col] = reduced[row][cols]
    if any(value.denominator != 1 for value in reduced_solution):
        return None
    solution: List[Optional[int]] = [None] * nvars
    for local_index, var_index in enumerate(active_vars):
        solution[var_index] = int(reduced_solution[local_index])
    return solution

def computeDistances(source: ArrayAccess, sink: ArrayAccess, loop_vars: List[str]) -> Optional[List[Optional[int]]]:
    if len(source.indices) != len(sink.indices):
        return None
    source_matrix = coefficientMatrix(source, loop_vars)
    sink_matrix = coefficientMatrix(sink, loop_vars)
    if source_matrix != sink_matrix:
        return None
    rhs = [src_index.const - sink_index.const for src_index, sink_index in zip(source.indices, sink.indices)]
    return solveIntegerSystem(source_matrix, rhs, len(loop_vars))

def computeDependenceVectors(nest: LoopNest) -> List[DependenceVector]:
    if nest.depth == 0 or not isAffineNest(nest):
        return []
    _, all_affine = accessStatus(nest.body, set(nest.vars))
    if not all_affine:
        return []
    accesses = collectAccesses(nest.body, set(nest.vars))
    dependencies: List[DependenceVector] = []
    seen: Set[Tuple[str, Tuple[Optional[int], ...], int, bool]] = set()
    for source in accesses:
        if not source.is_write:
            continue
        for sink in accesses:
            if source.array_name != sink.array_name:
                continue
            if source.order == sink.order:
                continue
            distances = computeDistances(source, sink, nest.vars)
            if distances is None:
                continue
            dep = DependenceVector(
                distances=distances,
                source=source,
                sink=sink,
            )
            key = (
                source.array_name.upper(),
                tuple(distances),
                dep.carrier if dep.carrier is not None else -1,
                sink.is_write,
            )
            if key in seen:
                continue
            seen.add(key)
            dependencies.append(dep)
    prefix_depth = effectiveStateCarriedPrefixDepth(nest)
    if prefix_depth > 0:
        self_arrays = selfDependentArrays(nest)
        extra_dependencies: List[DependenceVector] = []
        for dep in dependencies:
            if dep.source.array_name.upper() not in self_arrays:
                continue
            temporal_distances = list(dep.distances)
            temporal_distances[0] = 1
            temporal_dep = DependenceVector(
                distances=temporal_distances,
                source=dep.source,
                sink=dep.sink,
            )
            key = (
                dep.source.array_name.upper(),
                tuple(temporal_distances),
                temporal_dep.carrier if temporal_dep.carrier is not None else -1,
                dep.sink.is_write,
            )
            if key in seen:
                continue
            seen.add(key)
            extra_dependencies.append(temporal_dep)
        dependencies.extend(extra_dependencies)
    return dependencies

def buildNest(loop: Statement) -> LoopNest:
    loops: List[LoopInfo] = []
    current = loop
    while isinstance(current, (DoLoop, LabeledDoLoop)):
        loops.append(LoopInfo(
            var=current.var,
            start=current.start,
            end=current.end,
            step=current.step,
            node=current,
        ))
        body = current.body
        inner_loops = [stmt for stmt in body if isinstance(stmt, (DoLoop, LabeledDoLoop))]
        non_loops = [stmt for stmt in body if not isinstance(stmt, (DoLoop, LabeledDoLoop))]
        if len(inner_loops) == 1 and len(non_loops) == 0:
            current = inner_loops[0]
        else:
            return LoopNest(loops=loops, body=body)
    return LoopNest(loops=loops, body=[])

def extractLoopNests(stmts: List[Statement]) -> List[Tuple[List[Statement], LoopNest, int]]:
    results = []
    for index, stmt in enumerate(stmts):
        if isinstance(stmt, (DoLoop, LabeledDoLoop)):
            nest = buildNest(stmt)
            if nest.depth >= 1:
                results.append((stmts[:index], nest, index))
    return results

def estimateTripCount(loop_info: LoopInfo) -> Optional[int]:
    start = constantInt(loop_info.start)
    end = constantInt(loop_info.end)
    step = constantInt(loop_info.step)
    if start is None or end is None or step in (None, 0):
        return None
    if step > 0:
        if end < start:
            return 0
        return ((end - start) // step) + 1
    if start < end:
        return 0
    return ((start - end) // (-step)) + 1

def estimateNestVolume(nest: LoopNest, limit_depth: Optional[int] = None) -> Optional[int]:
    volume = 1
    depth = nest.depth if limit_depth is None else min(nest.depth, limit_depth)
    for loop_info in nest.loops[:depth]:
        trip_count = estimateTripCount(loop_info)
        if trip_count is None:
            return None
        volume *= max(trip_count, 0)
    return volume

def hasArrayAccesses(nest: LoopNest) -> bool:
    has_access, _ = accessStatus(nest.body, set(nest.vars))
    return has_access

def isGeneratedLoopVar(var: str) -> bool:
    return var.startswith(("tile_", "skew_"))

def baseActiveLoopVars(nest: LoopNest) -> Set[str]:
    active: Set[str] = set()
    for access in collectAccesses(nest.body, set(nest.vars)):
        for index in access.indices:
            for var, coeff in index.coeffs.items():
                if coeff != 0:
                    active.add(var)
    return active

def selfDependentArrays(nest: LoopNest) -> Set[str]:
    accesses = collectAccesses(nest.body, set(nest.vars))
    read_arrays = {access.array_name.upper() for access in accesses if not access.is_write}
    write_arrays = {access.array_name.upper() for access in accesses if access.is_write}
    return read_arrays & write_arrays

def stateCarriedPrefixDepth(nest: LoopNest) -> int:
    if nest.depth != 3:
        return 0
    base_active = baseActiveLoopVars(nest)
    if not base_active:
        return 0
    if not selfDependentArrays(nest):
        return 0
    depth = 0
    for loop_info in nest.loops:
        if isGeneratedLoopVar(loop_info.var):
            return 0
        if loop_info.var in base_active:
            break
        depth += 1
    return depth if depth < nest.depth else 0

def activeLoopVars(nest: LoopNest) -> Set[str]:
    active = baseActiveLoopVars(nest)
    prefix_depth = effectiveStateCarriedPrefixDepth(nest)
    for loop_info in nest.loops[:prefix_depth]:
        active.add(loop_info.var)
    return active

def localityScore(accesses: List[ArrayAccess], var: str) -> int:
    score = 0
    for access in accesses:
        rank = len(access.indices)
        for dim, index in enumerate(access.indices):
            coeff = abs(index.coeff(var))
            if coeff == 0:
                continue
            score += coeff * (rank - dim)
    return score

def spatialCarrierDepth(dep: DependenceVector, prefix_depth: int) -> Optional[int]:
    for index, distance in enumerate(dep.distances[prefix_depth:]):
        if distance not in (None, 0):
            return index
    return None

def spatialDependenceBandDepth(nest: LoopNest, prefix_depth: int) -> int:
    dependencies = computeDependenceVectors(nest)
    deepest = 0
    seen = False
    for dep in dependencies:
        carrier = spatialCarrierDepth(dep, prefix_depth)
        if carrier is None:
            continue
        deepest = max(deepest, carrier + 1)
        seen = True
    return deepest if seen else 1

def isStencilLikeNest(nest: LoopNest) -> bool:
    if not selfDependentArrays(nest):
        return False
    accesses = collectAccesses(nest.body, set(nest.vars))
    if countArrayAccesses(nest) < 4:
        return False
    if len({access.array_name.upper() for access in accesses}) > 6:
        return False
    if any(not access.indices for access in accesses):
        return False
    return True

def uniqueArrayCount(nest: LoopNest) -> int:
    return len({access.array_name.upper() for access in collectAccesses(nest.body, set(nest.vars))})

def isCoefficientHeavyStencil(nest: LoopNest) -> bool:
    return isStencilLikeNest(nest) and uniqueArrayCount(nest) >= 6 and countArrayAccesses(nest) >= 8

def isSimpleSingleArrayStencil(nest: LoopNest) -> bool:
    return isStencilLikeNest(nest) and uniqueArrayCount(nest) == 1

def effectiveStateCarriedPrefixDepth(nest: LoopNest) -> int:
    if nest.depth >= 3 and not isGeneratedLoopVar(nest.loops[0].var):
        if any(loop_info.var.startswith("skew_") for loop_info in nest.loops[1:]) and selfDependentArrays(nest):
            return 1
    base_active = baseActiveLoopVars(nest)
    if not base_active:
        return 0
    if not selfDependentArrays(nest):
        return 0
    depth = 0
    for loop_info in nest.loops:
        if loop_info.var in base_active:
            break
        depth += 1
    return depth if 0 < depth < nest.depth else 0

def stencilFamily(nest: LoopNest) -> str:
    prefix_depth = effectiveStateCarriedPrefixDepth(nest)
    if nest.depth >= 3 and not isGeneratedLoopVar(nest.loops[0].var) and any(loop_info.var.startswith("skew_") for loop_info in nest.loops[1:]) and selfDependentArrays(nest):
        if isCoefficientHeavyStencil(nest):
            return "dirichlet_gs"
        if isSimpleSingleArrayStencil(nest):
            return "gauss_seidel"
    if isCoefficientHeavyStencil(nest) and prefix_depth > 0:
        return "dirichlet_gs"
    if isSimpleSingleArrayStencil(nest) and prefix_depth > 0:
        return "gauss_seidel"
    if isCoefficientHeavyStencil(nest):
        return "coefficient_stencil"
    if isSimpleSingleArrayStencil(nest):
        return "single_array_stencil"
    if isStencilLikeNest(nest):
        return "stencil"
    if hasArrayAccesses(nest):
        return "affine_loop_nest"
    return "non_array_nest"

def stencilReuseScore(nest: LoopNest) -> int:
    score = 0
    loop_vars = nest.vars
    for dep in computeDependenceVectors(nest):
        if dep.carrier is None:
            continue
        tail = dep.distances[stateCarriedPrefixDepth(nest):]
        for distance in tail:
            if distance in (None, 0):
                continue
            if abs(distance) == 1:
                score += 2
            else:
                score += 1
    for access in collectAccesses(nest.body, set(loop_vars)):
        if access.is_write:
            score += 1
    return score

def axisDependenceScore(nest: LoopNest, var: str) -> int:
    if var not in nest.vars:
        return 0
    var_index = nest.vars.index(var)
    score = 0
    for dep in computeDependenceVectors(nest):
        if var_index >= len(dep.distances):
            continue
        distance = dep.distances[var_index]
        if distance in (None, 0):
            continue
        if abs(distance) == 1:
            score += 6
        else:
            score += 2
        if dep.carrier == var_index:
            score += 4
    return score

def referencedLoopVars(expr: Expression, loop_vars: Set[str]) -> Set[str]:
    if isinstance(expr, Variable):
        return {expr.name} if expr.name in loop_vars else set()
    if isinstance(expr, ArrayRef):
        result: Set[str] = set()
        for index in expr.indices:
            result.update(referencedLoopVars(index, loop_vars))
        return result
    if isinstance(expr, BinaryOp):
        return referencedLoopVars(expr.left, loop_vars) | referencedLoopVars(expr.right, loop_vars)
    if isinstance(expr, UnaryOp):
        return referencedLoopVars(expr.operand, loop_vars)
    if isinstance(expr, FunctionCall):
        result: Set[str] = set()
        for arg in expr.args:
            result.update(referencedLoopVars(arg, loop_vars))
        return result
    return set()

def chooseIntraTileLoopOrder(nest: LoopNest) -> List[int]:
    if nest.depth <= 1:
        return list(range(nest.depth))
    prefix_depth = effectiveStateCarriedPrefixDepth(nest)
    spatial_depth = nest.depth - prefix_depth
    if spatial_depth <= 1:
        return list(range(nest.depth))
    if articleModeEnabled():
        from src.optimizations.article_core import articlePointOrder

        return articlePointOrder(prefix_depth, spatial_depth)
    innermost = nest.depth - 1
    middle = list(range(prefix_depth, nest.depth - 1))
    return list(range(prefix_depth)) + [innermost] + middle

def preferInterchange(nest: LoopNest) -> bool:
    if nest.depth < 2 or not isAffineNest(nest):
        return False
    if hasObservableSideEffects(nest.body):
        return False
    has_access, all_affine = accessStatus(nest.body, set(nest.vars))
    if not has_access or not all_affine:
        return False
    if stateCarriedPrefixDepth(nest) > 0:
        return False
    if len(activeLoopVars(nest)) != nest.depth:
        return False
    volume = estimateNestVolume(nest, limit_depth=2)
    if volume is not None and volume <= 16:
        return False
    dependencies = computeDependenceVectors(nest)
    if any(dep.carrier is not None for dep in dependencies):
        return False
    accesses = collectAccesses(nest.body, set(nest.vars))
    outer_var = nest.vars[0]
    inner_var = nest.vars[1]
    outer_score = localityScore(accesses, outer_var)
    inner_score = localityScore(accesses, inner_var)
    if outer_score <= inner_score:
        return False
    if any(dep.carrier == 0 and (dep.distances[1] or 0) != 0 for dep in dependencies if nest.depth >= 2):
        return False
    return True

def canInterchange(nest: LoopNest) -> bool:
    if nest.depth < 2 or not isAffineNest(nest):
        return False
    if hasObservableSideEffects(nest.body):
        return False
    _, all_affine = accessStatus(nest.body, set(nest.vars))
    if not all_affine:
        return False
    if len(activeLoopVars(nest)) != nest.depth:
        return False
    dependencies = computeDependenceVectors(nest)
    return all(dep.allNonNegative() for dep in dependencies)

def tileDecision(nest: LoopNest, tile_size: int, min_depth: int) -> Tuple[bool, str]:
    if nest.depth < min_depth:
        return False, "depth below tiling threshold"
    if not isAffineNest(nest):
        return False, "non-affine nest"
    if hasObservableSideEffects(nest.body):
        return False, "observable side effects in loop body"
    has_access, all_affine = accessStatus(nest.body, set(nest.vars))
    if not has_access or not all_affine:
        return False, "body is not affine-access dominated"
    if len(activeLoopVars(nest)) != nest.depth:
        return False, "not all loop variables are active in accesses"
    family = stencilFamily(nest)
    if isStencilLikeNest(nest):
        if family == "dirichlet_gs":
            return True, "iterative coefficient stencil follows article tiling path"
        if family == "gauss_seidel":
            return True, "iterative Gauss-Seidel stencil follows article tiling path"
        return True, "iterative stencil follows article tiling path"
    return True, "affine iterative nest follows O3 tiling path"

def shouldTileNest(nest: LoopNest, tile_size: int, min_depth: int) -> bool:
    return tileDecision(nest, tile_size, min_depth)[0]

def dependenceBandDepth(nest: LoopNest) -> int:
    prefix_depth = stateCarriedPrefixDepth(nest)
    spatial_depth = spatialDependenceBandDepth(nest, prefix_depth)
    return min(nest.depth, prefix_depth + spatial_depth)

def needsSkewing(nest: LoopNest) -> bool:
    if nest.depth < 2 or not isAffineNest(nest):
        return False
    if hasObservableSideEffects(nest.body):
        return False
    _, all_affine = accessStatus(nest.body, set(nest.vars))
    if not all_affine:
        return False
    if len(activeLoopVars(nest)) != nest.depth:
        return False
    prefix_depth = effectiveStateCarriedPrefixDepth(nest)
    for dep in computeDependenceVectors(nest):
        for index, distance in enumerate(dep.distances):
            if index < prefix_depth:
                continue
            if distance is not None and distance < 0:
                return True
    return False

def skewDecision(nest: LoopNest) -> Tuple[bool, str]:
    if not needsSkewing(nest):
        return False, "dependence band does not require skewing"
    family = stencilFamily(nest)
    if family == "dirichlet_gs":
        return True, "negative carried dependences require skewing before article tiling"
    return True, "dependence vectors require article skewing"

def shouldSkewNest(nest: LoopNest) -> bool:
    return skewDecision(nest)[0]

def getSkewMatrix(nest: LoopNest) -> List[List[int]]:
    if articleModeEnabled():
        from src.optimizations.article_core import articleSkewMatrix

        return articleSkewMatrix(nest, needsSkewing(nest))
    matrix = [[0 for _ in range(nest.depth)] for _ in range(nest.depth)]
    if not needsSkewing(nest):
        return matrix
    for dep in computeDependenceVectors(nest):
        for index, distance in enumerate(dep.distances):
            if distance is None or distance >= 0:
                continue
            factor = abs(distance)
            for outer_index in range(index):
                matrix[index][outer_index] = max(matrix[index][outer_index], factor)
            for inner_index in range(index + 1, nest.depth):
                matrix[inner_index][index] = max(matrix[inner_index][index], factor)
    return matrix

def getSkewFactors(nest: LoopNest) -> List[int]:
    matrix = getSkewMatrix(nest)
    factors = [0] * nest.depth
    for inner_index in range(1, nest.depth):
        factors[inner_index] = sum(matrix[inner_index][:inner_index])
    return factors

def countArrayAccesses(nest: LoopNest) -> int:
    return len(collectAccesses(nest.body, set(nest.vars)))

def estimateWorkingSet(nest: LoopNest, bytes_per_element: int = 8) -> Optional[int]:
    accesses = collectAccesses(nest.body, set(nest.vars))
    if not accesses:
        return 0
    volume = estimateNestVolume(nest, limit_depth=min(3, nest.depth))
    if volume is None:
        return None
    unique_arrays = len({access.array_name.upper() for access in accesses})
    return unique_arrays * volume * bytes_per_element

def prefixLoopDepth(nest: LoopNest, prefix: str) -> int:
    depth = 0
    for loop_info in nest.loops:
        if loop_info.var.startswith(prefix):
            depth += 1
        else:
            break
    return depth

def isTileLoopVar(var: str) -> bool:
    return var.startswith("tile_")

def temporalPrefixDepth(nest: LoopNest) -> int:
    return effectiveStateCarriedPrefixDepth(nest)

def isIterativeTypeNest(nest: LoopNest) -> bool:
    if nest.depth < 3 or not isAffineNest(nest):
        return False
    if not selfDependentArrays(nest):
        return False
    has_access, all_affine = accessStatus(nest.body, set(nest.vars))
    if not has_access or not all_affine:
        return False
    if articleModeEnabled():
        from src.optimizations.article_core import isCanonicalIterativeNest

        if isCanonicalIterativeNest(nest):
            return True
    family = stencilFamily(nest)
    if family in {"gauss_seidel", "dirichlet_gs"}:
        return True
    prefix_depth = effectiveStateCarriedPrefixDepth(nest)
    if prefix_depth <= 0:
        return False
    base_active = baseActiveLoopVars(nest)
    for loop_info in nest.loops[:prefix_depth]:
        if loop_info.var in base_active:
            return False
    return True

def tileBandStart(nest: LoopNest) -> int:
    return 0 if isIterativeTypeNest(nest) else temporalPrefixDepth(nest)

def sideSliceBandStart(nest: LoopNest) -> int:
    tile_start, tile_count = tileBandSpan(nest)
    if tile_count > 0:
        point_start = tile_start + tile_count
        active = baseActiveLoopVars(nest)
        for index in range(point_start, nest.depth):
            loop_info = nest.loops[index]
            if loop_info.var in active or loop_info.var.startswith("skew_"):
                return index
        return point_start
    if isIterativeTypeNest(nest):
        return temporalPrefixDepth(nest)
    return tileBandStart(nest)

def tileBandSpan(nest: LoopNest) -> Tuple[int, int]:
    start: Optional[int] = None
    for index, loop_info in enumerate(nest.loops):
        if isTileLoopVar(loop_info.var):
            if start is None:
                start = index
        elif start is not None:
            break
    if start is None:
        return 0, 0
    count = 0
    for loop_info in nest.loops[start:]:
        if isTileLoopVar(loop_info.var):
            count += 1
        else:
            break
    return start, count

def pointBandSpan(nest: LoopNest) -> Tuple[int, int]:
    tile_start, tile_count = tileBandSpan(nest)
    if tile_count > 0:
        point_start = tile_start + tile_count
        return point_start, nest.depth - point_start
    prefix = temporalPrefixDepth(nest)
    return prefix, nest.depth - prefix

def articleOptimalTileSide(
    l1_bytes: int = 32 * 1024,
    elem_size: int = 8,
    spatial_dims: int = 2,
    n_arrays: int = 1,
) -> int:
    budget = max(l1_bytes // (max(n_arrays, 1) * elem_size), 16)
    if spatial_dims <= 1:
        return max(2, int(math.sqrt(budget + 1)) - 1)
    if spatial_dims == 2:
        return max(2, int(math.sqrt(budget + 4)) - 2)
    return max(2, int(round((budget + 8) ** (1.0 / spatial_dims) - 2)))

def postSkewDependencesLegal(nest: LoopNest) -> bool:
    if not needsSkewing(nest):
        return True
    return all(dep.allNonNegative() for dep in computeDependenceVectors(nest))

def transformedDependencesLegal(nest: LoopNest, matrix: List[List[int]]) -> bool:
    if nest.depth == 0 or len(matrix) != nest.depth:
        return False
    for row in matrix:
        if len(row) != nest.depth:
            return False
    for dep in computeDependenceVectors(nest):
        transformed: List[Optional[int]] = []
        for row_index, distance in enumerate(dep.distances):
            value = distance
            if value is None:
                transformed.append(None)
                continue
            for col_index in range(row_index):
                coeff = matrix[row_index][col_index]
                if coeff == 0:
                    continue
                outer_distance = dep.distances[col_index]
                if outer_distance is None:
                    value = None
                    break
                value += coeff * outer_distance
            transformed.append(value)
        if any(distance is not None and distance < 0 for distance in transformed):
            return False
    return True

def canInterchangePointOrder(nest: LoopNest, point_start: int, point_order: List[int]) -> bool:
    point_depth = len(point_order)
    if point_order == list(range(point_depth)):
        return False
    point_vars = [loop_info.var for loop_info in nest.loops[point_start:point_start + point_depth]]
    if (
        len(point_vars) == 2
        and all(var.startswith("skew_") for var in point_vars)
    ):
        from src.optimizations.side_slice import sideSlicePointInfos

        point_infos = [
            (loop_info.var, loop_info.start, loop_info.end, loop_info.step)
            for loop_info in nest.loops[point_start:point_start + point_depth]
        ]
        line = nest.loops[point_start].node.line
        col = nest.loops[point_start].node.col
        return sideSlicePointInfos(point_infos, point_order, set(nest.vars), line, col) is not None
    dependencies = computeDependenceVectors(nest)
    if not dependencies:
        return False
    for dep in dependencies:
        distances = dep.distances
        if len(distances) < point_start + point_depth:
            continue
        spatial = distances[point_start:point_start + point_depth]
        permuted = [spatial[index] for index in point_order]
        new_distances = list(distances[:point_start]) + permuted + list(distances[point_start + point_depth:])
        if any(distance is not None and distance < 0 for distance in new_distances):
            return False
    return True

def shouldApplySideSliceInterchange(nest: LoopNest, point_start: int) -> bool:
    _, point_depth = pointBandSpan(nest)
    if point_depth < 2:
        return False
    point_loops = nest.loops[point_start:point_start + point_depth]
    if all(loop_info.var.startswith("skew_") for loop_info in point_loops):
        return stencilFamily(nest) in {
            "gauss_seidel",
            "dirichlet_gs",
            "coefficient_stencil",
            "single_array_stencil",
            "stencil",
        }
    return True

def pointSkewDepth(nest: LoopNest) -> int:
    point_start, point_depth = pointBandSpan(nest)
    return sum(1 for loop_info in nest.loops[point_start:point_start + point_depth] if loop_info.var.startswith("skew_"))

def estimatePrefixVolume(nest: LoopNest, depth: int) -> Optional[int]:
    if depth <= 0:
        return None
    volume = 1
    for loop_info in nest.loops[:min(depth, nest.depth)]:
        trip_count = estimateTripCount(loop_info)
        if trip_count is None:
            return None
        volume *= max(trip_count, 0)
    return volume

def estimateTileFootprint(nest: LoopNest) -> Optional[int]:
    tile_start, tile_depth = tileBandSpan(nest)
    if tile_depth == 0:
        return None
    point_start = tile_start + tile_depth
    if nest.depth <= point_start:
        return None
    footprint = 1
    known = False
    for index in range(tile_depth):
        point_index = point_start + index
        if point_index >= nest.depth:
            break
        tile_step = constantInt(nest.loops[index].step)
        point_step = constantInt(nest.loops[point_index].step)
        if tile_step is None or point_step in (None, 0):
            continue
        extent = max(tile_step // abs(point_step), 1)
        footprint *= extent
        known = True
    if not known:
        return None
    return footprint

def estimateTransformedWorkingSet(nest: LoopNest, bytes_per_element: int = 8) -> Optional[int]:
    accesses = collectAccesses(nest.body, set(nest.vars))
    if not accesses:
        return 0
    tile_footprint = estimateTileFootprint(nest)
    unique_arrays = len({access.array_name.upper() for access in accesses})
    if tile_footprint is not None:
        return unique_arrays * tile_footprint * bytes_per_element
    return estimateWorkingSet(nest, bytes_per_element=bytes_per_element)


def describeNest(nest: LoopNest, tile_size: int = 32, min_depth: int = 2) -> Dict[str, object]:
    tile_ok, tile_reason = tileDecision(nest, tile_size, min_depth)
    skew_ok, skew_reason = skewDecision(nest)
    return {
        "vars": list(nest.vars),
        "depth": nest.depth,
        "family": stencilFamily(nest),
        "trip_counts": [estimateTripCount(loop_info) for loop_info in nest.loops],
        "accesses": countArrayAccesses(nest),
        "unique_arrays": uniqueArrayCount(nest),
        "working_set": estimateWorkingSet(nest),
        "state_prefix_depth": effectiveStateCarriedPrefixDepth(nest),
        "reuse_score": stencilReuseScore(nest) if isStencilLikeNest(nest) else 0,
        "skew": {"apply": skew_ok, "reason": skew_reason, "matrix": getSkewMatrix(nest)},
        "tile": {"apply": tile_ok, "reason": tile_reason},
        "point_order": [nest.loops[index].var for index in chooseIntraTileLoopOrder(nest)],
    }
