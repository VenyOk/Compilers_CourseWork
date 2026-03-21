from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Dict, List, Optional, Set, Tuple

from src.core import (
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


@dataclass
class DependenceVector:
    distances: List[Optional[int]]
    source: ArrayAccess
    sink: ArrayAccess
    carrier: Optional[int] = field(init=False)
    is_loop_independent: bool = field(init=False)

    def __post_init__(self):
        self.carrier = None
        for index, distance in enumerate(self.distances):
            if distance not in (None, 0):
                self.carrier = index
                break
        self.is_loop_independent = self.carrier is None and all(
            distance == 0 for distance in self.distances if distance is not None
        )

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
            if dep.hasNegative():
                continue
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


def activeLoopVars(nest: LoopNest) -> Set[str]:
    active: Set[str] = set()
    for access in collectAccesses(nest.body, set(nest.vars)):
        for index in access.indices:
            for var, coeff in index.coeffs.items():
                if coeff != 0:
                    active.add(var)
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


def preferInterchange(nest: LoopNest) -> bool:
    if nest.depth < 2 or not isAffineNest(nest):
        return False
    has_access, all_affine = accessStatus(nest.body, set(nest.vars))
    if not has_access or not all_affine:
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
    _, all_affine = accessStatus(nest.body, set(nest.vars))
    if not all_affine:
        return False
    if len(activeLoopVars(nest)) != nest.depth:
        return False
    dependencies = computeDependenceVectors(nest)
    return all(dep.allNonNegative() for dep in dependencies)


def shouldTileNest(nest: LoopNest, tile_size: int, min_depth: int) -> bool:
    if nest.depth < min_depth or not isAffineNest(nest):
        return False
    has_access, all_affine = accessStatus(nest.body, set(nest.vars))
    if not has_access or not all_affine:
        return False
    if len(activeLoopVars(nest)) != nest.depth:
        return False
    trip_counts = [estimateTripCount(loop_info) for loop_info in nest.loops[:min(4, nest.depth)]]
    known = [count for count in trip_counts if count is not None]
    if known:
        if max(known) <= 3:
            return False
        if len(known) >= 2 and known[0] * known[1] <= 16:
            return False
    volume = estimateNestVolume(nest, limit_depth=min(3, nest.depth))
    if volume is not None and volume <= max(32, tile_size * 2):
        return False
    dependencies = computeDependenceVectors(nest)
    if dependencies and not needsSkewing(nest):
        deepest = max((dep.carrier for dep in dependencies if dep.carrier is not None), default=0)
        if deepest >= 2:
            return False
    return True


def dependenceBandDepth(nest: LoopNest) -> int:
    dependencies = computeDependenceVectors(nest)
    deepest = max((dep.carrier for dep in dependencies if dep.carrier is not None), default=None)
    if deepest is None:
        return 1
    return deepest + 1


def needsSkewing(nest: LoopNest) -> bool:
    if nest.depth < 2 or not isAffineNest(nest):
        return False
    _, all_affine = accessStatus(nest.body, set(nest.vars))
    if not all_affine:
        return False
    if len(activeLoopVars(nest)) != nest.depth:
        return False
    dependencies = computeDependenceVectors(nest)
    return any(dep.carrier is not None and dep.carrier > 0 for dep in dependencies)


def getSkewMatrix(nest: LoopNest) -> List[List[int]]:
    matrix = [[0 for _ in range(nest.depth)] for _ in range(nest.depth)]
    if not needsSkewing(nest):
        return matrix
    band_depth = min(dependenceBandDepth(nest), nest.depth)
    for inner_index in range(1, band_depth):
        for outer_index in range(inner_index):
            matrix[inner_index][outer_index] = 1
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
    tile_depth = prefixLoopDepth(nest, "tile_")
    if tile_depth == 0:
        return None
    if nest.depth <= tile_depth:
        return None
    footprint = 1
    known = False
    for index in range(tile_depth):
        point_index = tile_depth + index
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


def containsUnsupportedParallelControl(stmts: List[Statement]) -> bool:
    for stmt in stmts:
        if isinstance(stmt, (ReadStatement, WriteStatement, PrintStatement, CallStatement, ReturnStatement, StopStatement, ContinueStatement, GotoStatement, ArithmeticIfStatement, DoWhile, LabeledDoWhile, ExitStatement)):
            return True
        if isinstance(stmt, (DoLoop, LabeledDoLoop)):
            if containsUnsupportedParallelControl(stmt.body):
                return True
            continue
        if isinstance(stmt, IfStatement):
            if containsUnsupportedParallelControl(stmt.then_body):
                return True
            for _, body in stmt.elif_parts:
                if containsUnsupportedParallelControl(body):
                    return True
            if stmt.else_body and containsUnsupportedParallelControl(stmt.else_body):
                return True
            continue
        if isinstance(stmt, SimpleIfStatement):
            if containsUnsupportedParallelControl([stmt.statement]):
                return True
    return False


def collectRegionLoopVars(stmts: List[Statement]) -> Set[str]:
    result: Set[str] = set()
    for stmt in stmts:
        if isinstance(stmt, (DoLoop, LabeledDoLoop)):
            result.add(stmt.var)
            result.update(collectRegionLoopVars(stmt.body))
        elif isinstance(stmt, IfStatement):
            result.update(collectRegionLoopVars(stmt.then_body))
            for _, body in stmt.elif_parts:
                result.update(collectRegionLoopVars(body))
            if stmt.else_body:
                result.update(collectRegionLoopVars(stmt.else_body))
        elif isinstance(stmt, SimpleIfStatement):
            result.update(collectRegionLoopVars([stmt.statement]))
    return result


def isPrivatizableScalarName(name: str) -> bool:
    return name.startswith(("cse_tmp_", "licm_tmp_", "tile_", "skew_", "wf_"))


def exprReadsVariable(expr: Expression, name: str) -> bool:
    if isinstance(expr, Variable):
        return expr.name == name
    if isinstance(expr, ArrayRef):
        return any(exprReadsVariable(index, name) for index in expr.indices)
    if isinstance(expr, BinaryOp):
        return exprReadsVariable(expr.left, name) or exprReadsVariable(expr.right, name)
    if isinstance(expr, UnaryOp):
        return exprReadsVariable(expr.operand, name)
    if isinstance(expr, FunctionCall):
        return any(exprReadsVariable(arg, name) for arg in expr.args)
    return False


def scalarAssignmentCount(stmts: List[Statement], name: str) -> int:
    count = 0
    for stmt in stmts:
        if isinstance(stmt, Assignment):
            if not stmt.indices and stmt.target == name:
                count += 1
            continue
        if isinstance(stmt, (DoLoop, LabeledDoLoop)):
            count += scalarAssignmentCount(stmt.body, name)
            continue
        if isinstance(stmt, IfStatement):
            count += scalarAssignmentCount(stmt.then_body, name)
            for _, body in stmt.elif_parts:
                count += scalarAssignmentCount(body, name)
            if stmt.else_body:
                count += scalarAssignmentCount(stmt.else_body, name)
            continue
        if isinstance(stmt, SimpleIfStatement):
            count += scalarAssignmentCount([stmt.statement], name)
    return count


def readBeforeAssignStatus(stmts: List[Statement], name: str, assigned: bool = False) -> Tuple[bool, bool]:
    current_assigned = assigned
    for stmt in stmts:
        if isinstance(stmt, Assignment):
            reads_name = any(exprReadsVariable(index, name) for index in stmt.indices) or exprReadsVariable(stmt.value, name)
            if reads_name and not current_assigned:
                return False, current_assigned
            if not stmt.indices and stmt.target == name:
                current_assigned = True
            continue
        if isinstance(stmt, (DoLoop, LabeledDoLoop)):
            loop_safe, _ = readBeforeAssignStatus(stmt.body, name, False)
            if not loop_safe:
                return False, current_assigned
            continue
        if isinstance(stmt, IfStatement):
            if exprReadsVariable(stmt.condition, name) and not current_assigned:
                return False, current_assigned
            then_safe, then_assigned = readBeforeAssignStatus(stmt.then_body, name, current_assigned)
            if not then_safe:
                return False, current_assigned
            branch_assigned = []
            for condition, body in stmt.elif_parts:
                if exprReadsVariable(condition, name) and not current_assigned:
                    return False, current_assigned
                branch_safe, branch_after = readBeforeAssignStatus(body, name, current_assigned)
                if not branch_safe:
                    return False, current_assigned
                branch_assigned.append(branch_after)
            if stmt.else_body is not None:
                else_safe, else_assigned = readBeforeAssignStatus(stmt.else_body, name, current_assigned)
                if not else_safe:
                    return False, current_assigned
                branch_assigned.append(else_assigned)
            current_assigned = current_assigned or (then_assigned and all(branch_assigned) if branch_assigned else then_assigned)
            continue
        if isinstance(stmt, SimpleIfStatement):
            if exprReadsVariable(stmt.condition, name) and not current_assigned:
                return False, current_assigned
            nested_safe, nested_assigned = readBeforeAssignStatus([stmt.statement], name, current_assigned)
            if not nested_safe:
                return False, current_assigned
            current_assigned = current_assigned or nested_assigned
    return True, current_assigned


def canPrivatizeUserScalar(stmts: List[Statement], name: str) -> bool:
    if scalarAssignmentCount(stmts, name) < 2:
        return False
    safe, assigned = readBeforeAssignStatus(stmts, name, False)
    return safe and assigned


def collectPrivatizableScalars(stmts: List[Statement], private_vars: Set[str]) -> Set[str]:
    result: Set[str] = set()
    for stmt in stmts:
        if isinstance(stmt, Assignment):
            if not stmt.indices and (stmt.target in private_vars or isPrivatizableScalarName(stmt.target) or canPrivatizeUserScalar(stmts, stmt.target)):
                result.add(stmt.target)
            continue
        if isinstance(stmt, (DoLoop, LabeledDoLoop)):
            nested_private = set(private_vars)
            nested_private.add(stmt.var)
            result.update(collectPrivatizableScalars(stmt.body, nested_private))
            continue
        if isinstance(stmt, IfStatement):
            result.update(collectPrivatizableScalars(stmt.then_body, private_vars))
            for _, body in stmt.elif_parts:
                result.update(collectPrivatizableScalars(body, private_vars))
            if stmt.else_body:
                result.update(collectPrivatizableScalars(stmt.else_body, private_vars))
            continue
        if isinstance(stmt, SimpleIfStatement):
            result.update(collectPrivatizableScalars([stmt.statement], private_vars))
    return result


def hasUnsafeScalarWrites(stmts: List[Statement], private_vars: Set[str]) -> bool:
    for stmt in stmts:
        if isinstance(stmt, Assignment):
            if not stmt.indices and stmt.target not in private_vars and not isPrivatizableScalarName(stmt.target):
                return True
            continue
        if isinstance(stmt, (DoLoop, LabeledDoLoop)):
            nested_private = set(private_vars)
            nested_private.add(stmt.var)
            if hasUnsafeScalarWrites(stmt.body, nested_private):
                return True
            continue
        if isinstance(stmt, IfStatement):
            if hasUnsafeScalarWrites(stmt.then_body, private_vars):
                return True
            for _, body in stmt.elif_parts:
                if hasUnsafeScalarWrites(body, private_vars):
                    return True
            if stmt.else_body and hasUnsafeScalarWrites(stmt.else_body, private_vars):
                return True
            continue
        if isinstance(stmt, SimpleIfStatement):
            if hasUnsafeScalarWrites([stmt.statement], private_vars):
                return True
    return False


def isSafeParallelBody(stmts: List[Statement], private_vars: Set[str]) -> bool:
    if containsUnsupportedParallelControl(stmts):
        return False
    effective_private = set(private_vars)
    effective_private.update(collectPrivatizableScalars(stmts, effective_private))
    if hasUnsafeScalarWrites(stmts, effective_private):
        return False
    return True


def shouldWavefrontNest(nest: LoopNest) -> bool:
    tile_depth = prefixLoopDepth(nest, "tile_")
    if tile_depth < 2:
        return False
    if not hasArrayAccesses(nest):
        return False
    if countArrayAccesses(nest) < 2:
        return False
    if tile_depth == 2 and countArrayAccesses(nest) < 6:
        return False
    band_volume = estimatePrefixVolume(nest, tile_depth)
    if band_volume is not None and band_volume < 2:
        return False
    tile_footprint = estimateTileFootprint(nest)
    if tile_footprint is not None and tile_footprint < 4:
        return False
    working_set = estimateTransformedWorkingSet(nest)
    if working_set is not None and working_set < 64:
        return False
    return True


def shouldParallelizeIndependentNest(nest: LoopNest) -> bool:
    if nest.depth == 0 or not isAffineNest(nest):
        return False
    if not hasArrayAccesses(nest):
        return False
    if nest.loops[0].var not in activeLoopVars(nest):
        return False
    dependencies = computeDependenceVectors(nest)
    if any(dep.carrier == 0 for dep in dependencies):
        return False
    outer_trip = estimateTripCount(nest.loops[0])
    if outer_trip is not None and outer_trip < 4:
        return False
    volume = estimateNestVolume(nest, limit_depth=min(3, nest.depth))
    if volume is not None and volume < 8192:
        return False
    working_set = estimateWorkingSet(nest)
    if working_set is not None and working_set < 8 * 1024:
        return False
    private_vars = collectRegionLoopVars(nest.body) | {nest.loops[0].var}
    return isSafeParallelBody(nest.body, private_vars)


def chooseParallelGrain(nest: LoopNest) -> int:
    outer_trip = estimateTripCount(nest.loops[0])
    if outer_trip is None or outer_trip <= 0:
        return 1
    tile_depth = prefixLoopDepth(nest, "tile_")
    tile_footprint = estimateTileFootprint(nest)
    if tile_footprint is not None:
        work_per_iter = max(1, tile_footprint * max(1, countArrayAccesses(nest)))
    else:
        volume = estimatePrefixVolume(nest, tile_depth) if tile_depth > 0 else estimateNestVolume(nest, limit_depth=min(3, nest.depth))
        if volume is None:
            work_per_iter = max(32, countArrayAccesses(nest) * 8)
        else:
            work_per_iter = max(1, volume // max(outer_trip, 1))
    if work_per_iter >= 4096:
        return 1
    if work_per_iter >= 1024:
        return 2
    if work_per_iter >= 256:
        return 4
    if work_per_iter >= 64:
        return 8
    return 16


def shouldParallelizeTiledBand(nest: LoopNest) -> bool:
    tile_depth = prefixLoopDepth(nest, "tile_")
    if tile_depth == 0:
        return False
    if not hasArrayAccesses(nest):
        return False
    if any(var.startswith("skew_") for var in nest.vars):
        return False
    outer_trip = estimateTripCount(nest.loops[0])
    if outer_trip is not None and outer_trip < 3:
        return False
    band_volume = estimatePrefixVolume(nest, tile_depth)
    if band_volume is not None and band_volume < 16:
        return False
    tile_footprint = estimateTileFootprint(nest)
    if tile_footprint is not None and tile_footprint < 256:
        return False
    working_set = estimateTransformedWorkingSet(nest)
    if working_set is not None and working_set < 8 * 1024:
        return False
    private_vars = collectRegionLoopVars(nest.body) | {nest.loops[0].var}
    return isSafeParallelBody(nest.body, private_vars)


def shouldParallelizeWavefrontBand(nest: LoopNest) -> bool:
    if nest.depth == 0:
        return False
    outer_trip = estimateTripCount(nest.loops[0])
    if outer_trip is not None and outer_trip < 2:
        return False
    tile_depth = prefixLoopDepth(nest, "tile_")
    band_volume = estimatePrefixVolume(nest, tile_depth if tile_depth > 0 else min(2, nest.depth))
    if band_volume is not None and band_volume < 4:
        return False
    tile_footprint = estimateTileFootprint(nest)
    if tile_footprint is not None and tile_footprint < 128:
        return False
    working_set = estimateTransformedWorkingSet(nest)
    if countArrayAccesses(nest) < 6:
        return False
    if working_set is not None and (tile_footprint is not None or tile_depth > 1) and working_set < 4 * 1024:
        return False
    private_vars = collectRegionLoopVars(nest.body) | {nest.loops[0].var}
    return isSafeParallelBody(nest.body, private_vars)
