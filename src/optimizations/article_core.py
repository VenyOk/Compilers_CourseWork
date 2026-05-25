from __future__ import annotations

from typing import List, Optional

from src.frontend.ast import Assignment, Statement
from src.optimizations.loop_analysis import (
    LoopNest,
    articleOptimalTileSide,
    collectAccesses,
    computeDependenceVectors,
    effectiveStateCarriedPrefixDepth,
    estimateTripCount,
    isAffineNest,
    parseAffine,
    selfDependentArrays,
    uniqueArrayCount,
)

def maxStencilOffset(nest: LoopNest) -> Optional[int]:
    prefix = effectiveStateCarriedPrefixDepth(nest)
    temporal_vars = {nest.loops[index].var for index in range(prefix)}
    spatial_vars = set(nest.vars[prefix:])
    limit = 0
    for access in collectAccesses(nest.body, set(nest.vars)):
        for index in access.indices:
            affine = parseAffine(index, set(nest.vars))
            if affine is None:
                return None
            for var, coeff in affine.coeffs.items():
                if var in temporal_vars and coeff != 0:
                    return None
                if var not in spatial_vars and coeff != 0:
                    return None
                limit = max(limit, abs(coeff))
            limit = max(limit, abs(affine.const))
    return limit

def countInnerAssignments(stmts: List[Statement]) -> int:
    total = 0
    for stmt in stmts:
        if isinstance(stmt, Assignment) and stmt.indices:
            total += 1
    return total

def isCanonicalIterativeNest(nest: LoopNest) -> bool:
    if nest.depth < 3 or not isAffineNest(nest):
        return False
    prefix = effectiveStateCarriedPrefixDepth(nest)
    if prefix <= 0 or prefix >= nest.depth:
        return False
    if not selfDependentArrays(nest):
        return False
    temporal_vars = {nest.loops[index].var for index in range(prefix)}
    for access in collectAccesses(nest.body, set(nest.vars)):
        for index in access.indices:
            affine = parseAffine(index, set(nest.vars))
            if affine is None:
                return False
            for var, coeff in affine.coeffs.items():
                if var in temporal_vars and coeff != 0:
                    return False
    offset = maxStencilOffset(nest)
    if offset is None or offset > 8:
        return False
    if countInnerAssignments(nest.body) < 1:
        return False
    return True

def articleSliceVolume(sizes: List[int]) -> int:
    if not sizes:
        return 0
    if len(sizes) == 1:
        return sizes[0] + 2
    if len(sizes) == 2:
        d1, d2 = sizes
        return (d1 + 2) * (d2 + 2) - 4
    volume = 1
    for size in sizes:
        volume *= size + 2
    return volume - 2 * len(sizes)

def fitTileSize(count: Optional[int], side: int) -> int:
    if count is None:
        return max(side, 2)
    if count <= 2:
        return max(2, count)
    if count <= side:
        return max(2, count)
    for candidate in [side, side - 2, side + 2, side * 2 // 3, side // 2, 16, 12, 8, 6, 4, 2]:
        if candidate >= 2 and candidate < count:
            return candidate
    return max(2, min(side, count))

def articleSkewMatrix(nest: LoopNest, needs_skew: bool) -> List[List[int]]:
    depth = nest.depth
    matrix = [[0 for _ in range(depth)] for _ in range(depth)]
    if not needs_skew:
        return matrix
    prefix = effectiveStateCarriedPrefixDepth(nest)
    for dep in computeDependenceVectors(nest):
        for m, distance in enumerate(dep.distances):
            if distance is None or distance >= 0:
                continue
            if m < prefix:
                continue
            factor = abs(distance)
            for carrier in dep.carriers:
                if carrier < m:
                    matrix[m][carrier] = max(matrix[m][carrier], factor)
            for inner in range(m + 1, depth):
                matrix[inner][m] = max(matrix[inner][m], factor)
    return matrix

def articleTileSizesForNest(
    nest: LoopNest,
    l1_bytes: int,
    elem_size: int,
    base_tile_size: Optional[int],
    override: Optional[List[int]],
) -> List[int]:
    prefix = effectiveStateCarriedPrefixDepth(nest)
    from src.optimizations.loop_analysis import isIterativeTypeNest

    iterative = isIterativeTypeNest(nest)
    n_arrays = uniqueArrayCount(nest)
    budget = max(l1_bytes // (max(n_arrays, 1) * elem_size), 16)
    if override:
        if iterative:
            values = list(override[:nest.depth])
            while len(values) < nest.depth:
                values.append(values[-1] if values else 32)
            if nest.depth >= prefix + 2:
                values[-1] = values[-2]
            return values
        spatial_override = list(override[: nest.depth - prefix])
        while len(spatial_override) < nest.depth - prefix:
            spatial_override.append(spatial_override[-1] if spatial_override else 32)
        return [1] * prefix + spatial_override
    if iterative:
        spatial_depth = nest.depth - prefix
        spatial_side = base_tile_size if base_tile_size is not None else articleOptimalTileSide(
            l1_bytes=l1_bytes,
            elem_size=elem_size,
            spatial_dims=max(spatial_depth, 2),
            n_arrays=n_arrays,
        )
        temporal_candidates = sorted({32, 40, 48, 64, spatial_side})
        best: Optional[List[int]] = None
        best_volume = -1
        for temporal in temporal_candidates:
            spatial_sizes = [spatial_side] * spatial_depth
            if spatial_depth >= 2:
                spatial_sizes[-1] = spatial_sizes[-2]
            trial = [temporal] + spatial_sizes
            fitted = [
                fitTileSize(estimateTripCount(loop_info), trial[index])
                for index, loop_info in enumerate(nest.loops)
            ]
            if spatial_depth >= 2:
                fitted[-1] = fitted[-2]
            if spatial_depth >= 3:
                volume = articleSliceVolume(fitted[prefix:prefix + spatial_depth])
            else:
                face = fitted[prefix:prefix + 2] if spatial_depth >= 2 else fitted[prefix:]
                volume = articleSliceVolume(face)
            if volume <= budget and volume >= best_volume:
                best_volume = volume
                best = fitted
        if best is not None:
            return best
        fallback = [spatial_side] * nest.depth
        if nest.depth >= prefix + 2:
            fallback[-1] = fallback[-2]
        return [
            fitTileSize(estimateTripCount(loop_info), fallback[index])
            for index, loop_info in enumerate(nest.loops)
        ]
    band_depth = nest.depth - prefix
    side = base_tile_size if base_tile_size is not None else articleOptimalTileSide(
        l1_bytes=l1_bytes,
        elem_size=elem_size,
        spatial_dims=max(band_depth, 2),
        n_arrays=n_arrays,
    )
    sizes: List[int] = []
    for index, loop_info in enumerate(nest.loops):
        if index < prefix:
            sizes.append(1)
        else:
            sizes.append(fitTileSize(estimateTripCount(loop_info), side))
    if band_depth >= 2:
        sizes[-1] = sizes[-2]
    return sizes

def articlePointOrder(prefix_depth: int, spatial_depth: int) -> List[int]:
    if spatial_depth <= 1:
        return list(range(prefix_depth + spatial_depth))
    if spatial_depth == 2:
        return list(range(prefix_depth)) + [prefix_depth + 1, prefix_depth + 0]
    outer = list(range(prefix_depth + 2, prefix_depth + spatial_depth))
    face = [prefix_depth + 1, prefix_depth + 0]
    return list(range(prefix_depth)) + outer + face
