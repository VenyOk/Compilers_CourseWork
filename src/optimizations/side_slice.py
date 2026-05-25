from __future__ import annotations

from typing import List, Optional, Set, Tuple

from src.frontend.ast import BinaryOp, DoLoop, Expression, FunctionCall, IntegerLiteral, Statement, Variable
from src.optimizations.loop_analysis import LoopNest, parseAffine

PointInfo = Tuple[str, Expression, Expression, Expression]

def intExpr(value: int, line: int, col: int) -> IntegerLiteral:
    return IntegerLiteral(value=value, line=line, col=col)

def addExpr(left: Expression, right: Expression, line: int, col: int) -> Expression:
    if isinstance(left, IntegerLiteral) and left.value == 0:
        return right
    if isinstance(right, IntegerLiteral) and right.value == 0:
        return left
    return BinaryOp(left=left, op="+", right=right, line=line, col=col)

def subExpr(left: Expression, right: Expression, line: int, col: int) -> Expression:
    if isinstance(right, IntegerLiteral) and right.value == 0:
        return left
    return BinaryOp(left=left, op="-", right=right, line=line, col=col)

def maxExpr(left: Expression, right: Expression, line: int, col: int) -> FunctionCall:
    return FunctionCall(name="MAX", args=[left, right], line=line, col=col)

def minExpr(left: Expression, right: Expression, line: int, col: int) -> FunctionCall:
    return FunctionCall(name="MIN", args=[left, right], line=line, col=col)

def boundCoreOffset(expr: Expression, var: str, loop_vars: Set[str]) -> Optional[int]:
    if isinstance(expr, FunctionCall) and expr.name in ("MAX", "MIN") and len(expr.args) == 2:
        for arg in expr.args:
            affine = parseAffine(arg, loop_vars)
            if affine is not None and affine.coeff(var) == 1:
                return affine.const
        return None
    affine = parseAffine(expr, loop_vars)
    if affine is not None and affine.coeff(var) == 1:
        return affine.const
    return None

def replaceBoundCore(expr: Expression, var: str, new_core: Expression, loop_vars: Set[str], line: int, col: int) -> Expression:
    if isinstance(expr, FunctionCall) and expr.name in ("MAX", "MIN") and len(expr.args) == 2:
        new_args: List[Expression] = []
        replaced = False
        for arg in expr.args:
            affine = parseAffine(arg, loop_vars)
            if not replaced and affine is not None and affine.coeff(var) != 0:
                new_args.append(new_core)
                replaced = True
            else:
                new_args.append(arg)
        if replaced:
            return FunctionCall(name=expr.name, args=new_args, line=line, col=col)
    return new_core

def sideSlicePair(point_infos: List[PointInfo], loop_vars: Set[str], line: int, col: int) -> Optional[List[PointInfo]]:
    if len(point_infos) != 2:
        return None
    return sideSlicePointInfos(point_infos, [1, 0], loop_vars, line, col)

def sideSlicePointInfos(
    point_infos: List[PointInfo],
    point_order: List[int],
    loop_vars: Set[str],
    line: int,
    col: int,
) -> Optional[List[PointInfo]]:
    if point_order == list(range(len(point_order))):
        return list(point_infos)
    if len(point_order) != 2 or point_order != [1, 0]:
        return None
    inner_info = point_infos[0]
    outer_info = point_infos[1]
    var_inner, start_inner, end_inner, step_inner = inner_info
    var_outer, start_outer, end_outer, step_outer = outer_info
    lower_offset = boundCoreOffset(start_outer, var_inner, loop_vars)
    upper_offset = boundCoreOffset(end_outer, var_inner, loop_vars)
    if lower_offset is None or upper_offset is None:
        return None
    new_outer_start = replaceBoundCore(
        start_outer,
        var_inner,
        addExpr(intExpr(lower_offset, line, col), start_inner, line, col),
        loop_vars,
        line,
        col,
    )
    new_outer_end = replaceBoundCore(
        end_outer,
        var_inner,
        addExpr(intExpr(upper_offset, line, col), end_inner, line, col),
        loop_vars,
        line,
        col,
    )
    outer_var_expr = Variable(name=var_outer, line=line, col=col)
    new_inner_start = maxExpr(
        start_inner,
        subExpr(outer_var_expr, intExpr(upper_offset, line, col), line, col),
        line,
        col,
    )
    new_inner_end = minExpr(
        end_inner,
        subExpr(outer_var_expr, intExpr(lower_offset, line, col), line, col),
        line,
        col,
    )
    return [
        (var_outer, new_outer_start, new_outer_end, step_outer),
        (var_inner, new_inner_start, new_inner_end, step_inner),
    ]

def applyArticleSideSlice(
    point_infos: List[PointInfo],
    point_order: List[int],
    loop_vars: Set[str],
    line: int,
    col: int,
) -> Tuple[List[PointInfo], bool]:
    count = len(point_infos)
    if count < 2 or point_order == list(range(count)):
        return list(point_infos), False
    if count == 2:
        remapped = sideSlicePointInfos(point_infos, point_order, loop_vars, line, col)
        if remapped is None:
            return list(point_infos), False
        return remapped, True
    face = list(point_infos[:2])
    remapped_face = sideSlicePointInfos(face, [1, 0], loop_vars, line, col)
    if remapped_face is None:
        return list(point_infos), False
    remapped = {
        0: remapped_face[1],
        1: remapped_face[0],
    }
    for index in range(2, count):
        remapped[index] = point_infos[index]
    ordered = [remapped[index] for index in point_order]
    return ordered, True

def applySideSliceToPointInfos(
    point_infos: List[PointInfo],
    point_order: List[int],
    loop_vars: Set[str],
    line: int,
    col: int,
) -> Tuple[List[PointInfo], bool]:
    from src.optimizations.loop_analysis import articleModeEnabled

    if articleModeEnabled():
        return applyArticleSideSlice(point_infos, point_order, loop_vars, line, col)
    if not point_infos or point_order == list(range(len(point_infos))):
        return list(point_infos), False
    if len(point_infos) == 2:
        remapped = sideSlicePointInfos(point_infos, point_order, loop_vars, line, col)
        if remapped is None:
            return list(point_infos), False
        return remapped, True
    if len(point_infos) >= 3:
        prefix = list(point_infos[:-2])
        pair = list(point_infos[-2:])
        remapped_pair = sideSlicePair(pair, loop_vars, line, col)
        if remapped_pair is None:
            return list(point_infos), False
        return prefix + remapped_pair, True
    return list(point_infos), False

def buildPointLoops(
    point_infos: List[PointInfo],
    body: List[Statement],
    line: int,
    col: int,
) -> List[Statement]:
    nested = body
    for var, point_start, point_end, point_step in reversed(point_infos):
        nested = [DoLoop(
            var=var,
            start=point_start,
            end=point_end,
            step=point_step,
            body=nested,
            stmt_label=None,
            line=line,
            col=col,
        )]
    return nested

def desiredPointVarOrder(nest: LoopNest, point_start: int, point_depth: int) -> List[str]:
    from src.optimizations.loop_analysis import chooseIntraTileLoopOrder

    point_loops = nest.loops[point_start:point_start + point_depth]
    point_nest = LoopNest(loops=point_loops, body=nest.body)
    order = chooseIntraTileLoopOrder(point_nest)
    return [point_loops[index].var for index in order]

def sideSliceOrderApplied(nest: LoopNest, point_start: int, point_depth: int) -> bool:
    if point_depth < 2:
        return False
    desired = desiredPointVarOrder(nest, point_start, point_depth)
    current = [loop_info.var for loop_info in nest.loops[point_start:point_start + point_depth]]
    return current == desired
