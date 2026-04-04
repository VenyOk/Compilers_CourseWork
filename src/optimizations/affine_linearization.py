from __future__ import annotations

from dataclasses import replace as dcReplace
from typing import List, Optional, Tuple

from src.core import ArrayRef, BinaryOp, DoLoop, Expression, FunctionCall, IfStatement, IntegerLiteral, LabeledDoLoop, Program, RealLiteral, SimpleIfStatement, UnaryOp, Variable
from src.optimizations.base import ASTOptimizationPass


def exprKey(expr: Expression) -> str:
    if isinstance(expr, IntegerLiteral):
        return f"i:{expr.value}"
    if isinstance(expr, RealLiteral):
        return f"r:{expr.value}"
    if isinstance(expr, Variable):
        return f"v:{expr.name}"
    if isinstance(expr, UnaryOp):
        return f"u:{expr.op}:{exprKey(expr.operand)}"
    if isinstance(expr, BinaryOp):
        return f"b:{expr.op}:{exprKey(expr.left)}:{exprKey(expr.right)}"
    if isinstance(expr, FunctionCall):
        return f"f:{expr.name.upper()}:" + ",".join(exprKey(arg) for arg in expr.args)
    if isinstance(expr, ArrayRef):
        return f"a:{expr.name}:" + ",".join(exprKey(index) for index in expr.indices)
    return repr(expr)


def intValue(expr: Expression) -> Optional[int]:
    if isinstance(expr, IntegerLiteral):
        return expr.value
    if isinstance(expr, UnaryOp) and expr.op == "-":
        inner = intValue(expr.operand)
        if inner is not None:
            return -inner
    return None


def realValue(expr: Expression) -> Optional[float]:
    if isinstance(expr, RealLiteral):
        return expr.value
    if isinstance(expr, IntegerLiteral):
        return float(expr.value)
    if isinstance(expr, UnaryOp) and expr.op == "-":
        inner = realValue(expr.operand)
        if inner is not None:
            return -inner
    return None


def makeInt(value: int, expr: Expression) -> IntegerLiteral:
    return IntegerLiteral(value=value, line=expr.line, col=expr.col)


def makeReal(value: float, expr: Expression) -> RealLiteral:
    return RealLiteral(value=value, line=expr.line, col=expr.col)


def isZero(expr: Expression) -> bool:
    return intValue(expr) == 0 or realValue(expr) == 0.0


def isOne(expr: Expression) -> bool:
    return intValue(expr) == 1 or realValue(expr) == 1.0


def negate(expr: Expression) -> Expression:
    iv = intValue(expr)
    if iv is not None:
        return makeInt(-iv, expr)
    rv = realValue(expr)
    if rv is not None:
        return makeReal(-rv, expr)
    return UnaryOp(op="-", operand=expr, line=expr.line, col=expr.col)


def flattenAdd(expr: Expression) -> Tuple[List[Expression], float, bool]:
    if isinstance(expr, BinaryOp) and expr.op == "+":
        left_terms, left_const, left_real = flattenAdd(expr.left)
        right_terms, right_const, right_real = flattenAdd(expr.right)
        return left_terms + right_terms, left_const + right_const, left_real or right_real
    if isinstance(expr, BinaryOp) and expr.op == "-":
        left_terms, left_const, left_real = flattenAdd(expr.left)
        right_terms, right_const, right_real = flattenAdd(expr.right)
        neg_terms = [negate(term) for term in right_terms]
        return left_terms + neg_terms, left_const - right_const, left_real or right_real
    iv = intValue(expr)
    if iv is not None:
        return [], float(iv), False
    rv = realValue(expr)
    if rv is not None:
        return [], rv, True
    return [expr], 0.0, False


def rebuildAdd(terms: List[Expression], const_value: float, use_real: bool, template: Expression) -> Expression:
    filtered_terms = [term for term in terms if not isZero(term)]
    const_expr: Optional[Expression] = None
    if abs(const_value) > 0.0:
        if use_real or abs(const_value - round(const_value)) > 1e-12:
            const_expr = makeReal(const_value, template)
        else:
            const_expr = makeInt(int(round(const_value)), template)
    if const_expr is not None:
        filtered_terms.append(const_expr)
    if not filtered_terms:
        return makeReal(0.0, template) if use_real else makeInt(0, template)
    result = filtered_terms[0]
    for term in filtered_terms[1:]:
        result = BinaryOp(left=result, op="+", right=term, line=template.line, col=template.col)
    return result


def simplifyFunction(expr: FunctionCall) -> Expression:
    upper_name = expr.name.upper()
    args = expr.args
    if upper_name in {"MIN", "MAX"} and len(args) == 2:
        if exprKey(args[0]) == exprKey(args[1]):
            return args[0]
        left_int = intValue(args[0])
        right_int = intValue(args[1])
        if left_int is not None and right_int is not None:
            value = min(left_int, right_int) if upper_name == "MIN" else max(left_int, right_int)
            return makeInt(value, expr)
        left_real = realValue(args[0])
        right_real = realValue(args[1])
        if left_real is not None and right_real is not None:
            value = min(left_real, right_real) if upper_name == "MIN" else max(left_real, right_real)
            return makeReal(value, expr)
    return expr


def simplifyBinary(expr: BinaryOp) -> Expression:
    left = expr.left
    right = expr.right
    left_int = intValue(left)
    right_int = intValue(right)
    left_real = realValue(left)
    right_real = realValue(right)

    if expr.op in {"+", "-"}:
        use_real = isinstance(left, RealLiteral) or isinstance(right, RealLiteral)
        terms, const_value, const_real = flattenAdd(expr)
        return rebuildAdd(terms, const_value, use_real or const_real, expr)

    if expr.op == "*":
        if isZero(left) or isZero(right):
            if isinstance(left, RealLiteral) or isinstance(right, RealLiteral):
                return makeReal(0.0, expr)
            return makeInt(0, expr)
        if isOne(left):
            return right
        if isOne(right):
            return left
        if left_int is not None and right_int is not None:
            return makeInt(left_int * right_int, expr)
        if left_real is not None and right_real is not None:
            return makeReal(left_real * right_real, expr)
        return expr

    if expr.op == "/":
        if isZero(left):
            return makeReal(0.0, expr) if isinstance(left, RealLiteral) or isinstance(right, RealLiteral) else makeInt(0, expr)
        if isOne(right):
            return left
        if left_int is not None and right_int not in (None, 0):
            if left_int % right_int == 0:
                return makeInt(left_int // right_int, expr)
        if left_real is not None and right_real not in (None, 0.0):
            return makeReal(left_real / right_real, expr)
        return expr

    if expr.op == "**":
        if right_int == 0:
            return makeReal(1.0, expr) if isinstance(left, RealLiteral) else makeInt(1, expr)
        if right_int == 1:
            return left
        if left_int is not None and right_int is not None and right_int >= 0:
            return makeInt(left_int ** right_int, expr)
        if left_real is not None and right_int is not None:
            return makeReal(left_real ** right_int, expr)
        return expr

    return expr


class AffineLinearization(ASTOptimizationPass):
    name = "AffineLinearization"

    def __init__(self):
        super().__init__()
        self.changed = 0

    def _hasTransformedLoops(self, stmts) -> bool:
        for stmt in stmts:
            if isinstance(stmt, (DoLoop, LabeledDoLoop)):
                if stmt.var.startswith(("tile_", "skew_", "wf_")):
                    return True
                if self._hasTransformedLoops(stmt.body):
                    return True
            elif isinstance(stmt, IfStatement):
                if self._hasTransformedLoops(stmt.then_body):
                    return True
                for _, body in stmt.elif_parts:
                    if self._hasTransformedLoops(body):
                        return True
                if stmt.else_body and self._hasTransformedLoops(stmt.else_body):
                    return True
            elif isinstance(stmt, SimpleIfStatement):
                if self._hasTransformedLoops([stmt.statement]):
                    return True
        return False

    def transformExpr(self, expr):
        original = expr
        expr = super().transformExpr(expr)
        if isinstance(expr, UnaryOp) and expr.op == "-":
            iv = intValue(expr)
            if iv is not None:
                expr = makeInt(iv, expr)
            else:
                rv = realValue(expr)
                if rv is not None:
                    expr = makeReal(rv, expr)
        elif isinstance(expr, BinaryOp):
            expr = simplifyBinary(expr)
        elif isinstance(expr, FunctionCall):
            expr = simplifyFunction(expr)
        if exprKey(expr) != exprKey(original):
            self.changed += 1
        return expr

    def run(self, program: Program) -> Program:
        if not self._hasTransformedLoops(program.statements) and not any(self._hasTransformedLoops(subroutine.statements) for subroutine in program.subroutines) and not any(self._hasTransformedLoops(function.statements) for function in program.functions):
            self.stats = {"linearized": 0}
            return program
        self.changed = 0
        new_statements = self.transformStmts(program.statements)
        new_subroutines = [dcReplace(subroutine, statements=self.transformStmts(subroutine.statements)) for subroutine in program.subroutines]
        new_functions = [dcReplace(function, statements=self.transformStmts(function.statements)) for function in program.functions]
        self.stats = {"linearized": self.changed}
        return dcReplace(program, statements=new_statements, subroutines=new_subroutines, functions=new_functions)
