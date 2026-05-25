from __future__ import annotations

from dataclasses import replace as dcReplace
from typing import List, Optional, Tuple
from src.frontend.ast import ArrayRef, BinaryOp, DoLoop, Expression, FunctionCall, IfStatement, IntegerLiteral, LabeledDoLoop, Program, RealLiteral, SimpleIfStatement, UnaryOp, Variable
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

def combineLikeTerms(terms: List[Expression], const_value: float, template: Expression) -> Tuple[List[Expression], float]:
    coeffs: dict = {}
    others: List[Expression] = []
    for term in terms:
        if isinstance(term, Variable):
            coeffs[term.name] = coeffs.get(term.name, 0) + 1
            continue
        if isinstance(term, UnaryOp) and term.op == "-" and isinstance(term.operand, Variable):
            coeffs[term.operand.name] = coeffs.get(term.operand.name, 0) - 1
            continue
        if isinstance(term, BinaryOp) and term.op == "*":
            factor = intValue(term.left)
            if factor is not None and isinstance(term.right, Variable):
                coeffs[term.right.name] = coeffs.get(term.right.name, 0) + factor
                continue
            factor = intValue(term.right)
            if factor is not None and isinstance(term.left, Variable):
                coeffs[term.left.name] = coeffs.get(term.left.name, 0) + factor
                continue
        others.append(term)
    combined: List[Expression] = list(others)
    for name in sorted(coeffs):
        coeff = coeffs[name]
        if coeff == 0:
            continue
        var = Variable(name=name, line=template.line, col=template.col)
        if coeff == 1:
            combined.append(var)
        elif coeff == -1:
            combined.append(UnaryOp(op="-", operand=var, line=template.line, col=template.col))
        else:
            combined.append(BinaryOp(left=makeInt(coeff, template), op="*", right=var, line=template.line, col=template.col))
    return combined, const_value

def linearizeAddExpr(expr: Expression) -> Expression:
    if isinstance(expr, BinaryOp) and expr.op in {"+", "-"}:
        use_real = isinstance(expr.left, RealLiteral) or isinstance(expr.right, RealLiteral)
        terms, const_value, const_real = flattenAdd(expr)
        terms, const_value = combineLikeTerms(terms, const_value, expr)
        return rebuildAdd(terms, const_value, use_real or const_real, expr)
    return expr

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

def linearizeExpr(expr: Expression) -> Expression:
    if isinstance(expr, BinaryOp):
        left = linearizeExpr(expr.left)
        right = linearizeExpr(expr.right)
        simplified = simplifyBinary(dcReplace(expr, left=left, right=right))
        if isinstance(simplified, BinaryOp) and simplified.op in {"+", "-"}:
            return linearizeAddExpr(simplified)
        return simplified
    if isinstance(expr, UnaryOp):
        operand = linearizeExpr(expr.operand)
        if isinstance(operand, IntegerLiteral):
            return makeInt(operand.value, expr) if expr.op != "-" else makeInt(-operand.value, expr)
        if isinstance(operand, RealLiteral):
            return makeReal(-operand.value, expr) if expr.op == "-" else operand
        return dcReplace(expr, operand=operand)
    if isinstance(expr, FunctionCall):
        args = [linearizeExpr(arg) for arg in expr.args]
        return simplifyFunction(dcReplace(expr, args=args))
    return expr

def simplifyFunction(expr: FunctionCall) -> Expression:
    upper_name = expr.name.upper()
    args = expr.args
    if upper_name in {"MIN", "MAX"} and len(args) == 2:
        left = linearizeAddExpr(args[0]) if isinstance(args[0], BinaryOp) else args[0]
        right = linearizeAddExpr(args[1]) if isinstance(args[1], BinaryOp) else args[1]
        if exprKey(left) == exprKey(right):
            return left
        left_int = intValue(left)
        right_int = intValue(right)
        if left_int is not None and right_int is not None:
            value = min(left_int, right_int) if upper_name == "MIN" else max(left_int, right_int)
            return makeInt(value, expr)
        left_real = realValue(left)
        right_real = realValue(right)
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
        return linearizeAddExpr(expr)

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

    def hasTransformedLoops(self, stmts) -> bool:
        for stmt in stmts:
            if isinstance(stmt, (DoLoop, LabeledDoLoop)):
                if stmt.var.startswith(("tile_", "skew_")):
                    return True
                if self.hasTransformedLoops(stmt.body):
                    return True
            elif isinstance(stmt, IfStatement):
                if self.hasTransformedLoops(stmt.then_body):
                    return True
                for _, body in stmt.elif_parts:
                    if self.hasTransformedLoops(body):
                        return True
                if stmt.else_body and self.hasTransformedLoops(stmt.else_body):
                    return True
            elif isinstance(stmt, SimpleIfStatement):
                if self.hasTransformedLoops([stmt.statement]):
                    return True
        return False

    def transformStmt(self, stmt):
        return super().transformStmt(stmt)

    def transformExpr(self, expr):
        original = expr
        expr = super().transformExpr(expr)
        expr = linearizeExpr(expr)
        if exprKey(expr) != exprKey(original):
            self.changed += 1
        return expr

    def run(self, program: Program) -> Program:
        if not self.hasTransformedLoops(program.statements) and not any(self.hasTransformedLoops(subroutine.statements) for subroutine in program.subroutines) and not any(self.hasTransformedLoops(function.statements) for function in program.functions):
            self.stats = {"linearized": 0}
            return program
        self.changed = 0
        new_statements = self.transformStmts(program.statements)
        new_subroutines = [dcReplace(subroutine, statements=self.transformStmts(subroutine.statements)) for subroutine in program.subroutines]
        new_functions = [dcReplace(function, statements=self.transformStmts(function.statements)) for function in program.functions]
        self.stats = {"linearized": self.changed}
        return dcReplace(program, statements=new_statements, subroutines=new_subroutines, functions=new_functions)
