from __future__ import annotations

from copy import deepcopy
from dataclasses import replace as dcReplace
from typing import Dict, List

from src.core import (
    ArrayRef,
    Assignment,
    BinaryOp,
    CallStatement,
    DoLoop,
    DoWhile,
    Expression,
    FunctionCall,
    IfStatement,
    IntegerLiteral,
    LabeledDoLoop,
    LabeledDoWhile,
    PrintStatement,
    Program,
    ReadStatement,
    SimpleIfStatement,
    Statement,
    UnaryOp,
    Variable,
    WriteStatement,
)
from src.optimizations.base import ASTOptimizationPass
from src.optimizations.loop_analysis import buildNest, getSkewMatrix, skewDecision, stencilFamily


def skewVarName(var: str) -> str:
    return f"skew_{var}"


def isSkewVar(var: str) -> bool:
    return var.startswith("skew_")


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


def mulExprByInt(expr: Expression, factor: int, line: int, col: int) -> Expression:
    if factor == 0:
        return intExpr(0, line, col)
    if factor == 1:
        return expr
    return BinaryOp(left=intExpr(factor, line, col), op="*", right=expr, line=line, col=col)


def substituteExpr(expr: Expression, substitutions: Dict[str, Expression]) -> Expression:
    if isinstance(expr, Variable) and expr.name in substitutions:
        replacement = deepcopy(substitutions[expr.name])
        if isinstance(replacement, Variable) and replacement.name == expr.name:
            return replacement
        return substituteExpr(replacement, substitutions)
    if isinstance(expr, BinaryOp):
        return dcReplace(
            expr,
            left=substituteExpr(expr.left, substitutions),
            right=substituteExpr(expr.right, substitutions),
        )
    if isinstance(expr, UnaryOp):
        return dcReplace(expr, operand=substituteExpr(expr.operand, substitutions))
    if isinstance(expr, FunctionCall):
        return dcReplace(expr, args=[substituteExpr(arg, substitutions) for arg in expr.args])
    if isinstance(expr, ArrayRef):
        return dcReplace(expr, indices=[substituteExpr(index, substitutions) for index in expr.indices])
    return expr


def substituteStmt(stmt: Statement, substitutions: Dict[str, Expression]) -> Statement:
    if isinstance(stmt, Assignment):
        return dcReplace(
            stmt,
            value=substituteExpr(stmt.value, substitutions),
            indices=[substituteExpr(index, substitutions) for index in stmt.indices],
        )
    if isinstance(stmt, (DoLoop, LabeledDoLoop)):
        return dcReplace(
            stmt,
            start=substituteExpr(stmt.start, substitutions),
            end=substituteExpr(stmt.end, substitutions),
            step=substituteExpr(stmt.step, substitutions) if stmt.step else stmt.step,
            body=[substituteStmt(inner, substitutions) for inner in stmt.body],
        )
    if isinstance(stmt, (DoWhile, LabeledDoWhile)):
        return dcReplace(
            stmt,
            condition=substituteExpr(stmt.condition, substitutions),
            body=[substituteStmt(inner, substitutions) for inner in stmt.body],
        )
    if isinstance(stmt, IfStatement):
        return dcReplace(
            stmt,
            condition=substituteExpr(stmt.condition, substitutions),
            then_body=[substituteStmt(inner, substitutions) for inner in stmt.then_body],
            elif_parts=[
                (substituteExpr(condition, substitutions), [substituteStmt(inner, substitutions) for inner in body])
                for condition, body in stmt.elif_parts
            ],
            else_body=[substituteStmt(inner, substitutions) for inner in stmt.else_body] if stmt.else_body else None,
        )
    if isinstance(stmt, SimpleIfStatement):
        return dcReplace(
            stmt,
            condition=substituteExpr(stmt.condition, substitutions),
            statement=substituteStmt(stmt.statement, substitutions),
        )
    if isinstance(stmt, PrintStatement):
        return dcReplace(stmt, items=[substituteExpr(item, substitutions) for item in stmt.items])
    if isinstance(stmt, WriteStatement):
        return dcReplace(stmt, items=[substituteExpr(item, substitutions) for item in stmt.items])
    if isinstance(stmt, CallStatement):
        return dcReplace(stmt, args=[substituteExpr(arg, substitutions) for arg in stmt.args])
    if isinstance(stmt, ReadStatement):
        return stmt
    return stmt


def buildSubstitutions(nest, matrix: List[List[int]]) -> Dict[str, Expression]:
    substitutions: Dict[str, Expression] = {}
    for index, loop_info in enumerate(nest.loops):
        if index == 0 or not any(matrix[index][:index]):
            substitutions[loop_info.var] = Variable(
                name=loop_info.var,
                line=loop_info.node.line,
                col=loop_info.node.col,
            )
            continue
        expr: Expression = Variable(
            name=skewVarName(loop_info.var),
            line=loop_info.node.line,
            col=loop_info.node.col,
        )
        for outer_index in range(index):
            coeff = matrix[index][outer_index]
            if coeff == 0:
                continue
            outer_expr = deepcopy(substitutions[nest.loops[outer_index].var])
            expr = subExpr(
                expr,
                mulExprByInt(outer_expr, coeff, loop_info.node.line, loop_info.node.col),
                loop_info.node.line,
                loop_info.node.col,
            )
        substitutions[loop_info.var] = expr
    return substitutions


def shiftedBound(expr: Expression, nest, matrix: List[List[int]], index: int) -> Expression:
    line = getattr(expr, "line", 0)
    col = getattr(expr, "col", 0)
    shifted = deepcopy(expr)
    for outer_index in range(index):
        coeff = matrix[index][outer_index]
        if coeff == 0:
            continue
        outer_var = Variable(name=nest.loops[outer_index].var, line=line, col=col)
        shifted = addExpr(shifted, mulExprByInt(outer_var, coeff, line, col), line, col)
    return shifted


def skewNest(nest, matrix: List[List[int]]) -> Statement:
    substitutions = buildSubstitutions(nest, matrix)
    transformed_body = [substituteStmt(stmt, substitutions) for stmt in nest.body]
    nested_body = transformed_body
    for index in reversed(range(nest.depth)):
        loop_info = nest.loops[index]
        has_skew = any(matrix[index][:index])
        loop_var = skewVarName(loop_info.var) if has_skew else loop_info.var
        start = shiftedBound(loop_info.start, nest, matrix, index) if has_skew else deepcopy(loop_info.start)
        end = shiftedBound(loop_info.end, nest, matrix, index) if has_skew else deepcopy(loop_info.end)
        start = substituteExpr(start, substitutions)
        end = substituteExpr(end, substitutions)
        step = substituteExpr(loop_info.step, substitutions) if loop_info.step else loop_info.step
        nested_body = [DoLoop(
            var=loop_var,
            start=start,
            end=end,
            step=step,
            body=nested_body,
            stmt_label=None,
            line=loop_info.node.line,
            col=loop_info.node.col,
        )]
    return nested_body[0]


def trySkew(loop: Statement, counter: List[int], diagnostics: List[Dict[str, object]]) -> Statement:
    if not isinstance(loop, (DoLoop, LabeledDoLoop)):
        return loop
    nest = buildNest(loop)
    if any(isSkewVar(var) for var in nest.vars):
        return dcReplace(loop, body=[trySkew(stmt, counter, diagnostics) for stmt in loop.body])
    should_skew, reason = skewDecision(nest)
    if nest.depth >= 2 and should_skew:
        matrix = getSkewMatrix(nest)
        if any(any(row) for row in matrix):
            counter[0] += 1
            diagnostics.append({
                "vars": list(nest.vars),
                "family": stencilFamily(nest),
                "reason": reason,
                "matrix": matrix,
            })
            return skewNest(nest, matrix)
    return dcReplace(loop, body=[trySkew(stmt, counter, diagnostics) for stmt in loop.body])


def processStmts(stmts: List[Statement], counter: List[int], diagnostics: List[Dict[str, object]]) -> List[Statement]:
    result = []
    for stmt in stmts:
        if isinstance(stmt, (DoLoop, LabeledDoLoop)):
            result.append(trySkew(stmt, counter, diagnostics))
        else:
            result.append(stmt)
    return result


class LoopSkewing(ASTOptimizationPass):
    name = "LoopSkewing"

    def run(self, program: Program) -> Program:
        counter = [0]
        diagnostics: List[Dict[str, object]] = []
        new_statements = processStmts(program.statements, counter, diagnostics)
        new_subroutines = [
            dcReplace(subroutine, statements=processStmts(subroutine.statements, counter, diagnostics))
            for subroutine in program.subroutines
        ]
        new_functions = [
            dcReplace(function, statements=processStmts(function.statements, counter, diagnostics))
            for function in program.functions
        ]
        self.stats = {"skewed": counter[0], "diagnostics": diagnostics}
        return dcReplace(
            program,
            statements=new_statements,
            subroutines=new_subroutines,
            functions=new_functions,
        )
