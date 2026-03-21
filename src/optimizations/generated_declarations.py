from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import replace as dcReplace
from typing import Dict, Iterable, List, Optional, Tuple

from src.core import (
    Program, Subroutine, FunctionDef, Declaration,
    ImplicitNone, ImplicitStatement,
    Statement, Assignment, DoLoop, LabeledDoLoop, DoWhile, LabeledDoWhile,
    IfStatement, SimpleIfStatement, PrintStatement, WriteStatement, ReadStatement, CallStatement,
    Expression, Variable, ArrayRef, IntegerLiteral, RealLiteral, LogicalLiteral, StringLiteral,
    BinaryOp, UnaryOp, FunctionCall,
)
from src.optimizations.base import ASTOptimizationPass


GENERATED_NAME_RE = re.compile(
    r"^(?:cse_tmp_\d+|licm_tmp_\d+|tile_[A-Za-z0-9_]+|skew_[A-Za-z0-9_]+|wf_[A-Za-z0-9_]+)$",
    re.IGNORECASE,
)


def isGeneratedName(name: str) -> bool:
    return bool(GENERATED_NAME_RE.match(name))


def normalizeType(typeName: Optional[str]) -> Optional[str]:
    if not typeName:
        return None
    upper = typeName.upper().replace(" ", "")
    if upper == "DOUBLEPRECISION":
        return "REAL"
    if upper.startswith("CHARACTER"):
        return "CHARACTER"
    if upper in {"INTEGER", "REAL", "LOGICAL", "COMPLEX"}:
        return upper
    return None


def buildTypeEnv(declarations: List) -> Dict[str, str]:
    env: Dict[str, str] = {}
    for decl in declarations:
        if not isinstance(decl, Declaration):
            continue
        norm = normalizeType(decl.type)
        if norm is None:
            continue
        for name, _ in decl.names:
            env[name] = norm
    return env


def mergeNumericTypes(left: Optional[str], right: Optional[str]) -> Optional[str]:
    if left == "COMPLEX" or right == "COMPLEX":
        return "COMPLEX"
    if left == "REAL" or right == "REAL":
        return "REAL"
    if left == "INTEGER" and right == "INTEGER":
        return "INTEGER"
    if left == "LOGICAL" and right == "LOGICAL":
        return "LOGICAL"
    if left == "CHARACTER" and right == "CHARACTER":
        return "CHARACTER"
    return left or right


def inferExprType(expr: Expression, env: Dict[str, str]) -> Optional[str]:
    if isinstance(expr, IntegerLiteral):
        return "INTEGER"
    if isinstance(expr, RealLiteral):
        return "REAL"
    if isinstance(expr, LogicalLiteral):
        return "LOGICAL"
    if isinstance(expr, StringLiteral):
        return "CHARACTER"
    if isinstance(expr, Variable):
        return env.get(expr.name)
    if isinstance(expr, ArrayRef):
        return env.get(expr.name)
    if isinstance(expr, UnaryOp):
        if expr.op == ".NOT.":
            return "LOGICAL"
        return inferExprType(expr.operand, env)
    if isinstance(expr, BinaryOp):
        if expr.op in {".LT.", "<", ".LE.", "<=", ".GT.", ">", ".GE.", ">=", ".EQ.", "==", ".NE.", "/=", ".AND.", "&", ".OR.", "|", ".EQV.", ".NEQV."}:
            return "LOGICAL"
        if expr.op == "//":
            return "CHARACTER"
        leftType = inferExprType(expr.left, env)
        rightType = inferExprType(expr.right, env)
        return mergeNumericTypes(leftType, rightType)
    if isinstance(expr, FunctionCall):
        name = expr.name.upper()
        if name in {"SQRT", "SIN", "COS", "TAN", "ASIN", "ACOS", "ATAN", "EXP", "LOG", "LOG10", "FLOAT", "REAL"}:
            return "REAL"
        if name in {"INT", "MOD"}:
            return "INTEGER"
        if name in {"MIN", "MAX", "ABS", "SIGN", "POW"}:
            current: Optional[str] = None
            for arg in expr.args:
                current = mergeNumericTypes(current, inferExprType(arg, env))
            return current
    return None


def collectGeneratedLoopVars(stmt: Statement, out: Dict[str, str]) -> None:
    if isinstance(stmt, (DoLoop, LabeledDoLoop)):
        if isGeneratedName(stmt.var):
            out.setdefault(stmt.var, "INTEGER")
        for inner in stmt.body:
            collectGeneratedLoopVars(inner, out)
    elif isinstance(stmt, (DoWhile, LabeledDoWhile)):
        for inner in stmt.body:
            collectGeneratedLoopVars(inner, out)
    elif isinstance(stmt, IfStatement):
        for inner in stmt.then_body:
            collectGeneratedLoopVars(inner, out)
        for _, body in stmt.elif_parts:
            for inner in body:
                collectGeneratedLoopVars(inner, out)
        if stmt.else_body:
            for inner in stmt.else_body:
                collectGeneratedLoopVars(inner, out)
    elif isinstance(stmt, SimpleIfStatement):
        collectGeneratedLoopVars(stmt.statement, out)


def iterStatements(stmts: List[Statement]) -> Iterable[Statement]:
    for stmt in stmts:
        yield stmt
        if isinstance(stmt, (DoLoop, LabeledDoLoop, DoWhile, LabeledDoWhile)):
            yield from iterStatements(stmt.body)
        elif isinstance(stmt, IfStatement):
            yield from iterStatements(stmt.then_body)
            for _, body in stmt.elif_parts:
                yield from iterStatements(body)
            if stmt.else_body:
                yield from iterStatements(stmt.else_body)
        elif isinstance(stmt, SimpleIfStatement):
            yield from iterStatements([stmt.statement])


def collectGeneratedAssignments(stmts: List[Statement]) -> Dict[str, List[Expression]]:
    assignments: Dict[str, List[Expression]] = defaultdict(list)
    for stmt in iterStatements(stmts):
        if isinstance(stmt, Assignment) and not stmt.indices and isGeneratedName(stmt.target):
            assignments[stmt.target].append(stmt.value)
    return assignments


def inferGeneratedTypes(declarations: List, statements: List[Statement]) -> Dict[str, str]:
    env = buildTypeEnv(declarations)
    generated: Dict[str, str] = {}
    for stmt in statements:
        collectGeneratedLoopVars(stmt, generated)
    assignments = collectGeneratedAssignments(statements)
    changed = True
    while changed:
        changed = False
        fullEnv = dict(env)
        fullEnv.update(generated)
        for name, exprs in assignments.items():
            if name in generated and generated[name] is not None:
                continue
            inferred: Optional[str] = None
            for expr in exprs:
                inferred = mergeNumericTypes(inferred, inferExprType(expr, fullEnv))
            if inferred is not None and generated.get(name) != inferred:
                generated[name] = inferred
                changed = True
    for name in assignments:
        generated.setdefault(name, "INTEGER")
    return generated


def existingDeclaredNames(declarations: List) -> Dict[str, str]:
    names: Dict[str, str] = {}
    for decl in declarations:
        if isinstance(decl, Declaration):
            for name, _ in decl.names:
                names[name] = name
    return names


def declarationInsertionIndex(declarations: List) -> int:
    idx = 0
    while idx < len(declarations) and isinstance(declarations[idx], (ImplicitNone, ImplicitStatement)):
        idx += 1
    return idx


def addGeneratedDeclarations(declarations: List, statements: List[Statement]) -> Tuple[List, int]:
    generated = inferGeneratedTypes(declarations, statements)
    declared = existingDeclaredNames(declarations)
    byType: Dict[str, List[str]] = defaultdict(list)
    for name, typeName in generated.items():
        if name in declared:
            continue
        if typeName not in {"INTEGER", "REAL", "LOGICAL", "COMPLEX", "CHARACTER"}:
            typeName = "INTEGER"
        byType[typeName].append(name)
    if not byType:
        return declarations, 0
    newDecls = list(declarations)
    insertAt = declarationInsertionIndex(newDecls)
    additions: List[Declaration] = []
    for typeName in ["INTEGER", "REAL", "LOGICAL", "COMPLEX", "CHARACTER"]:
        names = byType.get(typeName)
        if not names:
            continue
        additions.append(Declaration(type=typeName, names=[(name, None) for name in sorted(names)], line=0, col=0))
    updated = newDecls[:insertAt] + additions + newDecls[insertAt:]
    count = sum(len(decl.names) for decl in additions)
    return updated, count


class GeneratedVariableDeclarations(ASTOptimizationPass):
    name = "GeneratedVariableDeclarations"

    def run(self, program: Program) -> Program:
        total = 0
        newProgramDecls, count = addGeneratedDeclarations(program.declarations, program.statements)
        total += count
        newSubs: List[Subroutine] = []
        for sub in program.subroutines:
            newDecls, count = addGeneratedDeclarations(sub.declarations, sub.statements)
            total += count
            newSubs.append(dcReplace(sub, declarations=newDecls))
        newFuncs: List[FunctionDef] = []
        for func in program.functions:
            newDecls, count = addGeneratedDeclarations(func.declarations, func.statements)
            total += count
            newFuncs.append(dcReplace(func, declarations=newDecls))
        self.stats = {"declared_generated": total}
        return dcReplace(program, declarations=newProgramDecls, subroutines=newSubs, functions=newFuncs)
