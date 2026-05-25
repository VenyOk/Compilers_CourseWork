from typing import List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field

@dataclass
class ASTNode:
    line: int = 0
    col: int = 0

@dataclass
class Program(ASTNode):
    name: str = ""
    declarations: List['Declaration'] = field(default_factory=list)
    statements: List['Statement'] = field(default_factory=list)
    statement_functions: List = field(default_factory=list)
    subroutines: List['Subroutine'] = field(default_factory=list)
    functions: List['FunctionDef'] = field(default_factory=list)

    def __str__(self):
        return f"Program({self.name}, {len(self.declarations)} decls, {len(self.statements)} stmts)"

@dataclass
class Subroutine(ASTNode):
    name: str = ""
    params: List[str] = field(default_factory=list)
    declarations: List['Declaration'] = field(default_factory=list)
    statements: List['Statement'] = field(default_factory=list)

    def __str__(self):
        return f"Subroutine({self.name})"

@dataclass
class FunctionDef(ASTNode):
    name: str = ""
    return_type: str = ""
    params: List[str] = field(default_factory=list)
    declarations: List['Declaration'] = field(default_factory=list)
    statements: List['Statement'] = field(default_factory=list)

    def __str__(self):
        return f"Function({self.name}: {self.return_type})"

@dataclass
class Declaration(ASTNode):
    type: str = ""
    names: List[Tuple[str, Optional[List[object]]]] = field(default_factory=list)
    type_size: Optional[int] = None

    def __str__(self):
        names_str = ", ".join(
            f"{name}{format_dimension_list(dim_ranges)}" if dim_ranges else name
            for name, dim_ranges in self.names
        )
        return f"{self.type} {names_str}"

@dataclass
class ImplicitNone(ASTNode):
    def __str__(self):
        return "IMPLICIT NONE"

@dataclass
class ImplicitRule(ASTNode):
    type_name: str = ""
    type_size: Optional[int] = None
    letters: List[str] = field(default_factory=list)

    def __str__(self):
        size_str = f"*{self.type_size}" if self.type_size else ""
        letters_str = ", ".join(self.letters)
        return f"IMPLICIT {self.type_name}{size_str}({letters_str})"

    def get_letters(self) -> Set[str]:
        result = set()
        for letter_spec in self.letters:
            if '-' in letter_spec:
                parts = letter_spec.split('-')
                if len(parts) == 2:
                    start = parts[0].strip().upper()
                    end = parts[1].strip().upper()
                    if len(start) == 1 and len(end) == 1 and start.isalpha() and end.isalpha():
                        start_ord = ord(start)
                        end_ord = ord(end)
                        if start_ord <= end_ord:
                            for i in range(start_ord, end_ord + 1):
                                result.add(chr(i))
            else:
                letter = letter_spec.strip().upper()
                if letter and letter.isalpha():
                    result.add(letter)
        return result

@dataclass
class ImplicitStatement(ASTNode):
    rules: List[ImplicitRule] = field(default_factory=list)

    def __str__(self):
        rules_str = ", ".join(str(rule) for rule in self.rules)
        return f"IMPLICIT {rules_str}"

@dataclass
class DimensionStatement(ASTNode):
    names: List[Tuple[str, List[object]]] = field(default_factory=list)

    def __str__(self):
        names_str = ", ".join(f"{name}{format_dimension_list(dim_ranges)}" for name, dim_ranges in self.names)
        return f"DIMENSION {names_str}"

@dataclass
class ParameterStatement(ASTNode):
    params: List[Tuple[str, 'Expression']] = field(default_factory=list)

    def __str__(self):
        params_str = ", ".join(f"{name}={expr}" for name, expr in self.params)
        return f"PARAMETER ({params_str})"

@dataclass
class Statement(ASTNode):
    stmt_label: Optional[str] = None

@dataclass
class DataItem(ASTNode):
    name: str = ""
    indices: List['Expression'] = field(default_factory=list)

    def __str__(self):
        if self.indices:
            indices_str = "(" + ", ".join(str(idx)
                                          for idx in self.indices) + ")"
            return f"{self.name}{indices_str}"
        return self.name

@dataclass
class DataStatement(Statement):
    items: List[Tuple[List[DataItem], List['Expression']]
                ] = field(default_factory=list)

    def __str__(self):
        items_str = ", ".join(
            f"{','.join(str(item) for item in vars)} / {','.join(str(v) for v in vals)} /"
            for vars, vals in self.items
        )
        return f"DATA {items_str}"

@dataclass
class Assignment(Statement):
    target: str = ""
    value: 'Expression' = None
    indices: List['Expression'] = field(default_factory=list)

    def __str__(self):
        return f"Assign({self.target} = ...)"

@dataclass
class DoLoop(Statement):
    var: str = ""
    start: 'Expression' = None
    end: 'Expression' = None
    step: Optional['Expression'] = None
    body: List[Statement] = field(default_factory=list)

    def __str__(self):
        return f"DO {self.var} = ... END DO"
@dataclass
class DoWhile(Statement):
    condition: 'Expression' = None
    body: List[Statement] = field(default_factory=list)

    def __str__(self):
        return f"DO WHILE (...) END DO"

@dataclass
class SimpleIfStatement(Statement):
    condition: 'Expression' = None
    statement: 'Statement' = None

    def __str__(self):
        return f"IF (...) S"

@dataclass
class IfStatement(Statement):
    condition: 'Expression' = None
    then_body: List[Statement] = field(default_factory=list)
    elif_parts: List[Tuple['Expression', List[Statement]]
                     ] = field(default_factory=list)
    else_body: Optional[List[Statement]] = None

    def __str__(self):
        return f"IF (...) THEN ... END IF"

@dataclass
class PrintStatement(Statement):
    items: List['Expression'] = field(default_factory=list)

    def __str__(self):
        return f"PRINT {len(self.items)} items"

@dataclass
class ReadStatement(Statement):
    unit: str = ""
    format: str = ""
    items: List[str] = field(default_factory=list)

    def __str__(self):
        return f"READ ({self.unit}, {self.format}) {len(self.items)} items"

@dataclass
class WriteStatement(Statement):
    unit: str = ""
    format: str = ""
    items: List['Expression'] = field(default_factory=list)

    def __str__(self):
        return f"WRITE ({self.unit}, {self.format}) {len(self.items)} items"

@dataclass
class CallStatement(Statement):
    name: str = ""
    args: List['Expression'] = field(default_factory=list)

    def __str__(self):
        return f"CALL {self.name}"

@dataclass
class ReturnStatement(Statement):
    def __str__(self):
        return "RETURN"

@dataclass
class StopStatement(Statement):
    def __str__(self):
        return "STOP"

@dataclass
class GotoStatement(Statement):
    label: str = ""

    def __str__(self):
        return f"GOTO {self.label}"

@dataclass
class ContinueStatement(Statement):
    label: Optional[str] = None

    def __str__(self):
        if self.label:
            return f"CONTINUE ({self.label})"
        return "CONTINUE"

@dataclass
class ExternalStatement(ASTNode):
    names: List[str] = field(default_factory=list)

    def __str__(self):
        return f"EXTERNAL {', '.join(self.names)}"

@dataclass
class CommonStatement(ASTNode):
    blocks: List[Tuple[str, List['Variable']]] = field(default_factory=list)

    def __str__(self):
        parts = []
        for block_name, vars in self.blocks:
            if block_name:
                parts.append(f"/{block_name}/ {', '.join(str(v) for v in vars)}")
            else:
                parts.append(f"{', '.join(str(v) for v in vars)}")
        return f"COMMON {', '.join(parts)}"

@dataclass
class ExitStatement(Statement):
    def __str__(self):
        return "EXIT"

@dataclass
class ArithmeticIfStatement(Statement):
    condition: 'Expression' = None
    label_neg: str = ""
    label_zero: str = ""
    label_pos: str = ""

    def __str__(self):
        return f"IF({self.condition}) {self.label_neg}, {self.label_zero}, {self.label_pos}"

@dataclass
class LabeledDoLoop(Statement):
    label: str = ""
    var: str = ""
    start: 'Expression' = None
    end: 'Expression' = None
    step: Optional['Expression'] = None
    body: List[Statement] = field(default_factory=list)

    def __str__(self):
        return f"DO {self.label} {self.var} = ... END DO"

@dataclass
class LabeledDoWhile(Statement):
    label: str = ""
    condition: 'Expression' = None
    body: List[Statement] = field(default_factory=list)

    def __str__(self):
        return f"DO {self.label} WHILE(...) END DO"

@dataclass
class Expression(ASTNode):
    pass

@dataclass
class BinaryOp(Expression):
    left: Expression = None
    op: str = ""
    right: Expression = None

    def __str__(self):
        return f"({self.op})"

@dataclass
class UnaryOp(Expression):
    op: str = ""
    operand: Expression = None

    def __str__(self):
        return f"({self.op} ...)"

@dataclass
class FunctionCall(Expression):
    name: str = ""
    args: List[Expression] = field(default_factory=list)

    def __str__(self):
        return f"{self.name}(...)"

@dataclass
class ArrayRef(Expression):
    name: str = ""
    indices: List['Expression'] = field(default_factory=list)

    def __str__(self):
        return f"{self.name}[...]"

@dataclass
class Variable(Expression):
    name: str = ""

    def __str__(self):
        return f"{self.name}"

@dataclass
class IntegerLiteral(Expression):
    value: int = 0

    def __str__(self):
        return str(self.value)

@dataclass
class RealLiteral(Expression):
    value: float = 0.0

    def __str__(self):
        return str(self.value)

@dataclass
class StringLiteral(Expression):
    value: str = ""

    def __str__(self):
        return repr(self.value)

@dataclass
class LogicalLiteral(Expression):
    value: bool = False

    def __str__(self):
        return ".TRUE." if self.value else ".FALSE."

@dataclass
class ComplexLiteral(Expression):
    real_part: float = 0.0
    imag_part: float = 0.0

    def __str__(self):
        return f"({self.real_part}, {self.imag_part})"

def format_dimension_bound(bound: object) -> str:
    if isinstance(bound, int):
        return str(bound)
    if isinstance(bound, IntegerLiteral):
        return str(bound.value)
    if isinstance(bound, Variable):
        return bound.name
    if isinstance(bound, UnaryOp):
        return f"{bound.op}{format_dimension_bound(bound.operand)}"
    if isinstance(bound, BinaryOp):
        left = format_dimension_bound(bound.left)
        right = format_dimension_bound(bound.right)
        return f"{left}{bound.op}{right}"
    if isinstance(bound, RealLiteral):
        return str(bound.value)
    if isinstance(bound, LogicalLiteral):
        return ".TRUE." if bound.value else ".FALSE."
    if isinstance(bound, StringLiteral):
        return repr(bound.value)
    return str(bound)

def format_dimension_spec(dim_spec: object) -> str:
    if isinstance(dim_spec, tuple) and len(dim_spec) == 2:
        lower, upper = dim_spec
        lower_text = format_dimension_bound(lower)
        upper_text = format_dimension_bound(upper)
        if lower_text == "1":
            return upper_text
        return f"{lower_text}:{upper_text}"
    return format_dimension_bound(dim_spec)

def format_dimension_list(dim_specs: List[object]) -> str:
    return "(" + ", ".join(format_dimension_spec(dim_spec) for dim_spec in dim_specs) + ")"

def pretty_print_ast(node: ASTNode, indent: int = 0) -> str:
    """Pretty print AST node with indentation."""
    prefix = "  " * indent
    if isinstance(node, Program):
        lines = [f"{prefix}Program: {node.name}"]
        for decl in node.declarations:
            lines.append(pretty_print_ast(decl, indent + 1))
        for stmt in node.statements:
            lines.append(pretty_print_ast(stmt, indent + 1))
        for sub in node.subroutines:
            lines.append(pretty_print_ast(sub, indent + 1))
        for func in node.functions:
            lines.append(pretty_print_ast(func, indent + 1))
        return "\n".join(lines)
    elif isinstance(node, Subroutine):
        lines = [f"{prefix}Subroutine: {node.name}"]
        for decl in node.declarations:
            lines.append(pretty_print_ast(decl, indent + 1))
        for stmt in node.statements:
            lines.append(pretty_print_ast(stmt, indent + 1))
        return "\n".join(lines)
    elif isinstance(node, FunctionDef):
        lines = [f"{prefix}Function: {node.name} -> {node.return_type}"]
        for decl in node.declarations:
            lines.append(pretty_print_ast(decl, indent + 1))
        for stmt in node.statements:
            lines.append(pretty_print_ast(stmt, indent + 1))
        return "\n".join(lines)
    elif isinstance(node, Declaration):
        names_str = ", ".join(
            f"{name}{format_dimension_list(dim_ranges)}" if dim_ranges else name
            for name, dim_ranges in node.names
        )
        return f"{prefix}Declaration: {node.type} {names_str}"
    elif isinstance(node, ImplicitNone):
        return f"{prefix}IMPLICIT NONE"
    elif isinstance(node, ImplicitStatement):
        return f"{prefix}IMPLICIT {', '.join(str(rule) for rule in node.rules)}"
    elif isinstance(node, DimensionStatement):
        return f"{prefix}DIMENSION: {node.names}"
    elif isinstance(node, ParameterStatement):
        return f"{prefix}PARAMETER: {node.params}"
    elif isinstance(node, Assignment):
        return f"{prefix}Assignment: {node.target} = ..."
    elif isinstance(node, DoLoop):
        return f"{prefix}DO: {node.var} = ... END DO"
    elif isinstance(node, DoWhile):
        return f"{prefix}DO WHILE: ... END DO"
    elif isinstance(node, IfStatement):
        return f"{prefix}IF: ... THEN ... END IF"
    elif isinstance(node, SimpleIfStatement):
        return f"{prefix}IF: ... statement"
    elif isinstance(node, PrintStatement):
        return f"{prefix}PRINT: {len(node.items)} items"
    elif isinstance(node, ReadStatement):
        return f"{prefix}READ: ({node.unit}, {node.format})"
    elif isinstance(node, WriteStatement):
        return f"{prefix}WRITE: ({node.unit}, {node.format})"
    elif isinstance(node, CallStatement):
        return f"{prefix}CALL: {node.name}"
    elif isinstance(node, ReturnStatement):
        return f"{prefix}RETURN"
    elif isinstance(node, StopStatement):
        return f"{prefix}STOP"
    elif isinstance(node, GotoStatement):
        return f"{prefix}GOTO: {node.label}"
    elif isinstance(node, ContinueStatement):
        return f"{prefix}CONTINUE"
    elif isinstance(node, ExitStatement):
        return f"{prefix}EXIT"
    elif isinstance(node, ExternalStatement):
        return f"{prefix}EXTERNAL: {', '.join(node.names)}"
    elif isinstance(node, CommonStatement):
        return f"{prefix}COMMON: {node.blocks}"
    elif isinstance(node, ArithmeticIfStatement):
        return f"{prefix}ARITH IF: {node.condition} -> {node.label_neg}, {node.label_zero}, {node.label_pos}"
    elif isinstance(node, LabeledDoLoop):
        return f"{prefix}DO {node.label}: {node.var} = ... END DO"
    elif isinstance(node, LabeledDoWhile):
        return f"{prefix}DO {node.label} WHILE: ... END DO"
    elif isinstance(node, BinaryOp):
        return f"{prefix}BinaryOp: {node.op}"
    elif isinstance(node, UnaryOp):
        return f"{prefix}UnaryOp: {node.op}"
    elif isinstance(node, FunctionCall):
        return f"{prefix}FunctionCall: {node.name}(...)"
    elif isinstance(node, ArrayRef):
        return f"{prefix}ArrayRef: {node.name}[...]"
    elif isinstance(node, Variable):
        return f"{prefix}Variable: {node.name}"
    elif isinstance(node, IntegerLiteral):
        return f"{prefix}Integer: {node.value}"
    elif isinstance(node, RealLiteral):
        return f"{prefix}Real: {node.value}"
    elif isinstance(node, StringLiteral):
        return f"{prefix}String: {repr(node.value)}"
    elif isinstance(node, LogicalLiteral):
        return f"{prefix}Logical: {node.value}"
    elif isinstance(node, ComplexLiteral):
        return f"{prefix}Complex: ({node.real_part}, {node.imag_part})"
    elif isinstance(node, DataStatement):
        return f"{prefix}DATA: {len(node.items)} items"
    else:
        return f"{prefix}{type(node).__name__}: {node}"
