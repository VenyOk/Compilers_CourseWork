from typing import Dict, List, Optional
from src.core import (
    Program, Declaration, Statement, Assignment, DoLoop, IfStatement,
    PrintStatement, ReadStatement, WriteStatement, CallStatement,
    BinaryOp, UnaryOp, Variable, IntegerLiteral, RealLiteral,
    StringLiteral, LogicalLiteral, FunctionCall, Expression,
    ReturnStatement, StopStatement, DoWhile, LabeledDoLoop, LabeledDoWhile,
    SimpleIfStatement, ArrayRef, DimensionStatement, GotoStatement, ContinueStatement,
    Subroutine, FunctionDef
)


class SSAGenerator:
    def __init__(self):
        self.instructions: List[str] = []
        self.var_versions: Dict[str, int] = {}
        self.temp_counter = 0
        self.var_types: Dict[str, str] = {}
        
    def generate(self, ast: Program) -> List[str]:
        self.instructions = []
        self.var_versions = {}
        self.temp_counter = 0
        self.var_types = {}
        
        self.instructions.append("====== SSA REPRESENTATION ======")
        self.instructions.append("")
        
        for decl in ast.declarations:
            if isinstance(decl, Declaration):
                self._process_declaration(decl)
            
        for stmt in ast.statements:
            self._process_statement(stmt)
            
        return self.instructions
    
    def _process_declaration(self, decl: Declaration):
        for name_info in decl.names:
            var_name = name_info[0]
            self.var_types[var_name] = decl.type
            self.var_versions[var_name] = 0
            self.instructions.append(f"{var_name} = alloca {decl.type}")
    
    def _process_statement(self, stmt: Statement):
        if isinstance(stmt, Assignment):
            self._process_assignment(stmt)
        elif isinstance(stmt, PrintStatement):
            self._process_print(stmt)
        elif isinstance(stmt, DoLoop):
            self._process_do_loop(stmt)
        elif isinstance(stmt, DoWhile):
            self._process_do_while(stmt)
        elif isinstance(stmt, IfStatement):
            self._process_if_statement(stmt)
        elif isinstance(stmt, SimpleIfStatement):
            self._process_simple_if(stmt)
        elif isinstance(stmt, ReadStatement):
            pass
        elif isinstance(stmt, WriteStatement):
            self._process_write(stmt)
        elif isinstance(stmt, CallStatement):
            self._process_call(stmt)
        elif isinstance(stmt, ReturnStatement):
            pass
        elif isinstance(stmt, StopStatement):
            pass
        elif isinstance(stmt, GotoStatement):
            pass
        elif isinstance(stmt, ContinueStatement):
            pass
    
    def _process_assignment(self, assign: Assignment):
        target = assign.target
        value_expr = self._process_expression(assign.value)
        
        if target not in self.var_versions:
            self.var_versions[target] = 0
        
        self.var_versions[target] += 1
        version = self.var_versions[target]
        
        self.instructions.append(f"{target}_{version} = assign {value_expr}")
    
    def _process_expression(self, expr: Expression) -> str:
        if isinstance(expr, IntegerLiteral):
            return str(expr.value)
        elif isinstance(expr, RealLiteral):
            return str(expr.value)
        elif isinstance(expr, StringLiteral):
            return f'"{expr.value}"'
        elif isinstance(expr, LogicalLiteral):
            return "TRUE" if expr.value else "FALSE"
        elif isinstance(expr, Variable):
            var_name = expr.name
            if var_name not in self.var_versions:
                self.var_versions[var_name] = 0
                return var_name
            version = self.var_versions[var_name]
            return f"{var_name}_{version}" if version > 0 else var_name
        elif isinstance(expr, BinaryOp):
            left = self._process_expression(expr.left)
            right = self._process_expression(expr.right)
            op = self._get_op_symbol(expr.op)
            self.temp_counter += 1
            temp_name = f"%tmp_{self.temp_counter}"
            self.instructions.append(f"{temp_name} = {op} {left} {right}")
            return temp_name
        elif isinstance(expr, UnaryOp):
            operand = self._process_expression(expr.operand)
            op = self._get_unary_op_symbol(expr.op)
            self.temp_counter += 1
            temp_name = f"%tmp_{self.temp_counter}"
            self.instructions.append(f"{temp_name} = {op} {operand}")
            return temp_name
        elif isinstance(expr, FunctionCall):
            args = [self._process_expression(arg) for arg in expr.args]
            args_str = " ".join(args)
            self.temp_counter += 1
            temp_name = f"%tmp_{self.temp_counter}"
            self.instructions.append(f"{temp_name} = call {expr.name} {args_str}")
            return temp_name
        elif isinstance(expr, ArrayRef):
            array_name = expr.name
            indices = [self._process_expression(idx) for idx in expr.indices]
            indices_str = " ".join(indices)
            if array_name not in self.var_versions:
                self.var_versions[array_name] = 0
            version = self.var_versions[array_name]
            base_name = f"{array_name}_{version}" if version > 0 else array_name
            self.temp_counter += 1
            temp_name = f"%tmp_{self.temp_counter}"
            self.instructions.append(f"{temp_name} = load {base_name} {indices_str}")
            return temp_name
        else:
            return "unknown"
    
    def _get_op_symbol(self, op: str) -> str:
        op_map = {
            '+': '+',
            '-': '-',
            '*': '*',
            '/': '/',
            '**': '**',
            '.EQ.': '==',
            '==': '==',
            '.NE.': '!=',
            '/=': '!=',
            '.LT.': '<',
            '<': '<',
            '.LE.': '<=',
            '<=': '<=',
            '.GT.': '>',
            '>': '>',
            '.GE.': '>=',
            '>=': '>=',
            '.AND.': 'and',
            '&': 'and',
            '.OR.': 'or',
            '|': 'or',
            '.EQV.': 'eqv',
            '.NEQV.': 'neqv',
        }
        return op_map.get(op, op)
    
    def _get_unary_op_symbol(self, op: str) -> str:
        op_map = {
            '-': '-',
            '+': '+',
            '.NOT.': 'not',
            '~': 'not',
            'NOT': 'not',
        }
        return op_map.get(op, op)
    
    def _process_print(self, stmt: PrintStatement):
        for item in stmt.items:
            value = self._process_expression(item)
            self.instructions.append(f"print {value}")
    
    def _process_write(self, stmt: WriteStatement):
        for item in stmt.items:
            value = self._process_expression(item)
            self.instructions.append(f"write {value}")
    
    def _process_call(self, stmt: CallStatement):
        args = [self._process_expression(arg) for arg in stmt.args]
        args_str = " ".join(args) if args else ""
        self.instructions.append(f"call {stmt.name} {args_str}")
    
    def _process_do_loop(self, stmt: DoLoop):
        var = stmt.var
        start = self._process_expression(stmt.start)
        end = self._process_expression(stmt.end)
        step = self._process_expression(stmt.step) if stmt.step else "1"
        
        if var not in self.var_versions:
            self.var_versions[var] = 0
        
        self.var_versions[var] += 1
        version = self.var_versions[var]
        
        self.instructions.append(f"do {var}_{version} = {start} {end} {step}")
        for body_stmt in stmt.body:
            self._process_statement(body_stmt)
        self.instructions.append("end do")
    
    def _process_do_while(self, stmt: DoWhile):
        condition = self._process_expression(stmt.condition)
        self.instructions.append(f"do while {condition}")
        for body_stmt in stmt.body:
            self._process_statement(body_stmt)
        self.instructions.append("end do")
    
    def _process_if_statement(self, stmt: IfStatement):
        condition = self._process_expression(stmt.condition)
        self.instructions.append(f"if {condition} then")
        for then_stmt in stmt.then_body:
            self._process_statement(then_stmt)
        for elif_cond, elif_body in stmt.elif_parts:
            elif_cond_str = self._process_expression(elif_cond)
            self.instructions.append(f"elseif {elif_cond_str} then")
            for elif_stmt in elif_body:
                self._process_statement(elif_stmt)
        if stmt.else_body:
            self.instructions.append("else")
            for else_stmt in stmt.else_body:
                self._process_statement(else_stmt)
        self.instructions.append("end if")
    
    def _process_simple_if(self, stmt: SimpleIfStatement):
        condition = self._process_expression(stmt.condition)
        self.instructions.append(f"if {condition} then")
        self._process_statement(stmt.statement)
        self.instructions.append("end if")
    
    def to_string(self, instructions: List[str]) -> str:
        return "\n".join(instructions)

