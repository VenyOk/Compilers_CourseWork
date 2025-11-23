from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, field
from src.core import (
    Program, Declaration, Statement, Assignment, DoLoop, IfStatement,
    PrintStatement, ReadStatement, WriteStatement, CallStatement,
    BinaryOp, UnaryOp, Variable, IntegerLiteral, RealLiteral,
    StringLiteral, LogicalLiteral, FunctionCall, Expression,
    ReturnStatement, StopStatement, GotoStatement, ContinueStatement,
    ExitStatement, SimpleIfStatement, DoWhile, LabeledDoLoop, LabeledDoWhile,
    ArrayRef
)
@dataclass
class SSAVariable:
    name: str
    version: int
    type_: str = "unknown"
    def __str__(self):
        return f"{self.name}_{self.version}"
@dataclass
class SSAInstruction:
    opcode: str
    operands: List[str] = field(default_factory=list)
    result: Optional[str] = None
    def __str__(self):
        if self.result:
            return f"{self.result} = {self.opcode} {' '.join(self.operands)}"
        else:
            return f"{self.opcode} {' '.join(self.operands)}"
class SSAGenerator:
    def __init__(self):
        self.instructions: List[SSAInstruction] = []
        self.var_versions: Dict[str, int] = {}
        self.phi_functions: Dict[str, List[Tuple[str, str]]] = {}
        self.blocks: Dict[str, List[SSAInstruction]] = {}
        self.current_block = "entry"
        self.block_counter = 0
        self.temp_counter = 0
    def generate(self, ast: Program) -> List[SSAInstruction]:
        self.blocks = {"entry": []}
        for decl in ast.declarations:
            self._process_declaration(decl)
        for stmt in ast.statements:
            self._process_statement(stmt)
        instructions = []
        for block_name in sorted(self.blocks.keys()):
            instructions.extend(self.blocks[block_name])
        return instructions
    def _process_declaration(self, decl: Declaration):
        if hasattr(decl, 'names'):
            for name, dims in decl.names:
                if dims:
                    dim_sizes = []
                    for d in dims:
                        if isinstance(d, tuple):
                            start, end = d
                            dim_sizes.append(str(end - start + 1))
                        else:
                            dim_sizes.append(str(d))
                    instr = SSAInstruction(
                        "alloca_array",
                        operands=dim_sizes,
                        result=name
                    )
                else:
                    instr = SSAInstruction(
                        "alloca",
                        operands=[decl.type],
                        result=name
                    )
                self._emit(instr)
                self.var_versions[name] = 0
    def _process_statement(self, stmt: Statement):
        if isinstance(stmt, Assignment):
            self._process_assignment(stmt)
        elif isinstance(stmt, DoLoop):
            self._process_do_loop(stmt)
        elif isinstance(stmt, DoWhile):
            self._process_do_while(stmt)
        elif isinstance(stmt, LabeledDoLoop):
            self._process_labeled_do_loop(stmt)
        elif isinstance(stmt, LabeledDoWhile):
            self._process_labeled_do_while(stmt)
        elif isinstance(stmt, IfStatement):
            self._process_if_statement(stmt)
        elif isinstance(stmt, SimpleIfStatement):
            self._process_simple_if_statement(stmt)
        elif isinstance(stmt, PrintStatement):
            self._process_print_statement(stmt)
        elif isinstance(stmt, (ReadStatement, WriteStatement)):
            self._process_io_statement(stmt)
        elif isinstance(stmt, CallStatement):
            self._process_call_statement(stmt)
        elif isinstance(stmt, StopStatement):
            self._emit(SSAInstruction("stop"))
        elif isinstance(stmt, ReturnStatement):
            self._emit(SSAInstruction("return"))
        elif isinstance(stmt, GotoStatement):
            self._emit(SSAInstruction("goto", operands=[str(stmt.label)]))
        elif isinstance(stmt, ContinueStatement):
            self._emit(SSAInstruction("continue"))
        elif isinstance(stmt, ExitStatement):
            self._emit(SSAInstruction("exit"))
    def _process_assignment(self, stmt: Assignment):
        rhs_val = self._process_expression(stmt.value)
        if stmt.target not in self.var_versions:
            self.var_versions[stmt.target] = 0
        else:
            self.var_versions[stmt.target] += 1
        new_version = self.var_versions[stmt.target]
        new_var = f"{stmt.target}_{new_version}"
        if stmt.indices:
            indices = [self._process_expression(idx) for idx in stmt.indices]
            instr = SSAInstruction(
                "store_array",
                operands=[rhs_val] + indices,
                result=new_var
            )
        else:
            instr = SSAInstruction(
                "assign",
                operands=[rhs_val],
                result=new_var
            )
        self._emit(instr)
    def _process_do_loop(self, stmt: DoLoop):
        init_val = self._process_expression(stmt.start)
        self._emit(SSAInstruction(
            "assign",
            operands=[init_val],
            result=f"{stmt.var}_loop_init"
        ))
        loop_block = self._new_block("loop")
        body_block = self._new_block("loop_body")
        exit_block = self._new_block("loop_exit")
        self._emit_phi(f"{stmt.var}_loop",
                      [(f"{stmt.var}_loop_init", self.current_block)])
        end_val = self._process_expression(stmt.end)
        step_val = "1"
        if stmt.step:
            step_val = self._process_expression(stmt.step)
        self._emit(SSAInstruction(
            "branch_if_le",
            operands=[f"{stmt.var}_loop", end_val, body_block, exit_block]
        ))
        self.current_block = body_block
        for body_stmt in stmt.body:
            self._process_statement(body_stmt)
        self._emit(SSAInstruction(
            "add",
            operands=[f"{stmt.var}_loop", step_val],
            result=f"{stmt.var}_loop_next"
        ))
        self._emit(SSAInstruction("branch", operands=[loop_block]))
        self.current_block = exit_block
    def _process_do_while(self, stmt: DoWhile):
        loop_block = self._new_block("do_while_loop")
        body_block = self._new_block("do_while_body")
        exit_block = self._new_block("do_while_exit")
        self._emit(SSAInstruction("branch", operands=[loop_block]))
        self.current_block = loop_block
        cond_val = self._process_expression(stmt.condition)
        self._emit(SSAInstruction(
            "branch_if",
            operands=[cond_val, body_block, exit_block]
        ))
        self.current_block = body_block
        for body_stmt in stmt.body:
            self._process_statement(body_stmt)
        self._emit(SSAInstruction("branch", operands=[loop_block]))
        self.current_block = exit_block
    def _process_labeled_do_loop(self, stmt: LabeledDoLoop):
        self._process_do_loop(stmt)
    def _process_labeled_do_while(self, stmt: LabeledDoWhile):
        self._process_do_while(stmt)
    def _process_if_statement(self, stmt: IfStatement):
        cond_val = self._process_expression(stmt.condition)
        then_block = self._new_block("then")
        exit_block = self._new_block("if_exit")
        else_block = exit_block
        if stmt.else_body or stmt.elif_parts:
            else_block = self._new_block("else")
        self._emit(SSAInstruction(
            "branch_if",
            operands=[cond_val, then_block, else_block]
        ))
        self.current_block = then_block
        for stmt_node in stmt.then_body:
            self._process_statement(stmt_node)
        self._emit(SSAInstruction("branch", operands=[exit_block]))
        if stmt.elif_parts:
            for elif_cond, elif_body in stmt.elif_parts:
                elif_block = self._new_block("elif")
                elif_then_block = self._new_block("elif_then")
                self.current_block = else_block
                elif_cond_val = self._process_expression(elif_cond)
                self._emit(SSAInstruction(
                    "branch_if",
                    operands=[elif_cond_val, elif_then_block, elif_block]
                ))
                self.current_block = elif_then_block
                for elif_stmt in elif_body:
                    self._process_statement(elif_stmt)
                self._emit(SSAInstruction("branch", operands=[exit_block]))
                else_block = elif_block
        if stmt.else_body:
            self.current_block = else_block
            for stmt_node in stmt.else_body:
                self._process_statement(stmt_node)
            self._emit(SSAInstruction("branch", operands=[exit_block]))
        self.current_block = exit_block
    def _process_simple_if_statement(self, stmt: SimpleIfStatement):
        cond_val = self._process_expression(stmt.condition)
        then_block = self._new_block("then")
        exit_block = self._new_block("if_exit")
        self._emit(SSAInstruction(
            "branch_if",
            operands=[cond_val, then_block, exit_block]
        ))
        self.current_block = then_block
        self._process_statement(stmt.statement)
        self._emit(SSAInstruction("branch", operands=[exit_block]))
        self.current_block = exit_block
    def _process_print_statement(self, stmt: PrintStatement):
        operands = []
        for item in stmt.items:
            val = self._process_expression(item)
            operands.append(val)
        self._emit(SSAInstruction(
            "print",
            operands=operands
        ))
    def _process_io_statement(self, stmt):
        if isinstance(stmt, ReadStatement):
            for item in stmt.items:
                if item not in self.var_versions:
                    self.var_versions[item] = 0
                new_version = self.var_versions[item]
                new_var = f"{item}_{new_version}"
                self._emit(SSAInstruction(
                    "read",
                    operands=[],
                    result=new_var
                ))
        elif isinstance(stmt, WriteStatement):
            operands = []
            for item in stmt.items:
                val = self._process_expression(item)
                operands.append(val)
            self._emit(SSAInstruction(
                "write",
                operands=operands
            ))
    def _process_call_statement(self, stmt: CallStatement):
        operands = []
        for arg in stmt.args:
            val = self._process_expression(arg)
            operands.append(val)
        self._emit(SSAInstruction(
            "call",
            operands=[stmt.name] + operands
        ))
    def _process_expression(self, expr: Expression) -> str:
        if isinstance(expr, IntegerLiteral):
            return str(expr.value)
        elif isinstance(expr, RealLiteral):
            return str(expr.value)
        elif isinstance(expr, StringLiteral):
            return f'"{expr.value}"'
        elif isinstance(expr, LogicalLiteral):
            return ".TRUE." if expr.value else ".FALSE."
        elif isinstance(expr, Variable):
            if expr.name in self.var_versions:
                version = self.var_versions[expr.name]
                return f"{expr.name}_{version}"
            return expr.name
        elif isinstance(expr, ArrayRef):
            array_name = expr.name
            indices = [self._process_expression(idx) for idx in expr.indices]
            result = self._new_temp()
            if array_name in self.var_versions:
                version = self.var_versions[array_name]
                array_var = f"{array_name}_{version}"
            else:
                array_var = array_name
            self._emit(SSAInstruction(
                "load_array",
                operands=[array_var] + indices,
                result=result
            ))
            return result
        elif isinstance(expr, BinaryOp):
            return self._process_binop(expr)
        elif isinstance(expr, UnaryOp):
            return self._process_unaryop(expr)
        elif isinstance(expr, FunctionCall):
            return self._process_function_call(expr)
        return "unknown"
    def _process_binop(self, expr: BinaryOp) -> str:
        left = self._process_expression(expr.left)
        right = self._process_expression(expr.right)
        result = self._new_temp()
        opcode = expr.op
        if opcode == "**":
            opcode = "pow"
        elif opcode in {".EQ.", "=="}:
            opcode = "eq"
        elif opcode in {".NE.", "/="}:
            opcode = "ne"
        elif opcode in {".LT.", "<"}:
            opcode = "lt"
        elif opcode in {".LE.", "<="}:
            opcode = "le"
        elif opcode in {".GT.", ">"}:
            opcode = "gt"
        elif opcode in {".GE.", ">="}:
            opcode = "ge"
        elif opcode in {".AND.", "&"}:
            opcode = "and"
        elif opcode in {".OR.", "|"}:
            opcode = "or"
        elif opcode in {".NOT.", "~", "NOT"}:
            opcode = "not"
        elif opcode in {".EQV.", ".EQV"}:
            opcode = "eqv"
        elif opcode in {".NEQV.", ".NEQV"}:
            opcode = "neqv"
        self._emit(SSAInstruction(
            opcode,
            operands=[left, right],
            result=result
        ))
        return result
    def _process_unaryop(self, expr: UnaryOp) -> str:
        operand = self._process_expression(expr.operand)
        result = self._new_temp()
        opcode = expr.op
        if opcode in {".NOT.", "~", "NOT"}:
            opcode = "not"
        self._emit(SSAInstruction(
            opcode,
            operands=[operand],
            result=result
        ))
        return result
    def _process_function_call(self, expr: FunctionCall) -> str:
        args = [self._process_expression(arg) for arg in expr.args]
        result = self._new_temp()
        self._emit(SSAInstruction(
            "call",
            operands=[expr.name] + args,
            result=result
        ))
        return result
    def _emit(self, instr: SSAInstruction):
        if self.current_block not in self.blocks:
            self.blocks[self.current_block] = []
        self.blocks[self.current_block].append(instr)
    def _emit_phi(self, var: str, predecessors: List[Tuple[str, str]]):
        operands = []
        for val, block in predecessors:
            operands.append(f"[{val}, {block}]")
        self._emit(SSAInstruction(
            "phi",
            operands=operands,
            result=var
        ))
    def _new_block(self, prefix: str = "block") -> str:
        self.block_counter += 1
        return f"{prefix}_{self.block_counter}"
    def _new_temp(self) -> str:
        self.temp_counter += 1
        return f"%tmp_{self.temp_counter}"
    def to_string(self, instructions: List[SSAInstruction]) -> str:
        lines = [
            "====== SSA REPRESENTATION ======",
            ""
        ]
        for instr in instructions:
            lines.append(str(instr))
        return "\n".join(lines)