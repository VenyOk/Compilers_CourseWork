from typing import Dict, List, Optional, Tuple
from src.core import (
    Program, Declaration, Statement, Assignment, DoLoop, IfStatement,
    PrintStatement, ReadStatement, WriteStatement, CallStatement,
    BinaryOp, UnaryOp, Variable, IntegerLiteral, RealLiteral,
    StringLiteral, LogicalLiteral, FunctionCall, Expression,
    ReturnStatement, StopStatement, DoWhile, LabeledDoLoop, LabeledDoWhile,
    SimpleIfStatement, ArrayRef
)
class LLVMGenerator:
    def __init__(self, target_triple: str = "x86_64-unknown-linux-gnu"):
        self.code_lines: List[str] = []
        self.var_alloc: Dict[str, Tuple[str, str]] = {}
        self.local_counter = 0
        self.string_counter = 0
        self.strings: Dict[str, str] = {}
        self.target_triple = target_triple
        self.type_map = {
            'INTEGER': 'i32',
            'REAL': 'double',
            'LOGICAL': 'i1',
            'CHARACTER': 'i8*',
        }
    def generate(self, ast: Program) -> str:
        self.code_lines = []
        self.strings = {}
        self.string_counter = 0
        self._collect_strings(ast)
        self._emit_header()
        self._emit_external_declarations()
        strings_line_start = len(self.code_lines)
        self.code_lines.append("; === Глобальные строки (будут заполнены) ===")
        self.local_counter = 0
        self.var_alloc = {}
        self._emit_program(ast)
        if self.strings:
            strings_lines = ["; === Глобальные строки ==="]
            for string_val, string_name in self.strings.items():
                escaped = string_val.replace("\\", "\\\\").replace('"', '\\"')
                length = len(escaped) + 1
                strings_lines.append(f'{string_name} = private constant [{length} x i8] c"{escaped}\\00"')
            strings_lines.append("")
            self.code_lines[strings_line_start:strings_line_start+1] = strings_lines
        return "\n".join(self.code_lines)
    def _collect_strings(self, node):
        if isinstance(node, StringLiteral):
            self._add_string(node.value)
        elif isinstance(node, list):
            for item in node:
                self._collect_strings(item)
        elif hasattr(node, '__dict__'):
            for attr_val in node.__dict__.values():
                self._collect_strings(attr_val)
    def _emit_header(self):
        self.code_lines.append("; ModuleID = 'fortran'")
        self.code_lines.append("source_filename = \"fortran\"")
        self.code_lines.append(f"target triple = \"{self.target_triple}\"")
        self.code_lines.append("")
    def _emit_external_declarations(self):
        self.code_lines.append("; === Объявления внешних Си функций ===")
        self.code_lines.append("declare i32 @printf(i8*, ...)")
        self.code_lines.append("declare i32 @scanf(i8*, ...)")
        self.code_lines.append("declare i32 @puts(i8*)")
        self.code_lines.append("declare double @sqrt(double)")
        self.code_lines.append("declare double @sin(double)")
        self.code_lines.append("declare double @cos(double)")
        self.code_lines.append("declare double @exp(double)")
        self.code_lines.append("declare double @log(double)")
        self.code_lines.append("declare double @log10(double)")
        self.code_lines.append("declare double @pow(double, double)")
        self.code_lines.append("declare i32 @abs(i32)")
        self.code_lines.append("declare double @fabs(double)")
        self.code_lines.append("")
    def _emit_program(self, ast: Program):
        self.code_lines.append("; === Основная программа ===")
        self.code_lines.append("define i32 @main() {")
        self.code_lines.append("entry:")
        self.local_counter = 0
        self.var_alloc = {}
        self._emit_declarations(ast.declarations)
        for stmt in ast.statements:
            self._emit_statement(stmt)
        self.code_lines.append("  ret i32 0")
        self.code_lines.append("}")
        self.code_lines.append("")
    def _emit_declarations(self, declarations: List[Declaration]):
        for decl in declarations:
            if hasattr(decl, 'names'):
                llvm_type = self.type_map.get(decl.type, 'i32')
                for name, dims in decl.names:
                    if dims:
                        size = 1
                        for d in dims:
                            if isinstance(d, tuple):
                                start, end = d
                                size *= (end - start + 1)
                            else:
                                size *= d
                        alloc_type = f"[{size} x {llvm_type}]"
                        local_name = f"%{name}"
                    else:
                        alloc_type = llvm_type
                        local_name = f"%{name}"
                    self.code_lines.append(f"  {local_name} = alloca {alloc_type}")
                    self.var_alloc[name] = (alloc_type, local_name)
    def _emit_statement(self, stmt: Statement):
        if isinstance(stmt, Assignment):
            self._emit_assignment(stmt)
        elif isinstance(stmt, DoLoop):
            self._emit_do_loop(stmt)
        elif isinstance(stmt, DoWhile):
            self._emit_do_while(stmt)
        elif isinstance(stmt, LabeledDoLoop):
            self._emit_do_loop(stmt)
        elif isinstance(stmt, LabeledDoWhile):
            self._emit_do_while(stmt)
        elif isinstance(stmt, IfStatement):
            self._emit_if_statement(stmt)
        elif isinstance(stmt, SimpleIfStatement):
            self._emit_simple_if_statement(stmt)
        elif isinstance(stmt, PrintStatement):
            self._emit_print_statement(stmt)
        elif isinstance(stmt, ReadStatement):
            self._emit_read_statement(stmt)
        elif isinstance(stmt, WriteStatement):
            self._emit_write_statement(stmt)
        elif isinstance(stmt, CallStatement):
            self._emit_call_statement(stmt)
        elif isinstance(stmt, StopStatement):
            self.code_lines.append("  ret i32 0")
    def _emit_assignment(self, stmt: Assignment):
        rhs_val, rhs_type = self._emit_expression(stmt.value)
        if stmt.indices:
            indices_vals = [self._emit_expression(idx)[0] for idx in stmt.indices]
            if stmt.target in self.var_alloc:
                alloc_type, ptr_name = self.var_alloc[stmt.target]
                base_type = self.type_map.get('REAL', 'double')
                if '[' in alloc_type:
                    elem_ptr = self._new_local()
                    idx = indices_vals[0]
                    adjusted_idx = self._new_local()
                    self.code_lines.append(f"  {adjusted_idx} = sub i32 {idx}, 1")
                    array_size = alloc_type.split('[')[1].split(']')[0].split(' x ')[0]
                    self.code_lines.append(
                        f"  {elem_ptr} = getelementptr inbounds {alloc_type}, "
                        f"{alloc_type}* {ptr_name}, i64 0, i32 {adjusted_idx}"
                    )
                    self.code_lines.append(f"  store {rhs_type} {rhs_val}, {rhs_type}* {elem_ptr}")
        else:
            if stmt.target in self.var_alloc:
                alloc_type, ptr_name = self.var_alloc[stmt.target]
                if '[' in alloc_type:
                    base_type = alloc_type.split('[')[1].split(']')[0].split(' x ')[1]
                else:
                    base_type = alloc_type
                self.code_lines.append(f"  store {base_type} {rhs_val}, {base_type}* {ptr_name}")
    def _emit_do_loop(self, stmt: DoLoop):
        loop_id = self._new_block_id()
        loop_label = f"loop_{loop_id}"
        loop_body_label = f"loop_body_{loop_id}"
        loop_end_label = f"loop_end_{loop_id}"
        start_val, _ = self._emit_expression(stmt.start)
        if stmt.var in self.var_alloc:
            alloc_type, ptr = self.var_alloc[stmt.var]
            base_type = alloc_type if '[' not in alloc_type else alloc_type.split('[')[1].split(']')[0].split(' x ')[1]
            self.code_lines.append(f"  store {base_type} {start_val}, {base_type}* {ptr}")
        self.code_lines.append(f"{loop_label}:")
        if stmt.var in self.var_alloc:
            alloc_type, ptr = self.var_alloc[stmt.var]
            base_type = alloc_type if '[' not in alloc_type else alloc_type.split('[')[1].split(']')[0].split(' x ')[1]
            loop_var = self._new_local()
            self.code_lines.append(f"  {loop_var} = load {base_type}, {base_type}* {ptr}")
            end_val, _ = self._emit_expression(stmt.end)
            cond = self._new_local()
            self.code_lines.append(f"  {cond} = icmp sle {base_type} {loop_var}, {end_val}")
            self.code_lines.append(f"  br i1 {cond}, label %{loop_body_label}, label %{loop_end_label}")
        self.code_lines.append(f"{loop_body_label}:")
        for body_stmt in stmt.body:
            self._emit_statement(body_stmt)
        step = 1
        if stmt.step:
            step_val, _ = self._emit_expression(stmt.step)
            step = step_val
        else:
            step_val = self._new_local()
            self.code_lines.append(f"  {step_val} = add {base_type} {loop_var}, 1")
            step = step_val
        if stmt.var in self.var_alloc:
            alloc_type, ptr = self.var_alloc[stmt.var]
            base_type = alloc_type if '[' not in alloc_type else alloc_type.split('[')[1].split(']')[0].split(' x ')[1]
            loop_var = self._new_local()
            self.code_lines.append(f"  {loop_var} = load {base_type}, {base_type}* {ptr}")
            next_val = self._new_local()
            self.code_lines.append(f"  {next_val} = add {base_type} {loop_var}, {step}")
            self.code_lines.append(f"  store {base_type} {next_val}, {base_type}* {ptr}")
        self.code_lines.append(f"  br label %{loop_label}")
        self.code_lines.append(f"{loop_end_label}:")
    def _emit_do_while(self, stmt: DoWhile):
        loop_id = self._new_block_id()
        loop_label = f"do_while_loop_{loop_id}"
        loop_body_label = f"do_while_body_{loop_id}"
        loop_end_label = f"do_while_end_{loop_id}"
        self.code_lines.append(f"  br label %{loop_label}")
        self.code_lines.append(f"{loop_label}:")
        cond_val, cond_type = self._emit_expression(stmt.condition)
        if cond_type != "i1":
            cond_bool = self._new_local()
            self.code_lines.append(f"  {cond_bool} = icmp ne {cond_type} {cond_val}, 0")
            cond_val = cond_bool
        self.code_lines.append(f"  br i1 {cond_val}, label %{loop_body_label}, label %{loop_end_label}")
        self.code_lines.append(f"{loop_body_label}:")
        for body_stmt in stmt.body:
            self._emit_statement(body_stmt)
        self.code_lines.append(f"  br label %{loop_label}")
        self.code_lines.append(f"{loop_end_label}:")
    def _emit_if_statement(self, stmt: IfStatement):
        if_id = self._new_block_id()
        then_label = f"if_then_{if_id}"
        endif_label = f"if_end_{if_id}"
        cond_val, cond_type = self._emit_expression(stmt.condition)
        if cond_type != "i1":
            cond_bool = self._new_local()
            self.code_lines.append(f"  {cond_bool} = icmp ne {cond_type} {cond_val}, 0")
            cond_val = cond_bool
        if stmt.elif_parts:
            elif_labels = []
            for i, (elif_cond, _) in enumerate(stmt.elif_parts):
                elif_labels.append(f"if_elif_{if_id}_{i}")
            else_label = f"if_else_{if_id}" if stmt.else_body else endif_label
            next_label = elif_labels[0] if elif_labels else (else_label if stmt.else_body else endif_label)
            self.code_lines.append(f"  br i1 {cond_val}, label %{then_label}, label %{next_label}")
            self.code_lines.append(f"{then_label}:")
            for s in stmt.then_body:
                self._emit_statement(s)
            self.code_lines.append(f"  br label %{endif_label}")
            for i, (elif_cond, elif_body) in enumerate(stmt.elif_parts):
                self.code_lines.append(f"{elif_labels[i]}:")
                elif_cond_val, elif_cond_type = self._emit_expression(elif_cond)
                if elif_cond_type != "i1":
                    elif_cond_bool = self._new_local()
                    self.code_lines.append(f"  {elif_cond_bool} = icmp ne {elif_cond_type} {elif_cond_val}, 0")
                    elif_cond_val = elif_cond_bool
                next_elif_label = elif_labels[i + 1] if i + 1 < len(elif_labels) else else_label
                self.code_lines.append(f"  br i1 {elif_cond_val}, label %if_elif_then_{if_id}_{i}, label %{next_elif_label}")
                self.code_lines.append(f"if_elif_then_{if_id}_{i}:")
                for s in elif_body:
                    self._emit_statement(s)
                self.code_lines.append(f"  br label %{endif_label}")
            if stmt.else_body:
                self.code_lines.append(f"{else_label}:")
                for s in stmt.else_body:
                    self._emit_statement(s)
                self.code_lines.append(f"  br label %{endif_label}")
        else:
            else_label = f"if_else_{if_id}" if stmt.else_body else endif_label
            self.code_lines.append(f"  br i1 {cond_val}, label %{then_label}, label %{else_label}")
            self.code_lines.append(f"{then_label}:")
            for s in stmt.then_body:
                self._emit_statement(s)
            self.code_lines.append(f"  br label %{endif_label}")
            if stmt.else_body:
                self.code_lines.append(f"{else_label}:")
                for s in stmt.else_body:
                    self._emit_statement(s)
                self.code_lines.append(f"  br label %{endif_label}")
        self.code_lines.append(f"{endif_label}:")
    def _emit_simple_if_statement(self, stmt: SimpleIfStatement):
        if_id = self._new_block_id()
        then_label = f"if_then_{if_id}"
        endif_label = f"if_end_{if_id}"
        cond_val, cond_type = self._emit_expression(stmt.condition)
        if cond_type != "i1":
            cond_bool = self._new_local()
            self.code_lines.append(f"  {cond_bool} = icmp ne {cond_type} {cond_val}, 0")
            cond_val = cond_bool
        self.code_lines.append(f"  br i1 {cond_val}, label %{then_label}, label %{endif_label}")
        self.code_lines.append(f"{then_label}:")
        self._emit_statement(stmt.statement)
        self.code_lines.append(f"  br label %{endif_label}")
        self.code_lines.append(f"{endif_label}:")
    def _emit_print_statement(self, stmt: PrintStatement):
        if not stmt.items:
            return
        for item in stmt.items:
            val, val_type = self._emit_expression(item)
            if val_type == "i8*":
                self.code_lines.append(f"  call i32 @puts(i8* {val})")
            elif val_type == "double":
                fmt_str = "%g\\n"
                str_name = self._add_string(fmt_str)
                str_ptr = self._new_local()
                self.code_lines.append(
                    f"  {str_ptr} = getelementptr inbounds [5 x i8], "
                    f"[5 x i8]* {str_name}, i64 0, i64 0"
                )
                self.code_lines.append(f"  call i32 (i8*, ...) @printf(i8* {str_ptr}, double {val})")
            else:
                fmt_str = "%d\\n"
                str_name = self._add_string(fmt_str)
                str_ptr = self._new_local()
                self.code_lines.append(
                    f"  {str_ptr} = getelementptr inbounds [4 x i8], "
                    f"[4 x i8]* {str_name}, i64 0, i64 0"
                )
                self.code_lines.append(f"  call i32 (i8*, ...) @printf(i8* {str_ptr}, i32 {val})")
    def _emit_read_statement(self, stmt: ReadStatement):
        for item in stmt.items:
            if item in self.var_alloc:
                alloc_type, ptr = self.var_alloc[item]
                base_type = alloc_type if '[' not in alloc_type else alloc_type.split('[')[1].split(']')[0].split(' x ')[1]
                format_str = "%lf" if base_type == 'double' else "%d"
                str_name = self._add_string(format_str)
                str_ptr = self._new_local()
                self.code_lines.append(f"  {str_ptr} = getelementptr inbounds [4 x i8], "
                                      f"[4 x i8]* {str_name}, i64 0, i64 0")
                self.code_lines.append(f"  call i32 (i8*, ...) @scanf(i8* {str_ptr}, {base_type}* {ptr})")
    def _emit_write_statement(self, stmt: WriteStatement):
        for item in stmt.items:
            val, val_type = self._emit_expression(item)
            if val_type == "i8*":
                self.code_lines.append(f"  call i32 @puts(i8* {val})")
            elif val_type == "double":
                fmt_str = "%g\\n"
                str_name = self._add_string(fmt_str)
                str_ptr = self._new_local()
                self.code_lines.append(
                    f"  {str_ptr} = getelementptr inbounds [5 x i8], "
                    f"[5 x i8]* {str_name}, i64 0, i64 0"
                )
                self.code_lines.append(f"  call i32 (i8*, ...) @printf(i8* {str_ptr}, double {val})")
            else:
                fmt_str = "%d\\n"
                str_name = self._add_string(fmt_str)
                str_ptr = self._new_local()
                self.code_lines.append(
                    f"  {str_ptr} = getelementptr inbounds [4 x i8], "
                    f"[4 x i8]* {str_name}, i64 0, i64 0"
                )
                self.code_lines.append(f"  call i32 (i8*, ...) @printf(i8* {str_ptr}, i32 {val})")
    def _emit_call_statement(self, stmt: CallStatement):
        args = [self._emit_expression(arg)[0] for arg in stmt.args]
        args_str = ", ".join(args)
        self.code_lines.append(f"  call void @{stmt.name}({args_str})")
    def _emit_expression(self, expr: Expression) -> Tuple[str, str]:
        if isinstance(expr, IntegerLiteral):
            return (str(expr.value), "i32")
        elif isinstance(expr, RealLiteral):
            return (str(expr.value), "double")
        elif isinstance(expr, StringLiteral):
            str_name = self._add_string(expr.value)
            ptr = self._new_local()
            self.code_lines.append(f"  {ptr} = getelementptr inbounds [{len(expr.value) + 1} x i8], "
                                  f"[{len(expr.value) + 1} x i8]* {str_name}, i64 0, i64 0")
            return (ptr, "i8*")
        elif isinstance(expr, LogicalLiteral):
            val = "1" if expr.value else "0"
            return (val, "i1")
        elif isinstance(expr, Variable):
            if expr.name in self.var_alloc:
                alloc_type, ptr = self.var_alloc[expr.name]
                base_type = alloc_type if '[' not in alloc_type else alloc_type.split('[')[1].split(']')[0].split(' x ')[1]
                loaded = self._new_local()
                self.code_lines.append(f"  {loaded} = load {base_type}, {base_type}* {ptr}")
                return (loaded, base_type)
            return (f"@{expr.name}", "unknown")
        elif isinstance(expr, ArrayRef):
            array_name = expr.name
            indices = [self._emit_expression(idx)[0] for idx in expr.indices]
            if array_name in self.var_alloc:
                alloc_type, ptr = self.var_alloc[array_name]
                base_type = alloc_type if '[' not in alloc_type else alloc_type.split('[')[1].split(']')[0].split(' x ')[1]
                elem_ptr = self._new_local()
                adjusted_idx = self._new_local()
                self.code_lines.append(f"  {adjusted_idx} = sub i32 {indices[0]}, 1")
                self.code_lines.append(
                    f"  {elem_ptr} = getelementptr inbounds {alloc_type}, "
                    f"{alloc_type}* {ptr}, i64 0, i32 {adjusted_idx}"
                )
                loaded = self._new_local()
                self.code_lines.append(f"  {loaded} = load {base_type}, {base_type}* {elem_ptr}")
                return (loaded, base_type)
            return ("0", "unknown")
        elif isinstance(expr, BinaryOp):
            return self._emit_binop(expr)
        elif isinstance(expr, UnaryOp):
            return self._emit_unaryop(expr)
        elif isinstance(expr, FunctionCall):
            return self._emit_function_call(expr)
        return ("0", "unknown")
    def _emit_binop(self, expr: BinaryOp) -> Tuple[str, str]:
        left_val, left_type = self._emit_expression(expr.left)
        right_val, right_type = self._emit_expression(expr.right)
        result = self._new_local()
        if expr.op == "+":
            if left_type == "double" or right_type == "double":
                if left_type != "double":
                    left_val = self._convert_to_double(left_val)
                if right_type != "double":
                    right_val = self._convert_to_double(right_val)
                self.code_lines.append(f"  {result} = fadd double {left_val}, {right_val}")
                return (result, "double")
            else:
                self.code_lines.append(f"  {result} = add i32 {left_val}, {right_val}")
                return (result, "i32")
        elif expr.op == "-":
            if left_type == "double" or right_type == "double":
                if left_type != "double":
                    left_val = self._convert_to_double(left_val)
                if right_type != "double":
                    right_val = self._convert_to_double(right_val)
                self.code_lines.append(f"  {result} = fsub double {left_val}, {right_val}")
                return (result, "double")
            else:
                self.code_lines.append(f"  {result} = sub i32 {left_val}, {right_val}")
                return (result, "i32")
        elif expr.op == "*":
            if left_type == "double" or right_type == "double":
                if left_type != "double":
                    left_val = self._convert_to_double(left_val)
                if right_type != "double":
                    right_val = self._convert_to_double(right_val)
                self.code_lines.append(f"  {result} = fmul double {left_val}, {right_val}")
                return (result, "double")
            else:
                self.code_lines.append(f"  {result} = mul i32 {left_val}, {right_val}")
                return (result, "i32")
        elif expr.op == "/":
            if left_type == "double" or right_type == "double":
                if left_type != "double":
                    left_val = self._convert_to_double(left_val)
                if right_type != "double":
                    right_val = self._convert_to_double(right_val)
                self.code_lines.append(f"  {result} = fdiv double {left_val}, {right_val}")
                return (result, "double")
            else:
                self.code_lines.append(f"  {result} = sdiv i32 {left_val}, {right_val}")
                return (result, "i32")
        elif expr.op == "**":
            left_double = self._convert_to_double(left_val) if left_type != "double" else left_val
            right_double = self._convert_to_double(right_val) if right_type != "double" else right_val
            self.code_lines.append(f"  {result} = call double @pow(double {left_double}, double {right_double})")
            return (result, "double")
        elif expr.op in {".EQ.", "=="}:
            self.code_lines.append(f"  {result} = icmp eq {left_type} {left_val}, {right_val}")
            return (result, "i1")
        elif expr.op in {".NE.", "/="}:
            self.code_lines.append(f"  {result} = icmp ne {left_type} {left_val}, {right_val}")
            return (result, "i1")
        elif expr.op in {".LT.", "<"}:
            if left_type == "double" or right_type == "double":
                if left_type != "double":
                    left_val = self._convert_to_double(left_val)
                if right_type != "double":
                    right_val = self._convert_to_double(right_val)
                self.code_lines.append(f"  {result} = fcmp olt double {left_val}, {right_val}")
            else:
                self.code_lines.append(f"  {result} = icmp slt {left_type} {left_val}, {right_val}")
            return (result, "i1")
        elif expr.op in {".LE.", "<="}:
            if left_type == "double" or right_type == "double":
                if left_type != "double":
                    left_val = self._convert_to_double(left_val)
                if right_type != "double":
                    right_val = self._convert_to_double(right_val)
                self.code_lines.append(f"  {result} = fcmp ole double {left_val}, {right_val}")
            else:
                self.code_lines.append(f"  {result} = icmp sle {left_type} {left_val}, {right_val}")
            return (result, "i1")
        elif expr.op in {".GT.", ">"}:
            if left_type == "double" or right_type == "double":
                if left_type != "double":
                    left_val = self._convert_to_double(left_val)
                if right_type != "double":
                    right_val = self._convert_to_double(right_val)
                self.code_lines.append(f"  {result} = fcmp ogt double {left_val}, {right_val}")
            else:
                self.code_lines.append(f"  {result} = icmp sgt {left_type} {left_val}, {right_val}")
            return (result, "i1")
        elif expr.op in {".GE.", ">="}:
            if left_type == "double" or right_type == "double":
                if left_type != "double":
                    left_val = self._convert_to_double(left_val)
                if right_type != "double":
                    right_val = self._convert_to_double(right_val)
                self.code_lines.append(f"  {result} = fcmp oge double {left_val}, {right_val}")
            else:
                self.code_lines.append(f"  {result} = icmp sge {left_type} {left_val}, {right_val}")
            return (result, "i1")
        elif expr.op in {".AND.", "&"}:
            if left_type == "i1" and right_type == "i1":
                self.code_lines.append(f"  {result} = and i1 {left_val}, {right_val}")
                return (result, "i1")
            else:
                left_bool = self._convert_to_bool(left_val, left_type)
                right_bool = self._convert_to_bool(right_val, right_type)
                self.code_lines.append(f"  {result} = and i1 {left_bool}, {right_bool}")
                return (result, "i1")
        elif expr.op in {".OR.", "|"}:
            if left_type == "i1" and right_type == "i1":
                self.code_lines.append(f"  {result} = or i1 {left_val}, {right_val}")
                return (result, "i1")
            else:
                left_bool = self._convert_to_bool(left_val, left_type)
                right_bool = self._convert_to_bool(right_val, right_type)
                self.code_lines.append(f"  {result} = or i1 {left_bool}, {right_bool}")
                return (result, "i1")
        elif expr.op in {".EQV.", ".EQV"}:
            if left_type == "i1" and right_type == "i1":
                self.code_lines.append(f"  {result} = icmp eq i1 {left_val}, {right_val}")
                return (result, "i1")
            else:
                left_bool = self._convert_to_bool(left_val, left_type)
                right_bool = self._convert_to_bool(right_val, right_type)
                self.code_lines.append(f"  {result} = icmp eq i1 {left_bool}, {right_bool}")
                return (result, "i1")
        elif expr.op in {".NEQV.", ".NEQV"}:
            if left_type == "i1" and right_type == "i1":
                self.code_lines.append(f"  {result} = icmp ne i1 {left_val}, {right_val}")
                return (result, "i1")
            else:
                left_bool = self._convert_to_bool(left_val, left_type)
                right_bool = self._convert_to_bool(right_val, right_type)
                self.code_lines.append(f"  {result} = icmp ne i1 {left_bool}, {right_bool}")
                return (result, "i1")
        return (result, "unknown")
    def _emit_unaryop(self, expr: UnaryOp) -> Tuple[str, str]:
        val, val_type = self._emit_expression(expr.operand)
        result = self._new_local()
        if expr.op == "-":
            if val_type == "double":
                self.code_lines.append(f"  {result} = fneg double {val}")
            else:
                self.code_lines.append(f"  {result} = sub i32 0, {val}")
            return (result, val_type)
        elif expr.op == "+":
            return (val, val_type)
        elif expr.op in {".NOT.", "~", "NOT"}:
            if val_type == "i1":
                self.code_lines.append(f"  {result} = xor i1 {val}, 1")
                return (result, "i1")
            else:
                bool_val = self._convert_to_bool(val, val_type)
                self.code_lines.append(f"  {result} = xor i1 {bool_val}, 1")
                return (result, "i1")
        return (val, val_type)
    def _emit_function_call(self, expr: FunctionCall) -> Tuple[str, str]:
        args = [self._emit_expression(arg)[0] for arg in expr.args]
        args_str = ", ".join(args)
        result = self._new_local()
        func_name = expr.name.upper()
        if func_name in {"SIN", "COS", "TAN", "ASIN", "ACOS", "ATAN", "EXP", "LOG", "LOG10", "SQRT"}:
            self.code_lines.append(f"  {result} = call double @{func_name.lower()}(double {args[0]})")
            return (result, "double")
        elif func_name == "ABS":
            if len(args) > 0:
                arg_type = self._emit_expression(expr.args[0])[1]
                if arg_type == "double":
                    self.code_lines.append(f"  {result} = call double @fabs(double {args[0]})")
                    return (result, "double")
                else:
                    self.code_lines.append(f"  {result} = call i32 @abs(i32 {args[0]})")
                    return (result, "i32")
        elif func_name == "INT":
            self.code_lines.append(f"  {result} = fptosi double {args[0]} to i32")
            return (result, "i32")
        elif func_name == "FLOAT" or func_name == "REAL":
            self.code_lines.append(f"  {result} = sitofp i32 {args[0]} to double")
            return (result, "double")
        else:
            self.code_lines.append(f"  {result} = call double @{func_name.lower()}({args_str})")
            return (result, "double")
    def _convert_to_double(self, val: str) -> str:
        result = self._new_local()
        self.code_lines.append(f"  {result} = sitofp i32 {val} to double")
        return result
    def _convert_to_bool(self, val: str, val_type: str) -> str:
        if val_type == "i1":
            return val
        result = self._new_local()
        self.code_lines.append(f"  {result} = icmp ne {val_type} {val}, 0")
        return result
    def _add_string(self, string_val: str) -> str:
        if string_val not in self.strings:
            str_name = f"@.str.{self.string_counter}"
            self.string_counter += 1
            self.strings[string_val] = str_name
        return self.strings[string_val]
    def _new_local(self) -> str:
        self.local_counter += 1
        return f"%{self.local_counter}"
    def _new_block_id(self) -> int:
        self.local_counter += 1
        return self.local_counter