from typing import Dict, List, Optional, Tuple
import platform
import sys
from src.core import (
    Program, Declaration, Statement, Assignment, DoLoop, IfStatement,
    PrintStatement, ReadStatement, WriteStatement, CallStatement,
    BinaryOp, UnaryOp, Variable, IntegerLiteral, RealLiteral,
    StringLiteral, LogicalLiteral, FunctionCall, Expression,
    ReturnStatement, StopStatement, DoWhile, LabeledDoLoop, LabeledDoWhile,
    SimpleIfStatement, ArrayRef, DimensionStatement, GotoStatement, ContinueStatement,
    Subroutine, FunctionDef, ImplicitNone, ImplicitStatement, ComplexLiteral
)


def _get_default_target_triple() -> str:
    system = platform.system().lower()
    machine = platform.machine().lower()
    
    if system == "windows":
        if "64" in machine or machine == "amd64" or machine == "x86_64":
            return "x86_64-w64-windows-gnu"
        else:
            return "i686-w64-windows-gnu"
    elif system == "linux":
        if "64" in machine or machine == "amd64" or machine == "x86_64":
            return "x86_64-unknown-linux-gnu"
        else:
            return "i686-unknown-linux-gnu"
    elif system == "darwin":
        if "64" in machine or machine == "amd64" or machine == "x86_64":
            return "x86_64-apple-darwin"
        else:
            return "i686-apple-darwin"
    else:
        return "x86_64-unknown-linux-gnu"


class LLVMGenerator:
    def __init__(self, target_triple: Optional[str] = None):
        self.code_lines: List[str] = []
        self.var_alloc: Dict[str, Tuple[str, str]] = {}
        self.local_counter = 0
        self.string_counter = 0
        self.strings: Dict[str, str] = {}
        self.target_triple = target_triple if target_triple is not None else _get_default_target_triple()
        self.type_map = {
            'INTEGER': 'i32',
            'REAL': 'double',
            'LOGICAL': 'i1',
            'CHARACTER': 'i8*',
            'COMPLEX': '{double, double}',
        }
        self.implicit_rules = {}
        self.implicit_none = False
        # Для отслеживания phi-функций
        self.var_ssa_versions: Dict[str, str] = {}  # Имя переменной -> текущая SSA версия
        self.phi_tracking: List[Dict[str, str]] = []  # Стек для отслеживания переменных в ветках

    def generate(self, ast: Program) -> str:
        self.code_lines = []
        self.strings = {}
        self.string_counter = 0
        self.block_counter = 0
        self.current_function = "main"
        self.labels = {}
        self.statement_functions = {}
        self.char_lengths = {}
        self.var_ssa_versions = {}
        self.phi_tracking = []
        
        for stmt_func in ast.statement_functions:
            if hasattr(stmt_func, 'name') and stmt_func.name:
                self.statement_functions[stmt_func.name.upper()] = stmt_func
        
        self._collect_strings(ast)
        self._emit_header()
        self._emit_external_declarations()
        strings_line_start = len(self.code_lines)
        self.code_lines.append("; === Глобальные строки (будут заполнены) ===")
        self.local_counter = 0
        self.var_alloc = {}
        self._emit_program(ast)
        
        for sub in ast.subroutines:
            self._emit_subroutine(sub)
        
        for func in ast.functions:
            self._emit_function(func)
        
        if self.strings:
            strings_lines = ["; === Глобальные строки ==="]
            for string_val, (string_name, str_size) in self.strings.items():
                escaped = string_val
                escaped = escaped.replace('\\n', '\\0A')
                escaped = escaped.replace('\\t', '\\09')
                escaped = escaped.replace('\\r', '\\0D')
                escaped = escaped.replace('"', '\\22')
                strings_lines.append(
                    f'{string_name} = private constant [{str_size} x i8] c"{escaped}\\00"')
            strings_lines.append("")
            self.code_lines[strings_line_start:strings_line_start +
                            1] = strings_lines
        
        if ast.subroutines or ast.functions:
            self.code_lines.append("; === Атрибуты функций для совместимости с Си ===")
            self.code_lines.append("attributes #0 = { nounwind }")
            self.code_lines.append("")
        
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
        self.code_lines.append("declare i8* @strcpy(i8*, i8*)")
        self.code_lines.append("declare i8* @strcat(i8*, i8*)")
        self.code_lines.append("declare i64 @strlen(i8*)")
        self.code_lines.append("declare i8* @malloc(i64)")
        self.code_lines.append("declare double @sqrt(double)")
        self.code_lines.append("declare double @sin(double)")
        self.code_lines.append("declare double @cos(double)")
        self.code_lines.append("declare double @tan(double)")
        self.code_lines.append("declare double @asin(double)")
        self.code_lines.append("declare double @acos(double)")
        self.code_lines.append("declare double @atan(double)")
        self.code_lines.append("declare double @exp(double)")
        self.code_lines.append("declare double @log(double)")
        self.code_lines.append("declare double @log10(double)")
        self.code_lines.append("declare double @pow(double, double)")
        self.code_lines.append("declare i32 @abs(i32)")
        self.code_lines.append("declare double @fabs(double)")
        self.code_lines.append("")

    def _get_implicit_type(self, name: str) -> Tuple[str, Optional[int]]:
        if not name:
            return ("UNKNOWN", None)
        first_char = name[0].upper()
        if first_char in self.implicit_rules:
            type_name = self.implicit_rules[first_char]
            return (type_name, None)
        if not self.implicit_none:
            if 'I' <= first_char <= 'N':
                return ("INTEGER", None)
            else:
                return ("REAL", None)
        return ("UNKNOWN", None)

    def _emit_program(self, ast: Program):
        for decl in ast.declarations:
            if isinstance(decl, ImplicitNone):
                self.implicit_none = True
            elif isinstance(decl, ImplicitStatement):
                if hasattr(decl, 'rules'):
                    for rule in decl.rules:
                        if hasattr(rule, 'get_letters') and hasattr(rule, 'type_name'):
                            letters = rule.get_letters()
                            type_name = rule.type_name.upper()
                            for letter in letters:
                                self.implicit_rules[letter.upper()] = type_name
        if ast.statements:
            self.code_lines.append("; === Основная программа ===")
            self.code_lines.append("define i32 @main() {")
            self.code_lines.append("entry:")
            self.local_counter = 0
            self.var_alloc = {}
            self.var_ssa_versions = {}
            self.phi_tracking = []
            self._emit_declarations(ast.declarations)
            for stmt in ast.statements:
                self._emit_statement(stmt)
            self.code_lines.append("  ret i32 0")
            self.code_lines.append("}")
            self.code_lines.append("")
        elif ast.declarations:
            self.code_lines.append("; === Основная программа (только объявления) ===")
            self.code_lines.append("define i32 @main() {")
            self.code_lines.append("entry:")
            self.local_counter = 0
            self.var_alloc = {}
            self.var_ssa_versions = {}
            self.phi_tracking = []
            self._emit_declarations(ast.declarations)
            self.code_lines.append("  ret i32 0")
            self.code_lines.append("}")
            self.code_lines.append("")

    def _emit_declarations(self, declarations: List[Declaration]):
        for decl in declarations:
            if isinstance(decl, DimensionStatement):
                for name, dims in decl.names:
                    if name in self.var_alloc:
                        continue
                    llvm_type = 'double'
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
                continue
            if hasattr(decl, 'names') and hasattr(decl, 'type'):
                llvm_type = self.type_map.get(decl.type, 'i32')
                for name, dims in decl.names:
                    if name in self.var_alloc:
                        continue
                    if decl.type.startswith('CHARACTER') and decl.type_size:
                        char_size = int(decl.type_size)
                        alloc_type = f"[{char_size} x i8]"
                        local_name = f"%{name}"
                        self.char_lengths[name] = char_size
                        self.code_lines.append(
                            f"  {local_name} = alloca {alloc_type}")
                        self.var_alloc[name] = (alloc_type, local_name)
                        continue
                    elif dims:
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
                    self.code_lines.append(
                        f"  {local_name} = alloca {alloc_type}")
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
        elif isinstance(stmt, ReturnStatement):
            self._emit_return_statement(stmt)
        elif isinstance(stmt, GotoStatement):
            self._emit_goto_statement(stmt)
        elif isinstance(stmt, ContinueStatement):
            self._emit_continue_statement(stmt)
        elif isinstance(stmt, StopStatement):
            self.code_lines.append("  ret i32 0")

    def _emit_assignment(self, stmt: Assignment):
        rhs_val, rhs_type = self._emit_expression(stmt.value)
        if stmt.indices:
            indices_vals = [self._emit_expression(
                idx)[0] for idx in stmt.indices]
            if stmt.target in self.var_alloc:
                alloc_type, ptr_name = self.var_alloc[stmt.target]
                base_type = self.type_map.get('REAL', 'double')
                if '[' in alloc_type:
                    idx = indices_vals[0]
                    adjusted_idx = self._new_local()
                    self.code_lines.append(
                        f"  {adjusted_idx} = sub i32 {idx}, 1")
                    elem_ptr = self._new_local()
                    self.code_lines.append(
                        f"  {elem_ptr} = getelementptr inbounds {alloc_type}, "
                        f"{alloc_type}* {ptr_name}, i64 0, i32 {adjusted_idx}"
                    )
                    self.code_lines.append(
                        f"  store {rhs_type} {rhs_val}, {rhs_type}* {elem_ptr}")
        else:
            if stmt.target in self.var_alloc:
                alloc_type, ptr_name = self.var_alloc[stmt.target]
                if '[' in alloc_type:
                    base_type = alloc_type.split('[')[1].split(']')[
                        0].split(' x ')[1]
                else:
                    base_type = alloc_type
                if rhs_type == "i8*" and base_type == "i8" and '[' in alloc_type:
                    str_ptr = self._new_local()
                    self.code_lines.append(
                        f"  {str_ptr} = getelementptr inbounds {alloc_type}, "
                        f"{alloc_type}* {ptr_name}, i64 0, i64 0")
                    self.code_lines.append(
                        f"  call i8* @strcpy(i8* {str_ptr}, i8* {rhs_val})")
                elif base_type == "{double, double}" and rhs_type == "{double, double}":
                    self.code_lines.append(
                        f"  store {{double, double}} {rhs_val}, {{double, double}}* {ptr_name}")
                else:
                    # Если мы в ветке условного оператора, отслеживаем для phi
                    if self.phi_tracking:
                        # Сохраняем значение
                        self.code_lines.append(
                            f"  store {base_type} {rhs_val}, {base_type}* {ptr_name}")
                        # Загружаем обратно для phi
                        new_ssa = self._new_local()
                        self.code_lines.append(
                            f"  {new_ssa} = load {base_type}, {base_type}* {ptr_name}")
                        # Сохраняем SSA версию для phi
                        self.phi_tracking[-1][stmt.target] = new_ssa
                    else:
                        self.code_lines.append(
                            f"  store {base_type} {rhs_val}, {base_type}* {ptr_name}")

    def _emit_do_loop(self, stmt: DoLoop):
        loop_id = self._new_block_id()
        loop_label = f"loop_{loop_id}"
        loop_body_label = f"loop_body_{loop_id}"
        loop_end_label = f"loop_end_{loop_id}"
        start_val, _ = self._emit_expression(stmt.start)
        if stmt.var in self.var_alloc:
            alloc_type, ptr = self.var_alloc[stmt.var]
            base_type = alloc_type if '[' not in alloc_type else alloc_type.split(
                '[')[1].split(']')[0].split(' x ')[1]
            self.code_lines.append(
                f"  store {base_type} {start_val}, {base_type}* {ptr}")
        self.code_lines.append(f"  br label %{loop_label}")
        self.code_lines.append(f"{loop_label}:")
        if stmt.var in self.var_alloc:
            alloc_type, ptr = self.var_alloc[stmt.var]
            base_type = alloc_type if '[' not in alloc_type else alloc_type.split(
                '[')[1].split(']')[0].split(' x ')[1]
            loop_var = self._new_local()
            self.code_lines.append(
                f"  {loop_var} = load {base_type}, {base_type}* {ptr}")
            end_val, _ = self._emit_expression(stmt.end)
            cond = self._new_local()
            self.code_lines.append(
                f"  {cond} = icmp sle {base_type} {loop_var}, {end_val}")
            self.code_lines.append(
                f"  br i1 {cond}, label %{loop_body_label}, label %{loop_end_label}")
        self.code_lines.append(f"{loop_body_label}:")
        for body_stmt in stmt.body:
            self._emit_statement(body_stmt)
        step = 1
        if stmt.step:
            step_val, _ = self._emit_expression(stmt.step)
            step = step_val
        else:
            step_val = self._new_local()
            self.code_lines.append(
                f"  {step_val} = add {base_type} {loop_var}, 1")
            step = step_val
        if stmt.var in self.var_alloc:
            alloc_type, ptr = self.var_alloc[stmt.var]
            base_type = alloc_type if '[' not in alloc_type else alloc_type.split(
                '[')[1].split(']')[0].split(' x ')[1]
            loop_var = self._new_local()
            self.code_lines.append(
                f"  {loop_var} = load {base_type}, {base_type}* {ptr}")
            next_val = self._new_local()
            self.code_lines.append(
                f"  {next_val} = add {base_type} {loop_var}, {step}")
            self.code_lines.append(
                f"  store {base_type} {next_val}, {base_type}* {ptr}")
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
            self.code_lines.append(
                f"  {cond_bool} = icmp ne {cond_type} {cond_val}, 0")
            cond_val = cond_bool
        self.code_lines.append(
            f"  br i1 {cond_val}, label %{loop_body_label}, label %{loop_end_label}")
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
            self.code_lines.append(
                f"  {cond_bool} = icmp ne {cond_type} {cond_val}, 0")
            cond_val = cond_bool
        
        before_phi_vars = {}
        for var_name in self.var_alloc:
            if var_name in self.var_ssa_versions:
                before_phi_vars[var_name] = self.var_ssa_versions[var_name]
            else:
                alloc_type, ptr_name = self.var_alloc[var_name]
                base_type = alloc_type if '[' not in alloc_type else alloc_type.split('[')[1].split(']')[0].split(' x ')[1]
                before_val = self._new_local()
                self.code_lines.append(
                    f"  {before_val} = load {base_type}, {base_type}* {ptr_name}")
                before_phi_vars[var_name] = before_val
        
        self.phi_tracking.append({})
        then_phi_vars = {}
        elif_phi_vars_list = []
        
        if stmt.elif_parts:
            elif_labels = []
            for i, (elif_cond, _) in enumerate(stmt.elif_parts):
                elif_labels.append(f"if_elif_{if_id}_{i}")
            else_label = f"if_else_{if_id}" if stmt.else_body else endif_label
            next_label = elif_labels[0] if elif_labels else (
                else_label if stmt.else_body else endif_label)
            self.code_lines.append(
                f"  br i1 {cond_val}, label %{then_label}, label %{next_label}")
            self.code_lines.append(f"{then_label}:")
            for s in stmt.then_body:
                self._emit_statement(s)
            then_phi_vars = self.phi_tracking[-1].copy()
            self.code_lines.append(f"  br label %{endif_label}")
            
            for i, (elif_cond, elif_body) in enumerate(stmt.elif_parts):
                self.code_lines.append(f"{elif_labels[i]}:")
                elif_cond_val, elif_cond_type = self._emit_expression(
                    elif_cond)
                if elif_cond_type != "i1":
                    elif_cond_bool = self._new_local()
                    self.code_lines.append(
                        f"  {elif_cond_bool} = icmp ne {elif_cond_type} {elif_cond_val}, 0")
                    elif_cond_val = elif_cond_bool
                next_elif_label = elif_labels[i + 1] if i + \
                    1 < len(elif_labels) else else_label
                self.code_lines.append(
                    f"  br i1 {elif_cond_val}, label %if_elif_then_{if_id}_{i}, label %{next_elif_label}")
                self.code_lines.append(f"if_elif_then_{if_id}_{i}:")
                self.phi_tracking[-1] = {}
                for s in elif_body:
                    self._emit_statement(s)
                elif_phi_vars_list.append(self.phi_tracking[-1].copy())
                self.code_lines.append(f"  br label %{endif_label}")
            
            else_phi_vars = {}
            if stmt.else_body:
                self.code_lines.append(f"{else_label}:")
                self.phi_tracking[-1] = {}
                for s in stmt.else_body:
                    self._emit_statement(s)
                else_phi_vars = self.phi_tracking[-1].copy()
                self.code_lines.append(f"  br label %{endif_label}")
        else:
            else_label = f"if_else_{if_id}" if stmt.else_body else endif_label
            self.code_lines.append(
                f"  br i1 {cond_val}, label %{then_label}, label %{else_label}")
            self.code_lines.append(f"{then_label}:")
            for s in stmt.then_body:
                self._emit_statement(s)
            then_phi_vars = self.phi_tracking[-1].copy()
            self.code_lines.append(f"  br label %{endif_label}")
            
            else_phi_vars = {}
            if stmt.else_body:
                self.code_lines.append(f"{else_label}:")
                self.phi_tracking[-1] = {}
                for s in stmt.else_body:
                    self._emit_statement(s)
                else_phi_vars = self.phi_tracking[-1].copy()
                self.code_lines.append(f"  br label %{endif_label}")
        
        phi_instructions = []
        all_modified_vars = set(then_phi_vars.keys())
        for elif_vars in elif_phi_vars_list:
            all_modified_vars.update(elif_vars.keys())
        all_modified_vars.update(else_phi_vars.keys())
        
        for var_name in all_modified_vars:
            if var_name not in self.var_alloc:
                continue
            
            alloc_type, ptr_name = self.var_alloc[var_name]
            base_type = alloc_type if '[' not in alloc_type else alloc_type.split('[')[1].split(']')[0].split(' x ')[1]
            
            phi_args = []
            
            if var_name in then_phi_vars:
                phi_args.append(f"[{then_phi_vars[var_name]}, %{then_label}]")
            else:
                phi_args.append(f"[{before_phi_vars[var_name]}, %{then_label}]")
            
            for i, elif_vars in enumerate(elif_phi_vars_list):
                elif_label = f"if_elif_then_{if_id}_{i}"
                if var_name in elif_vars:
                    phi_args.append(f"[{elif_vars[var_name]}, %{elif_label}]")
                else:
                    phi_args.append(f"[{before_phi_vars[var_name]}, %{elif_label}]")
            
            if stmt.else_body:
                if var_name in else_phi_vars:
                    phi_args.append(f"[{else_phi_vars[var_name]}, %{else_label}]")
                else:
                    phi_args.append(f"[{before_phi_vars[var_name]}, %{else_label}]")
            
            if len(phi_args) > 1:
                phi_result = self._new_local()
                phi_args_str = ", ".join(phi_args)
                phi_instructions.append(
                    f"  {phi_result} = phi {base_type} {phi_args_str}")
                self.var_ssa_versions[var_name] = phi_result
        
        self.code_lines.append(f"{endif_label}:")
        for phi_line in phi_instructions:
            self.code_lines.append(phi_line)
        
        for var_name in all_modified_vars:
            if var_name not in self.var_alloc:
                continue
            if var_name in self.var_ssa_versions:
                alloc_type, ptr_name = self.var_alloc[var_name]
                base_type = alloc_type if '[' not in alloc_type else alloc_type.split('[')[1].split(']')[0].split(' x ')[1]
                phi_result = self.var_ssa_versions[var_name]
                self.code_lines.append(
                    f"  store {base_type} {phi_result}, {base_type}* {ptr_name}")
        
        if self.phi_tracking:
            self.phi_tracking.pop()

    def _emit_simple_if_statement(self, stmt: SimpleIfStatement):
        if_id = self._new_block_id()
        then_label = f"if_then_{if_id}"
        endif_label = f"if_end_{if_id}"
        cond_val, cond_type = self._emit_expression(stmt.condition)
        if cond_type != "i1":
            cond_bool = self._new_local()
            self.code_lines.append(
                f"  {cond_bool} = icmp ne {cond_type} {cond_val}, 0")
            cond_val = cond_bool
        
        before_phi_vars = {}
        for var_name in self.var_alloc:
            if var_name in self.var_ssa_versions:
                before_phi_vars[var_name] = self.var_ssa_versions[var_name]
            else:
                alloc_type, ptr_name = self.var_alloc[var_name]
                base_type = alloc_type if '[' not in alloc_type else alloc_type.split('[')[1].split(']')[0].split(' x ')[1]
                before_val = self._new_local()
                self.code_lines.append(
                    f"  {before_val} = load {base_type}, {base_type}* {ptr_name}")
                before_phi_vars[var_name] = before_val
        
        self.phi_tracking.append({})
        
        self.code_lines.append(
            f"  br i1 {cond_val}, label %{then_label}, label %{endif_label}")
        self.code_lines.append(f"{then_label}:")
        self._emit_statement(stmt.statement)
        
        then_phi_vars = self.phi_tracking[-1].copy()
        self.code_lines.append(f"  br label %{endif_label}")
        
        phi_instructions = []
        for var_name in then_phi_vars:
            if var_name not in self.var_alloc:
                continue
            
            alloc_type, ptr_name = self.var_alloc[var_name]
            base_type = alloc_type if '[' not in alloc_type else alloc_type.split('[')[1].split(']')[0].split(' x ')[1]
            
            before_val = before_phi_vars[var_name]
            phi_result = self._new_local()
            prev_label = "entry"
            phi_args_str = f"[{before_val}, %{prev_label}], [{then_phi_vars[var_name]}, %{then_label}]"
            phi_instructions.append(
                f"  {phi_result} = phi {base_type} {phi_args_str}")
            self.var_ssa_versions[var_name] = phi_result
        
        self.code_lines.append(f"{endif_label}:")
        for phi_line in phi_instructions:
            self.code_lines.append(phi_line)
        
        for var_name in then_phi_vars:
            if var_name in self.var_ssa_versions:
                alloc_type, ptr_name = self.var_alloc[var_name]
                base_type = alloc_type if '[' not in alloc_type else alloc_type.split('[')[1].split(']')[0].split(' x ')[1]
                phi_result = self.var_ssa_versions[var_name]
                self.code_lines.append(
                    f"  store {base_type} {phi_result}, {base_type}* {ptr_name}")
        
        if self.phi_tracking:
            self.phi_tracking.pop()

    def _emit_print_statement(self, stmt: PrintStatement):
        if not stmt.items:
            return
        for item in stmt.items:
            val, val_type = self._emit_expression(item)
            if val_type == "i8*":
                self.code_lines.append(f"  call i32 @puts(i8* {val})")
            elif val_type == "i1":
                conv_val = self._new_local()
                self.code_lines.append(f"  {conv_val} = zext i1 {val} to i32")
                fmt_str = "%d\\n"
                str_name, str_size = self._add_string(fmt_str)
                str_ptr = self._new_local()
                self.code_lines.append(
                    f"  {str_ptr} = getelementptr inbounds [{str_size} x i8], "
                    f"[{str_size} x i8]* {str_name}, i64 0, i64 0"
                )
                self.code_lines.append(
                    f"  call i32 (i8*, ...) @printf(i8* {str_ptr}, i32 {conv_val})")
            elif val_type == "double":
                fmt_str = "%g\\n"
                str_name, str_size = self._add_string(fmt_str)
                str_ptr = self._new_local()
                self.code_lines.append(
                    f"  {str_ptr} = getelementptr inbounds [{str_size} x i8], "
                    f"[{str_size} x i8]* {str_name}, i64 0, i64 0"
                )
                self.code_lines.append(
                    f"  call i32 (i8*, ...) @printf(i8* {str_ptr}, double {val})")
            elif val_type == "{double, double}":
                real_ptr = self._new_local()
                self.code_lines.append(f"  {real_ptr} = extractvalue {{double, double}} {val}, 0")
                imag_ptr = self._new_local()
                self.code_lines.append(f"  {imag_ptr} = extractvalue {{double, double}} {val}, 1")
                fmt_str = "(%g, %g)\\n"
                str_name, str_size = self._add_string(fmt_str)
                str_ptr = self._new_local()
                self.code_lines.append(
                    f"  {str_ptr} = getelementptr inbounds [{str_size} x i8], "
                    f"[{str_size} x i8]* {str_name}, i64 0, i64 0"
                )
                self.code_lines.append(
                    f"  call i32 (i8*, ...) @printf(i8* {str_ptr}, double {real_ptr}, double {imag_ptr})")
            else:
                fmt_str = "%d\\n"
                str_name, str_size = self._add_string(fmt_str)
                str_ptr = self._new_local()
                self.code_lines.append(
                    f"  {str_ptr} = getelementptr inbounds [{str_size} x i8], "
                    f"[{str_size} x i8]* {str_name}, i64 0, i64 0"
                )
                self.code_lines.append(
                    f"  call i32 (i8*, ...) @printf(i8* {str_ptr}, i32 {val})")

    def _emit_read_statement(self, stmt: ReadStatement):
        for item in stmt.items:
            if item in self.var_alloc:
                alloc_type, ptr = self.var_alloc[item]
                base_type = alloc_type if '[' not in alloc_type else alloc_type.split(
                    '[')[1].split(']')[0].split(' x ')[1]
                format_str = "%lf" if base_type == 'double' else "%d"
                str_name, str_size = self._add_string(format_str)
                str_ptr = self._new_local()
                self.code_lines.append(f"  {str_ptr} = getelementptr inbounds [{str_size} x i8], "
                                       f"[{str_size} x i8]* {str_name}, i64 0, i64 0")
                self.code_lines.append(
                    f"  call i32 (i8*, ...) @scanf(i8* {str_ptr}, {base_type}* {ptr})")

    def _emit_write_statement(self, stmt: WriteStatement):
        for item in stmt.items:
            val, val_type = self._emit_expression(item)
            if val_type == "i8*":
                self.code_lines.append(f"  call i32 @puts(i8* {val})")
            elif val_type == "i1":
                conv_val = self._new_local()
                self.code_lines.append(f"  {conv_val} = zext i1 {val} to i32")
                fmt_str = "%d\\n"
                str_name, str_size = self._add_string(fmt_str)
                str_ptr = self._new_local()
                self.code_lines.append(
                    f"  {str_ptr} = getelementptr inbounds [{str_size} x i8], "
                    f"[{str_size} x i8]* {str_name}, i64 0, i64 0"
                )
                self.code_lines.append(
                    f"  call i32 (i8*, ...) @printf(i8* {str_ptr}, i32 {conv_val})")
            elif val_type == "double":
                fmt_str = "%g\\n"
                str_name, str_size = self._add_string(fmt_str)
                str_ptr = self._new_local()
                self.code_lines.append(
                    f"  {str_ptr} = getelementptr inbounds [{str_size} x i8], "
                    f"[{str_size} x i8]* {str_name}, i64 0, i64 0"
                )
                self.code_lines.append(
                    f"  call i32 (i8*, ...) @printf(i8* {str_ptr}, double {val})")
            else:
                fmt_str = "%d\\n"
                str_name, str_size = self._add_string(fmt_str)
                str_ptr = self._new_local()
                self.code_lines.append(
                    f"  {str_ptr} = getelementptr inbounds [{str_size} x i8], "
                    f"[{str_size} x i8]* {str_name}, i64 0, i64 0"
                )
                self.code_lines.append(
                    f"  call i32 (i8*, ...) @printf(i8* {str_ptr}, i32 {val})")

    def _emit_call_statement(self, stmt: CallStatement):
        args = []
        arg_types = []
        for arg in stmt.args:
            if isinstance(arg, Variable):
                if arg.name in self.var_alloc:
                    alloc_type, ptr_name = self.var_alloc[arg.name]
                    base_type = alloc_type if '[' not in alloc_type else alloc_type.split('[')[1].split(']')[0].split(' x ')[1]
                    args.append(f"{base_type}* {ptr_name}")
                    arg_types.append(f"{base_type}*")
                else:
                    val, val_type = self._emit_expression(arg)
                    args.append(f"{val_type} {val}")
                    arg_types.append(val_type)
            else:
                val, val_type = self._emit_expression(arg)
                args.append(f"{val_type} {val}")
                arg_types.append(val_type)
        args_str = ", ".join(args)
        self.code_lines.append(f"  call void @{stmt.name.upper()}({args_str})")

    def _emit_expression(self, expr: Expression) -> Tuple[str, str]:
        if isinstance(expr, IntegerLiteral):
            return (str(expr.value), "i32")
        elif isinstance(expr, RealLiteral):
            return (str(expr.value), "double")
        elif isinstance(expr, ComplexLiteral):
            complex_val1 = self._new_local()
            self.code_lines.append(f"  {complex_val1} = insertvalue {{double, double}} undef, double {expr.real_part}, 0")
            complex_val2 = self._new_local()
            self.code_lines.append(f"  {complex_val2} = insertvalue {{double, double}} {complex_val1}, double {expr.imag_part}, 1")
            return (complex_val2, "{double, double}")
        elif isinstance(expr, StringLiteral):
            str_name, str_size = self._add_string(expr.value)
            ptr = self._new_local()
            self.code_lines.append(f"  {ptr} = getelementptr inbounds [{str_size} x i8], "
                                   f"[{str_size} x i8]* {str_name}, i64 0, i64 0")
            return (ptr, "i8*")
        elif isinstance(expr, LogicalLiteral):
            val = "1" if expr.value else "0"
            return (val, "i1")
        elif isinstance(expr, Variable):
            if expr.name in self.var_alloc:
                alloc_type, ptr = self.var_alloc[expr.name]
                if '[' in alloc_type and alloc_type.split('[')[1].split(']')[0].split(' x ')[1] == "i8":
                    str_ptr = self._new_local()
                    self.code_lines.append(
                        f"  {str_ptr} = getelementptr inbounds {alloc_type}, "
                        f"{alloc_type}* {ptr}, i64 0, i64 0")
                    return (str_ptr, "i8*")
                base_type = alloc_type if '[' not in alloc_type else alloc_type.split(
                    '[')[1].split(']')[0].split(' x ')[1]
                loaded = self._new_local()
                self.code_lines.append(
                    f"  {loaded} = load {base_type}, {base_type}* {ptr}")
                return (loaded, base_type)
            else:
                var_name = expr.name.upper()
                implicit_type_name, _ = self._get_implicit_type(var_name)
                if implicit_type_name and implicit_type_name != "UNKNOWN":
                    alloc_type = self.type_map.get(implicit_type_name, 'i32')
                    ptr = self._new_local()
                    self.code_lines.append(f"  {ptr} = alloca {alloc_type}")
                    self.var_alloc[var_name] = (alloc_type, ptr)
                    loaded = self._new_local()
                    self.code_lines.append(f"  {loaded} = load {alloc_type}, {alloc_type}* {ptr}")
                    return (loaded, alloc_type)
            return (f"@{expr.name}", "unknown")
        elif isinstance(expr, ArrayRef):
            array_name = expr.name
            indices = [self._emit_expression(idx)[0] for idx in expr.indices]
            if array_name in self.var_alloc:
                alloc_type, ptr = self.var_alloc[array_name]
                base_type = alloc_type if '[' not in alloc_type else alloc_type.split(
                    '[')[1].split(']')[0].split(' x ')[1]
                adjusted_idx = self._new_local()
                self.code_lines.append(
                    f"  {adjusted_idx} = sub i32 {indices[0]}, 1")
                elem_ptr = self._new_local()
                self.code_lines.append(
                    f"  {elem_ptr} = getelementptr inbounds {alloc_type}, "
                    f"{alloc_type}* {ptr}, i64 0, i32 {adjusted_idx}"
                )
                loaded = self._new_local()
                self.code_lines.append(
                    f"  {loaded} = load {base_type}, {base_type}* {elem_ptr}")
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
                self.code_lines.append(
                    f"  {result} = fadd double {left_val}, {right_val}")
                return (result, "double")
            else:
                self.code_lines.append(
                    f"  {result} = add i32 {left_val}, {right_val}")
                return (result, "i32")
        elif expr.op == "-":
            if left_type == "double" or right_type == "double":
                if left_type != "double":
                    left_val = self._convert_to_double(left_val)
                if right_type != "double":
                    right_val = self._convert_to_double(right_val)
                self.code_lines.append(
                    f"  {result} = fsub double {left_val}, {right_val}")
                return (result, "double")
            else:
                self.code_lines.append(
                    f"  {result} = sub i32 {left_val}, {right_val}")
                return (result, "i32")
        elif expr.op == "*":
            if left_type == "double" or right_type == "double":
                if left_type != "double":
                    left_val = self._convert_to_double(left_val)
                if right_type != "double":
                    right_val = self._convert_to_double(right_val)
                self.code_lines.append(
                    f"  {result} = fmul double {left_val}, {right_val}")
                return (result, "double")
            else:
                self.code_lines.append(
                    f"  {result} = mul i32 {left_val}, {right_val}")
                return (result, "i32")
        elif expr.op == "/":
            if left_type == "double" or right_type == "double":
                if left_type != "double":
                    left_val = self._convert_to_double(left_val)
                if right_type != "double":
                    right_val = self._convert_to_double(right_val)
                self.code_lines.append(
                    f"  {result} = fdiv double {left_val}, {right_val}")
                return (result, "double")
            else:
                self.code_lines.append(
                    f"  {result} = sdiv i32 {left_val}, {right_val}")
                return (result, "i32")
        elif expr.op == "**":
            left_double = self._convert_to_double(
                left_val) if left_type != "double" else left_val
            right_double = self._convert_to_double(
                right_val) if right_type != "double" else right_val
            self.code_lines.append(
                f"  {result} = call double @pow(double {left_double}, double {right_double})")
            return (result, "double")
        elif expr.op in {".EQ.", "=="}:
            self.code_lines.append(
                f"  {result} = icmp eq {left_type} {left_val}, {right_val}")
            return (result, "i1")
        elif expr.op in {".NE.", "/="}:
            self.code_lines.append(
                f"  {result} = icmp ne {left_type} {left_val}, {right_val}")
            return (result, "i1")
        elif expr.op in {".LT.", "<"}:
            if left_type == "double" or right_type == "double":
                if left_type != "double":
                    left_val = self._convert_to_double(left_val)
                if right_type != "double":
                    right_val = self._convert_to_double(right_val)
                self.code_lines.append(
                    f"  {result} = fcmp olt double {left_val}, {right_val}")
            else:
                self.code_lines.append(
                    f"  {result} = icmp slt {left_type} {left_val}, {right_val}")
            return (result, "i1")
        elif expr.op in {".LE.", "<="}:
            if left_type == "double" or right_type == "double":
                if left_type != "double":
                    left_val = self._convert_to_double(left_val)
                if right_type != "double":
                    right_val = self._convert_to_double(right_val)
                self.code_lines.append(
                    f"  {result} = fcmp ole double {left_val}, {right_val}")
            else:
                self.code_lines.append(
                    f"  {result} = icmp sle {left_type} {left_val}, {right_val}")
            return (result, "i1")
        elif expr.op in {".GT.", ">"}:
            if left_type == "double" or right_type == "double":
                if left_type != "double":
                    left_val = self._convert_to_double(left_val)
                if right_type != "double":
                    right_val = self._convert_to_double(right_val)
                self.code_lines.append(
                    f"  {result} = fcmp ogt double {left_val}, {right_val}")
            else:
                self.code_lines.append(
                    f"  {result} = icmp sgt {left_type} {left_val}, {right_val}")
            return (result, "i1")
        elif expr.op in {".GE.", ">="}:
            if left_type == "double" or right_type == "double":
                if left_type != "double":
                    left_val = self._convert_to_double(left_val)
                if right_type != "double":
                    right_val = self._convert_to_double(right_val)
                self.code_lines.append(
                    f"  {result} = fcmp oge double {left_val}, {right_val}")
            else:
                self.code_lines.append(
                    f"  {result} = icmp sge {left_type} {left_val}, {right_val}")
            return (result, "i1")
        elif expr.op in {".AND.", "&"}:
            if left_type == "i1" and right_type == "i1":
                self.code_lines.append(
                    f"  {result} = and i1 {left_val}, {right_val}")
                return (result, "i1")
            else:
                left_bool = self._convert_to_bool(left_val, left_type)
                right_bool = self._convert_to_bool(right_val, right_type)
                self.code_lines.append(
                    f"  {result} = and i1 {left_bool}, {right_bool}")
                return (result, "i1")
        elif expr.op in {".OR.", "|"}:
            if left_type == "i1" and right_type == "i1":
                self.code_lines.append(
                    f"  {result} = or i1 {left_val}, {right_val}")
                return (result, "i1")
            else:
                left_bool = self._convert_to_bool(left_val, left_type)
                right_bool = self._convert_to_bool(right_val, right_type)
                self.code_lines.append(
                    f"  {result} = or i1 {left_bool}, {right_bool}")
                return (result, "i1")
        elif expr.op == "//":
            if left_type == "i8*" and right_type == "i8*":
                left_len = self._new_local()
                right_len = self._new_local()
                total_len = self._new_local()
                new_str = self._new_local()
                self.code_lines.append(f"  {left_len} = call i64 @strlen(i8* {left_val})")
                self.code_lines.append(f"  {right_len} = call i64 @strlen(i8* {right_val})")
                self.code_lines.append(f"  {total_len} = add i64 {left_len}, {right_len}")
                alloc_size = self._new_local()
                self.code_lines.append(f"  {alloc_size} = add i64 {total_len}, 1")
                self.code_lines.append(f"  {new_str} = call i8* @malloc(i64 {alloc_size})")
                self.code_lines.append(f"  call i8* @strcpy(i8* {new_str}, i8* {left_val})")
                self.code_lines.append(f"  call i8* @strcat(i8* {new_str}, i8* {right_val})")
                return (new_str, "i8*")
            else:
                return ("0", "unknown")
        elif expr.op in {".EQV.", ".EQV"}:
            if left_type == "i1" and right_type == "i1":
                self.code_lines.append(
                    f"  {result} = icmp eq i1 {left_val}, {right_val}")
                return (result, "i1")
            else:
                left_bool = self._convert_to_bool(left_val, left_type)
                right_bool = self._convert_to_bool(right_val, right_type)
                self.code_lines.append(
                    f"  {result} = icmp eq i1 {left_bool}, {right_bool}")
                return (result, "i1")
        elif expr.op in {".NEQV.", ".NEQV"}:
            if left_type == "i1" and right_type == "i1":
                self.code_lines.append(
                    f"  {result} = icmp ne i1 {left_val}, {right_val}")
                return (result, "i1")
            else:
                left_bool = self._convert_to_bool(left_val, left_type)
                right_bool = self._convert_to_bool(right_val, right_type)
                self.code_lines.append(
                    f"  {result} = icmp ne i1 {left_bool}, {right_bool}")
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
        result = self._new_local()
        func_name = expr.name.upper()
        if func_name in {"SIN", "COS", "TAN", "ASIN", "ACOS", "ATAN", "EXP", "LOG", "LOG10", "SQRT"}:
            arg_val, arg_type = self._emit_expression(expr.args[0])
            arg_double = self._convert_to_double(arg_val) if arg_type != "double" else arg_val
            self.code_lines.append(
                f"  {result} = call double @{func_name.lower()}(double {arg_double})")
            return (result, "double")
        elif func_name == "ABS":
            if len(expr.args) > 0:
                arg_val, arg_type = self._emit_expression(expr.args[0])
                if arg_type == "double":
                    self.code_lines.append(
                        f"  {result} = call double @fabs(double {arg_val})")
                    return (result, "double")
                else:
                    self.code_lines.append(
                        f"  {result} = call i32 @abs(i32 {arg_val})")
                    return (result, "i32")
        elif func_name == "INT":
            arg_val, arg_type = self._emit_expression(expr.args[0])
            arg_double = self._convert_to_double(arg_val) if arg_type != "double" else arg_val
            self.code_lines.append(
                f"  {result} = fptosi double {arg_double} to i32")
            return (result, "i32")
        elif func_name == "FLOAT" or func_name == "REAL":
            arg_val, arg_type = self._emit_expression(expr.args[0])
            self.code_lines.append(
                f"  {result} = sitofp i32 {arg_val} to double")
            return (result, "double")
        elif func_name == "MOD":
            left_val, left_type = self._emit_expression(expr.args[0])
            right_val, right_type = self._emit_expression(expr.args[1])
            self.code_lines.append(
                f"  {result} = srem i32 {left_val}, {right_val}")
            return (result, "i32")
        elif func_name == "MIN":
            left_val, left_type = self._emit_expression(expr.args[0])
            right_val, right_type = self._emit_expression(expr.args[1])
            cmp_result = self._new_local()
            self.code_lines.append(
                f"  {cmp_result} = icmp slt i32 {left_val}, {right_val}")
            self.code_lines.append(
                f"  {result} = select i1 {cmp_result}, i32 {left_val}, i32 {right_val}")
            return (result, "i32")
        elif func_name == "POW":
            left_val, left_type = self._emit_expression(expr.args[0])
            right_val, right_type = self._emit_expression(expr.args[1])
            left_double = self._convert_to_double(left_val) if left_type != "double" else left_val
            right_double = self._convert_to_double(right_val) if right_type != "double" else right_val
            self.code_lines.append(
                f"  {result} = call double @pow(double {left_double}, double {right_double})")
            return (result, "double")
        elif func_name == "MAX":
            left_val, left_type = self._emit_expression(expr.args[0])
            right_val, right_type = self._emit_expression(expr.args[1])
            cmp_result = self._new_local()
            self.code_lines.append(
                f"  {cmp_result} = icmp sgt i32 {left_val}, {right_val}")
            self.code_lines.append(
                f"  {result} = select i1 {cmp_result}, i32 {left_val}, i32 {right_val}")
            return (result, "i32")
        else:
            args = [self._emit_expression(arg)[0] for arg in expr.args]
            args_str = ", ".join(args)
            self.code_lines.append(
                f"  {result} = call double @{func_name.lower()}({args_str})")
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

    def _add_string(self, string_val: str) -> Tuple[str, int]:
        if string_val not in self.strings:
            str_name = f"@.str.{self.string_counter}"
            self.string_counter += 1
            byte_count = 0
            i = 0
            while i < len(string_val):
                if i + 1 < len(string_val) and string_val[i] == '\\':
                    byte_count += 1
                    i += 2
                else:
                    byte_count += 1
                    i += 1
            byte_count += 1
            self.strings[string_val] = (str_name, byte_count)
        return self.strings[string_val]

    def _new_local(self) -> str:
        self.local_counter += 1
        return f"%t{self.local_counter}"

    def _new_block_id(self) -> int:
        self.block_counter += 1
        return self.block_counter

    def _emit_return_statement(self, stmt: ReturnStatement):
        """Генерирует LLVM код для RETURN"""
        if self.current_function == "main":
            self.code_lines.append("  ret i32 0")
        else:
            func_name = self.current_function
            if func_name in self.var_alloc:
                alloc_type, ptr = self.var_alloc[func_name]
                result = self._new_local()
                self.code_lines.append(f"  {result} = load {alloc_type}, {alloc_type}* {ptr}")
                self.code_lines.append(f"  ret {alloc_type} {result}")
            else:
                self.code_lines.append("  ret void")

    def _emit_goto_statement(self, stmt: GotoStatement):
        """Генерирует LLVM код для GOTO"""
        label_name = f"label_{stmt.label}"
        self.code_lines.append(f"  br label %{label_name}")

    def _emit_continue_statement(self, stmt: ContinueStatement):
        """Генерирует LLVM код для CONTINUE"""
        if stmt.label:
            label_name = f"label_{stmt.label}"
            self.code_lines.append(f"  br label %{label_name}")
            self.code_lines.append(f"{label_name}:")

    def _emit_subroutine(self, sub: Subroutine):
        """Генерирует LLVM код для SUBROUTINE"""
        old_function = self.current_function
        old_var_alloc = self.var_alloc.copy()
        old_local_counter = self.local_counter
        old_block_counter = self.block_counter
        
        self.current_function = sub.name.upper()
        self.var_alloc = {}
        self.local_counter = 0
        self.block_counter = 0
        self.var_ssa_versions = {}
        self.phi_tracking = []
        
        param_list = []
        param_types = {}
        for param in sub.params:
            param_type = "i32"
            param_list.append(f"{param_type}* %param_{param}")
            param_types[param.upper()] = param_type
        
        self.code_lines.append(f"; === Подпрограмма {sub.name} ===")
        self.code_lines.append(f"define void @{sub.name.upper()}({', '.join(param_list)}) #0 {{")
        self.code_lines.append("entry:")
        
        for param in sub.params:
            param_upper = param.upper()
            param_type = param_types[param_upper]
            self.var_alloc[param_upper] = (param_type, f"%param_{param}")
        
        self._emit_declarations(sub.declarations)
        
        for stmt in sub.statements:
            self._emit_statement(stmt)
        
        if not any(isinstance(s, ReturnStatement) for s in sub.statements):
            self.code_lines.append("  ret void")
        
        self.code_lines.append("}")
        self.code_lines.append("")
        
        self.current_function = old_function
        self.var_alloc = old_var_alloc
        self.local_counter = old_local_counter
        self.block_counter = old_block_counter

    def _emit_function(self, func: FunctionDef):
        """Генерирует LLVM код для FUNCTION"""
        old_function = self.current_function
        old_var_alloc = self.var_alloc.copy()
        old_local_counter = self.local_counter
        old_block_counter = self.block_counter
        
        func_name_upper = func.name.upper()
        self.current_function = func_name_upper
        self.var_alloc = {}
        self.local_counter = 0
        self.block_counter = 0
        self.var_ssa_versions = {}
        self.phi_tracking = []
        
        return_type = "i32"
        if func.return_type:
            type_map = {'INTEGER': 'i32', 'REAL': 'double', 'LOGICAL': 'i1'}
            return_type = type_map.get(func.return_type.upper(), 'i32')
        
        param_list = []
        param_types = {}
        for param in func.params:
            param_type = "i32"
            param_list.append(f"{param_type} %param_{param}")
            param_types[param.upper()] = param_type
        
        self.code_lines.append(f"; === Функция {func.name} ===")
        self.code_lines.append(f"define {return_type} @{func_name_upper}({', '.join(param_list)}) #0 {{")
        self.code_lines.append("entry:")
        
        result_ptr = f"%{func_name_upper}"
        self.code_lines.append(f"  {result_ptr} = alloca {return_type}")
        self.var_alloc[func_name_upper] = (return_type, result_ptr)
        
        for param in func.params:
            param_upper = param.upper()
            param_type = param_types[param_upper]
            local_name = f"%{param_upper}"
            self.code_lines.append(f"  {local_name} = alloca {param_type}")
            self.code_lines.append(f"  store {param_type} %param_{param}, {param_type}* {local_name}")
            self.var_alloc[param_upper] = (param_type, local_name)
        
        self._emit_declarations(func.declarations)
        
        for stmt in func.statements:
            self._emit_statement(stmt)
        
        result = self._new_local()
        self.code_lines.append(f"  {result} = load {return_type}, {return_type}* {result_ptr}")
        self.code_lines.append(f"  ret {return_type} {result}")
        
        self.code_lines.append("}")
        self.code_lines.append("")
        
        self.current_function = old_function
        self.var_alloc = old_var_alloc
        self.local_counter = old_local_counter
        self.block_counter = old_block_counter
