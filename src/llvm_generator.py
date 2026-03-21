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
    Subroutine, FunctionDef, ImplicitNone, ImplicitStatement, ComplexLiteral,
    ParameterStatement, ArithmeticIfStatement, CommonStatement, ExternalStatement,
    ExitStatement, ParallelDoLoop
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
            'DOUBLEPRECISION': 'double',
        }
        self.implicit_rules = {}
        self.implicit_none = False

        self.var_ssa_versions: Dict[str, str] = {}
        self.phi_tracking: List[Dict[str, str]] = []
        self.last_block_before_endif: Dict[str, str] = {}
        self.array_dimensions: Dict[str, List[int]] = {}
        self.array_bounds: Dict[str, List[Tuple[int, int]]] = {}
        self.last_block_label: str = ""
        self.current_block: str = "entry"
        self.current_subroutine_params: List[str] = []
        self.heap_array_threshold_bytes = 1048576
        self.loop_exit_stack: List[str] = []
        self.common_globals: Dict[str, Tuple[str, str]] = {}
        self.parallel_struct_counter = 0
        self.parallel_worker_counter = 0
        self.parallel_type_insert_pos = 0
        self.parallel_strings_insert_pos = 0
        self.deferred_functions: List[str] = []

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
        self.array_dimensions = {}
        self.array_bounds = {}
        self.current_subroutine_params = []
        self.loop_exit_stack = []
        self.common_globals = {}
        self.parallel_struct_counter = 0
        self.parallel_worker_counter = 0
        self.parallel_type_insert_pos = 0
        self.parallel_strings_insert_pos = 0
        self.deferred_functions = []

        for stmt_func in ast.statement_functions:
            if hasattr(stmt_func, 'name') and stmt_func.name:
                self.statement_functions[stmt_func.name.upper()] = stmt_func

        self.user_functions: Dict[str, tuple] = {}
        for func in ast.functions:
            fname = func.name.upper()
            ret_type = "i32"
            if func.return_type:
                ret_type = self.type_map.get(func.return_type.upper(), 'i32')
            param_type_map = {}
            for decl in func.declarations:
                if hasattr(decl, 'names') and hasattr(decl, 'type'):
                    llvm_t = self.type_map.get(decl.type, 'i32')
                    for pname, _ in decl.names:
                        param_type_map[pname.upper()] = llvm_t
            param_types = []
            for p in func.params:
                param_types.append(param_type_map.get(p.upper(), 'i32'))
            self.user_functions[fname] = (ret_type, param_types)

        self._collect_strings(ast)
        self._emit_header()
        self._emit_external_declarations()
        self._emit_common_globals(ast)
        self.parallel_type_insert_pos = len(self.code_lines)
        self.parallel_strings_insert_pos = len(self.code_lines)
        self.code_lines.append("; === Глобальные строки (будут заполнены) ===")
        self.local_counter = 0
        self.var_alloc = {}
        self._emit_program(ast)

        for sub in ast.subroutines:
            self._emit_subroutine(sub)

        for func in ast.functions:
            self._emit_function(func)

        if self.deferred_functions:
            self.code_lines.extend(self.deferred_functions)

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
            self.code_lines[self.parallel_strings_insert_pos:self.parallel_strings_insert_pos +
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
        self.code_lines.append("declare void @fortran_parallel_for_i32(i32, i32, i32, i32, i8*, i8*)")
        self.code_lines.append("")

    def _get_scope_implicit_type(self, name: str, implicit_none: bool, implicit_rules: Dict[str, str]) -> Tuple[str, Optional[int]]:
        if not name:
            return ("UNKNOWN", None)
        first_char = name[0].upper()
        if first_char in implicit_rules:
            return (implicit_rules[first_char], None)
        if not implicit_none:
            if 'I' <= first_char <= 'N':
                return ("INTEGER", None)
            return ("REAL", None)
        return ("UNKNOWN", None)

    def _evaluate_constant_expression(self, expr: Expression, parameters: Dict[str, object]):
        if isinstance(expr, IntegerLiteral):
            return expr.value
        if isinstance(expr, RealLiteral):
            return expr.value
        if isinstance(expr, LogicalLiteral):
            return expr.value
        if isinstance(expr, StringLiteral):
            return expr.value
        if isinstance(expr, Variable):
            return parameters.get(expr.name.upper())
        if isinstance(expr, UnaryOp):
            operand = self._evaluate_constant_expression(expr.operand, parameters)
            if operand is None:
                return None
            if expr.op == "+":
                return operand
            if expr.op == "-":
                return -operand
            if expr.op == ".NOT.":
                return not operand
            return None
        if isinstance(expr, BinaryOp):
            left = self._evaluate_constant_expression(expr.left, parameters)
            right = self._evaluate_constant_expression(expr.right, parameters)
            if left is None or right is None:
                return None
            if expr.op == "+":
                return left + right
            if expr.op == "-":
                return left - right
            if expr.op == "*":
                return left * right
            if expr.op == "/":
                if isinstance(left, int) and isinstance(right, int):
                    return left // right
                return left / right
            if expr.op == "**":
                return left ** right
            return None
        return None

    def _resolve_dimension_ranges(self, dims, parameters: Dict[str, object]) -> List[Tuple[int, int]]:
        resolved = []
        if not dims:
            return resolved
        for dim_spec in dims:
            if isinstance(dim_spec, tuple) and len(dim_spec) == 2:
                lower_bound, upper_bound = dim_spec
            else:
                lower_bound, upper_bound = 1, dim_spec
            lower_value = lower_bound if isinstance(lower_bound, int) else self._evaluate_constant_expression(lower_bound, parameters)
            upper_value = upper_bound if isinstance(upper_bound, int) else self._evaluate_constant_expression(upper_bound, parameters)
            if not isinstance(lower_value, int) or isinstance(lower_value, bool):
                lower_value = 1
            if not isinstance(upper_value, int) or isinstance(upper_value, bool):
                upper_value = lower_value
            resolved.append((lower_value, upper_value))
        return resolved

    def _collect_scope_declaration_info(self, declarations):
        implicit_none = False
        implicit_rules: Dict[str, str] = {}
        parameters: Dict[str, object] = {}
        declared_types: Dict[str, str] = {}
        declared_dims: Dict[str, List[Tuple[int, int]]] = {}
        common_blocks: List[Tuple[str, str]] = []
        for decl in declarations:
            if isinstance(decl, ImplicitNone):
                implicit_none = True
            elif isinstance(decl, ImplicitStatement):
                for rule in decl.rules:
                    for letter in rule.get_letters():
                        implicit_rules[letter.upper()] = rule.type_name.upper()
            elif isinstance(decl, ParameterStatement):
                for param_name, param_expr in decl.params:
                    value = self._evaluate_constant_expression(param_expr, parameters)
                    if value is not None:
                        parameters[param_name.upper()] = value
            elif isinstance(decl, DimensionStatement):
                for name, dims in decl.names:
                    declared_dims[name.upper()] = self._resolve_dimension_ranges(dims, parameters)
            elif isinstance(decl, Declaration):
                llvm_type = self.type_map.get(decl.type, 'i32')
                for name, dims in decl.names:
                    name_upper = name.upper()
                    declared_types[name_upper] = llvm_type
                    if dims:
                        declared_dims[name_upper] = self._resolve_dimension_ranges(dims, parameters)
            elif isinstance(decl, CommonStatement):
                for block_name, variables in decl.blocks:
                    for var in variables:
                        common_blocks.append((block_name.upper(), var.name.upper()))
        return {
            "implicit_none": implicit_none,
            "implicit_rules": implicit_rules,
            "parameters": parameters,
            "declared_types": declared_types,
            "declared_dims": declared_dims,
            "common_blocks": common_blocks,
        }

    def _common_global_key(self, block_name: str, var_name: str) -> str:
        block_part = block_name.upper() if block_name else "BLANK"
        return f"{block_part}:{var_name.upper()}"

    def _common_global_symbol(self, block_name: str, var_name: str) -> str:
        block_part = block_name.upper() if block_name else "BLANK"
        return f"@COMMON_{block_part}_{var_name.upper()}"

    def _ensure_common_global(self, block_name: str, var_name: str, llvm_type: str, dims: List[Tuple[int, int]]):
        key = self._common_global_key(block_name, var_name)
        if key in self.common_globals:
            return
        symbol = self._common_global_symbol(block_name, var_name)
        if dims:
            total_size, dim_sizes = self._compute_array_layout(dims)
            alloc_type = f"[{total_size} x {llvm_type}]"
            self.array_bounds[var_name.upper()] = dims
            self.array_dimensions[var_name.upper()] = dim_sizes
            self.code_lines.append(f"{symbol} = global {alloc_type} zeroinitializer")
            self.common_globals[key] = (alloc_type, symbol)
        else:
            self.code_lines.append(f"{symbol} = global {llvm_type} zeroinitializer")
            self.common_globals[key] = (llvm_type, symbol)

    def _emit_common_globals(self, ast: Program):
        scopes = [ast.declarations]
        scopes.extend(sub.declarations for sub in ast.subroutines)
        scopes.extend(func.declarations for func in ast.functions)
        for declarations in scopes:
            scope_info = self._collect_scope_declaration_info(declarations)
            for block_name, var_name in scope_info["common_blocks"]:
                llvm_type = scope_info["declared_types"].get(var_name)
                if llvm_type is None:
                    implicit_type_name, _ = self._get_scope_implicit_type(
                        var_name, scope_info["implicit_none"], scope_info["implicit_rules"]
                    )
                    llvm_type = self.type_map.get(implicit_type_name, 'double')
                dims = scope_info["declared_dims"].get(var_name, [])
                self._ensure_common_global(block_name, var_name, llvm_type, dims)
        if self.common_globals:
            self.code_lines.append("")

    def _insert_parallel_type(self, line: str):
        self.code_lines.insert(self.parallel_type_insert_pos, line)
        self.parallel_type_insert_pos += 1
        self.parallel_strings_insert_pos += 1

    def _is_phi_trackable_var(self, var_name: str, alloc_type: str) -> bool:
        if alloc_type.startswith("["):
            return False
        if var_name.upper() in self.array_dimensions:
            return False
        return True

    def _lookup_var_alloc(self, name: str):
        if name in self.var_alloc:
            return self.var_alloc[name]
        name_upper = name.upper()
        for key, value in self.var_alloc.items():
            if key.upper() == name_upper:
                return value
        return None

    def _collect_parallel_loop_vars(self, stmts: List[Statement]) -> List[str]:
        result: List[str] = []
        for stmt in stmts:
            if isinstance(stmt, (DoLoop, LabeledDoLoop, ParallelDoLoop)):
                result.append(stmt.var)
                result.extend(self._collect_parallel_loop_vars(stmt.body))
            elif isinstance(stmt, IfStatement):
                result.extend(self._collect_parallel_loop_vars(stmt.then_body))
                for _, body in stmt.elif_parts:
                    result.extend(self._collect_parallel_loop_vars(body))
                if stmt.else_body:
                    result.extend(self._collect_parallel_loop_vars(stmt.else_body))
            elif isinstance(stmt, SimpleIfStatement):
                result.extend(self._collect_parallel_loop_vars([stmt.statement]))
        ordered = []
        seen = set()
        for name in result:
            if name not in seen:
                seen.add(name)
                ordered.append(name)
        return ordered

    def _emit_parallel_worker_function(self, stmt: ParallelDoLoop, env_type_name: str, worker_name: str, captures, private_allocs):
        old_code_lines = self.code_lines
        old_var_alloc = self.var_alloc.copy()
        old_local_counter = self.local_counter
        old_block_counter = self.block_counter
        old_current_block = self.current_block
        old_current_function = getattr(self, "current_function", "main")
        old_phi_tracking = self.phi_tracking
        old_var_ssa_versions = self.var_ssa_versions
        old_last_block_before_endif = self.last_block_before_endif
        old_loop_exit_stack = self.loop_exit_stack
        lines: List[str] = []
        self.code_lines = lines
        self.var_alloc = {}
        self.local_counter = 0
        self.block_counter = 0
        self.current_block = "entry"
        self.current_function = worker_name
        self.phi_tracking = []
        self.var_ssa_versions = {}
        self.last_block_before_endif = {}
        self.loop_exit_stack = []

        self.code_lines.append(f"define internal void @{worker_name}(i32 %chunk_start, i32 %chunk_end, i8* %env) {{")
        self.code_lines.append("entry:")
        self._set_current_block("entry")

        if captures:
            env_cast = self._new_local()
            self.code_lines.append(f"  {env_cast} = bitcast i8* %env to {env_type_name}*")
            for index, (name, alloc_type, _) in enumerate(captures):
                field_type = f"{alloc_type}*"
                field_ptr = self._new_local()
                self.code_lines.append(
                    f"  {field_ptr} = getelementptr inbounds {env_type_name}, {env_type_name}* {env_cast}, i32 0, i32 {index}"
                )
                loaded_ptr = self._new_local()
                self.code_lines.append(f"  {loaded_ptr} = load {field_type}, {field_type}* {field_ptr}")
                self.var_alloc[name] = (alloc_type, loaded_ptr)

        for loop_var, alloc_type in private_allocs:
            local_name = f"%{loop_var}"
            self.code_lines.append(f"  {local_name} = alloca {alloc_type}")
            self.var_alloc[loop_var] = (alloc_type, local_name)

        loop_ptr = self._lookup_var_alloc(stmt.var)[1]
        step_val, _ = self._emit_expression(stmt.step if stmt.step is not None else IntegerLiteral(value=1))
        loop_id = self._new_block_id()
        loop_label = f"parallel_loop_{loop_id}"
        body_label = f"parallel_body_{loop_id}"
        end_label = f"parallel_end_{loop_id}"
        self.code_lines.append(f"  store i32 %chunk_start, i32* {loop_ptr}")
        self.code_lines.append(f"  br label %{loop_label}")
        self.code_lines.append(f"{loop_label}:")
        self._set_current_block(loop_label)
        loop_var = self._new_local()
        self.code_lines.append(f"  {loop_var} = load i32, i32* {loop_ptr}")
        cond = self._new_local()
        self.code_lines.append(f"  {cond} = icmp sle i32 {loop_var}, %chunk_end")
        self.code_lines.append(f"  br i1 {cond}, label %{body_label}, label %{end_label}")
        self.code_lines.append(f"{body_label}:")
        self._set_current_block(body_label)
        for body_stmt in stmt.body:
            self._emit_statement(body_stmt)
        current_val = self._new_local()
        self.code_lines.append(f"  {current_val} = load i32, i32* {loop_ptr}")
        next_val = self._new_local()
        self.code_lines.append(f"  {next_val} = add i32 {current_val}, {step_val}")
        self.code_lines.append(f"  store i32 {next_val}, i32* {loop_ptr}")
        self._emit_br_if_not_terminated(loop_label)
        self.code_lines.append(f"{end_label}:")
        self._set_current_block(end_label)
        self.code_lines.append("  ret void")
        self.code_lines.append("}")
        self.code_lines.append("")

        self.code_lines = old_code_lines
        self.var_alloc = old_var_alloc
        self.local_counter = old_local_counter
        self.block_counter = old_block_counter
        self.current_block = old_current_block
        self.current_function = old_current_function
        self.phi_tracking = old_phi_tracking
        self.var_ssa_versions = old_var_ssa_versions
        self.last_block_before_endif = old_last_block_before_endif
        self.loop_exit_stack = old_loop_exit_stack
        return "\n".join(lines)

    def _emit_parallel_do_loop(self, stmt: ParallelDoLoop):
        private_names = [stmt.var] + self._collect_parallel_loop_vars(stmt.body) + list(stmt.private_vars)
        ordered_private_names = []
        private_seen = set()
        for name in private_names:
            name_upper = name.upper()
            if name_upper in private_seen:
                continue
            private_seen.add(name_upper)
            ordered_private_names.append(name)
        private_set = {name.upper() for name in ordered_private_names}
        captures = [
            (name, alloc_type, ptr_name)
            for name, (alloc_type, ptr_name) in self.var_alloc.items()
            if name.upper() not in private_set
        ]
        private_allocs = []
        for name in ordered_private_names:
            alloc = self._lookup_var_alloc(name)
            alloc_type = alloc[0] if alloc is not None else "i32"
            private_allocs.append((name, alloc_type))
        env_type_name = f"%parallel_env_{self.parallel_struct_counter}"
        self.parallel_struct_counter += 1
        if captures:
            env_fields = ", ".join(f"{alloc_type}*" for _, alloc_type, _ in captures)
        else:
            env_fields = "i8"
        self._insert_parallel_type(f"{env_type_name} = type {{ {env_fields} }}")
        env_alloc = self._new_local()
        self.code_lines.append(f"  {env_alloc} = alloca {env_type_name}")
        if captures:
            for index, (_, alloc_type, ptr_name) in enumerate(captures):
                field_type = f"{alloc_type}*"
                field_ptr = self._new_local()
                self.code_lines.append(
                    f"  {field_ptr} = getelementptr inbounds {env_type_name}, {env_type_name}* {env_alloc}, i32 0, i32 {index}"
                )
                self.code_lines.append(f"  store {field_type} {ptr_name}, {field_type}* {field_ptr}")
        else:
            field_ptr = self._new_local()
            self.code_lines.append(
                f"  {field_ptr} = getelementptr inbounds {env_type_name}, {env_type_name}* {env_alloc}, i32 0, i32 0"
            )
            self.code_lines.append(f"  store i8 0, i8* {field_ptr}")

        worker_name = f"parallel_worker_{self.parallel_worker_counter}"
        self.parallel_worker_counter += 1
        self.deferred_functions.append(
            self._emit_parallel_worker_function(stmt, env_type_name, worker_name, captures, private_allocs)
        )

        start_val, _ = self._emit_expression(stmt.start)
        end_val, _ = self._emit_expression(stmt.end)
        step_val, _ = self._emit_expression(stmt.step if stmt.step is not None else IntegerLiteral(value=1))
        env_i8 = self._new_local()
        self.code_lines.append(f"  {env_i8} = bitcast {env_type_name}* {env_alloc} to i8*")
        worker_ptr = self._new_local()
        self.code_lines.append(f"  {worker_ptr} = bitcast void (i32, i32, i8*)* @{worker_name} to i8*")

        loop_alloc = self._lookup_var_alloc(stmt.var)
        if loop_alloc is not None:
            loop_alloc_type, loop_ptr = loop_alloc
            self.code_lines.append(f"  store i32 {start_val}, i32* {loop_ptr}")
        cond = self._new_local()
        self.code_lines.append(f"  {cond} = icmp sle i32 {start_val}, {end_val}")
        exec_label = f"parallel_exec_{self._new_block_id()}"
        skip_label = f"parallel_skip_{self._new_block_id()}"
        done_label = f"parallel_done_{self._new_block_id()}"
        self.code_lines.append(f"  br i1 {cond}, label %{exec_label}, label %{skip_label}")
        self.code_lines.append(f"{exec_label}:")
        self._set_current_block(exec_label)
        self.code_lines.append(
            f"  call void @fortran_parallel_for_i32(i32 {start_val}, i32 {end_val}, i32 {step_val}, i32 {stmt.grain}, i8* {env_i8}, i8* {worker_ptr})"
        )
        if loop_alloc is not None:
            final_val = self._new_local()
            self.code_lines.append(f"  {final_val} = add i32 {end_val}, {step_val}")
            self.code_lines.append(f"  store i32 {final_val}, i32* {loop_ptr}")
        self.code_lines.append(f"  br label %{done_label}")
        self.code_lines.append(f"{skip_label}:")
        self._set_current_block(skip_label)
        self.code_lines.append(f"  br label %{done_label}")
        self.code_lines.append(f"{done_label}:")
        self._set_current_block(done_label)

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
            self._set_current_block("entry")
            self.local_counter = 0
            self.var_alloc = {}
            self.var_ssa_versions = {}
            self.phi_tracking = []
            self._emit_declarations(ast.declarations)
            # Mark insert position for deferred alloca of optimization temporaries
            self._entry_alloca_insert_pos = len(self.code_lines)
            for stmt in ast.statements:
                self._emit_statement(stmt)
            self.code_lines.append("  ret i32 0")
            self.code_lines.append("}")
            self.code_lines.append("")
        elif ast.declarations:
            self.code_lines.append("; === Основная программа (только объявления) ===")
            self.code_lines.append("define i32 @main() {")
            self.code_lines.append("entry:")
            self._set_current_block("entry")
            self.local_counter = 0
            self.var_alloc = {}
            self.var_ssa_versions = {}
            self.phi_tracking = []
            self._emit_declarations(ast.declarations)
            self.code_lines.append("  ret i32 0")
            self.code_lines.append("}")
            self.code_lines.append("")

    def _emit_declarations(self, declarations: List[Declaration]):
        scope_info = self._collect_scope_declaration_info(declarations)
        common_names = {var_name for _, var_name in scope_info["common_blocks"]}

        for decl in declarations:
            if isinstance(decl, ParameterStatement):
                for param_name, param_expr in decl.params:
                    val, val_type = self._emit_expression(param_expr)
                    if param_name not in self.var_alloc:
                        local_name = f"%{param_name}"
                        self.code_lines.append(f"  {local_name} = alloca {val_type}")
                        self.var_alloc[param_name] = (val_type, local_name)
                    alloc_type, ptr_name = self.var_alloc[param_name]
                    self.code_lines.append(f"  store {val_type} {val}, {val_type}* {ptr_name}")
                continue
            if isinstance(decl, CommonStatement):
                for block_name, variables in decl.blocks:
                    for var in variables:
                        name = var.name
                        name_upper = name.upper()
                        key = self._common_global_key(block_name, name)
                        if key not in self.common_globals:
                            llvm_type = scope_info["declared_types"].get(name_upper)
                            if llvm_type is None:
                                implicit_type_name, _ = self._get_scope_implicit_type(
                                    name_upper, scope_info["implicit_none"], scope_info["implicit_rules"]
                                )
                                llvm_type = self.type_map.get(implicit_type_name, 'double')
                            dims = scope_info["declared_dims"].get(name_upper, [])
                            self._ensure_common_global(block_name, name, llvm_type, dims)
                        alloc_type, symbol = self.common_globals[key]
                        self.var_alloc[name] = (alloc_type, symbol)
                        if name_upper in scope_info["declared_dims"]:
                            dims = scope_info["declared_dims"][name_upper]
                            _, dim_sizes = self._compute_array_layout(dims)
                            self.array_bounds[name_upper] = dims
                            self.array_dimensions[name_upper] = dim_sizes
                continue
            if hasattr(decl, 'names') and hasattr(decl, 'type'):
                llvm_type = self.type_map.get(decl.type, 'i32')
                for name, dims in decl.names:
                    name_upper = name.upper()
                    if name_upper in self.user_functions or name_upper in common_names:
                        if dims:
                            resolved_dims = scope_info["declared_dims"].get(name_upper, [])
                            if resolved_dims:
                                _, dim_sizes = self._compute_array_layout(resolved_dims)
                                self.array_bounds[name_upper] = resolved_dims
                                self.array_dimensions[name_upper] = dim_sizes
                        continue
                    resolved_dims = scope_info["declared_dims"].get(name_upper, [])
                    if name in self.var_alloc:
                        old_alloc_type, old_ptr = self.var_alloc[name]
                        old_base = old_alloc_type.split(' x ')[-1].rstrip(']') if '[' in old_alloc_type else old_alloc_type
                        if old_base != llvm_type and '[' in old_alloc_type:
                            total_size = int(old_alloc_type.split('[')[1].split(']')[0].split(' x ')[0])
                            new_alloc_type = f"[{total_size} x {llvm_type}]"
                            idx = None
                            for ci, line in enumerate(self.code_lines):
                                if f"{old_ptr} = alloca {old_alloc_type}" in line:
                                    idx = ci
                                    break
                            if idx is not None:
                                self.code_lines[idx] = f"  {old_ptr} = alloca {new_alloc_type}"
                            self.var_alloc[name] = (new_alloc_type, old_ptr)
                        continue
                    if decl.type.startswith('CHARACTER') and decl.type_size:
                        char_size = int(decl.type_size)
                        alloc_type = f"[{char_size} x i8]"
                        local_name = f"%{name}"
                        self.char_lengths[name] = char_size
                        self.code_lines.append(f"  {local_name} = alloca {alloc_type}")
                        self.var_alloc[name] = (alloc_type, local_name)
                        continue
                    if resolved_dims:
                        size, dim_sizes = self._compute_array_layout(resolved_dims)
                        local_name = f"%{name}"
                        alloc_type, local_name = self._allocate_array_storage(local_name, llvm_type, size)
                        self.array_bounds[name_upper] = resolved_dims
                        self.array_dimensions[name_upper] = dim_sizes
                    else:
                        alloc_type = llvm_type
                        local_name = f"%{name}"
                        self.code_lines.append(f"  {local_name} = alloca {alloc_type}")
                    self.var_alloc[name] = (alloc_type, local_name)

        for name_upper, dims in scope_info["declared_dims"].items():
            if name_upper in common_names:
                continue
            if any(existing_name.upper() == name_upper for existing_name in self.var_alloc):
                continue
            llvm_type = scope_info["declared_types"].get(name_upper)
            if llvm_type is None:
                implicit_type_name, _ = self._get_scope_implicit_type(
                    name_upper, scope_info["implicit_none"], scope_info["implicit_rules"]
                )
                llvm_type = self.type_map.get(implicit_type_name, 'double')
            size, dim_sizes = self._compute_array_layout(dims)
            name = name_upper
            local_name = f"%{name}"
            alloc_type, local_name = self._allocate_array_storage(local_name, llvm_type, size)
            self.array_bounds[name_upper] = dims
            self.array_dimensions[name_upper] = dim_sizes
            self.var_alloc[name] = (alloc_type, local_name)

    def _compute_array_layout(self, dims):
        size = 1
        dim_sizes = []
        for d in dims:
            if isinstance(d, tuple):
                start, end = d
                dim_size = end - start + 1
            else:
                dim_size = d
            size *= dim_size
            dim_sizes.append(dim_size)
        return size, dim_sizes

    def _allocate_array_storage(self, local_name: str, llvm_type: str, size: int):
        total_bytes = size * self._type_size_bytes(llvm_type)
        if total_bytes > self.heap_array_threshold_bytes:
            raw_ptr = self._new_local()
            self.code_lines.append(f"  {raw_ptr} = call i8* @malloc(i64 {total_bytes})")
            self.code_lines.append(f"  {local_name} = bitcast i8* {raw_ptr} to {llvm_type}*")
            return llvm_type, local_name
        alloc_type = f"[{size} x {llvm_type}]"
        self.code_lines.append(f"  {local_name} = alloca {alloc_type}")
        return alloc_type, local_name

    def _type_size_bytes(self, llvm_type: str) -> int:
        if llvm_type == "double":
            return 8
        if llvm_type == "i32":
            return 4
        if llvm_type == "i1":
            return 1
        if llvm_type == "{double, double}":
            return 16
        return 8

    def _emit_statement(self, stmt: Statement):
        if hasattr(stmt, 'stmt_label') and stmt.stmt_label and not isinstance(stmt, ContinueStatement):
            label_name = f"label_{stmt.stmt_label}"
            self._emit_br_if_not_terminated(label_name)
            self.code_lines.append(f"{label_name}:")
            self._set_current_block(label_name)
        if isinstance(stmt, ParallelDoLoop):
            self._emit_parallel_do_loop(stmt)
        elif isinstance(stmt, Assignment):
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
        elif isinstance(stmt, ArithmeticIfStatement):
            self._emit_arithmetic_if(stmt)
        elif isinstance(stmt, StopStatement):
            self.code_lines.append("  ret i32 0")
        elif isinstance(stmt, ExitStatement):
            self._emit_exit_statement()

    def _emit_assignment(self, stmt: Assignment):
        rhs_val, rhs_type = self._emit_expression(stmt.value)
        if stmt.indices:
            indices_vals = [self._emit_expression(
                idx)[0] for idx in stmt.indices]
            if stmt.target in self.var_alloc:
                alloc_type, ptr_name = self.var_alloc[stmt.target]
                base_type = alloc_type if '[' not in alloc_type else alloc_type.split(
                    '[')[1].split(']')[0].split(' x ')[1]

                if base_type != rhs_type and rhs_type in ('i32', 'double') and base_type in ('i32', 'double'):
                    conv = self._new_local()
                    if base_type == 'double' and rhs_type == 'i32':
                        self.code_lines.append(f"  {conv} = sitofp i32 {rhs_val} to double")
                    else:
                        self.code_lines.append(f"  {conv} = fptosi double {rhs_val} to i32")
                    rhs_val = conv
                    rhs_type = base_type

                linear_idx = self._compute_linear_index(indices_vals, stmt.target)
                if '[' not in alloc_type:
                    elem_ptr = self._new_local()
                    self.code_lines.append(
                        f"  {elem_ptr} = getelementptr inbounds {base_type}, "
                        f"{base_type}* {ptr_name}, i32 {linear_idx}")
                else:
                    elem_ptr = self._new_local()
                    self.code_lines.append(
                        f"  {elem_ptr} = getelementptr inbounds {alloc_type}, "
                        f"{alloc_type}* {ptr_name}, i64 0, i32 {linear_idx}")
                self.code_lines.append(
                    f"  store {base_type} {rhs_val}, {base_type}* {elem_ptr}")
        else:
            if stmt.target not in self.var_alloc:
                var_name = stmt.target.upper()
                implicit_type_name, _ = self._get_implicit_type(var_name)
                if implicit_type_name and implicit_type_name != "UNKNOWN":
                    alloc_type = self.type_map.get(implicit_type_name, 'i32')
                else:
                    alloc_type = rhs_type if rhs_type in ('i32', 'double', 'i1') else 'i32'
                local_name = f"%{stmt.target}"
                alloca_instr = f"  {local_name} = alloca {alloc_type}"
                # Emit alloca in entry block to avoid stack growth in loops
                if hasattr(self, '_entry_alloca_insert_pos') and self.current_block != "entry":
                    self.code_lines.insert(self._entry_alloca_insert_pos, alloca_instr)
                    self._entry_alloca_insert_pos += 1
                else:
                    self.code_lines.append(alloca_instr)
                self.var_alloc[stmt.target] = (alloc_type, local_name)
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
                    # Type coercion for optimization-generated temporaries
                    if base_type != rhs_type and rhs_type in ('i32', 'double') and base_type in ('i32', 'double'):
                        conv = self._new_local()
                        if base_type == 'double' and rhs_type == 'i32':
                            self.code_lines.append(f"  {conv} = sitofp i32 {rhs_val} to double")
                        else:
                            self.code_lines.append(f"  {conv} = fptosi double {rhs_val} to i32")
                        rhs_val = conv

                    if self.phi_tracking:

                        self.code_lines.append(
                            f"  store {base_type} {rhs_val}, {base_type}* {ptr_name}")

                        new_ssa = self._new_local()
                        self.code_lines.append(
                            f"  {new_ssa} = load {base_type}, {base_type}* {ptr_name}")

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
        self._set_current_block(loop_label)
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
        self._set_current_block(loop_body_label)
        self.loop_exit_stack.append(loop_end_label)
        for body_stmt in stmt.body:
            self._emit_statement(body_stmt)
        self.loop_exit_stack.pop()
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
        self._emit_br_if_not_terminated(loop_label)
        self.code_lines.append(f"{loop_end_label}:")
        self._set_current_block(loop_end_label)


        for if_id_key in list(self.last_block_before_endif.keys()):
            self.last_block_before_endif[if_id_key] = loop_end_label

    def _emit_do_while(self, stmt: DoWhile):
        loop_id = self._new_block_id()
        loop_label = f"do_while_loop_{loop_id}"
        loop_body_label = f"do_while_body_{loop_id}"
        loop_end_label = f"do_while_end_{loop_id}"
        self.code_lines.append(f"  br label %{loop_label}")
        self.code_lines.append(f"{loop_label}:")
        self._set_current_block(loop_label)
        cond_val, cond_type = self._emit_expression(stmt.condition)
        if cond_type != "i1":
            cond_bool = self._new_local()
            if cond_type == "double":
                self.code_lines.append(
                    f"  {cond_bool} = fcmp one double {cond_val}, 0.0")
            else:
                self.code_lines.append(
                    f"  {cond_bool} = icmp ne {cond_type} {cond_val}, 0")
            cond_val = cond_bool
        self.code_lines.append(
            f"  br i1 {cond_val}, label %{loop_body_label}, label %{loop_end_label}")
        self.code_lines.append(f"{loop_body_label}:")
        self._set_current_block(loop_body_label)
        self.loop_exit_stack.append(loop_end_label)
        for body_stmt in stmt.body:
            self._emit_statement(body_stmt)
        self.loop_exit_stack.pop()
        self._emit_br_if_not_terminated(loop_label)
        self.code_lines.append(f"{loop_end_label}:")
        self._set_current_block(loop_end_label)


        for if_id_key in list(self.last_block_before_endif.keys()):
            self.last_block_before_endif[if_id_key] = loop_end_label

    def _emit_if_statement(self, stmt: IfStatement):
        if_id = self._new_block_id()
        then_label = f"if_then_{if_id}"
        endif_label = f"if_end_{if_id}"

        self.last_block_before_endif[str(if_id)] = then_label
        cond_val, cond_type = self._emit_expression(stmt.condition)
        if cond_type != "i1":
            cond_bool = self._new_local()
            if cond_type == "double":
                self.code_lines.append(
                    f"  {cond_bool} = fcmp one double {cond_val}, 0.0")
            else:
                self.code_lines.append(
                    f"  {cond_bool} = icmp ne {cond_type} {cond_val}, 0")
            cond_val = cond_bool

        before_phi_vars = {}
        for var_name in self.var_alloc:
            if var_name in self.var_ssa_versions:
                before_phi_vars[var_name] = self.var_ssa_versions[var_name]
            else:
                alloc_type, ptr_name = self.var_alloc[var_name]
                if not self._is_phi_trackable_var(var_name, alloc_type):
                    continue
                base_type = alloc_type if '[' not in alloc_type else alloc_type.split('[')[1].split(']')[0].split(' x ')[1]
                before_val = self._new_local()
                self.code_lines.append(
                    f"  {before_val} = load {base_type}, {base_type}* {ptr_name}")
                before_phi_vars[var_name] = before_val

        self.phi_tracking.append({})
        then_phi_vars = {}
        elif_phi_vars_list = []
        then_end_block = then_label
        else_end_block = None

        if stmt.elif_parts:
            elif_labels = []
            elif_end_blocks = []
            for i, (elif_cond, _) in enumerate(stmt.elif_parts):
                elif_labels.append(f"if_elif_{if_id}_{i}")
            else_label = f"if_else_{if_id}" if stmt.else_body else endif_label
            next_label = elif_labels[0] if elif_labels else (
                else_label if stmt.else_body else endif_label)
            self.code_lines.append(
                f"  br i1 {cond_val}, label %{then_label}, label %{next_label}")
            self.code_lines.append(f"{then_label}:")
            self._set_current_block(then_label)
            for s in stmt.then_body:
                self._emit_statement(s)
            then_phi_vars = self.phi_tracking[-1].copy()
            then_end_block = self.last_block_before_endif.get(str(if_id), then_label)
            self._emit_br_if_not_terminated(endif_label)

            for i, (elif_cond, elif_body) in enumerate(stmt.elif_parts):
                self.code_lines.append(f"{elif_labels[i]}:")
                elif_cond_val, elif_cond_type = self._emit_expression(
                    elif_cond)
                if elif_cond_type != "i1":
                    elif_cond_bool = self._new_local()
                    if elif_cond_type == "double":
                        self.code_lines.append(
                            f"  {elif_cond_bool} = fcmp one double {elif_cond_val}, 0.0")
                    else:
                        self.code_lines.append(
                            f"  {elif_cond_bool} = icmp ne {elif_cond_type} {elif_cond_val}, 0")
                    elif_cond_val = elif_cond_bool
                next_elif_label = elif_labels[i + 1] if i + \
                    1 < len(elif_labels) else else_label
                self.code_lines.append(
                    f"  br i1 {elif_cond_val}, label %if_elif_then_{if_id}_{i}, label %{next_elif_label}")
                self.code_lines.append(f"if_elif_then_{if_id}_{i}:")
                elif_then_label = f"if_elif_then_{if_id}_{i}"
                self._set_current_block(elif_then_label)
                self.last_block_before_endif[str(if_id)] = elif_then_label
                self.phi_tracking[-1] = {}
                for s in elif_body:
                    self._emit_statement(s)
                elif_phi_vars_list.append(self.phi_tracking[-1].copy())
                elif_end_blocks.append(self.last_block_before_endif.get(str(if_id), elif_then_label))
                self._emit_br_if_not_terminated(endif_label)

            else_phi_vars = {}
            else_end_block = else_label
            if stmt.else_body:
                self.code_lines.append(f"{else_label}:")
                self._set_current_block(else_label)
                self.last_block_before_endif[str(if_id)] = else_label
                self.phi_tracking[-1] = {}
                for s in stmt.else_body:
                    self._emit_statement(s)
                else_phi_vars = self.phi_tracking[-1].copy()
                else_end_block = self.last_block_before_endif.get(str(if_id), else_label)
                self._emit_br_if_not_terminated(endif_label)

            self.last_block_before_endif[str(if_id)] = else_end_block if else_end_block else then_end_block
        else:
            else_label = f"if_else_{if_id}" if stmt.else_body else endif_label
            self.code_lines.append(
                f"  br i1 {cond_val}, label %{then_label}, label %{else_label}")
            self.code_lines.append(f"{then_label}:")
            self._set_current_block(then_label)
            for s in stmt.then_body:
                self._emit_statement(s)
            then_phi_vars = self.phi_tracking[-1].copy()
            then_end_block = self.last_block_before_endif.get(str(if_id), then_label)
            self._emit_br_if_not_terminated(endif_label)

            else_phi_vars = {}
            else_end_block = else_label
            if stmt.else_body:
                self.code_lines.append(f"{else_label}:")
                self._set_current_block(else_label)
                self.last_block_before_endif[str(if_id)] = else_label
                self.phi_tracking[-1] = {}
                for s in stmt.else_body:
                    self._emit_statement(s)
                else_phi_vars = self.phi_tracking[-1].copy()
                else_end_block = self.last_block_before_endif.get(str(if_id), else_label)
                self._emit_br_if_not_terminated(endif_label)

            self.last_block_before_endif[str(if_id)] = else_end_block

        phi_instructions = []
        all_modified_vars = set(then_phi_vars.keys())
        for elif_vars in elif_phi_vars_list:
            all_modified_vars.update(elif_vars.keys())
        all_modified_vars.update(else_phi_vars.keys())

        for var_name in all_modified_vars:
            if var_name not in self.var_alloc:
                continue

            alloc_type, ptr_name = self.var_alloc[var_name]
            if not self._is_phi_trackable_var(var_name, alloc_type):
                continue
            base_type = alloc_type if '[' not in alloc_type else alloc_type.split('[')[1].split(']')[0].split(' x ')[1]

            phi_args = []

            if var_name in then_phi_vars:
                phi_args.append(f"[{then_phi_vars[var_name]}, %{then_end_block}]")
            else:
                phi_args.append(f"[{before_phi_vars[var_name]}, %{then_end_block}]")

            if stmt.elif_parts:
                for i, elif_vars in enumerate(elif_phi_vars_list):
                    elif_eb = elif_end_blocks[i] if i < len(elif_end_blocks) else f"if_elif_then_{if_id}_{i}"
                    if var_name in elif_vars:
                        phi_args.append(f"[{elif_vars[var_name]}, %{elif_eb}]")
                    else:
                        phi_args.append(f"[{before_phi_vars[var_name]}, %{elif_eb}]")
            else:
                pass

            if stmt.else_body:
                if var_name in else_phi_vars:
                    phi_args.append(f"[{else_phi_vars[var_name]}, %{else_end_block}]")
                else:
                    phi_args.append(f"[{before_phi_vars[var_name]}, %{else_end_block}]")

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
                if not self._is_phi_trackable_var(var_name, alloc_type):
                    continue
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
            if cond_type == "double":
                self.code_lines.append(
                    f"  {cond_bool} = fcmp one double {cond_val}, 0.0")
            else:
                self.code_lines.append(
                    f"  {cond_bool} = icmp ne {cond_type} {cond_val}, 0")
            cond_val = cond_bool

        prev_block = self.current_block

        before_phi_vars = {}
        for var_name in self.var_alloc:
            if var_name in self.var_ssa_versions:
                before_phi_vars[var_name] = self.var_ssa_versions[var_name]
            else:
                alloc_type, ptr_name = self.var_alloc[var_name]
                if not self._is_phi_trackable_var(var_name, alloc_type):
                    continue
                base_type = alloc_type if '[' not in alloc_type else alloc_type.split('[')[1].split(']')[0].split(' x ')[1]
                before_val = self._new_local()
                self.code_lines.append(
                    f"  {before_val} = load {base_type}, {base_type}* {ptr_name}")
                before_phi_vars[var_name] = before_val

        self.phi_tracking.append({})

        self.code_lines.append(
            f"  br i1 {cond_val}, label %{then_label}, label %{endif_label}")
        self.code_lines.append(f"{then_label}:")
        self._set_current_block(then_label)
        self._emit_statement(stmt.statement)

        then_phi_vars = self.phi_tracking[-1].copy()
        self._emit_br_if_not_terminated(endif_label)

        phi_instructions = []
        for var_name in then_phi_vars:
            if var_name not in self.var_alloc:
                continue

            alloc_type, ptr_name = self.var_alloc[var_name]
            if not self._is_phi_trackable_var(var_name, alloc_type):
                continue
            base_type = alloc_type if '[' not in alloc_type else alloc_type.split('[')[1].split(']')[0].split(' x ')[1]

            before_val = before_phi_vars[var_name]
            phi_result = self._new_local()
            phi_args_str = f"[{before_val}, %{prev_block}], [{then_phi_vars[var_name]}, %{then_label}]"
            phi_instructions.append(
                f"  {phi_result} = phi {base_type} {phi_args_str}")
            self.var_ssa_versions[var_name] = phi_result

        self.code_lines.append(f"{endif_label}:")
        self._set_current_block(endif_label)
        for phi_line in phi_instructions:
            self.code_lines.append(phi_line)

        for var_name in then_phi_vars:
            if var_name in self.var_ssa_versions:
                alloc_type, ptr_name = self.var_alloc[var_name]
                if not self._is_phi_trackable_var(var_name, alloc_type):
                    continue
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

                    if '[' in alloc_type:

                        array_ptr = self._new_local()
                        self.code_lines.append(
                            f"  {array_ptr} = getelementptr inbounds {alloc_type}, {alloc_type}* {ptr_name}, i64 0, i64 0"
                        )
                        args.append(f"{base_type}* {array_ptr}")
                    else:

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
            return (self._format_double(expr.value), "double")
        elif isinstance(expr, ComplexLiteral):
            complex_val1 = self._new_local()
            self.code_lines.append(f"  {complex_val1} = insertvalue {{double, double}} undef, double {self._format_double(expr.real_part)}, 0")
            complex_val2 = self._new_local()
            self.code_lines.append(f"  {complex_val2} = insertvalue {{double, double}} {complex_val1}, double {self._format_double(expr.imag_part)}, 1")
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

            builtin_functions = {"SIN", "COS", "TAN", "ASIN", "ACOS", "ATAN",
                                "EXP", "LOG", "LOG10", "SQRT", "ABS", "INT", "FLOAT",
                                "REAL", "MOD", "MIN", "MAX", "POW"}
            if array_name.upper() in builtin_functions:
                func_call = FunctionCall(name=array_name.upper(), args=expr.indices, line=expr.line, col=expr.col)
                return self._emit_function_call(func_call)

            if array_name.upper() in self.user_functions:
                func_call = FunctionCall(name=array_name.upper(), args=expr.indices, line=expr.line, col=expr.col)
                return self._emit_function_call(func_call)

            indices = [self._emit_expression(idx)[0] for idx in expr.indices]
            if array_name in self.var_alloc:
                alloc_type, ptr = self.var_alloc[array_name]
                base_type = alloc_type if '[' not in alloc_type else alloc_type.split(
                    '[')[1].split(']')[0].split(' x ')[1]

                linear_idx = self._compute_linear_index(indices, array_name)
                if '[' not in alloc_type:
                    elem_ptr = self._new_local()
                    self.code_lines.append(
                        f"  {elem_ptr} = getelementptr inbounds {base_type}, "
                        f"{base_type}* {ptr}, i32 {linear_idx}")
                else:
                    elem_ptr = self._new_local()
                    self.code_lines.append(
                        f"  {elem_ptr} = getelementptr inbounds {alloc_type}, "
                        f"{alloc_type}* {ptr}, i64 0, i32 {linear_idx}")
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
        cplx = "{double, double}"
        if expr.op == "+":
            if left_type == cplx or right_type == cplx:
                if left_type != cplx:
                    left_val = self._make_complex(left_val if left_type == "double" else self._convert_to_double(left_val), "0.0")
                if right_type != cplx:
                    right_val = self._make_complex(right_val if right_type == "double" else self._convert_to_double(right_val), "0.0")
                return (self._complex_add(left_val, right_val), cplx)
            elif left_type == "double" or right_type == "double":
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
            if left_type == cplx or right_type == cplx:
                if left_type != cplx:
                    left_val = self._make_complex(left_val if left_type == "double" else self._convert_to_double(left_val), "0.0")
                if right_type != cplx:
                    right_val = self._make_complex(right_val if right_type == "double" else self._convert_to_double(right_val), "0.0")
                return (self._complex_sub(left_val, right_val), cplx)
            elif left_type == "double" or right_type == "double":
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
            if left_type == cplx or right_type == cplx:
                if left_type != cplx:
                    left_val = self._make_complex(left_val if left_type == "double" else self._convert_to_double(left_val), "0.0")
                if right_type != cplx:
                    right_val = self._make_complex(right_val if right_type == "double" else self._convert_to_double(right_val), "0.0")
                return (self._complex_mul(left_val, right_val), cplx)
            elif left_type == "double" or right_type == "double":
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
            if left_type == cplx or right_type == cplx:
                if left_type != cplx:
                    left_val = self._make_complex(left_val if left_type == "double" else self._convert_to_double(left_val), "0.0")
                if right_type != cplx:
                    right_val = self._make_complex(right_val if right_type == "double" else self._convert_to_double(right_val), "0.0")
                return (self._complex_div(left_val, right_val), cplx)
            elif left_type == "double" or right_type == "double":
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
            if left_type == "double" or right_type == "double":
                if left_type != "double":
                    left_val = self._convert_to_double(left_val)
                if right_type != "double":
                    right_val = self._convert_to_double(right_val)
                self.code_lines.append(
                    f"  {result} = fcmp oeq double {left_val}, {right_val}")
            else:
                self.code_lines.append(
                    f"  {result} = icmp eq {left_type} {left_val}, {right_val}")
            return (result, "i1")
        elif expr.op in {".NE.", "/="}:
            if left_type == "double" or right_type == "double":
                if left_type != "double":
                    left_val = self._convert_to_double(left_val)
                if right_type != "double":
                    right_val = self._convert_to_double(right_val)
                self.code_lines.append(
                    f"  {result} = fcmp one double {left_val}, {right_val}")
            else:
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
            if val_type == "{double, double}":
                re_part, im_part = self._extract_complex_parts(val)
                neg_re = self._new_local()
                self.code_lines.append(f"  {neg_re} = fneg double {re_part}")
                neg_im = self._new_local()
                self.code_lines.append(f"  {neg_im} = fneg double {im_part}")
                return (self._make_complex(neg_re, neg_im), val_type)
            elif val_type == "double":
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
            if func_name in self.user_functions:
                ret_type, param_types = self.user_functions[func_name]
                arg_parts = []
                for i, arg in enumerate(expr.args):
                    expected_type = param_types[i] if i < len(param_types) else 'i32'
                    arg_val, arg_type = self._emit_expression(arg)
                    if expected_type == 'double' and arg_type == 'i32':
                        arg_val = self._convert_to_double(arg_val)
                        arg_type = 'double'
                    elif expected_type == 'i32' and arg_type == 'double':
                        conv = self._new_local()
                        self.code_lines.append(f"  {conv} = fptosi double {arg_val} to i32")
                        arg_val = conv
                        arg_type = 'i32'
                    tmp_ptr = self._new_local()
                    self.code_lines.append(f"  {tmp_ptr} = alloca {expected_type}")
                    self.code_lines.append(f"  store {expected_type} {arg_val}, {expected_type}* {tmp_ptr}")
                    arg_parts.append(f"{expected_type}* {tmp_ptr}")
                args_str = ", ".join(arg_parts)
                self.code_lines.append(
                    f"  {result} = call {ret_type} @{func_name}({args_str})")
                return (result, ret_type)
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

    def _extract_complex_parts(self, val: str) -> Tuple[str, str]:
        re_part = self._new_local()
        self.code_lines.append(f"  {re_part} = extractvalue {{double, double}} {val}, 0")
        im_part = self._new_local()
        self.code_lines.append(f"  {im_part} = extractvalue {{double, double}} {val}, 1")
        return (re_part, im_part)

    def _make_complex(self, re_val: str, im_val: str) -> str:
        tmp = self._new_local()
        self.code_lines.append(f"  {tmp} = insertvalue {{double, double}} undef, double {re_val}, 0")
        result = self._new_local()
        self.code_lines.append(f"  {result} = insertvalue {{double, double}} {tmp}, double {im_val}, 1")
        return result

    def _complex_add(self, left: str, right: str) -> str:
        a, b = self._extract_complex_parts(left)
        c, d = self._extract_complex_parts(right)
        re = self._new_local()
        self.code_lines.append(f"  {re} = fadd double {a}, {c}")
        im = self._new_local()
        self.code_lines.append(f"  {im} = fadd double {b}, {d}")
        return self._make_complex(re, im)

    def _complex_sub(self, left: str, right: str) -> str:
        a, b = self._extract_complex_parts(left)
        c, d = self._extract_complex_parts(right)
        re = self._new_local()
        self.code_lines.append(f"  {re} = fsub double {a}, {c}")
        im = self._new_local()
        self.code_lines.append(f"  {im} = fsub double {b}, {d}")
        return self._make_complex(re, im)

    def _complex_mul(self, left: str, right: str) -> str:
        a, b = self._extract_complex_parts(left)
        c, d = self._extract_complex_parts(right)
        ac = self._new_local()
        self.code_lines.append(f"  {ac} = fmul double {a}, {c}")
        bd = self._new_local()
        self.code_lines.append(f"  {bd} = fmul double {b}, {d}")
        ad = self._new_local()
        self.code_lines.append(f"  {ad} = fmul double {a}, {d}")
        bc = self._new_local()
        self.code_lines.append(f"  {bc} = fmul double {b}, {c}")
        re = self._new_local()
        self.code_lines.append(f"  {re} = fsub double {ac}, {bd}")
        im = self._new_local()
        self.code_lines.append(f"  {im} = fadd double {ad}, {bc}")
        return self._make_complex(re, im)

    def _complex_div(self, left: str, right: str) -> str:
        a, b = self._extract_complex_parts(left)
        c, d = self._extract_complex_parts(right)
        cc = self._new_local()
        self.code_lines.append(f"  {cc} = fmul double {c}, {c}")
        dd = self._new_local()
        self.code_lines.append(f"  {dd} = fmul double {d}, {d}")
        denom = self._new_local()
        self.code_lines.append(f"  {denom} = fadd double {cc}, {dd}")
        ac = self._new_local()
        self.code_lines.append(f"  {ac} = fmul double {a}, {c}")
        bd = self._new_local()
        self.code_lines.append(f"  {bd} = fmul double {b}, {d}")
        bc = self._new_local()
        self.code_lines.append(f"  {bc} = fmul double {b}, {c}")
        ad = self._new_local()
        self.code_lines.append(f"  {ad} = fmul double {a}, {d}")
        re_num = self._new_local()
        self.code_lines.append(f"  {re_num} = fadd double {ac}, {bd}")
        im_num = self._new_local()
        self.code_lines.append(f"  {im_num} = fsub double {bc}, {ad}")
        re = self._new_local()
        self.code_lines.append(f"  {re} = fdiv double {re_num}, {denom}")
        im = self._new_local()
        self.code_lines.append(f"  {im} = fdiv double {im_num}, {denom}")
        return self._make_complex(re, im)

    def _compute_linear_index(self, indices_vals, array_name):
        array_dims = self.array_dimensions.get(array_name.upper(), [])
        array_bounds = self.array_bounds.get(array_name.upper(), [])
        if len(indices_vals) == 1:
            adjusted = self._new_local()
            lower_bound = array_bounds[0][0] if array_bounds else 1
            self.code_lines.append(f"  {adjusted} = sub i32 {indices_vals[0]}, {lower_bound}")
            return adjusted
        if not array_dims:
            array_dims = [8] * len(indices_vals)
        while len(array_dims) < len(indices_vals):
            array_dims.append(8)
        while len(array_bounds) < len(indices_vals):
            array_bounds.append((1, array_dims[len(array_bounds)]))
        adj_first = self._new_local()
        self.code_lines.append(f"  {adj_first} = sub i32 {indices_vals[0]}, {array_bounds[0][0]}")
        linear = adj_first
        stride = array_dims[0]
        for k in range(1, len(indices_vals)):
            adj_k = self._new_local()
            self.code_lines.append(f"  {adj_k} = sub i32 {indices_vals[k]}, {array_bounds[k][0]}")
            mul_tmp = self._new_local()
            self.code_lines.append(f"  {mul_tmp} = mul i32 {adj_k}, {stride}")
            next_linear = self._new_local()
            self.code_lines.append(f"  {next_linear} = add i32 {linear}, {mul_tmp}")
            linear = next_linear
            stride *= array_dims[k]
        return linear

    def _convert_to_bool(self, val: str, val_type: str) -> str:
        if val_type == "i1":
            return val
        result = self._new_local()
        if val_type == "double":
            self.code_lines.append(f"  {result} = fcmp one double {val}, 0.0")
        else:
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

    def _is_block_terminated(self) -> bool:
        for i in range(len(self.code_lines) - 1, -1, -1):
            line = self.code_lines[i].strip()
            if not line:
                continue
            if line.endswith(':'):
                return False
            if line.startswith('br ') or line.startswith('ret '):
                return True
            return False
        return False

    def _emit_br_if_not_terminated(self, target_label: str):
        if not self._is_block_terminated():
            self.code_lines.append(f"  br label %{target_label}")

    def _set_current_block(self, block_name: str):
        self.current_block = block_name

    @staticmethod
    def _format_double(value) -> str:
        s = str(value)
        if 'e' in s.lower() or 'E' in s:
            s = f"{value:.15f}"
        if '.' not in s:
            s += '.0'
        return s

    def _emit_return_statement(self, stmt: ReturnStatement):
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

    def _emit_arithmetic_if(self, stmt: ArithmeticIfStatement):
        val, val_type = self._emit_expression(stmt.condition)
        if val_type == "double":
            cmp_neg = self._new_local()
            self.code_lines.append(f"  {cmp_neg} = fcmp olt double {val}, 0.0")
            cmp_zero = self._new_local()
            self.code_lines.append(f"  {cmp_zero} = fcmp oeq double {val}, 0.0")
        else:
            cmp_neg = self._new_local()
            self.code_lines.append(f"  {cmp_neg} = icmp slt i32 {val}, 0")
            cmp_zero = self._new_local()
            self.code_lines.append(f"  {cmp_zero} = icmp eq i32 {val}, 0")
        arif_id = self._new_block_id()
        check_zero_label = f"arif_checkzero_{arif_id}"
        self.code_lines.append(f"  br i1 {cmp_neg}, label %label_{stmt.label_neg}, label %{check_zero_label}")
        self.code_lines.append(f"{check_zero_label}:")
        self.code_lines.append(f"  br i1 {cmp_zero}, label %label_{stmt.label_zero}, label %label_{stmt.label_pos}")

    def _emit_goto_statement(self, stmt: GotoStatement):
        label_name = f"label_{stmt.label}"
        self.code_lines.append(f"  br label %{label_name}")

    def _emit_continue_statement(self, stmt: ContinueStatement):
        if stmt.label:
            label_name = f"label_{stmt.label}"
            self._emit_br_if_not_terminated(label_name)
            self.code_lines.append(f"{label_name}:")
            self._set_current_block(label_name)

    def _emit_exit_statement(self):
        if self.loop_exit_stack:
            self.code_lines.append(f"  br label %{self.loop_exit_stack[-1]}")

    def _emit_subroutine(self, sub: Subroutine):
        old_function = self.current_function
        old_var_alloc = self.var_alloc.copy()
        old_local_counter = self.local_counter
        old_block_counter = self.block_counter
        old_array_dimensions = self.array_dimensions.copy()
        old_array_bounds = self.array_bounds.copy()

        self.current_function = sub.name.upper()
        self.var_alloc = {}
        self.local_counter = 0
        self.block_counter = 0
        self.var_ssa_versions = {}
        self.phi_tracking = []
        self.current_subroutine_params = sub.params
        scope_info = self._collect_scope_declaration_info(sub.declarations)

        param_list = []
        param_types = {}
        for param in sub.params:
            param_upper = param.upper()
            param_type = scope_info["declared_types"].get(param_upper, "i32")
            param_list.append(f"{param_type}* %param_{param}")
            param_types[param_upper] = param_type

        self.code_lines.append(f"; === Подпрограмма {sub.name} ===")
        self.code_lines.append(f"define void @{sub.name.upper()}({', '.join(param_list)}) #0 {{")
        self.code_lines.append("entry:")
        self._set_current_block("entry")

        for param in sub.params:
            param_upper = param.upper()
            param_type = param_types[param_upper]
            self.var_alloc[param_upper] = (param_type, f"%param_{param}")
            if param_upper in scope_info["declared_dims"]:
                dims = scope_info["declared_dims"][param_upper]
                _, dim_sizes = self._compute_array_layout(dims)
                self.array_bounds[param_upper] = dims
                self.array_dimensions[param_upper] = dim_sizes

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
        self.current_subroutine_params = []
        self.array_dimensions = old_array_dimensions
        self.array_bounds = old_array_bounds

    def _emit_function(self, func: FunctionDef):
        old_function = self.current_function
        old_var_alloc = self.var_alloc.copy()
        old_local_counter = self.local_counter
        old_block_counter = self.block_counter
        old_array_dimensions = self.array_dimensions.copy()
        old_array_bounds = self.array_bounds.copy()

        func_name_upper = func.name.upper()
        self.current_function = func_name_upper
        self.var_alloc = {}
        self.local_counter = 0
        self.block_counter = 0
        self.var_ssa_versions = {}
        self.phi_tracking = []
        scope_info = self._collect_scope_declaration_info(func.declarations)

        return_type = "i32"
        if func.return_type:
            type_map = {'INTEGER': 'i32', 'REAL': 'double', 'LOGICAL': 'i1'}
            return_type = type_map.get(func.return_type.upper(), 'i32')

        param_list = []
        param_types = {}
        for param in func.params:
            param_type = scope_info["declared_types"].get(param.upper(), "i32")
            param_list.append(f"{param_type}* %param_{param}")
            param_types[param.upper()] = param_type

        self.code_lines.append(f"; === Функция {func.name} ===")
        self.code_lines.append(f"define {return_type} @{func_name_upper}({', '.join(param_list)}) #0 {{")
        self.code_lines.append("entry:")
        self._set_current_block("entry")

        result_ptr = f"%{func_name_upper}"
        self.code_lines.append(f"  {result_ptr} = alloca {return_type}")
        self.var_alloc[func_name_upper] = (return_type, result_ptr)

        for param in func.params:
            param_upper = param.upper()
            param_type = param_types[param_upper]
            self.var_alloc[param_upper] = (param_type, f"%param_{param}")
            if param_upper in scope_info["declared_dims"]:
                dims = scope_info["declared_dims"][param_upper]
                _, dim_sizes = self._compute_array_layout(dims)
                self.array_bounds[param_upper] = dims
                self.array_dimensions[param_upper] = dim_sizes

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
        self.array_dimensions = old_array_dimensions
        self.array_bounds = old_array_bounds
