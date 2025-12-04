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
        # Для отслеживания переменных, измененных в разных ветках
        self.branch_var_versions: List[Dict[str, int]] = []
        
    def generate(self, ast: Program) -> List[str]:
        self.instructions = []
        self.var_versions = {}
        self.temp_counter = 0
        self.var_types = {}
        self.branch_var_versions = []
        
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
        
        # Сохраняем состояние до цикла
        before_loop_versions = self.var_versions.copy()
        
        # Инициализация переменной цикла
        self.var_versions[var] += 1
        init_version = self.var_versions[var]
        
        self.instructions.append(f"do {var}_{init_version} = {start} {end} {step}")
        
        # Обрабатываем тело цикла
        for body_stmt in stmt.body:
            self._process_statement(body_stmt)
        
        # Состояние после тела цикла
        after_body_versions = self.var_versions.copy()
        
        self.instructions.append("end do")
        
        # Создаем phi-функции для переменных, измененных в цикле
        # Для переменной цикла: phi(начальное значение, значение после итерации)
        if var in after_body_versions and after_body_versions[var] > init_version:
            self.var_versions[var] = after_body_versions[var] + 1
            new_version = self.var_versions[var]
            init_val = f"{var}_{init_version}"
            body_val = f"{var}_{after_body_versions[var]}"
            self.instructions.append(f"{var}_{new_version} = phi {init_val} {body_val}")
        
        # Для других переменных, измененных в цикле
        modified_vars = set()
        for v in after_body_versions:
            if v != var and after_body_versions[v] > before_loop_versions.get(v, 0):
                modified_vars.add(v)
        
        for v in modified_vars:
            before_val = f"{v}_{before_loop_versions.get(v, 0)}" if before_loop_versions.get(v, 0) > 0 else v
            after_val = f"{v}_{after_body_versions[v]}"
            self.var_versions[v] = after_body_versions[v] + 1
            new_version = self.var_versions[v]
            self.instructions.append(f"{v}_{new_version} = phi {before_val} {after_val}")
    
    def _process_do_while(self, stmt: DoWhile):
        condition = self._process_expression(stmt.condition)
        self.instructions.append(f"do while {condition}")
        for body_stmt in stmt.body:
            self._process_statement(body_stmt)
        self.instructions.append("end do")
    
    def _process_if_statement(self, stmt: IfStatement):
        condition = self._process_expression(stmt.condition)
        
        # Сохраняем состояние переменных до ветвления
        before_versions = self.var_versions.copy()
        
        self.instructions.append(f"if {condition} then")
        
        # Обрабатываем then-ветку
        then_versions = self.var_versions.copy()
        for then_stmt in stmt.then_body:
            self._process_statement(then_stmt)
        then_versions = self.var_versions.copy()
        
        # Восстанавливаем состояние для elif-веток
        self.var_versions = before_versions.copy()
        elif_versions_list = []
        
        for elif_cond, elif_body in stmt.elif_parts:
            elif_cond_str = self._process_expression(elif_cond)
            self.instructions.append(f"elseif {elif_cond_str} then")
            for elif_stmt in elif_body:
                self._process_statement(elif_stmt)
            elif_versions_list.append(self.var_versions.copy())
            self.var_versions = before_versions.copy()
        
        # Обрабатываем else-ветку
        else_versions = None
        if stmt.else_body:
            self.instructions.append("else")
            for else_stmt in stmt.else_body:
                self._process_statement(else_stmt)
            else_versions = self.var_versions.copy()
        
        self.instructions.append("end if")
        
        # Создаем phi-функции для переменных, измененных в разных ветках
        all_modified_vars = set()
        all_modified_vars.update(then_versions.keys())
        for elif_vers in elif_versions_list:
            all_modified_vars.update(elif_vers.keys())
        if else_versions:
            all_modified_vars.update(else_versions.keys())
        
        # Находим переменные, которые были изменены хотя бы в одной ветке
        modified_vars = set()
        for var in all_modified_vars:
            if var in then_versions and then_versions[var] > before_versions.get(var, 0):
                modified_vars.add(var)
            for elif_vers in elif_versions_list:
                if var in elif_vers and elif_vers[var] > before_versions.get(var, 0):
                    modified_vars.add(var)
            if else_versions and var in else_versions and else_versions[var] > before_versions.get(var, 0):
                modified_vars.add(var)
        
        # Создаем phi-функции
        merged_versions = {}
        for var in modified_vars:
            phi_args = []
            
            # Добавляем значение из then-ветки
            if var in then_versions:
                then_ver = then_versions[var]
                if then_ver > before_versions.get(var, 0):
                    phi_args.append(f"{var}_{then_ver}")
                else:
                    phi_args.append(f"{var}_{before_versions.get(var, 0)}" if before_versions.get(var, 0) > 0 else var)
            else:
                phi_args.append(f"{var}_{before_versions.get(var, 0)}" if before_versions.get(var, 0) > 0 else var)
            
            # Добавляем значения из elif-веток
            for elif_vers in elif_versions_list:
                if var in elif_vers:
                    elif_ver = elif_vers[var]
                    if elif_ver > before_versions.get(var, 0):
                        phi_args.append(f"{var}_{elif_ver}")
                    else:
                        phi_args.append(f"{var}_{before_versions.get(var, 0)}" if before_versions.get(var, 0) > 0 else var)
                else:
                    phi_args.append(f"{var}_{before_versions.get(var, 0)}" if before_versions.get(var, 0) > 0 else var)
            
            # Добавляем значение из else-ветки
            if else_versions:
                if var in else_versions:
                    else_ver = else_versions[var]
                    if else_ver > before_versions.get(var, 0):
                        phi_args.append(f"{var}_{else_ver}")
                    else:
                        phi_args.append(f"{var}_{before_versions.get(var, 0)}" if before_versions.get(var, 0) > 0 else var)
                else:
                    phi_args.append(f"{var}_{before_versions.get(var, 0)}" if before_versions.get(var, 0) > 0 else var)
            
            # Создаем phi-функцию
            if len(phi_args) > 1 and len(set(phi_args)) > 1:
                # Только если есть разные версии
                self.var_versions[var] = max(then_versions.get(var, 0), 
                                            max([ev.get(var, 0) for ev in elif_versions_list], default=0),
                                            else_versions.get(var, 0) if else_versions else 0) + 1
                new_version = self.var_versions[var]
                phi_args_str = " ".join(phi_args)
                self.instructions.append(f"{var}_{new_version} = phi {phi_args_str}")
                merged_versions[var] = new_version
            else:
                # Если все версии одинаковые, просто используем последнюю
                if phi_args:
                    last_ver = phi_args[-1]
                    if "_" in last_ver:
                        ver_num = int(last_ver.split("_")[-1])
                        self.var_versions[var] = ver_num
                    else:
                        self.var_versions[var] = before_versions.get(var, 0)
        
        # Обновляем версии для переменных, которые не были изменены, но могли использоваться
        for var in before_versions:
            if var not in merged_versions:
                max_ver = max(
                    then_versions.get(var, before_versions[var]),
                    max([ev.get(var, before_versions[var]) for ev in elif_versions_list], default=before_versions[var]),
                    else_versions.get(var, before_versions[var]) if else_versions else before_versions[var]
                )
                self.var_versions[var] = max_ver
    
    def _process_simple_if(self, stmt: SimpleIfStatement):
        condition = self._process_expression(stmt.condition)
        
        # Сохраняем состояние переменных до ветвления
        before_versions = self.var_versions.copy()
        
        self.instructions.append(f"if {condition} then")
        
        # Обрабатываем then-ветку
        for then_stmt in [stmt.statement]:
            self._process_statement(then_stmt)
        then_versions = self.var_versions.copy()
        
        self.instructions.append("end if")
        
        # Создаем phi-функции для переменных, измененных в then-ветке
        modified_vars = set()
        for var in then_versions:
            if then_versions[var] > before_versions.get(var, 0):
                modified_vars.add(var)
        
        # Создаем phi-функции (объединяем значение из then и исходное значение)
        for var in modified_vars:
            then_ver = then_versions[var]
            before_ver = before_versions.get(var, 0)
            
            if then_ver > before_ver:
                self.var_versions[var] = then_ver + 1
                new_version = self.var_versions[var]
                before_val = f"{var}_{before_ver}" if before_ver > 0 else var
                then_val = f"{var}_{then_ver}"
                self.instructions.append(f"{var}_{new_version} = phi {before_val} {then_val}")
            else:
                self.var_versions[var] = then_ver
    
    def to_string(self, instructions: List[str]) -> str:
        return "\n".join(instructions)

