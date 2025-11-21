from typing import Dict, Set, Tuple, Optional, List
from enum import Enum
from dataclasses import dataclass
from src.core import (
    ASTNode, Program, Declaration, Statement, Assignment,
    DoLoop, LabeledDoLoop, IfStatement, SimpleIfStatement, PrintStatement, ReadStatement, WriteStatement,
    CallStatement, BinaryOp, UnaryOp, Variable, IntegerLiteral,
    RealLiteral, StringLiteral, LogicalLiteral, FunctionCall, Expression,
    DoWhile, LabeledDoWhile, ArrayRef, DimensionStatement, ParameterStatement,
    DataStatement, DataItem, ArithmeticIfStatement, ImplicitNone, ImplicitStatement, ImplicitRule
)
class TypeKind(Enum):
    INTEGER = "INTEGER"
    REAL = "REAL"
    LOGICAL = "LOGICAL"
    CHARACTER = "CHARACTER"
    COMPLEX = "COMPLEX"
    UNKNOWN = "UNKNOWN"
@dataclass
class VariableInfo:
    name: str
    type_kind: TypeKind
    is_array: bool = False
    dimensions: List[Tuple[int, int]] = None
    is_parameter: bool = False
    value: Optional[object] = None
    def __post_init__(self):
        if self.dimensions is None:
            self.dimensions = []
    def get_dimension_size(self, dim_index: int) -> int:
        if dim_index < len(self.dimensions):
            k, l = self.dimensions[dim_index]
            return l - k + 1
        return 0
class SemanticAnalyzer:
    def __init__(self):
        self.symbol_table: Dict[str, VariableInfo] = {}
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.current_scope = "global"
        self.implicit_none = False
        self.implicit_rules: Dict[str, Tuple[TypeKind, Optional[int]]] = {}
        self.builtin_functions = {
            "REAL": (TypeKind.REAL, [TypeKind.INTEGER, TypeKind.REAL, TypeKind.LOGICAL]),
            "INT": (TypeKind.INTEGER, [TypeKind.REAL, TypeKind.INTEGER, TypeKind.LOGICAL]),
            "FLOAT": (TypeKind.REAL, [TypeKind.INTEGER, TypeKind.REAL, TypeKind.LOGICAL]),
            "ABS": (None, []),
            "SQRT": (TypeKind.REAL, [TypeKind.REAL, TypeKind.INTEGER]),
            "SIN": (TypeKind.REAL, [TypeKind.REAL, TypeKind.INTEGER]),
            "COS": (TypeKind.REAL, [TypeKind.REAL, TypeKind.INTEGER]),
            "TAN": (TypeKind.REAL, [TypeKind.REAL, TypeKind.INTEGER]),
            "ASIN": (TypeKind.REAL, [TypeKind.REAL, TypeKind.INTEGER]),
            "ACOS": (TypeKind.REAL, [TypeKind.REAL, TypeKind.INTEGER]),
            "ATAN": (TypeKind.REAL, [TypeKind.REAL, TypeKind.INTEGER]),
            "EXP": (TypeKind.REAL, [TypeKind.REAL, TypeKind.INTEGER]),
            "LOG": (TypeKind.REAL, [TypeKind.REAL, TypeKind.INTEGER]),
            "LOG10": (TypeKind.REAL, [TypeKind.REAL, TypeKind.INTEGER]),
            "MOD": (None, []),
            "MIN": (None, []),
            "MAX": (None, []),
        }
    def analyze(self, ast: Program) -> bool:
        try:
            self._analyze_declarations(ast.declarations)
            self._analyze_statements(ast.statements)
            return len(self.errors) == 0
        except Exception as e:
            self.errors.append(f"Внутренняя ошибка: {e}")
            return False
    def _get_implicit_type(self, name: str) -> Tuple[TypeKind, Optional[int]]:
        if not name:
            return (TypeKind.UNKNOWN, None)
        first_char = name[0].upper()
        if first_char in self.implicit_rules:
            return self.implicit_rules[first_char]
        if not self.implicit_none:
            if 'I' <= first_char <= 'M':
                return (TypeKind.INTEGER, None)
            else:
                return (TypeKind.REAL, None)
        return (TypeKind.UNKNOWN, None)
    def _analyze_declarations(self, declarations: List[ASTNode]):
        for decl in declarations:
            if isinstance(decl, ImplicitNone):
                self.implicit_none = True
            elif isinstance(decl, ImplicitStatement):
                for rule in decl.rules:
                    letters = rule.get_letters()
                    type_kind = TypeKind[rule.type_name] if rule.type_name in TypeKind.__members__ else TypeKind.UNKNOWN
                    for letter in letters:
                        if letter in self.implicit_rules:
                            self.warnings.append(
                                f"Правило IMPLICIT для буквы '{letter}' переопределено: "
                                f"{self.implicit_rules[letter][0].value} -> {type_kind.value}"
                            )
                        self.implicit_rules[letter] = (type_kind, rule.type_size)
            elif isinstance(decl, Declaration):
                type_kind = TypeKind[decl.type] if decl.type in TypeKind.__members__ else TypeKind.UNKNOWN
                for name, dims in decl.names:
                    if name in self.symbol_table:
                        var_info = self.symbol_table[name]
                        if var_info.is_parameter:
                            self.errors.append(
                                f"Переменная '{name}' уже объявлена как PARAMETER, повторное объявление типа {decl.type} недопустимо"
                            )
                        else:
                            self.errors.append(
                                f"Переменная '{name}' уже объявлена, повторное объявление типа {decl.type} недопустимо"
                            )
                    else:
                        var_info = VariableInfo(
                            name=name,
                            type_kind=type_kind,
                            is_array=dims is not None,
                            dimensions=dims or []
                        )
                        self.symbol_table[name] = var_info
            elif isinstance(decl, DimensionStatement):
                self._analyze_dimension_statement(decl)
            elif isinstance(decl, ParameterStatement):
                self._analyze_parameter_statement(decl)
            elif isinstance(decl, DataStatement):
                self._analyze_data_statement(decl)
    def _analyze_statements(self, statements: List[Statement]):
        for stmt in statements:
            if isinstance(stmt, Assignment):
                self._analyze_assignment(stmt)
            elif isinstance(stmt, DoLoop):
                self._analyze_do_loop(stmt)
            elif isinstance(stmt, LabeledDoLoop):
                self._analyze_labeled_do_loop(stmt)
            elif isinstance(stmt, (DoWhile, LabeledDoWhile)):
                self._analyze_do_while(stmt)
            elif isinstance(stmt, SimpleIfStatement):
                self._analyze_simple_if_statement(stmt)
            elif isinstance(stmt, IfStatement):
                self._analyze_if_statement(stmt)
            elif isinstance(stmt, ArithmeticIfStatement):
                self._analyze_arithmetic_if_statement(stmt)
            elif isinstance(stmt, PrintStatement):
                self._analyze_print_statement(stmt)
            elif isinstance(stmt, (ReadStatement, WriteStatement)):
                self._analyze_io_statement(stmt)
            elif isinstance(stmt, CallStatement):
                self._analyze_call_statement(stmt)
            elif isinstance(stmt, DataStatement):
                self._analyze_data_statement(stmt)
    def _analyze_assignment(self, stmt: Assignment):
        if stmt.target not in self.symbol_table:
            if not self.implicit_none:
                implicit_type, type_size = self._get_implicit_type(stmt.target)
                if implicit_type != TypeKind.UNKNOWN:
                    var_info = VariableInfo(
                        name=stmt.target,
                        type_kind=implicit_type,
                        is_array=False,
                        dimensions=[]
                    )
                    self.symbol_table[stmt.target] = var_info
                    rule_source = "явным правилом IMPLICIT" if stmt.target[0].upper() in self.implicit_rules else "правилом I-M"
                    self.warnings.append(
                        f"Переменная '{stmt.target}' не объявлена, используется неявный тип {implicit_type.value} "
                        f"по {rule_source}"
                    )
                else:
                    self.errors.append(
                        f"Переменная '{stmt.target}' не объявлена и не может быть определена неявно"
                    )
                    return
            else:
                self.errors.append(
                    f"Переменная '{stmt.target}' не объявлена (IMPLICIT NONE установлен)"
                )
                return
        var_info = self.symbol_table[stmt.target]
        if var_info.is_parameter:
            self.errors.append(
                f"Попытка изменить константу PARAMETER '{stmt.target}'"
            )
        if stmt.indices:
            if not var_info.is_array:
                self.errors.append(
                    f"'{stmt.target}' не является массивом"
                )
            elif len(stmt.indices) != len(var_info.dimensions):
                self.errors.append(
                    f"Неверное количество индексов для '{stmt.target}': "
                    f"ожидается {len(var_info.dimensions)}, получено {len(stmt.indices)}"
                )
            else:
                for idx in stmt.indices:
                    idx_type = self._infer_expression_type(idx)
                    if idx_type != TypeKind.INTEGER and idx_type != TypeKind.UNKNOWN:
                        self.errors.append(
                            f"Индекс массива должен быть INTEGER, получено {idx_type.value}"
                        )
        expr_type = self._infer_expression_type(stmt.value)
        if expr_type != TypeKind.UNKNOWN and var_info.type_kind != TypeKind.UNKNOWN:
            if not self._are_compatible(var_info.type_kind, expr_type):
                self.warnings.append(
                    f"Неявное преобразование типа при присваивании '{stmt.target}': "
                    f"{expr_type.value} -> {var_info.type_kind.value}"
                )
    def _analyze_do_loop(self, stmt: DoLoop):
        if stmt.var not in self.symbol_table:
            self.errors.append(
                f"Переменная цикла '{stmt.var}' не объявлена"
            )
        else:
            var_info = self.symbol_table[stmt.var]
            if var_info.type_kind != TypeKind.INTEGER:
                self.errors.append(
                    f"Переменная цикла '{stmt.var}' должна быть INTEGER"
                )
        start_type = self._infer_expression_type(stmt.start)
        end_type = self._infer_expression_type(stmt.end)
        if start_type != TypeKind.INTEGER:
            self.errors.append("Начальное значение цикла должно быть INTEGER")
        if end_type != TypeKind.INTEGER:
            self.errors.append("Конечное значение цикла должно быть INTEGER")
        if stmt.step:
            step_type = self._infer_expression_type(stmt.step)
            if step_type != TypeKind.INTEGER:
                self.errors.append("Шаг цикла должен быть INTEGER")
        if not self._is_constant_expression(stmt.start):
            self.errors.append("Начальное значение цикла DO должно быть константой")
        if not self._is_constant_expression(stmt.end):
            self.errors.append("Конечное значение цикла DO должно быть константой")
        if stmt.step and not self._is_constant_expression(stmt.step):
            self.errors.append("Шаг цикла DO должен быть константой")
        self._analyze_statements(stmt.body)
    def _analyze_labeled_do_loop(self, stmt: LabeledDoLoop):
        if stmt.var not in self.symbol_table:
            self.errors.append(
                f"Переменная цикла '{stmt.var}' не объявлена"
            )
        else:
            var_info = self.symbol_table[stmt.var]
            if var_info.type_kind != TypeKind.INTEGER:
                self.errors.append(
                    f"Переменная цикла '{stmt.var}' должна быть INTEGER"
                )
        start_type = self._infer_expression_type(stmt.start)
        end_type = self._infer_expression_type(stmt.end)
        if start_type != TypeKind.INTEGER:
            self.errors.append("Начальное значение цикла должно быть INTEGER")
        if end_type != TypeKind.INTEGER:
            self.errors.append("Конечное значение цикла должно быть INTEGER")
        if stmt.step:
            step_type = self._infer_expression_type(stmt.step)
            if step_type != TypeKind.INTEGER:
                self.errors.append("Шаг цикла должен быть INTEGER")
        if not self._is_constant_expression(stmt.start):
            self.errors.append("Начальное значение цикла DO должно быть константой")
        if not self._is_constant_expression(stmt.end):
            self.errors.append("Конечное значение цикла DO должно быть константой")
        if stmt.step and not self._is_constant_expression(stmt.step):
            self.errors.append("Шаг цикла DO должен быть константой")
        self._analyze_statements(stmt.body)
    def _analyze_do_while(self, stmt):
        cond_type = self._infer_expression_type(stmt.condition)
        if cond_type != TypeKind.LOGICAL and cond_type != TypeKind.UNKNOWN:
            self.errors.append(
                f"Условие DO WHILE должно быть LOGICAL, получено {cond_type.value}"
            )
        self._analyze_statements(stmt.body)
    def _analyze_simple_if_statement(self, stmt: SimpleIfStatement):
        cond_type = self._infer_expression_type(stmt.condition)
        if cond_type != TypeKind.LOGICAL and cond_type != TypeKind.UNKNOWN:
            self.errors.append(
                f"Условие простого IF должно быть LOGICAL, получено {cond_type.value}"
            )
        self._analyze_statements([stmt.statement])
    def _analyze_if_statement(self, stmt: IfStatement):
        cond_type = self._infer_expression_type(stmt.condition)
        if cond_type != TypeKind.LOGICAL and cond_type != TypeKind.UNKNOWN:
            self.errors.append(
                f"Условие IF должно быть LOGICAL, получено {cond_type.value}"
            )
        self._analyze_statements(stmt.then_body)
        for elif_cond, elif_body in stmt.elif_parts:
            elif_type = self._infer_expression_type(elif_cond)
            if elif_type != TypeKind.LOGICAL and elif_type != TypeKind.UNKNOWN:
                self.errors.append(
                    f"Условие ELSEIF должно быть LOGICAL, получено {elif_type.value}"
                )
            self._analyze_statements(elif_body)
        if stmt.else_body:
            self._analyze_statements(stmt.else_body)
    def _analyze_arithmetic_if_statement(self, stmt: ArithmeticIfStatement):
        cond_type = self._infer_expression_type(stmt.condition)
        if cond_type not in {TypeKind.INTEGER, TypeKind.REAL} and cond_type != TypeKind.UNKNOWN:
            self.errors.append(
                f"Условие арифметического IF должно быть INTEGER или REAL, получено {cond_type.value}"
            )
    def _analyze_print_statement(self, stmt: PrintStatement):
        for item in stmt.items:
            self._infer_expression_type(item)
    def _analyze_io_statement(self, stmt):
        if isinstance(stmt, ReadStatement):
            for item in stmt.items:
                if item not in self.symbol_table:
                    if not self.implicit_none:
                        implicit_type, type_size = self._get_implicit_type(item)
                        if implicit_type != TypeKind.UNKNOWN:
                            var_info = VariableInfo(
                                name=item,
                                type_kind=implicit_type,
                                is_array=False,
                                dimensions=[]
                            )
                            self.symbol_table[item] = var_info
                        else:
                            self.errors.append(
                                f"Переменная '{item}' в READ не объявлена и не может быть определена через implicit typing"
                            )
                    else:
                        self.errors.append(
                            f"Переменная '{item}' в READ не объявлена"
                        )
                else:
                    var_info = self.symbol_table[item]
                    if var_info.is_parameter:
                        self.errors.append(
                            f"Попытка чтения в константу PARAMETER '{item}' через READ"
                        )
        elif isinstance(stmt, WriteStatement):
            for item in stmt.items:
                self._infer_expression_type(item)
    def _analyze_call_statement(self, stmt: CallStatement):
        func_name = stmt.name
        if func_name in self.builtin_functions:
            pass
        else:
            self.warnings.append(
                f"Пользовательская подпрограмма '{func_name}' может быть не определена"
            )
        for arg in stmt.args:
            self._infer_expression_type(arg)
    def _infer_expression_type(self, expr: Expression) -> TypeKind:
        if isinstance(expr, IntegerLiteral):
            return TypeKind.INTEGER
        elif isinstance(expr, RealLiteral):
            return TypeKind.REAL
        elif isinstance(expr, StringLiteral):
            return TypeKind.CHARACTER
        elif isinstance(expr, LogicalLiteral):
            return TypeKind.LOGICAL
        elif isinstance(expr, Variable):
            if expr.name in self.symbol_table:
                return self.symbol_table[expr.name].type_kind
            else:
                if not self.implicit_none:
                    implicit_type, type_size = self._get_implicit_type(expr.name)
                    if implicit_type != TypeKind.UNKNOWN:
                        var_info = VariableInfo(
                            name=expr.name,
                            type_kind=implicit_type,
                            is_array=False,
                            dimensions=[]
                        )
                        self.symbol_table[expr.name] = var_info
                        rule_source = "явным правилом IMPLICIT" if expr.name[0].upper() in self.implicit_rules else "правилом I-M"
                        self.warnings.append(
                            f"Переменная '{expr.name}' не объявлена, используется неявный тип {implicit_type.value} "
                            f"по {rule_source}"
                        )
                        return implicit_type
                    else:
                        self.errors.append(f"Переменная '{expr.name}' не объявлена и не может быть определена неявно")
                        return TypeKind.UNKNOWN
                else:
                    self.errors.append(f"Переменная '{expr.name}' не объявлена (IMPLICIT NONE установлен)")
                    return TypeKind.UNKNOWN
        elif isinstance(expr, ArrayRef):
            if expr.name not in self.symbol_table:
                self.errors.append(f"Массив '{expr.name}' не объявлен")
                return TypeKind.UNKNOWN
            var_info = self.symbol_table[expr.name]
            if not var_info.is_array:
                self.errors.append(
                    f"'{expr.name}' не является массивом, но используется с индексами"
                )
            elif len(expr.indices) != len(var_info.dimensions):
                self.errors.append(
                    f"Неверное количество индексов для '{expr.name}': "
                    f"ожидается {len(var_info.dimensions)}, получено {len(expr.indices)}"
                )
            else:
                for idx in expr.indices:
                    idx_type = self._infer_expression_type(idx)
                    if idx_type != TypeKind.INTEGER and idx_type != TypeKind.UNKNOWN:
                        self.errors.append(
                            f"Индекс массива должен быть INTEGER, получено {idx_type.value}"
                        )
            return var_info.type_kind
        elif isinstance(expr, BinaryOp):
            return self._infer_binop_type(expr)
        elif isinstance(expr, UnaryOp):
            return self._infer_unaryop_type(expr)
        elif isinstance(expr, FunctionCall):
            return self._infer_function_call_type(expr)
        return TypeKind.UNKNOWN
    def _infer_binop_type(self, expr: BinaryOp) -> TypeKind:
        left_type = self._infer_expression_type(expr.left)
        right_type = self._infer_expression_type(expr.right)
        if expr.op == "//":
            if left_type != TypeKind.CHARACTER or right_type != TypeKind.CHARACTER:
                if left_type != TypeKind.UNKNOWN and right_type != TypeKind.UNKNOWN:
                    self.errors.append(
                        f"Операция конкатенации '//' требует CHARACTER операндов, "
                        f"получено {left_type.value} и {right_type.value}"
                    )
                return TypeKind.UNKNOWN
            return TypeKind.CHARACTER
        if expr.op in {".OR.", ".AND.", "|", "&"}:
            if left_type != TypeKind.LOGICAL or right_type != TypeKind.LOGICAL:
                self.errors.append(
                    f"Логическая операция '{expr.op}' требует LOGICAL операндов"
                )
            return TypeKind.LOGICAL
        if expr.op in {".EQV.", ".NEQV."}:
            if left_type != TypeKind.LOGICAL or right_type != TypeKind.LOGICAL:
                self.errors.append(
                    f"Логическая операция '{expr.op}' требует LOGICAL операндов"
                )
            return TypeKind.LOGICAL
        if expr.op in {".EQ.", ".NE.", ".LT.", ".LE.", ".GT.", ".GE.", "==", "/=", "<", "<=", ">", ">="}:
            if not self._are_comparable(left_type, right_type):
                self.errors.append(
                    f"Операция сравнения '{expr.op}' требует совместимых типов, "
                    f"получено {left_type.value} и {right_type.value}"
                )
            return TypeKind.LOGICAL
        if expr.op in {"+", "-", "*", "/", "**"}:
            if left_type not in {TypeKind.INTEGER, TypeKind.REAL} or\
               right_type not in {TypeKind.INTEGER, TypeKind.REAL}:
                if left_type != TypeKind.UNKNOWN and right_type != TypeKind.UNKNOWN:
                    self.errors.append(
                        f"Арифметическая операция '{expr.op}' требует числовых операндов, "
                        f"получено {left_type.value} и {right_type.value}"
                    )
                return TypeKind.UNKNOWN
            if left_type == TypeKind.INTEGER and right_type == TypeKind.INTEGER:
                return TypeKind.INTEGER
            if left_type in {TypeKind.REAL, TypeKind.INTEGER} and\
               right_type in {TypeKind.REAL, TypeKind.INTEGER}:
                if left_type == TypeKind.REAL or right_type == TypeKind.REAL:
                    return TypeKind.REAL
                return TypeKind.INTEGER
        return TypeKind.UNKNOWN
    def _infer_unaryop_type(self, expr: UnaryOp) -> TypeKind:
        operand_type = self._infer_expression_type(expr.operand)
        if expr.op == ".NOT.":
            if operand_type != TypeKind.LOGICAL:
                self.errors.append(
                    f"Операция '.NOT.' требует LOGICAL операнда"
                )
            return TypeKind.LOGICAL
        elif expr.op in {"+", "-"}:
            return operand_type
        return TypeKind.UNKNOWN
    def _infer_function_call_type(self, expr: FunctionCall) -> TypeKind:
        func_name = expr.name
        if func_name in self.symbol_table:
            var_info = self.symbol_table[func_name]
            if var_info.is_array:
                if len(expr.args) != len(var_info.dimensions):
                    self.errors.append(
                        f"Неверное количество индексов для массива '{func_name}': "
                        f"ожидается {len(var_info.dimensions)}, получено {len(expr.args)}"
                    )
                else:
                    for idx in expr.args:
                        idx_type = self._infer_expression_type(idx)
                        if idx_type != TypeKind.INTEGER and idx_type != TypeKind.UNKNOWN:
                            self.errors.append(
                                f"Индекс массива должен быть INTEGER, получено {idx_type.value}"
                            )
                return var_info.type_kind
        if func_name in self.builtin_functions:
            ret_type, arg_types = self.builtin_functions[func_name]
            if arg_types and expr.args:
                for i, arg in enumerate(expr.args):
                    arg_type = self._infer_expression_type(arg)
                    if arg_type != TypeKind.UNKNOWN:
                        if not any(self._are_compatible(expected_type, arg_type) for expected_type in arg_types):
                            expected_str = " или ".join([t.value for t in arg_types])
                            self.errors.append(
                                f"Аргумент {i+1} функции '{func_name}' должен быть {expected_str}, "
                                f"получено {arg_type.value}"
                            )
            if ret_type is None:
                if expr.args:
                    return self._infer_expression_type(expr.args[0])
            return ret_type if ret_type else TypeKind.UNKNOWN
        else:
            if func_name not in self.symbol_table:
                self.errors.append(f"Неизвестная функция или переменная '{func_name}'")
        return TypeKind.UNKNOWN
    def _are_compatible(self, target: TypeKind, source: TypeKind) -> bool:
        if target == TypeKind.INTEGER and source == TypeKind.INTEGER:
            return True
        if target == TypeKind.REAL and source in {TypeKind.REAL, TypeKind.INTEGER}:
            return True
        if target == TypeKind.LOGICAL and source == TypeKind.LOGICAL:
            return True
        if target == TypeKind.CHARACTER and source == TypeKind.CHARACTER:
            return True
        if target == TypeKind.COMPLEX:
            return True
        return False
    def _are_comparable(self, left: TypeKind, right: TypeKind) -> bool:
        if left == TypeKind.UNKNOWN or right == TypeKind.UNKNOWN:
            return True
        if left in {TypeKind.INTEGER, TypeKind.REAL} and right in {TypeKind.INTEGER, TypeKind.REAL}:
            return True
        if left == TypeKind.LOGICAL and right == TypeKind.LOGICAL:
            return True
        if left == TypeKind.CHARACTER and right == TypeKind.CHARACTER:
            return True
        return False
    def get_errors(self) -> List[str]:
        return self.errors
    def get_warnings(self) -> List[str]:
        return self.warnings
    def get_symbol_table(self) -> Dict[str, VariableInfo]:
        return self.symbol_table
    def _analyze_dimension_statement(self, stmt: DimensionStatement):
        for name, dim_ranges in stmt.names:
            if len(dim_ranges) > 7:
                self.errors.append(
                    f"Массив '{name}' имеет {len(dim_ranges)} измерений, максимум допускается 7"
                )
                continue
            for i, (k, l) in enumerate(dim_ranges):
                if k > l:
                    self.errors.append(
                        f"Неверный диапазон для измерения {i+1} массива '{name}': "
                        f"начальное значение ({k}) больше конечного ({l})"
                    )
            if name in self.symbol_table:
                var_info = self.symbol_table[name]
                if var_info.is_array:
                    if var_info.dimensions != dim_ranges:
                        self.errors.append(
                            f"Массив '{name}' уже объявлен с другими размерностями: "
                            f"ожидается {var_info.dimensions}, получено {dim_ranges}"
                        )
                    else:
                        self.errors.append(
                            f"Массив '{name}' уже объявлен в DIMENSION, повторное объявление недопустимо"
                        )
                elif var_info.is_parameter:
                    self.errors.append(
                        f"Переменная '{name}' уже объявлена как PARAMETER, объявление в DIMENSION недопустимо"
                    )
                else:
                    var_info.is_array = True
                    var_info.dimensions = dim_ranges
            else:
                if not self.implicit_none:
                    implicit_type, type_size = self._get_implicit_type(name)
                    if implicit_type != TypeKind.UNKNOWN:
                        var_info = VariableInfo(
                            name=name,
                            type_kind=implicit_type,
                            is_array=True,
                            dimensions=dim_ranges
                        )
                        self.symbol_table[name] = var_info
                        rule_source = "явным правилом IMPLICIT" if name[0].upper() in self.implicit_rules else "правилом I-M"
                        self.warnings.append(
                            f"Массив '{name}' объявлен в DIMENSION без указания типа, используется неявный тип {implicit_type.value} "
                            f"по {rule_source}"
                        )
                    else:
                        var_info = VariableInfo(
                            name=name,
                            type_kind=TypeKind.UNKNOWN,
                            is_array=True,
                            dimensions=dim_ranges
                        )
                        self.symbol_table[name] = var_info
                        self.warnings.append(
                            f"Массив '{name}' объявлен в DIMENSION без указания типа"
                        )
                else:
                    var_info = VariableInfo(
                        name=name,
                        type_kind=TypeKind.UNKNOWN,
                        is_array=True,
                        dimensions=dim_ranges
                    )
                    self.symbol_table[name] = var_info
                    self.warnings.append(
                        f"Массив '{name}' объявлен в DIMENSION без указания типа"
                    )
    def _is_constant_expression(self, expr: Expression) -> bool:
        return self._is_constant_expression_with_params(expr, set())
    def _is_constant_expression_with_params(self, expr: Expression, declared_params) -> bool:
        if isinstance(expr, (IntegerLiteral, RealLiteral, LogicalLiteral, StringLiteral)):
            return True
        elif isinstance(expr, Variable):
            if expr.name in declared_params:
                return True
            if expr.name in self.symbol_table:
                var_info = self.symbol_table[expr.name]
                return var_info.is_parameter
            return False
        elif isinstance(expr, BinaryOp):
            return (self._is_constant_expression_with_params(expr.left, declared_params) and 
                   self._is_constant_expression_with_params(expr.right, declared_params))
        elif isinstance(expr, UnaryOp):
            return self._is_constant_expression_with_params(expr.operand, declared_params)
        elif isinstance(expr, FunctionCall):
            if expr.name in self.builtin_functions:
                return all(self._is_constant_expression_with_params(arg, declared_params) for arg in expr.args)
            return False
        elif isinstance(expr, ArrayRef):
            return False
        return False
    def _evaluate_constant_expression(self, expr: Expression):
        return self._evaluate_constant_expression_with_params(expr, set())
    def _evaluate_constant_expression_with_params(self, expr: Expression, declared_params: Dict[str, object]):
        if isinstance(expr, IntegerLiteral):
            return expr.value
        elif isinstance(expr, RealLiteral):
            return expr.value
        elif isinstance(expr, LogicalLiteral):
            return expr.value
        elif isinstance(expr, StringLiteral):
            return expr.value
        elif isinstance(expr, Variable):
            if expr.name in declared_params:
                return declared_params[expr.name]
            if expr.name in self.symbol_table:
                var_info = self.symbol_table[expr.name]
                if var_info.is_parameter and var_info.value is not None:
                    return var_info.value
            return None
        elif isinstance(expr, BinaryOp):
            left_val = self._evaluate_constant_expression_with_params(expr.left, declared_params)
            right_val = self._evaluate_constant_expression_with_params(expr.right, declared_params)
            if left_val is None or right_val is None:
                return None
            op = expr.op
            if op == "+":
                return left_val + right_val
            elif op == "-":
                return left_val - right_val
            elif op == "*":
                return left_val * right_val
            elif op == "/":
                if isinstance(left_val, int) and isinstance(right_val, int):
                    return left_val // right_val
                return left_val / right_val
            elif op == "**":
                return left_val ** right_val
            elif op == "//":
                return str(left_val) + str(right_val)
            elif op == ".AND.":
                return left_val and right_val
            elif op == ".OR.":
                return left_val or right_val
            elif op == ".EQV.":
                return left_val == right_val
            elif op == ".NEQV.":
                return left_val != right_val
            elif op in {".EQ.", "=="}:
                return left_val == right_val
            elif op in {".NE.", "/="}:
                return left_val != right_val
            elif op in {".LT.", "<"}:
                return left_val < right_val
            elif op in {".LE.", "<="}:
                return left_val <= right_val
            elif op in {".GT.", ">"}:
                return left_val > right_val
            elif op in {".GE.", ">="}:
                return left_val >= right_val
        elif isinstance(expr, UnaryOp):
            operand_val = self._evaluate_constant_expression_with_params(expr.operand, declared_params)
            if operand_val is None:
                return None
            if expr.op == "+":
                return operand_val
            elif expr.op == "-":
                return -operand_val
            elif expr.op == ".NOT.":
                return not operand_val
        elif isinstance(expr, FunctionCall):
            pass
        return None
    def _analyze_parameter_statement(self, stmt: ParameterStatement):
        declared_in_this_stmt = {}
        for name, expr in stmt.params:
            declared_names = set(declared_in_this_stmt.keys())
            if not self._is_constant_expression_with_params(expr, declared_names):
                self.errors.append(
                    f"Выражение для PARAMETER '{name}' должно быть константным "
                    f"(содержать только константы и ранее объявленные параметры)"
                )
                continue
            if name in self.symbol_table:
                var_info = self.symbol_table[name]
                if var_info.is_parameter:
                    self.errors.append(
                        f"Параметр '{name}' уже объявлен, повторное объявление недопустимо"
                    )
                else:
                    self.errors.append(
                        f"Переменная '{name}' уже объявлена, объявление как PARAMETER недопустимо"
                    )
            else:
                expr_type = self._infer_expression_type(expr)
                if expr_type == TypeKind.UNKNOWN:
                    if isinstance(expr, IntegerLiteral):
                        expr_type = TypeKind.INTEGER
                    elif isinstance(expr, RealLiteral):
                        expr_type = TypeKind.REAL
                    elif isinstance(expr, LogicalLiteral):
                        expr_type = TypeKind.LOGICAL
                    elif isinstance(expr, StringLiteral):
                        expr_type = TypeKind.CHARACTER
                    else:
                        self.errors.append(
                            f"Не удалось определить тип константы '{name}' в PARAMETER"
                        )
                        expr_type = TypeKind.INTEGER
                value = self._evaluate_constant_expression_with_params(expr, declared_in_this_stmt)
                var_info = VariableInfo(
                    name=name,
                    type_kind=expr_type,
                    is_array=False,
                    dimensions=[],
                    is_parameter=True,
                    value=value
                )
                self.symbol_table[name] = var_info
                if value is not None:
                    declared_in_this_stmt[name] = value
    def _analyze_data_statement(self, stmt: DataStatement):
        for vars_list, values in stmt.items:
            if len(vars_list) != len(values):
                self.errors.append(
                    f"Количество переменных/массивов ({len(vars_list)}) не совпадает с количеством значений ({len(values)}) в DATA"
                )
            for data_item, value in zip(vars_list, values):
                var_name = data_item.name
                indices = data_item.indices
                if var_name not in self.symbol_table:
                    self.errors.append(
                        f"Переменная '{var_name}' в DATA не объявлена"
                    )
                    continue
                var_info = self.symbol_table[var_name]
                if var_info.is_parameter:
                    self.errors.append(
                        f"Попытка инициализировать константу PARAMETER '{var_name}' через DATA"
                    )
                    continue
                if indices:
                    if not var_info.is_array:
                        self.errors.append(
                            f"'{var_name}' не является массивом, но используется с индексами в DATA"
                        )
                        continue
                    if len(indices) != len(var_info.dimensions):
                        self.errors.append(
                            f"Неверное количество индексов для массива '{var_name}' в DATA: "
                            f"ожидается {len(var_info.dimensions)}, получено {len(indices)}"
                        )
                        continue
                    for idx in indices:
                        idx_type = self._infer_expression_type(idx)
                        if idx_type != TypeKind.INTEGER and idx_type != TypeKind.UNKNOWN:
                            self.errors.append(
                                f"Индекс массива в DATA должен быть INTEGER, получено {idx_type.value}"
                            )
                    for i, idx_expr in enumerate(indices):
                        if isinstance(idx_expr, IntegerLiteral):
                            idx_value = idx_expr.value
                            k, l = var_info.dimensions[i]
                            if idx_value < k or idx_value > l:
                                self.errors.append(
                                    f"Индекс {idx_value} для измерения {i+1} массива '{var_name}' вне диапазона [{k}:{l}] в DATA"
                                )
                else:
                    if var_info.is_array:
                        self.errors.append(
                            f"Массив '{var_name}' должен быть инициализирован с указанием индексов в DATA"
                        )
                        continue
                value_type = self._infer_expression_type(value)
                if value_type != TypeKind.UNKNOWN and var_info.type_kind != TypeKind.UNKNOWN:
                    if not self._are_compatible(var_info.type_kind, value_type):
                        self.warnings.append(
                            f"Несоответствие типа при инициализации '{var_name}' в DATA: "
                            f"{value_type.value} -> {var_info.type_kind.value}"
                        )
