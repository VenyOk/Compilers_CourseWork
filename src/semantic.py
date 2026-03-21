from typing import Dict, Tuple, Optional, List
from enum import Enum
from dataclasses import dataclass
from src.core import (
    ASTNode, Program, Declaration, Statement, Assignment,
    DoLoop, LabeledDoLoop, IfStatement, SimpleIfStatement, PrintStatement, ReadStatement, WriteStatement,
    CallStatement, BinaryOp, UnaryOp, Variable, IntegerLiteral,
    RealLiteral, StringLiteral, LogicalLiteral, FunctionCall, Expression,
    DoWhile, LabeledDoWhile, ArrayRef, DimensionStatement, ParameterStatement,
    DataStatement, DataItem, ArithmeticIfStatement, ImplicitNone, ImplicitStatement, ImplicitRule,
    Subroutine, FunctionDef, ReturnStatement, ExternalStatement, CommonStatement,
    GotoStatement, ContinueStatement, StopStatement, ExitStatement
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
    explicitly_declared: bool = False

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
        self.loop_nesting = 0
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
            "POW": (TypeKind.REAL, [TypeKind.REAL, TypeKind.INTEGER]),
        }

    def _format_error(self, message: str, node: Optional[ASTNode] = None,
                      context: Optional[str] = None, suggestion: Optional[str] = None) -> str:
        parts = []
        if node and hasattr(node, 'line') and node.line > 0:
            location = f"строка {node.line}"
            if hasattr(node, 'col') and node.col > 0:
                location += f", колонка {node.col}"
            parts.append(f"[{location}]")
        parts.append(message)
        if context:
            parts.append(f"(контекст: {context})")
        if suggestion:
            parts.append(f"Подсказка: {suggestion}")
        return " ".join(parts)

    def _format_warning(self, message: str, node: Optional[ASTNode] = None,
                        context: Optional[str] = None) -> str:
        parts = []
        if node and hasattr(node, 'line') and node.line > 0:
            location = f"строка {node.line}"
            if hasattr(node, 'col') and node.col > 0:
                location += f", колонка {node.col}"
            parts.append(f"[{location}]")
        parts.append(message)
        if context:
            parts.append(f"(контекст: {context})")
        return " ".join(parts)

    def analyze(self, ast: Program) -> bool:
        try:
            self.loop_nesting = 0
            type_map = {
                'INTEGER': TypeKind.INTEGER,
                'REAL': TypeKind.REAL,
                'LOGICAL': TypeKind.LOGICAL,
                'COMPLEX': TypeKind.COMPLEX,
                'CHARACTER': TypeKind.CHARACTER
            }
            for func in ast.functions:
                fname = func.name.upper()
                ret_type = TypeKind.INTEGER
                if func.return_type:
                    ret_type = type_map.get(func.return_type.upper(), TypeKind.INTEGER)
                self.builtin_functions[fname] = (ret_type, None)

            self._analyze_declarations(ast.declarations)
            self._analyze_statements(ast.statements)

            for sub in ast.subroutines:
                self._analyze_subroutine(sub)

            for func in ast.functions:
                self._analyze_function(func)

            return len(self.errors) == 0
        except Exception as e:
            import traceback
            error_msg = self._format_error(
                f"Внутренняя ошибка семантического анализа: {type(e).__name__}: {str(e)}",
                node=ast,
                context="семантический анализ программы",
                suggestion="Проверьте синтаксис и правильность использования конструкций языка"
            )
            self.errors.append(error_msg)
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
                            old_type = self.implicit_rules[letter][0].value
                            warning_msg = self._format_warning(
                                f"Правило IMPLICIT для буквы '{letter}' переопределено",
                                node=rule,
                                context=f"было: {old_type}, стало: {type_kind.value}"
                            )
                            self.warnings.append(warning_msg)
                        self.implicit_rules[letter] = (
                            type_kind, rule.type_size)
            elif isinstance(decl, Declaration):
                type_kind = TypeKind[decl.type] if decl.type in TypeKind.__members__ else TypeKind.UNKNOWN
                for name, dims in decl.names:
                    resolved_dims = self._resolve_dimension_ranges(dims, decl, name) if dims is not None else None
                    if name not in self.symbol_table:
                        self.symbol_table[name] = VariableInfo(
                            name=name,
                            type_kind=type_kind,
                            is_array=resolved_dims is not None,
                            dimensions=resolved_dims or [],
                            explicitly_declared=True
                        )
                        continue
                    var_info = self.symbol_table[name]
                    existing_type = var_info.type_kind.value if var_info.type_kind else "неизвестный"
                    if var_info.explicitly_declared:
                        error_msg = self._format_error(
                            f"Переменная '{name}' уже объявлена как {existing_type}",
                            node=decl,
                            context=f"повторное объявление типа {decl.type}",
                            suggestion=f"Оставьте только одно явное объявление для переменной '{name}'"
                        )
                        self.errors.append(error_msg)
                        continue
                    if var_info.is_parameter and resolved_dims is not None:
                        error_msg = self._format_error(
                            f"Константа PARAMETER '{name}' не может быть объявлена как массив",
                            node=decl,
                            context=f"объявление типа {decl.type}",
                            suggestion=f"Уберите размерности у '{name}' или используйте обычную переменную вместо PARAMETER"
                        )
                        self.errors.append(error_msg)
                        continue
                    if resolved_dims is not None:
                        if var_info.is_array and var_info.dimensions and var_info.dimensions != resolved_dims:
                            old_dims = ", ".join(f"{left}:{right}" for left, right in var_info.dimensions)
                            new_dims = ", ".join(f"{left}:{right}" for left, right in resolved_dims)
                            error_msg = self._format_error(
                                f"Массив '{name}' уже объявлен с другими размерностями",
                                node=decl,
                                context=f"было: [{old_dims}], получено: [{new_dims}]",
                                suggestion=f"Используйте одинаковые размерности для массива '{name}' во всех объявлениях"
                            )
                            self.errors.append(error_msg)
                            continue
                        var_info.is_array = True
                        var_info.dimensions = resolved_dims
                    var_info.type_kind = type_kind
                    var_info.explicitly_declared = True
            elif isinstance(decl, DimensionStatement):
                self._analyze_dimension_statement(decl)
            elif isinstance(decl, ParameterStatement):
                self._analyze_parameter_statement(decl)
            elif isinstance(decl, DataStatement):
                self._analyze_data_statement(decl)
            elif isinstance(decl, ExternalStatement):
                self._analyze_external_statement(decl)
            elif isinstance(decl, CommonStatement):
                self._analyze_common_statement(decl)

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
            elif isinstance(stmt, GotoStatement):
                self._analyze_goto_statement(stmt)
            elif isinstance(stmt, ContinueStatement):
                self._analyze_continue_statement(stmt)
            elif isinstance(stmt, ReturnStatement):
                self._analyze_return_statement(stmt)
            elif isinstance(stmt, StopStatement):
                self._analyze_stop_statement(stmt)
            elif isinstance(stmt, ExitStatement):
                self._analyze_exit_statement(stmt)

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
                    rule_source = "явным правилом IMPLICIT" if stmt.target[0].upper(
                    ) in self.implicit_rules else "правилом I-M"
                    warning_msg = self._format_warning(
                        f"Переменная '{stmt.target}' не объявлена явно, используется неявный тип {implicit_type.value}",
                        node=stmt,
                        context=f"присваивание в {rule_source}"
                    )
                    self.warnings.append(warning_msg)
                else:
                    error_msg = self._format_error(
                        f"Переменная '{stmt.target}' не объявлена и не может быть определена неявно",
                        node=stmt,
                        context="присваивание значения",
                        suggestion=f"Добавьте объявление переменной '{stmt.target}' перед использованием (например: INTEGER {stmt.target})"
                    )
                    self.errors.append(error_msg)
                    return
            else:
                error_msg = self._format_error(
                    f"Переменная '{stmt.target}' не объявлена",
                    node=stmt,
                    context="присваивание (IMPLICIT NONE установлен)",
                    suggestion=f"Добавьте объявление переменной '{stmt.target}' перед использованием (например: INTEGER {stmt.target})"
                )
                self.errors.append(error_msg)
                return
        var_info = self.symbol_table[stmt.target]
        if var_info.is_parameter:
            error_msg = self._format_error(
                f"Попытка изменить константу PARAMETER '{stmt.target}'",
                node=stmt,
                context="присваивание значения",
                suggestion=f"PARAMETER '{stmt.target}' является константой и не может быть изменён. спользуйте обычную переменную вместо PARAMETER"
            )
            self.errors.append(error_msg)
        if stmt.indices:
            if not var_info.is_array:
                error_msg = self._format_error(
                    f"Переменная '{stmt.target}' не является массивом, но используется с индексами",
                    node=stmt,
                    context=f"присваивание с индексами {len(stmt.indices)}",
                    suggestion=f"Объявите '{stmt.target}' как массив (например: INTEGER {stmt.target}(10)) или уберите индексы"
                )
                self.errors.append(error_msg)
            elif len(stmt.indices) != len(var_info.dimensions):
                dims_str = ", ".join([f"{d[0]}:{d[1]}" if isinstance(
                    d, tuple) else str(d) for d in var_info.dimensions])
                error_msg = self._format_error(
                    f"Неверное количество индексов для массива '{stmt.target}'",
                    node=stmt,
                    context=f"массив имеет {len(var_info.dimensions)} измерений [{dims_str}], использовано {len(stmt.indices)} индексов",
                    suggestion=f"спользуйте {len(var_info.dimensions)} индексов для доступа к массиву '{stmt.target}'"
                )
                self.errors.append(error_msg)
            else:
                for i, idx in enumerate(stmt.indices):
                    idx_type = self._infer_expression_type(idx)
                    if idx_type != TypeKind.INTEGER and idx_type != TypeKind.UNKNOWN:
                        error_msg = self._format_error(
                            f"ндекс массива должен быть INTEGER, получен тип {idx_type.value}",
                            node=idx if hasattr(idx, 'line') else stmt,
                            context=f"индекс {i+1} массива '{stmt.target}'",
                            suggestion=f"спользуйте целочисленное выражение для индекса (например: INTEGER переменную или целочисленную константу)"
                        )
                        self.errors.append(error_msg)
        expr_type = self._infer_expression_type(stmt.value)
        if expr_type != TypeKind.UNKNOWN and var_info.type_kind != TypeKind.UNKNOWN:
            if not self._are_compatible(var_info.type_kind, expr_type):
                warning_msg = self._format_warning(
                    f"Неявное преобразование типа при присваивании '{stmt.target}'",
                    node=stmt,
                    context=f"{expr_type.value} -> {var_info.type_kind.value}"
                )
                self.warnings.append(warning_msg)

    def _analyze_do_loop(self, stmt: DoLoop):
        if stmt.var not in self.symbol_table:
            error_msg = self._format_error(
                f"Переменная цикла '{stmt.var}' не объявлена",
                node=stmt,
                context="цикл DO",
                suggestion=f"Объявите переменную цикла перед использованием (например: INTEGER {stmt.var})"
            )
            self.errors.append(error_msg)
        else:
            var_info = self.symbol_table[stmt.var]
            if var_info.type_kind != TypeKind.INTEGER:
                error_msg = self._format_error(
                    f"Переменная цикла '{stmt.var}' должна быть типа INTEGER",
                    node=stmt,
                    context=f"объявлена как {var_info.type_kind.value}",
                    suggestion=f"змените тип переменной '{stmt.var}' на INTEGER"
                )
                self.errors.append(error_msg)
        start_type = self._infer_expression_type(stmt.start)
        end_type = self._infer_expression_type(stmt.end)
        if start_type != TypeKind.INTEGER:
            error_msg = self._format_error(
                f"Начальное значение цикла DO должно быть типа INTEGER",
                node=stmt.start if hasattr(stmt.start, 'line') else stmt,
                context=f"получен тип {start_type.value}",
                suggestion="спользуйте целочисленное выражение для начального значения цикла"
            )
            self.errors.append(error_msg)
        if end_type != TypeKind.INTEGER:
            error_msg = self._format_error(
                f"Конечное значение цикла DO должно быть типа INTEGER",
                node=stmt.end if hasattr(stmt.end, 'line') else stmt,
                context=f"получен тип {end_type.value}",
                suggestion="спользуйте целочисленное выражение для конечного значения цикла"
            )
            self.errors.append(error_msg)
        if stmt.step:
            step_type = self._infer_expression_type(stmt.step)
            if step_type != TypeKind.INTEGER:
                error_msg = self._format_error(
                    f"Шаг цикла DO должен быть типа INTEGER",
                    node=stmt.step if hasattr(stmt.step, 'line') else stmt,
                    context=f"получен тип {step_type.value}",
                    suggestion="спользуйте целочисленное выражение для шага цикла"
                )
                self.errors.append(error_msg)
        self.loop_nesting += 1
        self._analyze_statements(stmt.body)
        self.loop_nesting -= 1

    def _analyze_labeled_do_loop(self, stmt: LabeledDoLoop):
        if stmt.var not in self.symbol_table:
            error_msg = self._format_error(
                f"Переменная цикла '{stmt.var}' не объявлена",
                node=stmt,
                context=f"цикл DO с меткой {stmt.label}",
                suggestion=f"Объявите переменную цикла перед использованием (например: INTEGER {stmt.var})"
            )
            self.errors.append(error_msg)
        else:
            var_info = self.symbol_table[stmt.var]
            if var_info.type_kind != TypeKind.INTEGER:
                error_msg = self._format_error(
                    f"Переменная цикла '{stmt.var}' должна быть типа INTEGER",
                    node=stmt,
                    context=f"объявлена как {var_info.type_kind.value}, метка {stmt.label}",
                    suggestion=f"змените тип переменной '{stmt.var}' на INTEGER"
                )
                self.errors.append(error_msg)
        start_type = self._infer_expression_type(stmt.start)
        end_type = self._infer_expression_type(stmt.end)
        if start_type != TypeKind.INTEGER:
            error_msg = self._format_error(
                f"Начальное значение цикла DO должно быть типа INTEGER",
                node=stmt.start if hasattr(stmt.start, 'line') else stmt,
                context=f"цикл с меткой {stmt.label}, получен тип {start_type.value}",
                suggestion="спользуйте целочисленное выражение для начального значения цикла"
            )
            self.errors.append(error_msg)
        if end_type != TypeKind.INTEGER:
            error_msg = self._format_error(
                f"Конечное значение цикла DO должно быть типа INTEGER",
                node=stmt.end if hasattr(stmt.end, 'line') else stmt,
                context=f"цикл с меткой {stmt.label}, получен тип {end_type.value}",
                suggestion="спользуйте целочисленное выражение для конечного значения цикла"
            )
            self.errors.append(error_msg)
        if stmt.step:
            step_type = self._infer_expression_type(stmt.step)
            if step_type != TypeKind.INTEGER:
                error_msg = self._format_error(
                    f"Шаг цикла DO должен быть типа INTEGER",
                    node=stmt.step if hasattr(stmt.step, 'line') else stmt,
                    context=f"цикл с меткой {stmt.label}, получен тип {step_type.value}",
                    suggestion="спользуйте целочисленное выражение для шага цикла"
                )
                self.errors.append(error_msg)
        self.loop_nesting += 1
        self._analyze_statements(stmt.body)
        self.loop_nesting -= 1

    def _analyze_do_while(self, stmt):
        cond_type = self._infer_expression_type(stmt.condition)
        if cond_type != TypeKind.LOGICAL and cond_type != TypeKind.UNKNOWN:
            error_msg = self._format_error(
                f"Условие цикла DO WHILE должно быть типа LOGICAL",
                node=stmt.condition if hasattr(
                    stmt.condition, 'line') else stmt,
                context=f"получен тип {cond_type.value}",
                suggestion="спользуйте логическое выражение или переменную типа LOGICAL (например: .TRUE., .FALSE., или сравнение)"
            )
            self.errors.append(error_msg)
        self.loop_nesting += 1
        self._analyze_statements(stmt.body)
        self.loop_nesting -= 1

    def _analyze_simple_if_statement(self, stmt: SimpleIfStatement):
        cond_type = self._infer_expression_type(stmt.condition)
        if cond_type != TypeKind.LOGICAL and cond_type != TypeKind.UNKNOWN:
            error_msg = self._format_error(
                f"Условие оператора IF должно быть типа LOGICAL",
                node=stmt.condition if hasattr(
                    stmt.condition, 'line') else stmt,
                context=f"простой IF, получен тип {cond_type.value}",
                suggestion="спользуйте логическое выражение или переменную типа LOGICAL (например: .TRUE., .FALSE., или сравнение)"
            )
            self.errors.append(error_msg)
        self._analyze_statements([stmt.statement])

    def _analyze_if_statement(self, stmt: IfStatement):
        cond_type = self._infer_expression_type(stmt.condition)
        if cond_type != TypeKind.LOGICAL and cond_type != TypeKind.UNKNOWN:
            error_msg = self._format_error(
                f"Условие оператора IF должен быть типа LOGICAL",
                node=stmt.condition if hasattr(
                    stmt.condition, 'line') else stmt,
                context=f"IF-THEN-ELSE, получен тип {cond_type.value}",
                suggestion="спользуйте логическое выражение или переменную типа LOGICAL (например: .TRUE., .FALSE., или сравнение)"
            )
            self.errors.append(error_msg)
        self._analyze_statements(stmt.then_body)
        for i, (elif_cond, elif_body) in enumerate(stmt.elif_parts, 1):
            elif_type = self._infer_expression_type(elif_cond)
            if elif_type != TypeKind.LOGICAL and elif_type != TypeKind.UNKNOWN:
                error_msg = self._format_error(
                    f"Условие ELSEIF должно быть типа LOGICAL",
                    node=elif_cond if hasattr(elif_cond, 'line') else stmt,
                    context=f"ELSEIF #{i} в IF-THEN-ELSE, получен тип {elif_type.value}",
                    suggestion="спользуйте логическое выражение или переменную типа LOGICAL (например: .TRUE., .FALSE., или сравнение)"
                )
                self.errors.append(error_msg)
            self._analyze_statements(elif_body)
        if stmt.else_body:
            self._analyze_statements(stmt.else_body)

    def _analyze_arithmetic_if_statement(self, stmt: ArithmeticIfStatement):
        cond_type = self._infer_expression_type(stmt.condition)
        if cond_type not in {TypeKind.INTEGER, TypeKind.REAL} and cond_type != TypeKind.UNKNOWN:
            error_msg = self._format_error(
                f"Условие арифметического IF должно быть типа INTEGER или REAL",
                node=stmt.condition if hasattr(
                    stmt.condition, 'line') else stmt,
                context=f"арифметический IF (метки: {stmt.label_neg}, {stmt.label_zero}, {stmt.label_pos}), получен тип {cond_type.value}",
                suggestion="спользуйте числовое выражение (INTEGER или REAL) для арифметического IF"
            )
            self.errors.append(error_msg)

    def _analyze_print_statement(self, stmt: PrintStatement):
        for item in stmt.items:
            self._infer_expression_type(item)

    def _analyze_io_statement(self, stmt):
        if isinstance(stmt, ReadStatement):
            for item in stmt.items:
                if item not in self.symbol_table:
                    if not self.implicit_none:
                        implicit_type, type_size = self._get_implicit_type(
                            item)
                        if implicit_type != TypeKind.UNKNOWN:
                            var_info = VariableInfo(
                                name=item,
                                type_kind=implicit_type,
                                is_array=False,
                                dimensions=[]
                            )
                            self.symbol_table[item] = var_info
                        else:
                            error_msg = self._format_error(
                                f"Переменная '{item}' в операторе READ не объявлена и не может быть определена неявно",
                                node=stmt,
                                context="оператор READ",
                                suggestion=f"Объявите переменную '{item}' перед использованием в READ (например: INTEGER {item})"
                            )
                            self.errors.append(error_msg)
                    else:
                        error_msg = self._format_error(
                            f"Переменная '{item}' в операторе READ не объявлена",
                            node=stmt,
                            context="оператор READ (IMPLICIT NONE установлен)",
                            suggestion=f"Объявите переменную '{item}' перед использованием в READ (например: INTEGER {item})"
                        )
                        self.errors.append(error_msg)
                else:
                    var_info = self.symbol_table[item]
                    if var_info.is_parameter:
                        error_msg = self._format_error(
                            f"Попытка чтения в константу PARAMETER '{item}' через оператор READ",
                            node=stmt,
                            context=f"PARAMETER '{item}' является константой",
                            suggestion=f"спользуйте обычную переменную вместо PARAMETER '{item}' для чтения значений"
                        )
                        self.errors.append(error_msg)
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
                    implicit_type, type_size = self._get_implicit_type(
                        expr.name)
                    if implicit_type != TypeKind.UNKNOWN:
                        var_info = VariableInfo(
                            name=expr.name,
                            type_kind=implicit_type,
                            is_array=False,
                            dimensions=[]
                        )
                        self.symbol_table[expr.name] = var_info
                        rule_source = "явным правилом IMPLICIT" if expr.name[0].upper(
                        ) in self.implicit_rules else "правилом I-M"
                        warning_msg = self._format_warning(
                            f"Переменная '{expr.name}' не объявлена явно, используется неявный тип {implicit_type.value}",
                            node=expr,
                            context=f"определение по {rule_source}"
                        )
                        self.warnings.append(warning_msg)
                        return implicit_type
                    else:
                        error_msg = self._format_error(
                            f"Переменная '{expr.name}' не объявлена и не может быть определена неявно",
                            node=expr,
                            context="использование в выражении",
                            suggestion=f"Объявите переменную '{expr.name}' перед использованием (например: INTEGER {expr.name})"
                        )
                        self.errors.append(error_msg)
                        return TypeKind.UNKNOWN
                else:
                    error_msg = self._format_error(
                        f"Переменная '{expr.name}' не объявлена",
                        node=expr,
                        context="использование в выражении (IMPLICIT NONE установлен)",
                        suggestion=f"Объявите переменную '{expr.name}' перед использованием (например: INTEGER {expr.name})"
                    )
                    self.errors.append(error_msg)
                    return TypeKind.UNKNOWN
        elif isinstance(expr, ArrayRef):

            if expr.name.upper() in self.builtin_functions:
                ret_type, arg_types = self.builtin_functions[expr.name.upper()]
                return ret_type if ret_type else TypeKind.UNKNOWN

            if expr.name not in self.symbol_table:
                error_msg = self._format_error(
                    f"Массив '{expr.name}' не объявлен",
                    node=expr,
                    context="доступ к элементу массива",
                    suggestion=f"Объявите массив '{expr.name}' перед использованием (например: INTEGER {expr.name}(10))"
                )
                self.errors.append(error_msg)
                return TypeKind.UNKNOWN
            var_info = self.symbol_table[expr.name]
            if not var_info.is_array:
                error_msg = self._format_error(
                    f"Переменная '{expr.name}' не является массивом, но используется с индексами",
                    node=expr,
                    context=f"использовано {len(expr.indices)} индексов",
                    suggestion=f"Объявите '{expr.name}' как массив (например: INTEGER {expr.name}(10)) или уберите индексы"
                )
                self.errors.append(error_msg)
            elif len(expr.indices) != len(var_info.dimensions):
                dims_str = ", ".join([f"{d[0]}:{d[1]}" if isinstance(
                    d, tuple) else str(d) for d in var_info.dimensions])
                error_msg = self._format_error(
                    f"Неверное количество индексов для массива '{expr.name}'",
                    node=expr,
                    context=f"массив имеет {len(var_info.dimensions)} измерений [{dims_str}], использовано {len(expr.indices)} индексов",
                    suggestion=f"спользуйте {len(var_info.dimensions)} индексов для доступа к массиву '{expr.name}'"
                )
                self.errors.append(error_msg)
            else:
                for i, idx in enumerate(expr.indices):
                    idx_type = self._infer_expression_type(idx)
                    if idx_type != TypeKind.INTEGER and idx_type != TypeKind.UNKNOWN:
                        error_msg = self._format_error(
                            f"ндекс массива должен быть типа INTEGER",
                            node=idx if hasattr(idx, 'line') else expr,
                            context=f"индекс {i+1} массива '{expr.name}', получен тип {idx_type.value}",
                            suggestion="спользуйте целочисленное выражение для индекса (например: INTEGER переменную или целочисленную константу)"
                        )
                        self.errors.append(error_msg)
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
                    error_msg = self._format_error(
                        f"Операция конкатенации '//' требует операндов типа CHARACTER",
                        node=expr,
                        context=f"получено {left_type.value} и {right_type.value}",
                        suggestion="спользуйте строковые константы или переменные типа CHARACTER для операции конкатенации"
                    )
                    self.errors.append(error_msg)
                return TypeKind.UNKNOWN
            return TypeKind.CHARACTER
        if expr.op in {".OR.", ".AND.", "|", "&"}:
            if left_type != TypeKind.LOGICAL or right_type != TypeKind.LOGICAL:
                error_msg = self._format_error(
                    f"Логическая операция '{expr.op}' требует операндов типа LOGICAL",
                    node=expr,
                    context=f"получено {left_type.value} и {right_type.value}",
                    suggestion="спользуйте логические выражения или переменные типа LOGICAL (например: .TRUE., .FALSE., или сравнения)"
                )
                self.errors.append(error_msg)
            return TypeKind.LOGICAL
        if expr.op in {".EQV.", ".NEQV."}:
            if left_type != TypeKind.LOGICAL or right_type != TypeKind.LOGICAL:
                error_msg = self._format_error(
                    f"Логическая операция '{expr.op}' требует операндов типа LOGICAL",
                    node=expr,
                    context=f"получено {left_type.value} и {right_type.value}",
                    suggestion="спользуйте логические выражения или переменные типа LOGICAL (например: .TRUE., .FALSE., или сравнения)"
                )
                self.errors.append(error_msg)
            return TypeKind.LOGICAL
        if expr.op in {".EQ.", ".NE.", ".LT.", ".LE.", ".GT.", ".GE.", "==", "/=", "<", "<=", ">", ">="}:
            if not self._are_comparable(left_type, right_type):
                error_msg = self._format_error(
                    f"Операция сравнения '{expr.op}' требует совместимых типов",
                    node=expr,
                    context=f"получено {left_type.value} и {right_type.value}",
                    suggestion="спользуйте операнды совместимых типов для сравнения (числовые типы с числовыми, логические с логическими, строковые со строковыми)"
                )
                self.errors.append(error_msg)
            return TypeKind.LOGICAL
        if expr.op in {"+", "-", "*", "/", "**"}:
            numeric_types = {TypeKind.INTEGER, TypeKind.REAL, TypeKind.COMPLEX}
            if left_type not in numeric_types or right_type not in numeric_types:
                if left_type != TypeKind.UNKNOWN and right_type != TypeKind.UNKNOWN:
                    error_msg = self._format_error(
                        f"Арифметическая операция '{expr.op}' требует числовых операндов",
                        node=expr,
                        context=f"получено {left_type.value} и {right_type.value}",
                        suggestion="спользуйте числовые выражения или переменные (INTEGER, REAL или COMPLEX) для арифметических операций"
                    )
                    self.errors.append(error_msg)
                return TypeKind.UNKNOWN
            if left_type == TypeKind.COMPLEX or right_type == TypeKind.COMPLEX:
                return TypeKind.COMPLEX
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
                error_msg = self._format_error(
                    f"Операция '.NOT.' требует операнда типа LOGICAL",
                    node=expr,
                    context=f"получен тип {operand_type.value}",
                    suggestion="спользуйте логическое выражение или переменную типа LOGICAL (например: .TRUE., .FALSE., или сравнение)"
                )
                self.errors.append(error_msg)
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
                    dims_str = ", ".join([f"{d[0]}:{d[1]}" if isinstance(
                        d, tuple) else str(d) for d in var_info.dimensions])
                    error_msg = self._format_error(
                        f"Неверное количество индексов для массива '{func_name}'",
                        node=expr,
                        context=f"массив имеет {len(var_info.dimensions)} измерений [{dims_str}], использовано {len(expr.args)} индексов",
                        suggestion=f"спользуйте {len(var_info.dimensions)} индексов для доступа к массиву '{func_name}'"
                    )
                    self.errors.append(error_msg)
                else:
                    for i, idx in enumerate(expr.args):
                        idx_type = self._infer_expression_type(idx)
                        if idx_type != TypeKind.INTEGER and idx_type != TypeKind.UNKNOWN:
                            error_msg = self._format_error(
                                f"ндекс массива должен быть типа INTEGER",
                                node=idx if hasattr(idx, 'line') else expr,
                                context=f"индекс {i+1} массива '{func_name}', получен тип {idx_type.value}",
                                suggestion="спользуйте целочисленное выражение для индекса (например: INTEGER переменную или целочисленную константу)"
                            )
                            self.errors.append(error_msg)
                return var_info.type_kind
        if func_name in self.builtin_functions:
            ret_type, arg_types = self.builtin_functions[func_name]
            if arg_types and expr.args:
                for i, arg in enumerate(expr.args):
                    arg_type = self._infer_expression_type(arg)
                    if arg_type != TypeKind.UNKNOWN:
                        if not any(self._are_compatible(expected_type, arg_type) for expected_type in arg_types):
                            expected_str = " или ".join(
                                [t.value for t in arg_types])
                            error_msg = self._format_error(
                                f"Неверный тип аргумента {i+1} функции '{func_name}'",
                                node=arg if hasattr(arg, 'line') else expr,
                                context=f"ожидается {expected_str}, получен {arg_type.value}",
                                suggestion=f"спользуйте аргумент типа {expected_str} для функции '{func_name}'"
                            )
                            self.errors.append(error_msg)
            if ret_type is None:
                if expr.args:
                    return self._infer_expression_type(expr.args[0])
            return ret_type if ret_type else TypeKind.UNKNOWN
        else:
            if func_name not in self.symbol_table:
                error_msg = self._format_error(
                    f"Неизвестная функция или переменная '{func_name}'",
                    node=expr,
                    context=f"вызов функции с {len(expr.args)} аргументами",
                    suggestion=f"Объявите функцию '{func_name}' или проверьте правильность написания имени. Доступные встроенные функции: {', '.join(sorted(self.builtin_functions.keys()))}"
                )
                self.errors.append(error_msg)
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
            resolved_dim_ranges = self._resolve_dimension_ranges(dim_ranges, stmt, name)
            if len(dim_ranges) > 7:
                error_msg = self._format_error(
                    f"Массив '{name}' имеет слишком много измерений",
                    node=stmt,
                    context=f"получено {len(dim_ranges)} измерений, максимум допускается 7",
                    suggestion=f"спользуйте не более 7 измерений для массива '{name}'"
                )
                self.errors.append(error_msg)
                continue
            for i, (k, l) in enumerate(resolved_dim_ranges):
                if k > l:
                    error_msg = self._format_error(
                        f"Неверный диапазон для измерения {i+1} массива '{name}'",
                        node=stmt,
                        context=f"начальное значение ({k}) больше конечного ({l})",
                        suggestion=f"справьте диапазон измерения {i+1}: начальное значение должно быть меньше или равно конечному (например: {l}:{k} → {k}:{l})"
                    )
                    self.errors.append(error_msg)
            if name in self.symbol_table:
                var_info = self.symbol_table[name]
                existing_dims_str = ", ".join([f"{d[0]}:{d[1]}" if isinstance(
                    d, tuple) else str(d) for d in var_info.dimensions])
                new_dims_str = ", ".join([f"{d[0]}:{d[1]}" if isinstance(
                    d, tuple) else str(d) for d in resolved_dim_ranges])
                if var_info.is_array:
                    if var_info.dimensions != resolved_dim_ranges:
                        error_msg = self._format_error(
                            f"Массив '{name}' уже объявлен с другими размерностями",
                            node=stmt,
                            context=f"было: [{existing_dims_str}], получено: [{new_dims_str}]",
                            suggestion=f"Убедитесь, что размерности массива '{name}' совпадают во всех объявлениях или удалите одно из объявлений"
                        )
                    else:
                        error_msg = self._format_error(
                            f"Массив '{name}' уже объявлен в DIMENSION, повторное объявление недопустимо",
                            node=stmt,
                            context=f"массив уже объявлен с размерностями [{existing_dims_str}]",
                            suggestion=f"Удалите одно из объявлений DIMENSION для массива '{name}'"
                        )
                    self.errors.append(error_msg)
                elif var_info.is_parameter:
                    error_msg = self._format_error(
                        f"Переменная '{name}' уже объявлена как PARAMETER, объявление в DIMENSION недопустимо",
                        node=stmt,
                        context=f"PARAMETER '{name}' уже существует",
                        suggestion=f"Нельзя объявить PARAMETER '{name}' как массив. спользуйте обычную переменную или удалите объявление PARAMETER"
                    )
                    self.errors.append(error_msg)
                else:
                    var_info.is_array = True
                    var_info.dimensions = resolved_dim_ranges
            else:
                if not self.implicit_none:
                    implicit_type, type_size = self._get_implicit_type(name)
                    if implicit_type != TypeKind.UNKNOWN:
                        var_info = VariableInfo(
                            name=name,
                            type_kind=implicit_type,
                            is_array=True,
                            dimensions=resolved_dim_ranges
                        )
                        self.symbol_table[name] = var_info
                        rule_source = "явным правилом IMPLICIT" if name[0].upper(
                        ) in self.implicit_rules else "правилом I-M"
                        self.warnings.append(
                            f"Массив '{name}' объявлен в DIMENSION без указания типа, используется неявный тип {implicit_type.value} "
                            f"по {rule_source}"
                        )
                    else:
                        var_info = VariableInfo(
                            name=name,
                            type_kind=TypeKind.UNKNOWN,
                            is_array=True,
                            dimensions=resolved_dim_ranges
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
            left_val = self._evaluate_constant_expression_with_params(
                expr.left, declared_params)
            right_val = self._evaluate_constant_expression_with_params(
                expr.right, declared_params)
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
            operand_val = self._evaluate_constant_expression_with_params(
                expr.operand, declared_params)
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
                error_msg = self._format_error(
                    f"Выражение для PARAMETER '{name}' должно быть константным",
                    node=expr if hasattr(expr, 'line') else stmt,
                    context="объявление PARAMETER",
                    suggestion="спользуйте только константы, числа или ранее объявленные параметры в выражении для PARAMETER"
                )
                self.errors.append(error_msg)
                continue
            if name in self.symbol_table:
                var_info = self.symbol_table[name]
                if var_info.is_parameter:
                    error_msg = self._format_error(
                        f"Параметр '{name}' уже объявлен, повторное объявление недопустимо",
                        node=stmt,
                        context=f"PARAMETER '{name}' уже существует",
                        suggestion=f"Удалите одно из объявлений PARAMETER '{name}' или используйте другое имя"
                    )
                    self.errors.append(error_msg)
                else:
                    value = self._evaluate_constant_expression_with_params(expr, declared_in_this_stmt)
                    var_info.is_parameter = True
                    var_info.value = value
                    if value is not None:
                        declared_in_this_stmt[name] = value
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
                        error_msg = self._format_error(
                            f"Не удалось определить тип константы '{name}' в PARAMETER",
                            node=expr if hasattr(expr, 'line') else stmt,
                            context="объявление PARAMETER",
                            suggestion="спользуйте явную константу известного типа (целое число, вещественное число, строку или логическое значение)"
                        )
                        self.errors.append(error_msg)
                        expr_type = TypeKind.INTEGER
                value = self._evaluate_constant_expression_with_params(
                    expr, declared_in_this_stmt)
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
                error_msg = self._format_error(
                    f"Количество переменных/массивов не совпадает с количеством значений в операторе DATA",
                    node=stmt,
                    context=f"переменных: {len(vars_list)}, значений: {len(values)}",
                    suggestion="Убедитесь, что количество переменных совпадает с количеством значений в списке DATA"
                )
                self.errors.append(error_msg)
            for data_item, value in zip(vars_list, values):
                var_name = data_item.name
                indices = data_item.indices
                if var_name not in self.symbol_table:
                    error_msg = self._format_error(
                        f"Переменная '{var_name}' в операторе DATA не объявлена",
                        node=data_item if hasattr(data_item, 'line') else stmt,
                        context="инициализация в DATA",
                        suggestion=f"Объявите переменную '{var_name}' перед использованием в DATA (например: INTEGER {var_name})"
                    )
                    self.errors.append(error_msg)
                    continue
                var_info = self.symbol_table[var_name]
                if var_info.is_parameter:
                    error_msg = self._format_error(
                        f"Попытка инициализировать константу PARAMETER '{var_name}' через оператор DATA",
                        node=data_item if hasattr(data_item, 'line') else stmt,
                        context=f"PARAMETER '{var_name}' является константой",
                        suggestion=f"PARAMETER не может быть инициализирован через DATA. спользуйте обычную переменную или инициализируйте PARAMETER при объявлении"
                    )
                    self.errors.append(error_msg)
                    continue
                if indices:
                    if not var_info.is_array:
                        error_msg = self._format_error(
                            f"Переменная '{var_name}' не является массивом, но используется с индексами в DATA",
                            node=data_item if hasattr(
                                data_item, 'line') else stmt,
                            context=f"использовано {len(indices)} индексов",
                            suggestion=f"Объявите '{var_name}' как массив (например: INTEGER {var_name}(10)) или уберите индексы"
                        )
                        self.errors.append(error_msg)
                        continue
                    if len(indices) != len(var_info.dimensions):
                        dims_str = ", ".join([f"{d[0]}:{d[1]}" if isinstance(
                            d, tuple) else str(d) for d in var_info.dimensions])
                        error_msg = self._format_error(
                            f"Неверное количество индексов для массива '{var_name}' в DATA",
                            node=data_item if hasattr(
                                data_item, 'line') else stmt,
                            context=f"массив имеет {len(var_info.dimensions)} измерений [{dims_str}], использовано {len(indices)} индексов",
                            suggestion=f"спользуйте {len(var_info.dimensions)} индексов для доступа к массиву '{var_name}'"
                        )
                        self.errors.append(error_msg)
                        continue
                    for idx in indices:
                        idx_type = self._infer_expression_type(idx)
                        if idx_type != TypeKind.INTEGER and idx_type != TypeKind.UNKNOWN:
                            error_msg = self._format_error(
                                f"ндекс массива в DATA должен быть типа INTEGER",
                                node=idx if hasattr(
                                    idx, 'line') else data_item,
                                context=f"массив '{var_name}', получен тип {idx_type.value}",
                                suggestion="спользуйте целочисленное выражение для индекса (например: INTEGER переменную или целочисленную константу)"
                            )
                            self.errors.append(error_msg)
                    for i, idx_expr in enumerate(indices):
                        if isinstance(idx_expr, IntegerLiteral):
                            idx_value = idx_expr.value
                            k, l = var_info.dimensions[i]
                            if idx_value < k or idx_value > l:
                                error_msg = self._format_error(
                                    f"ндекс {idx_value} для измерения {i+1} массива '{var_name}' вне диапазона",
                                    node=idx_expr if hasattr(
                                        idx_expr, 'line') else data_item,
                                    context=f"допустимый диапазон: [{k}:{l}]",
                                    suggestion=f"спользуйте индекс в диапазоне от {k} до {l} для измерения {i+1} массива '{var_name}'"
                                )
                                self.errors.append(error_msg)
                else:
                    if var_info.is_array:
                        error_msg = self._format_error(
                            f"Массив '{var_name}' должен быть инициализирован с указанием индексов в DATA",
                            node=data_item if hasattr(
                                data_item, 'line') else stmt,
                            context="инициализация массива в DATA",
                            suggestion=f"Укажите индексы для инициализации элемента массива (например: {var_name}(1, 1) вместо {var_name})"
                        )
                        self.errors.append(error_msg)
                        continue
                value_type = self._infer_expression_type(value)
                if value_type != TypeKind.UNKNOWN and var_info.type_kind != TypeKind.UNKNOWN:
                    if not self._are_compatible(var_info.type_kind, value_type):
                        warning_msg = self._format_warning(
                            f"Несоответствие типа при инициализации '{var_name}' в DATA",
                            node=value if hasattr(value, 'line') else stmt,
                            context=f"{value_type.value} -> {var_info.type_kind.value}"
                        )
                        self.warnings.append(warning_msg)

    def _analyze_subroutine(self, sub: 'Subroutine'):
        old_symbol_table = self.symbol_table
        old_implicit_none = self.implicit_none
        old_implicit_rules = self.implicit_rules.copy()
        old_loop_nesting = self.loop_nesting

        self.symbol_table = {}
        self.implicit_none = False
        self.implicit_rules = {}
        self.loop_nesting = 0

        for param in sub.params:
            param_upper = param.upper()
            self.symbol_table[param_upper] = VariableInfo(
                name=param_upper,
                type_kind=TypeKind.UNKNOWN,
                is_array=False,
                dimensions=[],
                is_parameter=False
            )

        self._analyze_declarations(sub.declarations)

        for param in sub.params:
            param_upper = param.upper()
            if param_upper in self.symbol_table:
                var_info = self.symbol_table[param_upper]
                if var_info.type_kind == TypeKind.UNKNOWN:
                    var_info.type_kind = self._get_implicit_type(param_upper)[0]

        self._analyze_statements(sub.statements)

        has_return = any(isinstance(s, ReturnStatement) for s in sub.statements)
        if not has_return:
            warning_msg = self._format_warning(
                f"Подпрограмма '{sub.name}' не содержит оператора RETURN",
                context="SUBROUTINE"
            )
            self.warnings.append(warning_msg)

        self.symbol_table = old_symbol_table
        self.implicit_none = old_implicit_none
        self.implicit_rules = old_implicit_rules
        self.loop_nesting = old_loop_nesting

    def _analyze_function(self, func: 'FunctionDef'):
        old_symbol_table = self.symbol_table
        old_implicit_none = self.implicit_none
        old_implicit_rules = self.implicit_rules.copy()
        old_loop_nesting = self.loop_nesting

        self.symbol_table = {}
        self.implicit_none = False
        self.implicit_rules = {}
        self.loop_nesting = 0

        return_type = TypeKind.UNKNOWN
        if func.return_type:
            type_map = {
                'INTEGER': TypeKind.INTEGER,
                'REAL': TypeKind.REAL,
                'LOGICAL': TypeKind.LOGICAL,
                'COMPLEX': TypeKind.COMPLEX,
                'CHARACTER': TypeKind.CHARACTER
            }
            return_type = type_map.get(func.return_type.upper(), TypeKind.UNKNOWN)

        func_name_upper = func.name.upper()
        self.symbol_table[func_name_upper] = VariableInfo(
            name=func_name_upper,
            type_kind=return_type,
            is_array=False,
            dimensions=[],
            is_parameter=False,
            explicitly_declared=func.return_type is not None
        )

        for param in func.params:
            param_upper = param.upper()
            self.symbol_table[param_upper] = VariableInfo(
                name=param_upper,
                type_kind=TypeKind.UNKNOWN,
                is_array=False,
                dimensions=[],
                is_parameter=False
            )

        self._analyze_declarations(func.declarations)

        if func_name_upper in self.symbol_table:
            func_info = self.symbol_table[func_name_upper]
            if func_info.type_kind == TypeKind.UNKNOWN:
                func_info.type_kind = self._get_implicit_type(func_name_upper)[0]

        for param in func.params:
            param_upper = param.upper()
            if param_upper in self.symbol_table:
                param_info = self.symbol_table[param_upper]
                if param_info.type_kind == TypeKind.UNKNOWN:
                    param_info.type_kind = self._get_implicit_type(param_upper)[0]

        self._analyze_statements(func.statements)

        has_return = any(isinstance(s, ReturnStatement) for s in func.statements)
        if not has_return:
            error_msg = self._format_error(
                f"Функция '{func.name}' должна содержать хотя бы один оператор RETURN",
                context="FUNCTION",
                suggestion="Добавьте оператор RETURN перед END"
            )
            self.errors.append(error_msg)

        self.symbol_table = old_symbol_table
        self.implicit_none = old_implicit_none
        self.implicit_rules = old_implicit_rules
        self.loop_nesting = old_loop_nesting

    def _analyze_external_statement(self, stmt: 'ExternalStatement'):
        for name in stmt.names:
            name_upper = name.upper()
            if name_upper in self.symbol_table:
                warning_msg = self._format_warning(
                    f"мя '{name}' в EXTERNAL уже объявлено как переменная",
                    node=stmt,
                    context="EXTERNAL должен объявлять только подпрограммы"
                )
                self.warnings.append(warning_msg)

    def _analyze_common_statement(self, stmt: 'CommonStatement'):
        for block_name, variables in stmt.blocks:
            for var in variables:
                var_name = var.name if hasattr(var, 'name') else str(var)
                var_upper = var_name.upper()
                if var_upper not in self.symbol_table:
                    type_kind, type_size = self._get_implicit_type(var_upper)
                    var_info = VariableInfo(
                        name=var_upper,
                        type_kind=type_kind,
                        is_array=False,
                        dimensions=[],
                        is_parameter=False
                    )
                    self.symbol_table[var_upper] = var_info

    def _analyze_goto_statement(self, stmt: 'GotoStatement'):
        pass

    def _analyze_continue_statement(self, stmt: 'ContinueStatement'):
        pass

    def _analyze_return_statement(self, stmt: 'ReturnStatement'):
        pass

    def _analyze_stop_statement(self, stmt: 'StopStatement'):
        pass

    def _analyze_exit_statement(self, stmt: 'ExitStatement'):
        if self.loop_nesting <= 0:
            error_msg = self._format_error(
                "РћРїРµСЂР°С‚РѕСЂ EXIT РјРѕР¶РЅРѕ РёСЃРїРѕР»СЊР·РѕРІР°С‚СЊ С‚РѕР»СЊРєРѕ РІРЅСѓС‚СЂРё С†РёРєР»Р°",
                node=stmt,
                context="EXIT",
                suggestion="РџРµСЂРµРјРµСЃС‚РёС‚Рµ EXIT РІРЅСѓС‚СЂСЊ DO/DO WHILE РёР»Рё РёСЃРїРѕР»СЊР·СѓР№С‚Рµ GOTO"
            )
            self.errors.append(error_msg)

    def _is_dimension_bound_constant(self, bound: object) -> bool:
        if isinstance(bound, int):
            return True
        return self._is_constant_expression(bound)

    def _evaluate_dimension_bound(self, bound: object):
        if isinstance(bound, int):
            return bound
        return self._evaluate_constant_expression(bound)

    def _resolve_dimension_ranges(self, dim_ranges: Optional[List[object]], node: ASTNode, name: str) -> Optional[List[Tuple[int, int]]]:
        if dim_ranges is None:
            return None
        resolved = []
        for i, dim_spec in enumerate(dim_ranges):
            if isinstance(dim_spec, tuple) and len(dim_spec) == 2:
                lower_bound, upper_bound = dim_spec
            else:
                lower_bound, upper_bound = 1, dim_spec
            if not self._is_dimension_bound_constant(lower_bound) or not self._is_dimension_bound_constant(upper_bound):
                error_msg = self._format_error(
                    f"Р Р°Р·РјРµСЂРЅРѕСЃС‚Рё РјР°СЃСЃРёРІР° '{name}' РґРѕР»Р¶РЅС‹ Р±С‹С‚СЊ РєРѕРЅСЃС‚Р°РЅС‚РЅС‹РјРё",
                    node=node,
                    context=f"РёР·РјРµСЂРµРЅРёРµ {i+1}",
                    suggestion="РСЃРїРѕР»СЊР·СѓР№С‚Рµ С†РµР»РѕС‡РёСЃР»РµРЅРЅС‹Рµ РєРѕРЅСЃС‚Р°РЅС‚С‹ РёР»Рё PARAMETER-РІС‹СЂР°Р¶РµРЅРёСЏ"
                )
                self.errors.append(error_msg)
                return []
            lower_value = self._evaluate_dimension_bound(lower_bound)
            upper_value = self._evaluate_dimension_bound(upper_bound)
            if not isinstance(lower_value, int) or isinstance(lower_value, bool) or not isinstance(upper_value, int) or isinstance(upper_value, bool):
                error_msg = self._format_error(
                    f"Р Р°Р·РјРµСЂРЅРѕСЃС‚Рё РјР°СЃСЃРёРІР° '{name}' РґРѕР»Р¶РЅС‹ Р±С‹С‚СЊ С†РµР»РѕС‡РёСЃР»РµРЅРЅС‹РјРё",
                    node=node,
                    context=f"РёР·РјРµСЂРµРЅРёРµ {i+1}",
                    suggestion="РСЃРїРѕР»СЊР·СѓР№С‚Рµ С†РµР»С‹Рµ РєРѕРЅСЃС‚Р°РЅС‚С‹ РёР»Рё INTEGER PARAMETER"
                )
                self.errors.append(error_msg)
                return []
            if lower_value > upper_value:
                error_msg = self._format_error(
                    f"РќРµРІРµСЂРЅС‹Р№ РґРёР°РїР°Р·РѕРЅ РґР»СЏ РёР·РјРµСЂРµРЅРёСЏ {i+1} РјР°СЃСЃРёРІР° '{name}'",
                    node=node,
                    context=f"РЅР°С‡Р°Р»СЊРЅРѕРµ Р·РЅР°С‡РµРЅРёРµ ({lower_value}) Р±РѕР»СЊС€Рµ РєРѕРЅРµС‡РЅРѕРіРѕ ({upper_value})",
                    suggestion="РЎРґРµР»Р°Р№С‚Рµ РЅРёР¶РЅСЋСЋ РіСЂР°РЅРёС†Сѓ РјРµРЅСЊС€Рµ РёР»Рё СЂР°РІРЅРѕ РІРµСЂС…РЅРµР№"
                )
                self.errors.append(error_msg)
                return []
            resolved.append((lower_value, upper_value))
        return resolved

