"""
Комплексные тесты для SSA и LLVM генераторов.
Тестирует сложные конструкции: вложенные циклы, условные операторы,
массивы, сложные выражения, функции.
"""
import unittest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.core import Lexer, Parser
from src.semantic import SemanticAnalyzer
from src.ssa_generator import SSAGenerator, SSAInstruction
from src.llvm_generator import LLVMGenerator
def compile_code(code: str, generate_ssa: bool = True, generate_llvm: bool = True):
    """Компилирует код и возвращает SSA и/или LLVM IR."""
    lexer = Lexer(code)
    tokens = lexer.tokenize()
    if lexer.get_errors():
        raise Exception(f"Lexer errors: {lexer.get_errors()}")
    parser = Parser(tokens)
    ast = parser.parse()
    semantic = SemanticAnalyzer()
    if not semantic.analyze(ast):
        raise Exception(f"Semantic errors: {semantic.get_errors()}")
    result = {'ssa': None, 'llvm': None}
    if generate_ssa:
        ssa_gen = SSAGenerator()
        ssa_instructions = ssa_gen.generate(ast)
        result['ssa'] = ssa_gen.to_string(ssa_instructions)
        result['ssa_instructions'] = ssa_instructions
    if generate_llvm:
        llvm_gen = LLVMGenerator()
        result['llvm'] = llvm_gen.generate(ast)
    return result
class TestSSAComplex(unittest.TestCase):
    """Комплексные тесты для SSA генератора."""
    def test_nested_loops_ssa(self):
        """Тест вложенных циклов DO с phi-функциями."""
        code = """PROGRAM NESTED
        IMPLICIT NONE
        INTEGER I, J, SUM
        SUM = 0
        DO I = 1, 10
            DO J = 1, 5
                SUM = SUM + I * J
            ENDDO
        ENDDO
        PRINT *, SUM
        END"""
        result = compile_code(code, generate_ssa=True, generate_llvm=False)
        ssa = result['ssa']
        instructions = result['ssa_instructions']
        phi_instrs = [i for i in instructions if i.opcode == "phi"]
        self.assertGreater(len(phi_instrs), 0, "Должны быть phi-функции для переменных циклов")
        self.assertIn("SUM_", ssa, "Должна быть версионированная переменная SUM")
        self.assertIn("I_", ssa, "Должна быть версионированная переменная I")
        self.assertIn("J_", ssa, "Должна быть версионированная переменная J")
        mul_instrs = [i for i in instructions if i.opcode == "*"]
        self.assertGreater(len(mul_instrs), 0, "Должны быть операции умножения")
        add_instrs = [i for i in instructions if i.opcode == "+"]
        self.assertGreater(len(add_instrs), 0, "Должны быть операции сложения")
    def test_complex_if_nested_ssa(self):
        """Тест вложенных условных операторов IF-THEN-ELSE."""
        code = """PROGRAM NESTED_IF
        IMPLICIT NONE
        INTEGER X, Y, Z, RESULT
        X = 10
        Y = 20
        Z = 30
        IF (X .GT. 0) THEN
            IF (Y .GT. 10) THEN
                RESULT = X + Y
            ELSE
                RESULT = X - Y
            ENDIF
        ELSE
            IF (Z .GT. 20) THEN
                RESULT = Z * 2
            ELSE
                RESULT = Z / 2
            ENDIF
        ENDIF
        PRINT *, RESULT
        END"""
        result = compile_code(code, generate_ssa=True, generate_llvm=False)
        ssa = result['ssa']
        instructions = result['ssa_instructions']
        br_instrs = [i for i in instructions if i.opcode == "br" or "br" in str(i.opcode).lower()]
        self.assertGreater(len(br_instrs), 0, "Должны быть условные переходы")
        result_versions = [str(i) for i in instructions if "RESULT" in str(i)]
        self.assertGreater(len(result_versions), 1, "Должно быть несколько версий RESULT")
        cmp_instrs = [i for i in instructions if ".GT." in str(i) or ">" in str(i.opcode)]
        self.assertGreater(len(cmp_instrs), 0, "Должны быть операции сравнения")
    def test_array_operations_ssa(self):
        """Тест операций с массивами и индексацией."""
        code = """PROGRAM ARRAY_OPS
        IMPLICIT NONE
        INTEGER A(10), B(10), C(10)
        INTEGER I
        DO I = 1, 10
            A(I) = I * 2
            B(I) = I * 3
            C(I) = A(I) + B(I)
        ENDDO
        PRINT *, C(5)
        END"""
        result = compile_code(code, generate_ssa=True, generate_llvm=False)
        ssa = result['ssa']
        instructions = result['ssa_instructions']
        array_allocas = [i for i in instructions if i.opcode == "alloca_array"]
        self.assertGreaterEqual(len(array_allocas), 3, "Должны быть аллокации для массивов A, B, C")
        array_accesses = [str(i) for i in instructions if "[" in str(i) or "getelementptr" in str(i.opcode).lower()]
        self.assertGreater(len(array_accesses), 0, "Должны быть операции доступа к массивам")
        self.assertIn("I_", ssa, "Должна быть версионированная переменная I")
    def test_complex_expressions_ssa(self):
        """Тест сложных арифметических и логических выражений."""
        code = """PROGRAM COMPLEX_EXPR
        IMPLICIT NONE
        INTEGER A, B, C, D, RESULT
        LOGICAL FLAG
        A = 10
        B = 20
        C = 30
        D = 5
        RESULT = (A + B) * C / D - A * 2
        FLAG = (A .GT. B) .AND. (C .LT. D)
        IF (FLAG .OR. (RESULT .GT. 0)) THEN
            RESULT = RESULT + 100
        ENDIF
        PRINT *, RESULT, FLAG
        END"""
        result = compile_code(code, generate_ssa=True, generate_llvm=False)
        ssa = result['ssa']
        instructions = result['ssa_instructions']
        mul_instrs = [i for i in instructions if i.opcode == "*"]
        div_instrs = [i for i in instructions if i.opcode == "/"]
        add_instrs = [i for i in instructions if i.opcode == "+"]
        sub_instrs = [i for i in instructions if i.opcode == "-"]
        self.assertGreater(len(mul_instrs), 0, "Должны быть операции умножения")
        self.assertGreater(len(div_instrs), 0, "Должны быть операции деления")
        self.assertGreater(len(add_instrs), 1, "Должны быть операции сложения")
        self.assertGreater(len(sub_instrs), 0, "Должны быть операции вычитания")
        and_instrs = [i for i in instructions if ".AND." in str(i) or "and" in str(i.opcode).lower()]
        or_instrs = [i for i in instructions if ".OR." in str(i) or "or" in str(i.opcode).lower()]
        self.assertGreater(len(and_instrs), 0, "Должны быть логические операции AND")
        self.assertGreater(len(or_instrs), 0, "Должны быть логические операции OR")
    def test_function_calls_ssa(self):
        """Тест вызовов встроенных функций."""
        code = """PROGRAM FUNC_CALLS
        IMPLICIT NONE
        REAL X, Y, Z
        INTEGER I
        X = 2.5
        Y = SQRT(X)
        Z = SIN(X) * COS(X) + EXP(X)
        I = INT(Z)
        PRINT *, Y, Z, I
        END"""
        result = compile_code(code, generate_ssa=True, generate_llvm=False)
        ssa = result['ssa']
        instructions = result['ssa_instructions']
        call_instrs = [i for i in instructions if i.opcode == "call"]
        self.assertGreaterEqual(len(call_instrs), 4, "Должны быть вызовы функций SQRT, SIN, COS, EXP, INT")
        ssa_upper = ssa.upper()
        ssa_lower = ssa.lower()
        self.assertTrue("SQRT" in ssa_upper or "sqrt" in ssa_lower, "Должен быть вызов SQRT")
        self.assertTrue("SIN" in ssa_upper or "sin" in ssa_lower, "Должен быть вызов SIN")
        self.assertTrue("COS" in ssa_upper or "cos" in ssa_lower, "Должен быть вызов COS")
    def test_do_while_loop_ssa(self):
        """Тест цикла DO WHILE."""
        code = """PROGRAM DO_WHILE
        IMPLICIT NONE
        INTEGER COUNTER, SUM
        LOGICAL COND
        COUNTER = 0
        SUM = 0
        DO WHILE (COUNTER .LT. 10)
            COUNTER = COUNTER + 1
            SUM = SUM + COUNTER
        ENDDO
        PRINT *, SUM
        END"""
        result = compile_code(code, generate_ssa=True, generate_llvm=False)
        ssa = result['ssa']
        instructions = result['ssa_instructions']
        phi_instrs = [i for i in instructions if i.opcode == "phi"]
        self.assertGreater(len(phi_instrs), 0, "Должны быть phi-функции для переменных цикла")
        self.assertIn("COUNTER_", ssa, "Должна быть версионированная переменная COUNTER")
        self.assertIn("SUM_", ssa, "Должна быть версионированная переменная SUM")
    def test_multiple_variable_updates_ssa(self):
        """Тест множественных обновлений переменных с версионированием."""
        code = """PROGRAM MULTI_UPDATE
        IMPLICIT NONE
        INTEGER X, Y, Z
        X = 1
        Y = 2
        Z = 3
        X = X + Y
        Y = Y + Z
        Z = Z + X
        X = X + Z
        Y = Y + X
        PRINT *, X, Y, Z
        END"""
        result = compile_code(code, generate_ssa=True, generate_llvm=False)
        ssa = result['ssa']
        instructions = result['ssa_instructions']
        x_versions = [str(i) for i in instructions if "X_" in str(i) or (i.result and "X" in i.result)]
        y_versions = [str(i) for i in instructions if "Y_" in str(i) or (i.result and "Y" in i.result)]
        z_versions = [str(i) for i in instructions if "Z_" in str(i) or (i.result and "Z" in i.result)]
        self.assertGreater(len(x_versions), 3, "Должно быть несколько версий X")
        self.assertGreater(len(y_versions), 3, "Должно быть несколько версий Y")
        self.assertGreater(len(z_versions), 3, "Должно быть несколько версий Z")
    def test_array_indexing_complex_ssa(self):
        """Тест сложной индексации массивов с вычисляемыми индексами."""
        code = """PROGRAM ARRAY_COMPLEX
        IMPLICIT NONE
        INTEGER A(20), B(20)
        INTEGER I, J
        DO I = 1, 10
            J = I * 2
            A(I) = I * 3
            B(J) = A(I) + I
            IF (I .GT. 5) THEN
                A(I + 5) = B(J - 1) * 2
            ENDIF
        ENDDO
        PRINT *, A(10), B(20)
        END"""
        result = compile_code(code, generate_ssa=True, generate_llvm=False)
        ssa = result['ssa']
        instructions = result['ssa_instructions']
        array_ops = [i for i in instructions if "alloca_array" in str(i.opcode) or
                     "getelementptr" in str(i.opcode).lower() or
                     any("[" in str(op) for op in (i.operands or []))]
        self.assertGreater(len(array_ops), 0, "Должны быть операции с массивами")
        mul_instrs = [i for i in instructions if i.opcode == "*"]
        self.assertGreater(len(mul_instrs), 0, "Должны быть вычисления индексов")
    def test_arithmetic_if_ssa(self):
        """Тест арифметического IF с тремя метками."""
        code = """PROGRAM ARITH_IF
        IMPLICIT NONE
        INTEGER X, RESULT
        X = 5
        IF (X - 3) 10, 20, 30
    10  RESULT = 1
        GOTO 40
    20  RESULT = 2
        GOTO 40
    30  RESULT = 3
    40  PRINT *, RESULT
        END"""
        result = compile_code(code, generate_ssa=True, generate_llvm=False)
        ssa = result['ssa']
        instructions = result['ssa_instructions']
        br_instrs = [i for i in instructions if "br" in str(i.opcode).lower()]
        self.assertGreater(len(br_instrs), 0, "Должны быть условные переходы для арифметического IF")
        sub_instrs = [i for i in instructions if i.opcode == "-"]
        self.assertGreater(len(sub_instrs), 0, "Должны быть операции вычитания для условия")
class TestLLVMComplex(unittest.TestCase):
    """Комплексные тесты для LLVM генератора."""
    def test_nested_loops_llvm(self):
        """Тест вложенных циклов с правильной структурой LLVM."""
        code = """PROGRAM NESTED_LOOPS
        IMPLICIT NONE
        INTEGER I, J, SUM
        SUM = 0
        DO I = 1, 5
            DO J = 1, 3
                SUM = SUM + I * J
            ENDDO
        ENDDO
        PRINT *, SUM
        END"""
        result = compile_code(code, generate_ssa=False, generate_llvm=True)
        llvm = result['llvm']
        self.assertIn("define i32 @main", llvm, "Должна быть функция main")
        self.assertIn("alloca i32", llvm, "Должны быть аллокации переменных")
        self.assertIn("phi", llvm, "Должны быть phi-функции для переменных циклов")
        self.assertIn("br", llvm, "Должны быть условные переходы")
        self.assertIn("mul", llvm, "Должны быть операции умножения")
        self.assertIn("add", llvm, "Должны быть операции сложения")
        self.assertIn("label", llvm, "Должны быть метки для циклов")
    def test_complex_if_structure_llvm(self):
        """Тест сложной структуры IF-THEN-ELSE с несколькими ветвями."""
        code = """PROGRAM COMPLEX_IF
        IMPLICIT NONE
        INTEGER X, Y, RESULT
        X = 10
        Y = 20
        IF (X .GT. 5) THEN
            IF (Y .GT. 15) THEN
                RESULT = X + Y
            ELSE
                RESULT = X - Y
            ENDIF
        ELSE
            RESULT = 0
        ENDIF
        PRINT *, RESULT
        END"""
        result = compile_code(code, generate_ssa=False, generate_llvm=True)
        llvm = result['llvm']
        self.assertIn("icmp", llvm, "Должны быть операции сравнения")
        self.assertIn("br i1", llvm, "Должны быть условные переходы")
        self.assertIn("add", llvm, "Должна быть операция сложения")
        self.assertIn("sub", llvm, "Должна быть операция вычитания")
        self.assertIn(":", llvm, "Должны быть метки для ветвлений")
    def test_array_operations_llvm(self):
        """Тест операций с массивами в LLVM IR."""
        code = """PROGRAM ARRAY_TEST
        IMPLICIT NONE
        INTEGER A(10), B(10)
        INTEGER I
        DO I = 1, 10
            A(I) = I * 2
            B(I) = A(I) + 5
        ENDDO
        PRINT *, A(5), B(5)
        END"""
        result = compile_code(code, generate_ssa=False, generate_llvm=True)
        llvm = result['llvm']
        self.assertIn("[10 x i32]", llvm, "Должна быть аллокация массива из 10 элементов")
        self.assertIn("getelementptr", llvm, "Должны быть операции getelementptr для доступа к массивам")
        self.assertIn("store", llvm, "Должны быть операции store для записи в массивы")
        self.assertIn("load", llvm, "Должны быть операции load для чтения из массивов")
    def test_function_calls_llvm(self):
        """Тест вызовов встроенных функций в LLVM IR."""
        code = """PROGRAM FUNC_TEST
        IMPLICIT NONE
        REAL X, Y, Z
        X = 2.5
        Y = SQRT(X)
        Z = SIN(X) + COS(X)
        PRINT *, Y, Z
        END"""
        result = compile_code(code, generate_ssa=False, generate_llvm=True)
        llvm = result['llvm']
        self.assertIn("declare double @sqrt", llvm, "Должно быть объявление функции sqrt")
        self.assertIn("declare double @sin", llvm, "Должно быть объявление функции sin")
        self.assertIn("declare double @cos", llvm, "Должно быть объявление функции cos")
        self.assertIn("call double @sqrt", llvm, "Должен быть вызов sqrt")
        self.assertIn("call double @sin", llvm, "Должен быть вызов sin")
        self.assertIn("call double @cos", llvm, "Должен быть вызов cos")
        self.assertIn("double", llvm, "Должен использоваться тип double")
    def test_complex_expressions_llvm(self):
        """Тест генерации сложных выражений в LLVM IR."""
        code = """PROGRAM EXPR_TEST
        IMPLICIT NONE
        INTEGER A, B, C, RESULT
        REAL X, Y, Z
        A = 10
        B = 20
        C = 30
        RESULT = (A + B) * C / 5 - A * 2
        X = 3.5
        Y = X ** 2.0
        Z = Y / X + 1.0
        PRINT *, RESULT, Z
        END"""
        result = compile_code(code, generate_ssa=False, generate_llvm=True)
        llvm = result['llvm']
        self.assertIn("mul", llvm, "Должны быть операции умножения")
        self.assertIn("sdiv", llvm or "udiv" in llvm, "Должны быть операции деления")
        self.assertIn("add", llvm, "Должны быть операции сложения")
        self.assertIn("sub", llvm, "Должны быть операции вычитания")
        self.assertIn("call double @pow", llvm, "Должен быть вызов pow для возведения в степень")
        self.assertTrue("fmul" in llvm or "mul" in llvm, "Должны быть операции умножения")
        self.assertTrue("fadd" in llvm or "add" in llvm, "Должны быть операции сложения")
        self.assertTrue("fdiv" in llvm or "div" in llvm or "sdiv" in llvm, "Должны быть операции деления")
    def test_do_while_llvm(self):
        """Тест цикла DO WHILE в LLVM IR."""
        code = """PROGRAM DO_WHILE_TEST
        IMPLICIT NONE
        INTEGER COUNTER, SUM
        COUNTER = 0
        SUM = 0
        DO WHILE (COUNTER .LT. 10)
            COUNTER = COUNTER + 1
            SUM = SUM + COUNTER
        ENDDO
        PRINT *, SUM
        END"""
        result = compile_code(code, generate_ssa=False, generate_llvm=True)
        llvm = result['llvm']
        self.assertIn("phi", llvm, "Должны быть phi-функции для переменных цикла")
        self.assertIn("br", llvm, "Должны быть условные переходы")
        self.assertIn("icmp", llvm, "Должна быть операция сравнения для условия цикла")
        self.assertIn("add", llvm, "Должны быть операции сложения для обновления переменных")
    def test_logical_operators_llvm(self):
        """Тест логических операторов в LLVM IR."""
        code = """PROGRAM LOGICAL_TEST
        IMPLICIT NONE
        LOGICAL A, B, C, D, E
        INTEGER X, Y
        X = 10
        Y = 20
        A = .TRUE.
        B = .FALSE.
        C = (X .GT. 5) .AND. (Y .LT. 30)
        D = (X .LT. 5) .OR. (Y .GT. 10)
        E = .NOT. C
        IF (C .EQV. D) THEN
            PRINT *, 'Equal'
        ENDIF
        END"""
        result = compile_code(code, generate_ssa=False, generate_llvm=True)
        llvm = result['llvm']
        self.assertIn("i1", llvm, "Должен использоваться тип i1 для логических значений")
        self.assertIn("icmp", llvm, "Должны быть операции сравнения")
        self.assertTrue("and" in llvm.lower() or "icmp" in llvm, "Должна быть логическая операция AND или сравнение")
        self.assertTrue("or" in llvm.lower() or "icmp" in llvm, "Должна быть логическая операция OR или сравнение")
        self.assertTrue("xor" in llvm.lower() or "icmp" in llvm, "Должны быть логические операции для EQV")
    def test_string_handling_llvm(self):
        """Тест обработки строк в LLVM IR."""
        code = """PROGRAM STRING_TEST
        IMPLICIT NONE
        CHARACTER*20 STR1, STR2, STR3
        STR1 = 'Hello'
        STR2 = 'World'
        STR3 = STR1 // ' ' // STR2
        PRINT *, STR3
        END"""
        result = compile_code(code, generate_ssa=False, generate_llvm=True)
        llvm = result['llvm']
        self.assertIn("@.str", llvm, "Должны быть константы строк")
        self.assertIn("private constant", llvm, "Должны быть приватные константы для строк")
        self.assertIn("i8*", llvm, "Должен использоваться тип i8* для строк")
    def test_multidimensional_array_llvm(self):
        """Тест многомерных массивов в LLVM IR."""
        code = """PROGRAM MULTIDIM_ARRAY
        IMPLICIT NONE
        INTEGER MATRIX(5, 5)
        INTEGER I, J
        DO I = 1, 5
            DO J = 1, 5
                MATRIX(I, J) = I * 5 + J
            ENDDO
        ENDDO
        PRINT *, MATRIX(3, 3)
        END"""
        result = compile_code(code, generate_ssa=False, generate_llvm=True)
        llvm = result['llvm']
        self.assertIn("[5 x", llvm, "Должна быть аллокация многомерного массива")
        self.assertIn("getelementptr", llvm, "Должны быть операции getelementptr для доступа")
        self.assertIn("mul", llvm, "Должны быть вычисления индексов")
    def test_parameter_and_data_llvm(self):
        """Тест PARAMETER и DATA в LLVM IR."""
        code = """PROGRAM PARAM_DATA
        IMPLICIT NONE
        INTEGER I
        PARAMETER (N = 100)
        REAL X(10)
        DATA (X(I), I = 1, 5) / 1.0, 2.0, 3.0, 4.0, 5.0 /
        PRINT *, N
        PRINT *, X(3)
        END"""
        result = compile_code(code, generate_ssa=False, generate_llvm=True)
        llvm = result['llvm']
        self.assertIn("100", llvm, "Должна использоваться константа из PARAMETER")
        self.assertIn("[10 x double]", llvm, "Должен быть массив")
    def test_arithmetic_if_llvm(self):
        """Тест арифметического IF в LLVM IR."""
        code = """PROGRAM ARITH_IF_LLVM
        IMPLICIT NONE
        INTEGER X, RESULT
        X = 5
        IF (X - 3) 10, 20, 30
    10  RESULT = -1
        GOTO 40
    20  RESULT = 0
        GOTO 40
    30  RESULT = 1
    40  PRINT *, RESULT
        END"""
        result = compile_code(code, generate_ssa=False, generate_llvm=True)
        llvm = result['llvm']
        self.assertIn("icmp", llvm, "Должны быть операции сравнения")
        self.assertIn("br", llvm, "Должны быть условные переходы")
        self.assertIn("sub", llvm, "Должна быть операция вычитания для условия")
    def test_complex_real_arithmetic_llvm(self):
        """Тест сложных вещественных вычислений."""
        code = """PROGRAM REAL_COMPLEX
        IMPLICIT NONE
        REAL A, B, C, D, RESULT
        A = 3.14159
        B = 2.71828
        C = SQRT(A * B)
        D = LOG(A) + LOG10(B)
        RESULT = (C ** 2.0) / D
        RESULT = ABS(RESULT) * EXP(-RESULT)
        PRINT *, RESULT
        END"""
        result = compile_code(code, generate_ssa=False, generate_llvm=True)
        llvm = result['llvm']
        self.assertIn("double", llvm, "Должен использоваться тип double")
        self.assertIn("fmul", llvm, "Должны быть операции fmul")
        self.assertIn("fdiv", llvm, "Должны быть операции fdiv")
        self.assertIn("fadd", llvm, "Должны быть операции fadd")
        self.assertIn("call double @sqrt", llvm, "Должен быть вызов sqrt")
        self.assertIn("call double @log", llvm, "Должен быть вызов log")
        self.assertIn("call double @log10", llvm, "Должен быть вызов log10")
        self.assertIn("call double @exp", llvm, "Должен быть вызов exp")
        self.assertIn("call double @fabs", llvm, "Должен быть вызов fabs")
        self.assertIn("call double @pow", llvm, "Должен быть вызов pow")
class TestSSAAndLLVMIntegration(unittest.TestCase):
    """Интеграционные тесты, проверяющие согласованность SSA и LLVM."""
    def test_ssa_llvm_consistency_simple(self):
        """Проверка согласованности SSA и LLVM для простой программы."""
        code = """PROGRAM SIMPLE
        IMPLICIT NONE
        INTEGER X, Y
        X = 10
        Y = X * 2
        PRINT *, Y
        END"""
        result = compile_code(code, generate_ssa=True, generate_llvm=True)
        self.assertIsNotNone(result['ssa'], "SSA должна быть сгенерирована")
        self.assertIsNotNone(result['llvm'], "LLVM IR должен быть сгенерирован")
        self.assertIn("X", result['ssa'], "SSA должна содержать X")
        self.assertIn("Y", result['ssa'], "SSA должна содержать Y")
        self.assertIn("X", result['llvm'], "LLVM должен содержать X")
        self.assertIn("Y", result['llvm'], "LLVM должен содержать Y")
    def test_ssa_llvm_consistency_loop(self):
        """Проверка согласованности для программы с циклом."""
        code = """PROGRAM LOOP_CONSISTENCY
        IMPLICIT NONE
        INTEGER I, SUM
        SUM = 0
        DO I = 1, 10
            SUM = SUM + I
        ENDDO
        PRINT *, SUM
        END"""
        result = compile_code(code, generate_ssa=True, generate_llvm=True)
        self.assertIn("SUM", result['ssa'], "SSA должна содержать SUM")
        self.assertIn("I", result['ssa'], "SSA должна содержать I")
        self.assertIn("SUM", result['llvm'], "LLVM должен содержать SUM")
        self.assertIn("I", result['llvm'], "LLVM должен содержать I")
        ssa_lower = result['ssa'].lower()
        self.assertTrue("+" in result['ssa'] or "add" in ssa_lower, "SSA должна содержать операцию сложения")
        self.assertIn("add", result['llvm'], "LLVM должен содержать операцию add")
    def test_matrix_multiplication_ssa_llvm(self):
        """Тест матричного умножения для проверки сложных массивов."""
        code = """PROGRAM MATRIX_MULT
        IMPLICIT NONE
        INTEGER A(3, 3), B(3, 3), C(3, 3)
        INTEGER I, J, K, SUM
        DO I = 1, 3
            DO J = 1, 3
                A(I, J) = I * 3 + J
                B(I, J) = (I + J) * 2
                SUM = 0
                DO K = 1, 3
                    SUM = SUM + A(I, K) * B(K, J)
                ENDDO
                C(I, J) = SUM
            ENDDO
        ENDDO
        PRINT *, C(2, 2)
        END"""
        result = compile_code(code, generate_ssa=True, generate_llvm=True)
        ssa_has_alloca = "alloca_array" in result['ssa'] or any("alloca" in str(i.opcode).lower() for i in result['ssa_instructions'])
        self.assertTrue(ssa_has_alloca, "SSA должна содержать аллокации массивов")
        self.assertIn("[3 x", result['llvm'], "LLVM должен содержать массивы размером 3")
        self.assertIn("getelementptr", result['llvm'], "LLVM должен использовать getelementptr для доступа к массивам")
    def test_recursive_control_flow_ssa_llvm(self):
        """Тест рекурсивной структуры управления (вложенные IF в циклах)."""
        code = """PROGRAM RECURSIVE_CF
        IMPLICIT NONE
        INTEGER I, J, COUNT
        COUNT = 0
        DO I = 1, 10
            IF (I .GT. 5) THEN
                DO J = 1, I
                    IF (J .LT. 5) THEN
                        COUNT = COUNT + 1
                    ELSE
                        COUNT = COUNT + 2
                    ENDIF
                ENDDO
            ELSE
                COUNT = COUNT + I
            ENDIF
        ENDDO
        PRINT *, COUNT
        END"""
        result = compile_code(code, generate_ssa=True, generate_llvm=True)
        instructions = result['ssa_instructions']
        branch_instrs = [i for i in instructions if "branch" in str(i.opcode).lower() or "br" in str(i.opcode).lower()]
        self.assertGreater(len(branch_instrs), 3, "Должно быть много условных переходов")
        phi_instrs = [i for i in instructions if i.opcode == "phi"]
        self.assertGreater(len(phi_instrs), 0, "Должны быть phi-функции")
        self.assertIn("br i1", result['llvm'], "LLVM должен содержать условные переходы")
        self.assertIn("phi", result['llvm'], "LLVM должен содержать phi-функции")
    def test_complex_numerical_computation_llvm(self):
        """Тест сложных численных вычислений (метод Ньютона)."""
        code = """PROGRAM NEWTON
        IMPLICIT NONE
        REAL X, EPS, F, FP, DELTA
        INTEGER ITER
        X = 2.0
        EPS = 1.0E-6
        ITER = 0
        DO WHILE (ABS(F) .GT. EPS .AND. ITER .LT. 100)
            F = X ** 2.0 - 2.0
            FP = 2.0 * X
            IF (ABS(FP) .LT. 1.0E-10) THEN
                STOP
            ENDIF
            DELTA = F / FP
            X = X - DELTA
            ITER = ITER + 1
        ENDDO
        PRINT *, 'Root:', X
        END"""
        result = compile_code(code, generate_ssa=False, generate_llvm=True)
        llvm = result['llvm']
        self.assertIn("fmul", llvm, "Должны быть операции fmul")
        self.assertIn("fdiv", llvm, "Должны быть операции fdiv")
        self.assertIn("fsub", llvm, "Должны быть операции fsub")
        self.assertIn("call double @pow", llvm, "Должен быть вызов pow")
        self.assertIn("call double @fabs", llvm, "Должен быть вызов fabs")
        self.assertIn("phi", llvm, "Должны быть phi-функции для цикла")
        self.assertIn("br", llvm, "Должны быть условные переходы")
    def test_array_sorting_pattern_ssa(self):
        """Тест паттерна сортировки массива (пузырьковая сортировка)."""
        code = """PROGRAM SORT
        IMPLICIT NONE
        INTEGER ARR(10), I, J, TEMP
        INTEGER N
        N = 10
        DO I = 1, N - 1
            DO J = 1, N - I
                IF (ARR(J) .GT. ARR(J + 1)) THEN
                    TEMP = ARR(J)
                    ARR(J) = ARR(J + 1)
                    ARR(J + 1) = TEMP
                ENDIF
            ENDDO
        ENDDO
        PRINT *, ARR(1), ARR(10)
        END"""
        result = compile_code(code, generate_ssa=True, generate_llvm=False)
        ssa = result['ssa']
        instructions = result['ssa_instructions']
        array_accesses = [str(i) for i in instructions if "store_array" in str(i.opcode) or
                         "load_array" in str(i.opcode) or "[" in str(i)]
        self.assertGreater(len(array_accesses), 5, "Должно быть много операций с массивами")
        self.assertIn("I_", ssa, "Должна быть переменная I")
        self.assertIn("J_", ssa, "Должна быть переменная J")
    def test_complex_expression_evaluation_llvm(self):
        """Тест оценки сложных выражений с приоритетами операций."""
        code = """PROGRAM EXPR_EVAL
        IMPLICIT NONE
        INTEGER A, B, C, D, E, RESULT
        REAL X, Y, Z
        A = 2
        B = 3
        C = 4
        D = 5
        E = 6
        RESULT = A + B * C ** D / E - A * B + C
        X = 3.0
        Y = 2.0
        Z = (X + Y) ** 2.0 / (X - Y) + SQRT(X * Y)
        PRINT *, RESULT, Z
        END"""
        result = compile_code(code, generate_ssa=False, generate_llvm=True)
        llvm = result['llvm']
        self.assertIn("mul", llvm, "Должны быть операции умножения")
        self.assertIn("sdiv", llvm or "udiv" in llvm, "Должны быть операции деления")
        self.assertIn("add", llvm, "Должны быть операции сложения")
        self.assertIn("sub", llvm, "Должны быть операции вычитания")
        self.assertIn("call double @pow", llvm, "Должен быть вызов pow для **")
        self.assertIn("fmul", llvm, "Должны быть операции fmul")
        self.assertIn("fdiv", llvm, "Должны быть операции fdiv")
        self.assertIn("fadd", llvm, "Должны быть операции fadd")
        self.assertIn("fsub", llvm, "Должны быть операции fsub")
    def test_conditional_assignment_ssa(self):
        """Тест условных присваиваний с множественными ветвями."""
        code = """PROGRAM COND_ASSIGN
        IMPLICIT NONE
        INTEGER X, Y, Z, RESULT
        X = 10
        Y = 20
        Z = 30
        IF (X .GT. Y) THEN
            RESULT = X * 2
        ELSE IF (Y .GT. Z) THEN
            RESULT = Y * 2
        ELSE IF (Z .GT. X) THEN
            RESULT = Z * 2
        ELSE
            RESULT = 0
        ENDIF
        IF (RESULT .GT. 50) THEN
            RESULT = RESULT / 2
        ENDIF
        PRINT *, RESULT
        END"""
        result = compile_code(code, generate_ssa=True, generate_llvm=False)
        ssa = result['ssa']
        instructions = result['ssa_instructions']
        branch_instrs = [i for i in instructions if "branch" in str(i.opcode).lower()]
        self.assertGreater(len(branch_instrs), 2, "Должно быть несколько условных переходов")
        result_ops = [str(i) for i in instructions if "RESULT" in str(i)]
        self.assertGreater(len(result_ops), 3, "Должно быть несколько операций с RESULT")
    def test_loop_with_break_condition_llvm(self):
        """Тест цикла с условием выхода в середине."""
        code = """PROGRAM LOOP_BREAK
        IMPLICIT NONE
        INTEGER I, SUM, TARGET
        SUM = 0
        TARGET = 100
        DO I = 1, 100
            SUM = SUM + I
            IF (SUM .GT. TARGET) THEN
                EXIT
            ENDIF
        ENDDO
        PRINT *, SUM, I
        END"""
        result = compile_code(code, generate_ssa=False, generate_llvm=True)
        llvm = result['llvm']
        self.assertIn("phi", llvm, "Должны быть phi-функции")
        self.assertIn("icmp", llvm, "Должна быть проверка условия")
        self.assertIn("br", llvm, "Должны быть условные переходы")
    def test_mixed_types_operations_llvm(self):
        """Тест операций со смешанными типами (целые и вещественные)."""
        code = """PROGRAM MIXED_TYPES
        IMPLICIT NONE
        INTEGER I, J
        REAL X, Y
        I = 10
        J = 3
        X = 5.5
        Y = REAL(I) + X / REAL(J)
        Y = Y * 2.0
        I = INT(Y)
        PRINT *, I, Y
        END"""
        result = compile_code(code, generate_ssa=False, generate_llvm=True)
        llvm = result['llvm']
        self.assertTrue("sitofp" in llvm or "fptosi" in llvm or "fpext" in llvm or "trunc" in llvm,
                       "Должны быть преобразования типов")
        self.assertIn("double", llvm, "Должен использоваться тип double")
        self.assertIn("i32", llvm, "Должен использоваться тип i32")
    def test_string_concatenation_llvm(self):
        """Тест конкатенации строк."""
        code = """PROGRAM STRING_CONCAT
        IMPLICIT NONE
        CHARACTER*20 STR1, STR2, STR3, RESULT
        STR1 = 'Hello'
        STR2 = 'World'
        STR3 = 'Test'
        RESULT = STR1 // ' ' // STR2 // ' ' // STR3
        PRINT *, RESULT
        END"""
        result = compile_code(code, generate_ssa=False, generate_llvm=True)
        llvm = result['llvm']
        self.assertIn("i8*", llvm, "Должен использоваться тип i8* для строк")
        self.assertIn("@.str", llvm, "Должны быть строковые константы")
        self.assertIn("private constant", llvm, "Должны быть приватные константы")
    def test_parameter_usage_in_expressions_llvm(self):
        """Тест использования PARAMETER в выражениях."""
        code = """PROGRAM PARAM_EXPR
        IMPLICIT NONE
        INTEGER I, J
        PARAMETER (N = 10, M = 5)
        REAL X
        PARAMETER (PI = 3.14159)
        DO I = 1, N
            J = I * M
            X = REAL(J) * PI
        ENDDO
        PRINT *, J, X
        END"""
        result = compile_code(code, generate_ssa=False, generate_llvm=True)
        llvm = result['llvm']
        self.assertIn("10", llvm, "Должна использоваться константа 10")
        self.assertIn("5", llvm, "Должна использоваться константа 5")
        self.assertIn("mul", llvm, "Должны быть операции с константами")
if __name__ == '__main__':
    unittest.main(verbosity=2)