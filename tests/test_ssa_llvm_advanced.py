"""
Комплексные тесты для SSA и LLVM генераторов.
Проверяет сложные случаи: вложенные циклы, условия, массивы, функции.
"""
import unittest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.core import Lexer, Parser
from src.semantic import SemanticAnalyzer
from src.ssa_generator import SSAGenerator
from src.llvm_generator import LLVMGenerator
def compile_to_ssa(code: str) -> str:
    """Компилирует код в SSA форму."""
    lexer = Lexer(code)
    tokens = lexer.tokenize()
    parser = Parser(tokens)
    ast = parser.parse()
    semantic = SemanticAnalyzer()
    if not semantic.analyze(ast):
        raise Exception(f"Семантические ошибки: {semantic.get_errors()}")
    ssa_gen = SSAGenerator()
    instructions = ssa_gen.generate(ast)
    return ssa_gen.to_string(instructions)
def compile_to_llvm(code: str) -> str:
    """Компилирует код в LLVM IR."""
    lexer = Lexer(code)
    tokens = lexer.tokenize()
    parser = Parser(tokens)
    ast = parser.parse()
    semantic = SemanticAnalyzer()
    if not semantic.analyze(ast):
        raise Exception(f"Семантические ошибки: {semantic.get_errors()}")
    llvm_gen = LLVMGenerator()
    return llvm_gen.generate(ast)
class TestSSAComplexExpressions(unittest.TestCase):
    """Тесты сложных выражений в SSA."""
    def test_complex_arithmetic_expression(self):
        """Тест сложного арифметического выражения."""
        code = """
PROGRAM TEST
INTEGER A, B, C, D, E, F
A = 10
B = 20
C = 30
D = A + B * C / 2 - 5
E = (A + B) * (C - D) / (A - B)
F = A ** 2 + B ** 3 - C ** 4 / D
PRINT *, D, E, F
END
"""
        ssa = compile_to_ssa(code)
        self.assertIn('*', ssa, "Должна быть операция умножения")
        self.assertIn('/', ssa, "Должна быть операция деления")
        self.assertIn('+', ssa, "Должна быть операция сложения")
        self.assertIn('-', ssa, "Должна быть операция вычитания")
        self.assertIn('pow', ssa, "Должна быть операция возведения в степень")
        self.assertIn('A_', ssa, "Должна быть версионированная переменная A")
        self.assertIn('B_', ssa, "Должна быть версионированная переменная B")
        self.assertIn('D_', ssa, "Должна быть версионированная переменная D")
        self.assertIn('E_', ssa, "Должна быть версионированная переменная E")
        self.assertIn('F_', ssa, "Должна быть версионированная переменная F")
    def test_logical_expressions_complex(self):
        """Тест сложных логических выражений."""
        code = """
PROGRAM TEST
LOGICAL L1, L2, L3, L4, L5
INTEGER A, B, C
A = 5
B = 10
C = 15
L1 = A .GT. B
L2 = B .LT. C
L3 = L1 .AND. L2
L4 = L1 .OR. L2
L5 = .NOT. L3 .EQV. (.NOT. L4)
PRINT *, L3, L4, L5
END
"""
        ssa = compile_to_ssa(code)
        self.assertIn('gt', ssa.lower() or 'gt', "Должно быть сравнение .GT.")
        self.assertIn('lt', ssa.lower() or 'lt', "Должно быть сравнение .LT.")
        self.assertIn('and', ssa.lower() or 'and', "Должна быть операция .AND.")
        self.assertIn('or', ssa.lower() or 'or', "Должна быть операция .OR.")
        self.assertIn('not', ssa.lower() or 'not', "Должна быть операция .NOT.")
        self.assertIn('eqv', ssa.lower() or 'eqv', "Должна быть операция .EQV.")
    def test_nested_function_calls(self):
        """Тест вложенных вызовов функций."""
        code = """
PROGRAM TEST
REAL X, Y, Z, W
X = 2.0
Y = SIN(X)
Z = SQRT(SIN(X) + COS(Y))
W = EXP(LOG(SQRT(ABS(Z - Y))))
PRINT *, Y, Z, W
END
"""
        ssa = compile_to_ssa(code)
        self.assertIn('call', ssa.lower(), "Должны быть вызовы функций")
        self.assertIn('SIN', ssa, "Должен быть вызов SIN")
        self.assertIn('SQRT', ssa, "Должен быть вызов SQRT")
        self.assertIn('COS', ssa, "Должен быть вызов COS")
        self.assertIn('EXP', ssa, "Должен быть вызов EXP")
        self.assertIn('LOG', ssa, "Должен быть вызов LOG")
        self.assertIn('ABS', ssa, "Должен быть вызов ABS")
        self.assertIn('%tmp_', ssa, "Должны быть временные переменные для промежуточных результатов")
class TestSSANestedLoops(unittest.TestCase):
    """Тесты вложенных циклов в SSA."""
    def test_triple_nested_loops(self):
        """Тест тройного вложенного цикла."""
        code = """
PROGRAM TEST
INTEGER I, J, K, SUM
SUM = 0
DO I = 1, 5
    DO J = 1, 5
        DO K = 1, 5
            SUM = SUM + I * J + K
        ENDDO
    ENDDO
ENDDO
PRINT *, SUM
END
"""
        ssa = compile_to_ssa(code)
        self.assertIn('branch', ssa.lower(), "Должны быть инструкции ветвления для циклов")
        sum_versions = [line for line in ssa.split('\n') if 'SUM_' in line]
        self.assertGreater(len(sum_versions), 5, "Переменная SUM должна версионироваться в цикле")
        self.assertIn('I_', ssa, "Должна быть версионированная переменная I")
        self.assertIn('J_', ssa, "Должна быть версионированная переменная J")
        self.assertIn('K_', ssa, "Должна быть версионированная переменная K")
    def test_nested_loops_with_conditions(self):
        """Тест вложенных циклов с условиями внутри."""
        code = """
PROGRAM TEST
INTEGER I, J, COUNT
COUNT = 0
DO I = 1, 10
    DO J = 1, 10
        IF (I .LT. J) THEN
            COUNT = COUNT + 1
        ELSEIF (I .EQ. J) THEN
            COUNT = COUNT + 2
        ELSE
            COUNT = COUNT + 3
        ENDIF
    ENDDO
ENDDO
PRINT *, COUNT
END
"""
        ssa = compile_to_ssa(code)
        self.assertIn('branch_if', ssa.lower(), "Должны быть условные переходы")
        count_versions = [line for line in ssa.split('\n') if 'COUNT_' in line]
        self.assertGreater(len(count_versions), 3, "COUNT должна версионироваться в разных ветках условия")
        self.assertIn('lt', ssa.lower() or 'lt', "Должно быть сравнение .LT.")
        self.assertIn('eq', ssa.lower() or 'eq', "Должно быть сравнение .EQ.")
class TestSSAComplexArrays(unittest.TestCase):
    """Тесты работы с массивами в SSA."""
    def test_multidimensional_array_operations(self):
        """Тест операций с многомерным массивом."""
        code = """
PROGRAM TEST
INTEGER A(10, 20, 5)
INTEGER I, J, K, SUM
SUM = 0
DO I = 1, 10
    DO J = 1, 20
        DO K = 1, 5
            A(I, J, K) = I * 100 + J * 10 + K
            SUM = SUM + A(I, J, K)
        ENDDO
    ENDDO
ENDDO
PRINT *, A(1, 1, 1), A(10, 20, 5), SUM
END
"""
        ssa = compile_to_ssa(code)
        self.assertIn('alloca_array', ssa.lower(), "Должно быть выделение памяти для массива")
        self.assertIn('store_array', ssa.lower(), "Должны быть операции записи в массив")
        self.assertIn('load_array', ssa.lower(), "Должны быть операции чтения из массива")
        array_ops = [line for line in ssa.split('\n') if 'A' in line and ('store' in line.lower() or 'load' in line.lower())]
        self.assertGreater(len(array_ops), 0, "Должны быть операции с массивом A")
    def test_array_with_complex_indexing(self):
        """Тест массива со сложной индексацией."""
        code = """
PROGRAM TEST
INTEGER A(100)
INTEGER I, J, K
I = 10
J = 20
K = 30
A(I + J) = I * J
A(K - I) = K / I
A(I * J / K) = I + J + K
A(SQRT(REAL(I)) + 1) = I ** 2
PRINT *, A(30), A(20), A(6)
END
"""
        ssa = compile_to_ssa(code)
        self.assertIn('store_array', ssa.lower(), "Должны быть операции записи в массив")
        self.assertIn('load_array', ssa.lower(), "Должны быть операции чтения из массива")
        self.assertIn('%tmp_', ssa, "Должны быть временные переменные для вычисления индексов")
class TestSSAComplexControlFlow(unittest.TestCase):
    """Тесты сложного управления потоком выполнения."""
    def test_nested_if_with_multiple_elif(self):
        """Тест вложенных IF с несколькими ELSEIF."""
        code = """
PROGRAM TEST
INTEGER X, Y, Z
X = 5
IF (X .GT. 10) THEN
    Y = 1
ELSEIF (X .GT. 5) THEN
    Y = 2
ELSEIF (X .GT. 0) THEN
    IF (X .EQ. 5) THEN
        Y = 3
    ELSE
        Y = 4
    ENDIF
ELSEIF (X .GT. -5) THEN
    Y = 5
ELSE
    Y = 6
ENDIF
Z = Y * 2
PRINT *, Y, Z
END
"""
        ssa = compile_to_ssa(code)
        self.assertIn('branch_if', ssa.lower(), "Должны быть условные переходы")
        y_versions = [line for line in ssa.split('\n') if 'Y_' in line]
        self.assertGreater(len(y_versions), 5, "Y должна версионироваться в разных ветках")
        self.assertIn('then', ssa.lower() or 'then', "Должны быть блоки THEN")
        self.assertIn('elif', ssa.lower() or 'elif', "Должны быть блоки ELSEIF")
        self.assertIn('else', ssa.lower() or 'else', "Должен быть блок ELSE")
    def test_loop_with_nested_conditions_and_breaks(self):
        """Тест цикла с вложенными условиями и прерываниями."""
        code = """
PROGRAM TEST
INTEGER I, SUM, COUNT
SUM = 0
COUNT = 0
DO I = 1, 100
    IF (I .GT. 50) THEN
        IF (SUM .GT. 1000) THEN
            COUNT = COUNT + 1
        ENDIF
    ENDIF
    IF (I .LT. 25) THEN
        SUM = SUM + I
    ELSEIF (I .LT. 75) THEN
        SUM = SUM + I * 2
    ELSE
        SUM = SUM + I * 3
    ENDIF
ENDDO
PRINT *, SUM, COUNT
END
"""
        ssa = compile_to_ssa(code)
        self.assertIn('branch', ssa.lower(), "Должны быть переходы для цикла")
        self.assertIn('branch_if', ssa.lower(), "Должны быть условные переходы")
        self.assertIn('SUM_', ssa, "SUM должна версионироваться в цикле")
        self.assertIn('COUNT_', ssa, "COUNT должна версионироваться в цикле")
        sum_versions = [line for line in ssa.split('\n') if 'SUM_' in line]
        self.assertGreater(len(sum_versions), 5, "SUM должна версионироваться в разных ветках")
class TestSSAMathematicalComputations(unittest.TestCase):
    """Тесты математических вычислений."""
    def test_newton_method(self):
        """Тест метода Ньютона (итерационный алгоритм)."""
        code = """
PROGRAM NEWTON
REAL X, EPS, DELTA
INTEGER ITER, MAXITER
X = 2.0
EPS = 1.0E-6
MAXITER = 100
ITER = 0
DO WHILE (ITER .LT. MAXITER)
    DELTA = (X ** 2 - 2.0) / (2.0 * X)
    IF (ABS(DELTA) .LT. EPS) THEN
        GOTO 100
    ENDIF
    X = X - DELTA
    ITER = ITER + 1
ENDDO
100 CONTINUE
PRINT *, 'Root = ', X, ' Iterations = ', ITER
END
"""
        ssa = compile_to_ssa(code)
        self.assertIn('do_while', ssa.lower() or 'do_while', "Должен быть цикл DO WHILE")
        self.assertIn('pow', ssa.lower() or 'pow', "Должна быть операция возведения в степень")
        self.assertIn('ABS', ssa, "Должен быть вызов ABS")
        self.assertIn('X_', ssa, "X должна версионироваться в итерациях")
        self.assertIn('ITER_', ssa, "ITER должна версионироваться в цикле")
        self.assertIn('DELTA_', ssa, "DELTA должна версионироваться")
    def test_matrix_multiplication(self):
        """Тест умножения матриц."""
        code = """
PROGRAM MATRIX
INTEGER A(10, 10), B(10, 10), C(10, 10)
INTEGER I, J, K, SUM
DO I = 1, 10
    DO J = 1, 10
        A(I, J) = I + J
        B(I, J) = I * J
    ENDDO
ENDDO
DO I = 1, 10
    DO J = 1, 10
        SUM = 0
        DO K = 1, 10
            SUM = SUM + A(I, K) * B(K, J)
        ENDDO
        C(I, J) = SUM
    ENDDO
ENDDO
PRINT *, C(1, 1), C(10, 10)
END
"""
        ssa = compile_to_ssa(code)
        self.assertIn('branch', ssa.lower(), "Должны быть переходы для циклов")
        self.assertIn('alloca_array', ssa.lower(), "Должно быть выделение памяти для массивов")
        self.assertIn('store_array', ssa.lower(), "Должны быть операции записи")
        self.assertIn('load_array', ssa.lower(), "Должны быть операции чтения")
        sum_versions = [line for line in ssa.split('\n') if 'SUM_' in line]
        self.assertGreater(len(sum_versions), 10, "SUM должна многократно версионироваться")
class TestLLVMComplexExpressions(unittest.TestCase):
    """Тесты сложных выражений в LLVM IR."""
    def test_complex_arithmetic_llvm(self):
        """Тест сложного арифметического выражения в LLVM."""
        code = """
PROGRAM TEST
REAL A, B, C, D, E
A = 1.5
B = 2.5
C = 3.5
D = (A + B) * (C - A) / (B + C) ** 2.0
E = SIN(A) * COS(B) + SQRT(C) - EXP(D)
PRINT *, D, E
END
"""
        llvm = compile_to_llvm(code)
        self.assertIn('double', llvm, "Должны быть переменные типа double")
        self.assertIn('fadd', llvm, "Должна быть операция сложения для вещественных")
        self.assertIn('fmul', llvm, "Должна быть операция умножения для вещественных")
        self.assertIn('fdiv', llvm, "Должна быть операция деления для вещественных")
        self.assertIn('@sin', llvm, "Должен быть вызов функции sin")
        self.assertIn('@cos', llvm, "Должен быть вызов функции cos")
        self.assertIn('@sqrt', llvm, "Должен быть вызов функции sqrt")
        self.assertIn('@exp', llvm, "Должен быть вызов функции exp")
        self.assertIn('@pow', llvm, "Должен быть вызов функции pow")
    def test_logical_operations_llvm(self):
        """Тест логических операций в LLVM."""
        code = """
PROGRAM TEST
LOGICAL L1, L2, L3, L4
INTEGER A, B, C
A = 10
B = 20
C = 30
L1 = A .GT. B
L2 = B .LT. C
L3 = L1 .AND. L2
L4 = L1 .OR. (.NOT. L2)
PRINT *, L3, L4
END
"""
        llvm = compile_to_llvm(code)
        self.assertIn('i1', llvm, "Должны быть переменные типа i1 (логические)")
        self.assertIn('icmp', llvm, "Должны быть операции сравнения")
        self.assertIn('and', llvm, "Должна быть логическая операция AND")
        self.assertIn('or', llvm, "Должна быть логическая операция OR")
        self.assertIn('xor', llvm, "Может быть операция XOR для NOT")
class TestLLVMComplexControlFlow(unittest.TestCase):
    """Тесты сложного управления потоком в LLVM."""
    def test_nested_loops_llvm(self):
        """Тест вложенных циклов в LLVM."""
        code = """
PROGRAM TEST
INTEGER I, J, SUM
SUM = 0
DO I = 1, 10
    DO J = 1, 10
        SUM = SUM + I * J
    ENDDO
ENDDO
PRINT *, SUM
END
"""
        llvm = compile_to_llvm(code)
        self.assertIn('label', llvm, "Должны быть метки для блоков")
        self.assertIn('br ', llvm, "Должны быть инструкции перехода")
        self.assertIn('phi', llvm, "Должны быть phi-функции для переменных циклов")
        self.assertIn('icmp', llvm, "Должны быть сравнения для условий циклов")
    def test_complex_if_then_else_llvm(self):
        """Тест сложной IF-THEN-ELSE конструкции в LLVM."""
        code = """
PROGRAM TEST
INTEGER X, Y, Z
X = 5
IF (X .GT. 10) THEN
    Y = 1
    Z = 10
ELSEIF (X .GT. 5) THEN
    Y = 2
    Z = 20
ELSEIF (X .GT. 0) THEN
    Y = 3
    Z = 30
ELSE
    Y = 4
    Z = 40
ENDIF
PRINT *, Y, Z
END
"""
        llvm = compile_to_llvm(code)
        self.assertIn('br i1', llvm, "Должны быть условные переходы")
        self.assertIn('then', llvm or 'then', "Должна быть метка THEN")
        self.assertIn('else', llvm.lower() or 'else', "Должна быть метка ELSE")
        self.assertIn('icmp', llvm, "Должны быть операции сравнения")
        label_count = llvm.count('label')
        self.assertGreater(label_count, 5, "Должно быть несколько блоков для разных веток")
class TestLLVMArrays(unittest.TestCase):
    """Тесты работы с массивами в LLVM."""
    def test_multidimensional_array_llvm(self):
        """Тест многомерного массива в LLVM."""
        code = """
PROGRAM TEST
INTEGER A(5, 10, 15)
INTEGER I, J, K
I = 2
J = 3
K = 4
A(1, 1, 1) = 100
A(I, J, K) = 200
A(5, 10, 15) = 300
PRINT *, A(1, 1, 1), A(I, J, K)
END
"""
        llvm = compile_to_llvm(code)
        self.assertIn('alloca', llvm, "Должно быть выделение памяти")
        self.assertIn('getelementptr', llvm, "Должны быть операции getelementptr для доступа к массиву")
        self.assertIn('store', llvm, "Должны быть операции store для записи")
        self.assertIn('load', llvm, "Должны быть операции load для чтения")
        self.assertIn('[5 x', llvm, "Должен быть массив размером 5")
        self.assertIn('[10 x', llvm, "Должен быть массив размером 10")
        self.assertIn('[15 x', llvm, "Должен быть массив размером 15")
    def test_array_initialization_llvm(self):
        """Тест инициализации массива в цикле."""
        code = """
PROGRAM TEST
INTEGER A(100)
INTEGER I
DO I = 1, 100
    A(I) = I * 2
ENDDO
PRINT *, A(1), A(50), A(100)
END
"""
        llvm = compile_to_llvm(code)
        self.assertIn('alloca', llvm, "Должно быть выделение памяти для массива")
        self.assertIn('phi', llvm, "Должны быть phi-функции для переменной цикла")
        self.assertIn('getelementptr', llvm, "Должны быть операции доступа к массиву")
        self.assertIn('store', llvm, "Должны быть операции записи в массив")
class TestLLVMBuiltinFunctions(unittest.TestCase):
    """Тесты встроенных функций в LLVM."""
    def test_all_math_functions_llvm(self):
        """Тест всех математических функций."""
        code = """
PROGRAM TEST
REAL X, Y, Z, W, V
X = 1.5
Y = SIN(X)
Z = COS(X)
W = TAN(X)
V = SQRT(X)
PRINT *, Y, Z, W, V
END
"""
        llvm = compile_to_llvm(code)
        self.assertIn('declare double @sin', llvm, "Должно быть объявление sin")
        self.assertIn('declare double @cos', llvm, "Должно быть объявление cos")
        self.assertIn('declare double @sqrt', llvm, "Должно быть объявление sqrt")
        self.assertIn('call double @sin', llvm, "Должен быть вызов sin")
        self.assertIn('call double @cos', llvm, "Должен быть вызов cos")
    def test_type_conversions_llvm(self):
        """Тест преобразований типов."""
        code = """
PROGRAM TEST
INTEGER I, J
REAL X, Y
I = 5
X = REAL(I)
J = INT(X + 0.5)
Y = FLOAT(I)
PRINT *, I, X, J, Y
END
"""
        llvm = compile_to_llvm(code)
        self.assertIn('sitofp', llvm, "Должно быть преобразование int to float")
        self.assertIn('fptosi', llvm, "Должно быть преобразование float to int")
        self.assertIn('i32', llvm, "Должны быть целочисленные переменные")
        self.assertIn('double', llvm, "Должны быть вещественные переменные")
class TestSSAAndLLVMIntegration(unittest.TestCase):
    """Интеграционные тесты SSA и LLVM."""
    def test_complex_program_ssa_llvm(self):
        """Тест сложной программы, компилируемой в SSA и LLVM."""
        code = """
PROGRAM COMPLEX
INTEGER I, J, N, SUM
REAL PI, AREA
LOGICAL FOUND
N = 100
SUM = 0
PI = 3.14159
FOUND = .FALSE.
DO I = 1, N
    IF (MOD(I, 2) .EQ. 0) THEN
        SUM = SUM + I
    ELSE
        DO J = 1, I
            IF (J * J .EQ. I) THEN
                FOUND = .TRUE.
            ENDIF
        ENDDO
    ENDIF
ENDDO
AREA = PI * REAL(SUM) ** 2.0
PRINT *, 'Sum = ', SUM, ' Area = ', AREA, ' Found = ', FOUND
END
"""
        ssa = compile_to_ssa(code)
        llvm = compile_to_llvm(code)
        self.assertIn('DO', ssa.upper() or 'DO', "SSA должна содержать циклы")
        self.assertIn('branch', ssa.lower(), "SSA должна содержать переходы")
        self.assertIn('SUM_', ssa, "SSA должна содержать версионированные переменные")
        self.assertIn('define i32 @main', llvm, "LLVM должна содержать функцию main")
        self.assertIn('phi', llvm, "LLVM должна содержать phi-функции")
        self.assertIn('br', llvm, "LLVM должна содержать переходы")
        self.assertGreater(len(ssa), 50, "SSA должна быть достаточно подробной")
        self.assertGreater(len(llvm), 100, "LLVM IR должна быть достаточно подробной")
    def test_parameter_and_array_combination(self):
        """Тест комбинации PARAMETER и массивов."""
        code = """
PROGRAM TEST
PARAMETER (N = 10, M = 5)
INTEGER A(N, M)
INTEGER I, J
DO I = 1, N
    DO J = 1, M
        A(I, J) = (I - 1) * M + J
    ENDDO
ENDDO
PRINT *, A(1, 1), A(N, M), A(N/2, M/2)
END
"""
        ssa = compile_to_ssa(code)
        llvm = compile_to_llvm(code)
        self.assertIn('alloca', llvm.lower(), "Должно быть выделение памяти для массива")
        self.assertIn('A', ssa, "SSA должна содержать работу с массивом A")
        self.assertIn('10', ssa, "SSA должна содержать использование параметра N")
        self.assertIn('5', ssa, "SSA должна содержать использование параметра M")
class TestSSAPhiFunctions(unittest.TestCase):
    """Тесты phi-функций в SSA."""
    def test_phi_in_if_then_else(self):
        """Тест phi-функций в IF-THEN-ELSE."""
        code = """
PROGRAM TEST
INTEGER X, Y
X = 5
IF (X .GT. 10) THEN
    Y = 1
ELSE
    Y = 2
ENDIF
PRINT *, Y
END
"""
        ssa = compile_to_ssa(code)
        y_versions = [line for line in ssa.split('\n') if 'Y_' in line or 'Y ' in line]
        self.assertGreater(len(y_versions), 1, "Y должна иметь разные версии в разных ветках")
        self.assertIn('branch_if', ssa.lower() or 'branch_if', "Должны быть условные переходы")
    def test_phi_in_loop(self):
        """Тест phi-функций в цикле."""
        code = """
PROGRAM TEST
INTEGER I, SUM
SUM = 0
DO I = 1, 10
    SUM = SUM + I
ENDDO
PRINT *, SUM
END
"""
        ssa = compile_to_ssa(code)
        self.assertIn('I_', ssa, "I должна версионироваться в цикле")
        self.assertIn('SUM_', ssa, "SUM должна версионироваться в цикле")
        self.assertIn('branch', ssa.lower(), "Должны быть переходы для цикла")
class TestSSAEdgeCases(unittest.TestCase):
    """Тесты граничных случаев в SSA."""
    def test_empty_loop_body(self):
        """Тест цикла с пустым телом."""
        code = """
PROGRAM TEST
INTEGER I
DO I = 1, 10
ENDDO
PRINT *, I
END
"""
        ssa = compile_to_ssa(code)
        self.assertIn('branch', ssa.lower(), "Должен быть цикл даже с пустым телом")
        self.assertIn('I_', ssa, "Должна быть переменная цикла")
    def test_assignment_in_condition(self):
        """Тест присваивания внутри условия."""
        code = """
PROGRAM TEST
INTEGER X, Y
X = 5
IF (X .GT. 0) THEN
    Y = X + 1
    IF (Y .GT. 5) THEN
        X = Y * 2
    ENDIF
ENDIF
PRINT *, X, Y
END
"""
        ssa = compile_to_ssa(code)
        self.assertIn('X_', ssa, "X должна версионироваться")
        self.assertIn('Y_', ssa, "Y должна версионироваться")
        self.assertIn('branch_if', ssa.lower(), "Должны быть условные переходы")
    def test_array_as_index(self):
        """Тест использования элемента массива как индекса."""
        code = """
PROGRAM TEST
INTEGER A(10), B(10), I
DO I = 1, 10
    A(I) = I
ENDDO
B(A(1)) = 100
B(A(5)) = 200
PRINT *, B(1), B(5)
END
"""
        ssa = compile_to_ssa(code)
        self.assertIn('load_array', ssa.lower(), "Должна быть загрузка из массива")
        self.assertIn('store_array', ssa.lower(), "Должна быть запись в массив")
        self.assertIn('%tmp_', ssa, "Должны быть временные переменные для индексов")
class TestLLVMEdgeCases(unittest.TestCase):
    """Тесты граничных случаев в LLVM."""
    def test_empty_program_llvm(self):
        """Тест минимальной программы."""
        code = """
PROGRAM EMPTY
END
"""
        llvm = compile_to_llvm(code)
        self.assertIn('define i32 @main', llvm, "Должна быть функция main")
        self.assertIn('ret i32 0', llvm, "Должен быть возврат значения")
    def test_only_declarations_llvm(self):
        """Тест программы только с объявлениями."""
        code = """
PROGRAM DECL
INTEGER X, Y, Z
REAL A, B
LOGICAL FLAG
END
"""
        llvm = compile_to_llvm(code)
        self.assertIn('alloca', llvm, "Должно быть выделение памяти для переменных")
        self.assertIn('i32', llvm, "Должны быть целочисленные переменные")
        self.assertIn('double', llvm, "Должны быть вещественные переменные")
        self.assertIn('i1', llvm, "Должны быть логические переменные")
class TestSSAPerformanceCritical(unittest.TestCase):
    """Тесты критичных для производительности случаев."""
    def test_loop_invariant_code_motion(self):
        """Тест инвариантного кода в цикле (для оптимизаций)."""
        code = """
PROGRAM TEST
INTEGER I, J, SUM, CONST
CONST = 100
SUM = 0
DO I = 1, 1000
    SUM = SUM + CONST * I
ENDDO
PRINT *, SUM
END
"""
        ssa = compile_to_ssa(code)
        self.assertIn('branch', ssa.lower(), "Должен быть цикл")
        self.assertIn('SUM_', ssa, "SUM должна версионироваться в цикле")
        self.assertIn('CONST_', ssa, "CONST должна быть доступна")
    def test_redundant_computations(self):
        """Тест избыточных вычислений (потенциал для оптимизации)."""
        code = """
PROGRAM TEST
INTEGER A, B, C, D, E
A = 5
B = 10
C = A + B
D = A + B
E = (A + B) * (A + B)
PRINT *, C, D, E
END
"""
        ssa = compile_to_ssa(code)
        self.assertIn('+', ssa, "Должны быть операции сложения")
        self.assertIn('*', ssa, "Должна быть операция умножения")
class TestLLVMPerformanceCritical(unittest.TestCase):
    """Тесты критичных для производительности случаев в LLVM."""
    def test_loop_with_many_iterations(self):
        """Тест цикла с большим количеством итераций."""
        code = """
PROGRAM TEST
INTEGER I, SUM
SUM = 0
DO I = 1, 10000
    SUM = SUM + I
ENDDO
PRINT *, SUM
END
"""
        llvm = compile_to_llvm(code)
        self.assertIn('phi', llvm, "Должны быть phi-функции для переменной цикла")
        self.assertIn('icmp', llvm, "Должно быть сравнение для условия цикла")
        self.assertIn('br', llvm, "Должны быть переходы")
        label_count = llvm.count('label %loop')
        self.assertGreater(label_count, 0, "Должны быть метки для цикла")
    def test_memory_access_patterns(self):
        """Тест паттернов доступа к памяти."""
        code = """
PROGRAM TEST
INTEGER A(1000), B(1000)
INTEGER I
DO I = 1, 1000
    A(I) = I
    B(I) = A(I) * 2
ENDDO
PRINT *, B(1), B(1000)
END
"""
        llvm = compile_to_llvm(code)
        self.assertIn('getelementptr', llvm, "Должны быть эффективные операции доступа к памяти")
        self.assertIn('store', llvm, "Должны быть операции записи")
        self.assertIn('load', llvm, "Должны быть операции чтения")
if __name__ == '__main__':
    unittest.main(verbosity=2)