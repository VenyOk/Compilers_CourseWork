"""
Комплексные тесты для SSA и LLVM генераторов.
Проверяют сложные случаи использования компилятора.
"""
import unittest
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from src.core import Lexer, Parser
from src.semantic import SemanticAnalyzer
from src.ssa_generator import SSAGenerator
from src.llvm_generator import LLVMGenerator
class TestSSAComplexExpressions(unittest.TestCase):
    """Тесты сложных выражений в SSA форме"""
    def test_nested_arithmetic_expressions(self):
        """Тест вложенных арифметических выражений"""
        code = """
PROGRAM TEST
INTEGER A, B, C, D, RESULT
A = 10
B = 5
C = 3
D = 2
RESULT = A + B * C - D / 2 + (A - B) ** 2
END
"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        semantic = SemanticAnalyzer()
        self.assertTrue(semantic.analyze(ast), "Семантический анализ должен пройти успешно")
        ssa_gen = SSAGenerator()
        instructions = ssa_gen.generate(ast)
        ssa_str = ssa_gen.to_string(instructions)
        self.assertIn("+", ssa_str, "Должна быть операция сложения")
        self.assertIn("*", ssa_str, "Должна быть операция умножения")
        self.assertIn("-", ssa_str, "Должна быть операция вычитания")
        self.assertIn("/", ssa_str, "Должна быть операция деления")
        self.assertIn("**", ssa_str, "Должна быть операция возведения в степень")
        self.assertIn("RESULT_", ssa_str, "Должна быть переменная RESULT с версией")
        self.assertGreater(len(instructions), 10, "Должно быть достаточно инструкций")
    def test_logical_expressions_chain(self):
        """Тест цепочки логических операций"""
        code = """
PROGRAM TEST
LOGICAL A, B, C, D, RESULT
A = .TRUE.
B = .FALSE.
C = .TRUE.
D = .FALSE.
RESULT = A .AND. B .OR. C .AND. .NOT. D .EQV. .TRUE.
END
"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        semantic = SemanticAnalyzer()
        self.assertTrue(semantic.analyze(ast))
        ssa_gen = SSAGenerator()
        instructions = ssa_gen.generate(ast)
        ssa_str = ssa_gen.to_string(instructions)
        self.assertIn(".AND.", ssa_str or "", "Должна быть операция AND")
        self.assertIn(".OR.", ssa_str or "", "Должна быть операция OR")
        self.assertIn(".NOT.", ssa_str or "", "Должна быть операция NOT")
        self.assertIn(".EQV.", ssa_str or "", "Должна быть операция EQV")
    def test_mixed_type_expressions(self):
        """Тест смешанных типов в выражениях"""
        code = """
PROGRAM TEST
INTEGER I
REAL X, Y, Z
I = 5
X = 3.14
Y = 2.71
Z = X + I * Y - I / 2.0
END
"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        semantic = SemanticAnalyzer()
        self.assertTrue(semantic.analyze(ast))
        ssa_gen = SSAGenerator()
        instructions = ssa_gen.generate(ast)
        ssa_str = ssa_gen.to_string(instructions)
        self.assertIn("X", ssa_str, "Должна быть переменная X")
        self.assertIn("Y", ssa_str, "Должна быть переменная Y")
        self.assertIn("Z", ssa_str, "Должна быть переменная Z")
        self.assertIn("I", ssa_str, "Должна быть переменная I")
class TestSSANestedControlFlow(unittest.TestCase):
    """Тесты вложенных конструкций управления потоком"""
    def test_nested_loops(self):
        """Тест вложенных циклов DO"""
        code = """
PROGRAM TEST
INTEGER I, J, SUM
SUM = 0
DO I = 1, 5
    DO J = 1, 5
        SUM = SUM + I * J
    ENDDO
ENDDO
PRINT *, SUM
END
"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        semantic = SemanticAnalyzer()
        self.assertTrue(semantic.analyze(ast))
        ssa_gen = SSAGenerator()
        instructions = ssa_gen.generate(ast)
        ssa_str = ssa_gen.to_string(instructions)
        self.assertIn("loop", ssa_str.lower(), "Должны быть инструкции циклов")
        self.assertIn("SUM_", ssa_str, "Должна быть переменная SUM с версионированием")
        self.assertIn("I", ssa_str, "Должна быть переменная I")
        self.assertIn("J", ssa_str, "Должна быть переменная J")
        self.assertGreater(len(instructions), 15, "Должно быть достаточно инструкций для вложенных циклов")
    def test_if_inside_loop(self):
        """Тест IF внутри цикла"""
        code = """
PROGRAM TEST
INTEGER I, SUM_EVEN, SUM_ODD
SUM_EVEN = 0
SUM_ODD = 0
DO I = 1, 10
    IF (MOD(I, 2) .EQ. 0) THEN
        SUM_EVEN = SUM_EVEN + I
    ELSE
        SUM_ODD = SUM_ODD + I
    ENDIF
ENDDO
PRINT *, SUM_EVEN, SUM_ODD
END
"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        semantic = SemanticAnalyzer()
        self.assertTrue(semantic.analyze(ast))
        ssa_gen = SSAGenerator()
        instructions = ssa_gen.generate(ast)
        ssa_str = ssa_gen.to_string(instructions)
        self.assertIn("SUM_EVEN", ssa_str, "Должна быть переменная SUM_EVEN")
        self.assertIn("SUM_ODD", ssa_str, "Должна быть переменная SUM_ODD")
        self.assertIn("loop", ssa_str.lower(), "Должен быть цикл")
    def test_loop_inside_if(self):
        """Тест цикла внутри IF"""
        code = """
PROGRAM TEST
INTEGER N, I, FACT
N = 5
FACT = 1
IF (N .GT. 0) THEN
    DO I = 1, N
        FACT = FACT * I
    ENDDO
ENDIF
PRINT *, FACT
END
"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        semantic = SemanticAnalyzer()
        self.assertTrue(semantic.analyze(ast))
        ssa_gen = SSAGenerator()
        instructions = ssa_gen.generate(ast)
        ssa_str = ssa_gen.to_string(instructions)
        self.assertIn("FACT", ssa_str, "Должна быть переменная FACT")
        self.assertIn("loop", ssa_str.lower(), "Должен быть цикл")
    def test_if_elseif_else_chain(self):
        """Тест цепочки IF-ELSEIF-ELSE"""
        code = """
PROGRAM TEST
INTEGER X, Y
X = 15
IF (X .LT. 10) THEN
    Y = 1
ELSEIF (X .LT. 20) THEN
    Y = 2
ELSEIF (X .LT. 30) THEN
    Y = 3
ELSE
    Y = 4
ENDIF
PRINT *, Y
END
"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        semantic = SemanticAnalyzer()
        self.assertTrue(semantic.analyze(ast))
        ssa_gen = SSAGenerator()
        instructions = ssa_gen.generate(ast)
        ssa_str = ssa_gen.to_string(instructions)
        self.assertIn("Y", ssa_str, "Должна быть переменная Y")
        self.assertGreater(len(instructions), 5, "Должно быть достаточно инструкций для цепочки условий")
class TestSSAArrays(unittest.TestCase):
    """Тесты массивов в SSA форме"""
    def test_one_dimensional_array(self):
        """Тест одномерного массива"""
        code = """
PROGRAM TEST
INTEGER A(10)
INTEGER I
DO I = 1, 10
    A(I) = I * 2
ENDDO
PRINT *, A(5)
END
"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        semantic = SemanticAnalyzer()
        self.assertTrue(semantic.analyze(ast))
        ssa_gen = SSAGenerator()
        instructions = ssa_gen.generate(ast)
        ssa_str = ssa_gen.to_string(instructions)
        self.assertIn("A", ssa_str, "Должен быть массив A")
        self.assertIn("alloca_array", ssa_str.lower() or "", "Должна быть инструкция alloca_array")
        self.assertIn("store_array", ssa_str.lower() or "", "Должна быть инструкция store_array")
    def test_two_dimensional_array(self):
        """Тест двумерного массива"""
        code = """
PROGRAM TEST
INTEGER MATRIX(5, 5)
INTEGER I, J
DO I = 1, 5
    DO J = 1, 5
        MATRIX(I, J) = I * J
    ENDDO
ENDDO
PRINT *, MATRIX(3, 3)
END
"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        semantic = SemanticAnalyzer()
        self.assertTrue(semantic.analyze(ast))
        ssa_gen = SSAGenerator()
        instructions = ssa_gen.generate(ast)
        ssa_str = ssa_gen.to_string(instructions)
        self.assertIn("MATRIX", ssa_str, "Должен быть массив MATRIX")
        self.assertIn("alloca_array", ssa_str.lower() or "", "Должна быть инструкция alloca_array")
        self.assertGreater(len(instructions), 20, "Должно быть достаточно инструкций для двумерного массива")
    def test_array_with_expressions_index(self):
        """Тест массива с выражениями в индексах"""
        code = """
PROGRAM TEST
INTEGER A(10), B(10), C(10)
INTEGER I
DO I = 1, 10
    A(I) = I
    B(I) = I * 2
    C(I) = A(I) + B(I + 1 - 1)
ENDDO
END
"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        semantic = SemanticAnalyzer()
        self.assertTrue(semantic.analyze(ast))
        ssa_gen = SSAGenerator()
        instructions = ssa_gen.generate(ast)
        ssa_str = ssa_gen.to_string(instructions)
        self.assertIn("A", ssa_str)
        self.assertIn("B", ssa_str)
        self.assertIn("C", ssa_str)
class TestSSAFunctions(unittest.TestCase):
    """Тесты встроенных функций"""
    def test_nested_function_calls(self):
        """Тест вложенных вызовов функций"""
        code = """
PROGRAM TEST
REAL X, Y, Z
X = 3.14159
Y = SIN(X)
Z = SQRT(ABS(Y)) + EXP(LOG(X))
PRINT *, Z
END
"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        semantic = SemanticAnalyzer()
        self.assertTrue(semantic.analyze(ast))
        ssa_gen = SSAGenerator()
        instructions = ssa_gen.generate(ast)
        ssa_str = ssa_gen.to_string(instructions)
        self.assertIn("SIN", ssa_str.upper() or "", "Должен быть вызов SIN")
        self.assertIn("SQRT", ssa_str.upper() or "", "Должен быть вызов SQRT")
        self.assertIn("ABS", ssa_str.upper() or "", "Должен быть вызов ABS")
        self.assertIn("EXP", ssa_str.upper() or "", "Должен быть вызов EXP")
        self.assertIn("LOG", ssa_str.upper() or "", "Должен быть вызов LOG")
        self.assertIn("call", ssa_str.lower() or "", "Должны быть инструкции call")
    def test_function_in_expression(self):
        """Тест функции в сложном выражении"""
        code = """
PROGRAM TEST
REAL X, Y, Z
X = 2.0
Y = 3.0
Z = X ** 2 + SQRT(Y) * COS(X) - SIN(Y) / TAN(X)
PRINT *, Z
END
"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        semantic = SemanticAnalyzer()
        self.assertTrue(semantic.analyze(ast))
        ssa_gen = SSAGenerator()
        instructions = ssa_gen.generate(ast)
        ssa_str = ssa_gen.to_string(instructions)
        self.assertIn("SQRT", ssa_str.upper() or "", "Должен быть вызов SQRT")
        self.assertIn("COS", ssa_str.upper() or "", "Должен быть вызов COS")
        self.assertIn("SIN", ssa_str.upper() or "", "Должен быть вызов SIN")
        self.assertIn("TAN", ssa_str.upper() or "", "Должен быть вызов TAN")
class TestLLVMComplexExpressions(unittest.TestCase):
    """Тесты сложных выражений в LLVM IR"""
    def test_arithmetic_precedence_llvm(self):
        """Тест приоритета операций в LLVM"""
        code = """
PROGRAM TEST
INTEGER A, B, C, RESULT
A = 10
B = 5
C = 3
RESULT = A + B * C ** 2 - A / B
END
"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        semantic = SemanticAnalyzer()
        self.assertTrue(semantic.analyze(ast))
        llvm_gen = LLVMGenerator()
        llvm_code = llvm_gen.generate(ast)
        self.assertIn("define i32 @main", llvm_code, "Должна быть функция main")
        self.assertIn("alloca i32", llvm_code, "Должны быть аллокации i32")
        self.assertIn("mul i32", llvm_code, "Должна быть операция умножения")
        self.assertIn("add i32", llvm_code, "Должна быть операция сложения")
        self.assertIn("sub i32", llvm_code, "Должна быть операция вычитания")
        self.assertIn("sdiv i32", llvm_code, "Должна быть операция деления")
    def test_real_arithmetic_llvm(self):
        """Тест вещественной арифметики в LLVM"""
        code = """
PROGRAM TEST
REAL X, Y, Z
X = 3.14159
Y = 2.71828
Z = X * Y + X / Y - Y ** 2.0
END
"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        semantic = SemanticAnalyzer()
        self.assertTrue(semantic.analyze(ast))
        llvm_gen = LLVMGenerator()
        llvm_code = llvm_gen.generate(ast)
        self.assertIn("alloca double", llvm_code, "Должны быть аллокации double")
        self.assertIn("fmul double", llvm_code, "Должна быть операция умножения для double")
        self.assertIn("fadd double", llvm_code, "Должна быть операция сложения для double")
        self.assertIn("fsub double", llvm_code, "Должна быть операция вычитания для double")
        self.assertIn("fdiv double", llvm_code, "Должна быть операция деления для double")
        self.assertIn("call double @pow", llvm_code, "Должен быть вызов pow для **")
    def test_logical_operations_llvm(self):
        """Тест логических операций в LLVM"""
        code = """
PROGRAM TEST
LOGICAL A, B, C, D, RESULT
A = .TRUE.
B = .FALSE.
C = .TRUE.
RESULT = A .AND. B .OR. C .AND. .NOT. B
END
"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        semantic = SemanticAnalyzer()
        self.assertTrue(semantic.analyze(ast))
        llvm_gen = LLVMGenerator()
        llvm_code = llvm_gen.generate(ast)
        self.assertIn("alloca i1", llvm_code, "Должны быть аллокации i1 для логических")
        self.assertIn("store i1", llvm_code, "Должны быть операции store i1")
        self.assertIn("and i1", llvm_code, "Должна быть операция and")
        self.assertIn("or i1", llvm_code, "Должна быть операция or")
        self.assertIn("xor i1", llvm_code, "Должна быть операция xor для NOT")
class TestLLVMControlFlow(unittest.TestCase):
    """Тесты конструкций управления потоком в LLVM"""
    def test_nested_loops_llvm(self):
        """Тест вложенных циклов в LLVM"""
        code = """
PROGRAM TEST
INTEGER I, J, SUM
SUM = 0
DO I = 1, 5
    DO J = 1, 5
        SUM = SUM + I * J
    ENDDO
ENDDO
END
"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        semantic = SemanticAnalyzer()
        self.assertTrue(semantic.analyze(ast))
        llvm_gen = LLVMGenerator()
        llvm_code = llvm_gen.generate(ast)
        self.assertIn("loop_", llvm_code, "Должны быть метки циклов")
        self.assertIn("loop_body_", llvm_code, "Должны быть метки тел циклов")
        self.assertIn("loop_end_", llvm_code, "Должны быть метки окончания циклов")
        self.assertIn("br label", llvm_code, "Должны быть инструкции ветвления")
        self.assertIn("icmp sle", llvm_code, "Должны быть сравнения для циклов")
        self.assertGreater(llvm_code.count("br label"), 3, "Должно быть достаточно инструкций ветвления")
    def test_if_then_else_llvm(self):
        """Тест IF-THEN-ELSE в LLVM"""
        code = """
PROGRAM TEST
INTEGER X, Y
X = 10
IF (X .GT. 5) THEN
    Y = 1
ELSE
    Y = 2
ENDIF
END
"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        semantic = SemanticAnalyzer()
        self.assertTrue(semantic.analyze(ast))
        llvm_gen = LLVMGenerator()
        llvm_code = llvm_gen.generate(ast)
        self.assertIn("if_then_", llvm_code, "Должна быть метка then")
        self.assertIn("if_else_", llvm_code, "Должна быть метка else")
        self.assertIn("if_end_", llvm_code, "Должна быть метка конца if")
        self.assertIn("icmp", llvm_code, "Должно быть сравнение")
        self.assertIn("br i1", llvm_code, "Должно быть условное ветвление")
    def test_if_elseif_else_llvm(self):
        """Тест IF-ELSEIF-ELSE в LLVM"""
        code = """
PROGRAM TEST
INTEGER X, Y
X = 15
IF (X .LT. 10) THEN
    Y = 1
ELSEIF (X .LT. 20) THEN
    Y = 2
ELSE
    Y = 3
ENDIF
END
"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        semantic = SemanticAnalyzer()
        self.assertTrue(semantic.analyze(ast))
        llvm_gen = LLVMGenerator()
        llvm_code = llvm_gen.generate(ast)
        self.assertIn("if_then_", llvm_code)
        self.assertIn("if_elif_", llvm_code, "Должны быть метки ELSEIF")
        self.assertIn("if_else_", llvm_code)
        self.assertGreater(llvm_code.count("icmp"), 1, "Должно быть несколько сравнений")
        self.assertGreater(llvm_code.count("br i1"), 2, "Должно быть несколько условных ветвлений")
    def test_do_while_llvm(self):
        """Тест цикла DO WHILE в LLVM"""
        code = """
PROGRAM TEST
INTEGER I, SUM
SUM = 0
I = 1
DO WHILE (I .LE. 10)
    SUM = SUM + I
    I = I + 1
ENDDO
END
"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        semantic = SemanticAnalyzer()
        self.assertTrue(semantic.analyze(ast))
        llvm_gen = LLVMGenerator()
        llvm_code = llvm_gen.generate(ast)
        self.assertIn("do_while_loop_", llvm_code, "Должна быть метка цикла DO WHILE")
        self.assertIn("do_while_body_", llvm_code, "Должно быть тело цикла")
        self.assertIn("do_while_end_", llvm_code, "Должен быть конец цикла")
        self.assertIn("br label", llvm_code, "Должны быть инструкции ветвления")
class TestLLVMArrays(unittest.TestCase):
    """Тесты массивов в LLVM IR"""
    def test_one_dimensional_array_llvm(self):
        """Тест одномерного массива в LLVM"""
        code = """
PROGRAM TEST
INTEGER A(10)
INTEGER I
DO I = 1, 10
    A(I) = I * 2
ENDDO
PRINT *, A(5)
END
"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        semantic = SemanticAnalyzer()
        self.assertTrue(semantic.analyze(ast))
        llvm_gen = LLVMGenerator()
        llvm_code = llvm_gen.generate(ast)
        self.assertIn("alloca [10 x i32]", llvm_code or "alloca [10", "Должна быть аллокация массива")
        self.assertIn("getelementptr", llvm_code, "Должен быть getelementptr для доступа к массиву")
        self.assertIn("store i32", llvm_code, "Должны быть операции store для массива")
        self.assertIn("load i32", llvm_code, "Должны быть операции load для массива")
    def test_array_index_arithmetic_llvm(self):
        """Тест арифметики индексов массива в LLVM"""
        code = """
PROGRAM TEST
INTEGER A(20)
INTEGER I, INDEX
DO I = 1, 10
    INDEX = I * 2
    A(INDEX) = I
ENDDO
END
"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        semantic = SemanticAnalyzer()
        self.assertTrue(semantic.analyze(ast))
        llvm_gen = LLVMGenerator()
        llvm_code = llvm_gen.generate(ast)
        self.assertIn("getelementptr", llvm_code, "Должен быть getelementptr")
        self.assertIn("mul i32", llvm_code, "Должно быть умножение для индекса")
class TestLLVMFunctions(unittest.TestCase):
    """Тесты функций в LLVM IR"""
    def test_builtin_function_declarations(self):
        """Тест объявлений встроенных функций"""
        code = """
PROGRAM TEST
REAL X, Y
X = 3.14
Y = SIN(X)
END
"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        semantic = SemanticAnalyzer()
        self.assertTrue(semantic.analyze(ast))
        llvm_gen = LLVMGenerator()
        llvm_code = llvm_gen.generate(ast)
        self.assertIn("declare double @sin(double)", llvm_code, "Должно быть объявление sin")
        self.assertIn("call double @sin", llvm_code, "Должен быть вызов sin")
    def test_multiple_function_calls(self):
        """Тест множественных вызовов функций"""
        code = """
PROGRAM TEST
REAL X, Y, Z, W
X = 1.5
Y = SIN(X)
Z = COS(X)
W = SQRT(Y ** 2 + Z ** 2)
END
"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        semantic = SemanticAnalyzer()
        self.assertTrue(semantic.analyze(ast))
        llvm_gen = LLVMGenerator()
        llvm_code = llvm_gen.generate(ast)
        self.assertIn("declare double @sin(double)", llvm_code)
        self.assertIn("declare double @cos(double)", llvm_code)
        self.assertIn("declare double @sqrt(double)", llvm_code)
        self.assertIn("call double @sin", llvm_code)
        self.assertIn("call double @cos", llvm_code)
        self.assertIn("call double @sqrt", llvm_code)
        self.assertIn("call double @pow", llvm_code, "Должен быть вызов pow для **")
class TestLLVMCompletePrograms(unittest.TestCase):
    """Тесты полных программ"""
    def test_matrix_multiplication_structure(self):
        """Тест структуры умножения матриц"""
        code = """
PROGRAM MATMUL
INTEGER A(3, 3), B(3, 3), C(3, 3)
INTEGER I, J, K
DO I = 1, 3
    DO J = 1, 3
        C(I, J) = 0
        DO K = 1, 3
            C(I, J) = C(I, J) + A(I, K) * B(K, J)
        ENDDO
    ENDDO
ENDDO
END
"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        semantic = SemanticAnalyzer()
        self.assertTrue(semantic.analyze(ast))
        llvm_gen = LLVMGenerator()
        llvm_code = llvm_gen.generate(ast)
        self.assertIn("define i32 @main", llvm_code, "Должна быть функция main")
        self.assertIn("alloca", llvm_code, "Должны быть аллокации")
        loop_blocks = llvm_code.count("loop_")
        self.assertGreaterEqual(loop_blocks, 3, "Должно быть минимум 3 цикла")
        self.assertIn("getelementptr", llvm_code, "Должен быть доступ к массивам")
    def test_newton_method_structure(self):
        """Тест структуры метода Ньютона"""
        code = """
PROGRAM NEWTON
REAL X, EPS, DELTA
INTEGER ITER
X = 1.0
EPS = 1.0E-6
ITER = 0
DO WHILE (ABS(DELTA) .GT. EPS .AND. ITER .LT. 100)
    DELTA = (X ** 2 - 2.0) / (2.0 * X)
    X = X - DELTA
    ITER = ITER + 1
ENDDO
PRINT *, 'Root:', X
END
"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        semantic = SemanticAnalyzer()
        self.assertTrue(semantic.analyze(ast))
        llvm_gen = LLVMGenerator()
        llvm_code = llvm_gen.generate(ast)
        self.assertIn("define i32 @main", llvm_code)
        self.assertIn("do_while_loop_", llvm_code, "Должен быть цикл DO WHILE")
        self.assertIn("call double @fabs", llvm_code, "Должен быть вызов ABS")
        self.assertIn("call double @pow", llvm_code, "Должен быть вызов pow для **")
        self.assertIn("icmp", llvm_code, "Должны быть сравнения")
    def test_parameter_and_data(self):
        """Тест PARAMETER и DATA"""
        code = """
PROGRAM TEST
PARAMETER (PI = 3.14159, E = 2.71828)
INTEGER A(5)
DATA A / 1, 2, 3, 4, 5 /
REAL X, Y
X = PI * 2.0
Y = E ** 2.0
PRINT *, X, Y, A(3)
END
"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        semantic = SemanticAnalyzer()
        self.assertTrue(semantic.analyze(ast))
        llvm_gen = LLVMGenerator()
        llvm_code = llvm_gen.generate(ast)
        self.assertIn("define i32 @main", llvm_code)
        self.assertIn("alloca", llvm_code)
        self.assertIn("3.14159", llvm_code or "3.14", "Должна быть константа PI")
    def test_arithmetic_if(self):
        """Тест арифметического IF"""
        code = """
PROGRAM TEST
INTEGER X, RESULT
READ(*, *) X
IF (X - 5) 10, 20, 30
10 RESULT = 1
GOTO 40
20 RESULT = 2
GOTO 40
30 RESULT = 3
40 PRINT *, RESULT
END
"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        semantic = SemanticAnalyzer()
        if semantic.analyze(ast):
            llvm_gen = LLVMGenerator()
            llvm_code = llvm_gen.generate(ast)
            self.assertIn("define i32 @main", llvm_code)
            self.assertIn("icmp", llvm_code, "Должны быть сравнения")
class TestSSAAndLLVMConsistency(unittest.TestCase):
    """Тесты согласованности между SSA и LLVM"""
    def test_same_program_both_generators(self):
        """Тест одной программы в обоих генераторах"""
        code = """
PROGRAM TEST
INTEGER I, SUM
SUM = 0
DO I = 1, 10
    SUM = SUM + I
ENDDO
IF (SUM .GT. 50) THEN
    PRINT *, 'Large sum:', SUM
ELSE
    PRINT *, 'Small sum:', SUM
ENDIF
END
"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        semantic = SemanticAnalyzer()
        self.assertTrue(semantic.analyze(ast))
        ssa_gen = SSAGenerator()
        ssa_instructions = ssa_gen.generate(ast)
        ssa_str = ssa_gen.to_string(ssa_instructions)
        llvm_gen = LLVMGenerator()
        llvm_code = llvm_gen.generate(ast)
        self.assertIn("SUM", ssa_str, "SSA должен содержать SUM")
        self.assertIn("SUM", llvm_code, "LLVM должен содержать SUM")
        self.assertIn("I", ssa_str, "SSA должен содержать I")
        self.assertIn("I", llvm_code, "LLVM должен содержать I")
        self.assertGreater(len(ssa_instructions), 5, "SSA должен иметь достаточно инструкций")
        self.assertGreater(len(llvm_code.split('\n')), 20, "LLVM должен иметь достаточно строк")
class TestComplexRealWorldPrograms(unittest.TestCase):
    """Тесты сложных реальных программ"""
    def test_quadratic_formula(self):
        """Тест решения квадратного уравнения"""
        code = """
PROGRAM QUADRATIC
REAL A, B, C, DISCRIM, X1, X2
A = 1.0
B = -5.0
C = 6.0
DISCRIM = B ** 2 - 4.0 * A * C
IF (DISCRIM .GE. 0.0) THEN
    X1 = (-B + SQRT(DISCRIM)) / (2.0 * A)
    X2 = (-B - SQRT(DISCRIM)) / (2.0 * A)
    PRINT *, 'Roots:', X1, X2
ELSE
    PRINT *, 'No real roots'
ENDIF
END
"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        semantic = SemanticAnalyzer()
        self.assertTrue(semantic.analyze(ast))
        llvm_gen = LLVMGenerator()
        llvm_code = llvm_gen.generate(ast)
        self.assertIn("call double @sqrt", llvm_code, "Должен быть вызов sqrt")
        self.assertIn("call double @pow", llvm_code, "Должен быть вызов pow")
        self.assertIn("if_then_", llvm_code, "Должна быть конструкция IF")
        self.assertIn("fmul double", llvm_code, "Должны быть операции с double")
    def test_bubble_sort_structure(self):
        """Тест структуры пузырьковой сортировки"""
        code = """
PROGRAM BUBBLE
INTEGER A(10)
INTEGER I, J, TEMP
LOGICAL SWAPPED
SWAPPED = .TRUE.
DO WHILE (SWAPPED)
    SWAPPED = .FALSE.
    DO I = 1, 9
        IF (A(I) .GT. A(I + 1)) THEN
            TEMP = A(I)
            A(I) = A(I + 1)
            A(I + 1) = TEMP
            SWAPPED = .TRUE.
        ENDIF
    ENDDO
ENDDO
END
"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        semantic = SemanticAnalyzer()
        self.assertTrue(semantic.analyze(ast))
        ssa_gen = SSAGenerator()
        ssa_instructions = ssa_gen.generate(ast)
        ssa_str = ssa_gen.to_string(ssa_instructions)
        llvm_gen = LLVMGenerator()
        llvm_code = llvm_gen.generate(ast)
        self.assertIn("SWAPPED", ssa_str)
        self.assertIn("A", ssa_str)
        self.assertIn("do_while_loop_", llvm_code, "Должен быть цикл DO WHILE")
        self.assertIn("loop_", llvm_code, "Должен быть цикл DO")
        self.assertIn("if_then_", llvm_code, "Должен быть IF внутри циклов")
        self.assertIn("getelementptr", llvm_code, "Должен быть доступ к массиву")
class TestComplexNestedStructures(unittest.TestCase):
    """Тесты сложных вложенных структур"""
    def test_nested_if_in_loop_with_array(self):
        """Тест вложенного IF в цикле с массивом"""
        code = """
PROGRAM TEST
INTEGER A(20), B(20), C(20)
INTEGER I
DO I = 1, 20
    A(I) = I
    IF (MOD(I, 2) .EQ. 0) THEN
        B(I) = A(I) * 2
        IF (B(I) .GT. 20) THEN
            C(I) = B(I) - 10
        ELSE
            C(I) = B(I) + 10
        ENDIF
    ELSE
        B(I) = A(I) + 1
        C(I) = B(I) - 1
    ENDIF
ENDDO
END
"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        semantic = SemanticAnalyzer()
        self.assertTrue(semantic.analyze(ast))
        ssa_gen = SSAGenerator()
        ssa_instructions = ssa_gen.generate(ast)
        ssa_str = ssa_gen.to_string(ssa_instructions)
        llvm_gen = LLVMGenerator()
        llvm_code = llvm_gen.generate(ast)
        self.assertIn("A", ssa_str)
        self.assertIn("B", ssa_str)
        self.assertIn("C", ssa_str)
        self.assertIn("loop", ssa_str.lower())
        self.assertIn("if_then_", llvm_code)
        self.assertIn("getelementptr", llvm_code)
    def test_complex_expression_with_functions_and_arrays(self):
        """Тест сложного выражения с функциями и массивами"""
        code = """
PROGRAM TEST
REAL X(10), Y(10), Z
INTEGER I
DO I = 1, 10
    X(I) = I * 0.5
    Y(I) = SQRT(X(I)) + SIN(X(I)) * COS(X(I))
ENDDO
Z = Y(1) * Y(2) + Y(3) ** 2.0 - ABS(Y(5) - Y(6))
PRINT *, Z
END
"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        semantic = SemanticAnalyzer()
        self.assertTrue(semantic.analyze(ast))
        llvm_gen = LLVMGenerator()
        llvm_code = llvm_gen.generate(ast)
        self.assertIn("call double @sqrt", llvm_code)
        self.assertIn("call double @sin", llvm_code)
        self.assertIn("call double @cos", llvm_code)
        self.assertIn("call double @fabs", llvm_code)
        self.assertIn("call double @pow", llvm_code)
        self.assertIn("getelementptr", llvm_code)
    def test_loop_with_conditional_break_logic(self):
        """Тест цикла с условной логикой прерывания"""
        code = """
PROGRAM TEST
INTEGER I, SUM, PRODUCT
SUM = 0
PRODUCT = 1
DO I = 1, 100
    SUM = SUM + I
    PRODUCT = PRODUCT * I
    IF (SUM .GT. 1000) THEN
        IF (PRODUCT .GT. 10000) THEN
            GOTO 100
        ENDIF
    ENDIF
ENDDO
100 CONTINUE
PRINT *, SUM, PRODUCT
END
"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        semantic = SemanticAnalyzer()
        if semantic.analyze(ast):
            llvm_gen = LLVMGenerator()
            llvm_code = llvm_gen.generate(ast)
            self.assertIn("loop_", llvm_code)
            self.assertIn("if_then_", llvm_code)
    def test_multiple_data_types_interactions(self):
        """Тест взаимодействия разных типов данных"""
        code = """
PROGRAM TEST
INTEGER I
REAL X, Y, Z
LOGICAL FLAG
CHARACTER*10 STR
I = 42
X = 3.14
Y = 2.71
Z = X + I
FLAG = (Z .GT. 40.0) .AND. (I .LT. 50)
IF (FLAG) THEN
    STR = 'Success'
    PRINT *, STR, Z
ELSE
    STR = 'Failed'
    PRINT *, STR
ENDIF
END
"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        semantic = SemanticAnalyzer()
        self.assertTrue(semantic.analyze(ast))
        llvm_gen = LLVMGenerator()
        llvm_code = llvm_gen.generate(ast)
        self.assertIn("alloca i32", llvm_code)
        self.assertIn("alloca double", llvm_code)
        self.assertIn("alloca i1", llvm_code)
        self.assertIn("alloca [10 x i8]", llvm_code or "alloca", "Должен быть массив символов")
        self.assertIn("if_then_", llvm_code)
class TestEdgeCases(unittest.TestCase):
    """Тесты граничных случаев"""
    def test_empty_loop_body(self):
        """Тест цикла с пустым телом"""
        code = """
PROGRAM TEST
INTEGER I
DO I = 1, 10
ENDDO
END
"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        semantic = SemanticAnalyzer()
        self.assertTrue(semantic.analyze(ast))
        llvm_gen = LLVMGenerator()
        llvm_code = llvm_gen.generate(ast)
        self.assertIn("loop_", llvm_code)
        self.assertIn("define i32 @main", llvm_code)
    def test_single_statement_if(self):
        """Тест IF с одним оператором"""
        code = """
PROGRAM TEST
INTEGER X, Y
X = 5
IF (X .GT. 3) Y = 10
END
"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        semantic = SemanticAnalyzer()
        self.assertTrue(semantic.analyze(ast))
        llvm_gen = LLVMGenerator()
        llvm_code = llvm_gen.generate(ast)
        self.assertIn("if_then_", llvm_code)
        self.assertIn("store i32 10", llvm_code)
    def test_array_index_with_expression(self):
        """Тест индекса массива с выражением"""
        code = """
PROGRAM TEST
INTEGER A(20), I, J
I = 5
J = 3
A(I + J) = 100
A(I * J - 2) = 200
PRINT *, A(8), A(13)
END
"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        semantic = SemanticAnalyzer()
        self.assertTrue(semantic.analyze(ast))
        llvm_gen = LLVMGenerator()
        llvm_code = llvm_gen.generate(ast)
        self.assertIn("getelementptr", llvm_code)
        self.assertIn("add i32", llvm_code)
        self.assertIn("mul i32", llvm_code)
if __name__ == '__main__':
    unittest.main()