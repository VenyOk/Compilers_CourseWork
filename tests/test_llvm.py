import unittest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llvm_generator import LLVMGenerator
from src.semantic import SemanticAnalyzer
from src.core import Lexer, Parser


def compile_code_llvm(code: str):
    lexer = Lexer(code)
    tokens = lexer.tokenize()
    if lexer.get_errors():
        raise Exception(f"Lexer errors: {lexer.get_errors()}")
    parser = Parser(tokens)
    ast = parser.parse()
    semantic = SemanticAnalyzer()
    if not semantic.analyze(ast):
        raise Exception(f"Semantic errors: {semantic.get_errors()}")
    llvm_gen = LLVMGenerator()
    llvm_code = llvm_gen.generate(ast)
    return llvm_code


class TestLLVMBasic(unittest.TestCase):
    def test_llvm_structure(self):
        code = """
PROGRAM LLVM
    INTEGER X
    X = 42
    PRINT *, X
END
"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        llvm_gen = LLVMGenerator()
        llvm_code = llvm_gen.generate(ast)
        self.assertIn("define i32 @main()", llvm_code)
        self.assertIn("alloca", llvm_code)
        self.assertIn("store", llvm_code)
        self.assertIn("ret i32 0", llvm_code)

    def test_simple_program_llvm(self):
        code = """
PROGRAM SIMPLE
    INTEGER X
    X = 42
    PRINT *, X
END
"""
        llvm_code = compile_code_llvm(code)
        self.assertIn("define i32 @main()", llvm_code, "Должна быть функция main")
        self.assertIn("alloca i32", llvm_code, "Должны быть аллокации переменных")
        self.assertIn("store", llvm_code, "Должны быть операции store")
        self.assertIn("ret i32 0", llvm_code, "Должен быть возврат из main")


class TestLLVMLoops(unittest.TestCase):
    def test_do_loop_llvm(self):
        code = """
PROGRAM DOLOOP
    INTEGER I, SUM
    SUM = 0
    DO I = 1, 10
        SUM = SUM + I
    ENDDO
    PRINT *, SUM
END
"""
        llvm_code = compile_code_llvm(code)
        self.assertIn("define i32 @main()", llvm_code, "Должна быть функция main")
        self.assertIn("alloca i32", llvm_code, "Должны быть аллокации переменных")
        self.assertIn("phi", llvm_code, "Должны быть phi-функции для переменных циклов")
        self.assertIn("br", llvm_code, "Должны быть условные переходы")
        self.assertIn("add", llvm_code, "Должны быть операции сложения")
        self.assertIn("label", llvm_code, "Должны быть метки для циклов")

    def test_nested_loops_llvm(self):
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
        llvm_code = compile_code_llvm(code)
        self.assertIn("define i32 @main()", llvm_code, "Должна быть функция main")
        self.assertIn("alloca i32", llvm_code, "Должны быть аллокации переменных")
        self.assertIn("phi", llvm_code, "Должны быть phi-функции для переменных циклов")
        self.assertIn("br", llvm_code, "Должны быть условные переходы")
        self.assertIn("mul", llvm_code, "Должны быть операции умножения")
        self.assertIn("add", llvm_code, "Должны быть операции сложения")
        self.assertIn("label", llvm_code, "Должны быть метки для циклов")


class TestLLVMControlFlow(unittest.TestCase):
    def test_if_statement_llvm(self):
        code = """
PROGRAM IFTEST
    INTEGER X, Y
    X = 10
    IF (X .GT. 5) THEN
        Y = 1
    ELSE
        Y = 0
    ENDIF
END
"""
        llvm_code = compile_code_llvm(code)
        self.assertIn("icmp", llvm_code, "Должны быть операции сравнения")
        self.assertIn("br i1", llvm_code, "Должны быть условные переходы")
        self.assertIn(":", llvm_code, "Должны быть метки для ветвлений")

    def test_complex_if_structure_llvm(self):
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
        llvm_code = compile_code_llvm(code)
        self.assertIn("icmp", llvm_code, "Должны быть операции сравнения")
        self.assertIn("br i1", llvm_code, "Должны быть условные переходы")
        self.assertIn("add", llvm_code, "Должна быть операция сложения")
        self.assertIn("sub", llvm_code, "Должна быть операция вычитания")
        self.assertIn(":", llvm_code, "Должны быть метки для ветвлений")


class TestLLVMExpressions(unittest.TestCase):
    def test_arithmetic_expressions_llvm(self):
        code = """
PROGRAM ARITH
    INTEGER A, B, C
    REAL X, Y, Z
    A = 10
    B = 3
    C = A + B
    C = A - B
    C = A * B
    C = A / B
    C = A ** 2
    X = 10.0
    Y = 3.0
    Z = X + Y
    Z = X - Y
    Z = X * Y
    Z = X / Y
    Z = X ** 2.0
END
"""
        llvm_code = compile_code_llvm(code)
        self.assertIn("add", llvm_code, "Должны быть операции сложения")
        self.assertIn("sub", llvm_code, "Должны быть операции вычитания")
        self.assertIn("mul", llvm_code, "Должны быть операции умножения")
        self.assertIn("sdiv", llvm_code, "Должны быть операции деления для целых")
        self.assertIn("fadd", llvm_code, "Должны быть операции сложения для вещественных")
        self.assertIn("fsub", llvm_code, "Должны быть операции вычитания для вещественных")
        self.assertIn("fmul", llvm_code, "Должны быть операции умножения для вещественных")
        self.assertIn("fdiv", llvm_code, "Должны быть операции деления для вещественных")

    def test_complex_expressions_llvm(self):
        code = """PROGRAM COMPLEX_EXPR
        IMPLICIT NONE
        INTEGER A, B, C, D, RESULT
        A = 10
        B = 20
        C = 30
        D = 5
        RESULT = (A + B) * C / D - A * 2
        PRINT *, RESULT
        END"""
        llvm_code = compile_code_llvm(code)
        self.assertIn("add", llvm_code, "Должны быть операции сложения")
        self.assertIn("mul", llvm_code, "Должны быть операции умножения")
        self.assertIn("sdiv", llvm_code, "Должны быть операции деления")
        self.assertIn("sub", llvm_code, "Должны быть операции вычитания")


class TestLLVMArrays(unittest.TestCase):
    def test_array_operations_llvm(self):
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
        llvm_code = compile_code_llvm(code)
        self.assertIn("[10 x i32]", llvm_code, "Должна быть аллокация массива из 10 элементов")
        self.assertIn("getelementptr", llvm_code, "Должны быть операции getelementptr для доступа к массивам")
        self.assertIn("store", llvm_code, "Должны быть операции store для записи в массивы")
        self.assertIn("load", llvm_code, "Должны быть операции load для чтения из массивов")


class TestLLVMFunctions(unittest.TestCase):
    def test_function_calls_llvm(self):
        code = """PROGRAM FUNC_TEST
        IMPLICIT NONE
        REAL X, Y, Z
        X = 2.5
        Y = SQRT(X)
        Z = SIN(X) + COS(X)
        PRINT *, Y, Z
        END"""
        llvm_code = compile_code_llvm(code)
        self.assertIn("declare double @sqrt", llvm_code, "Должно быть объявление функции sqrt")
        self.assertIn("declare double @sin", llvm_code, "Должно быть объявление функции sin")
        self.assertIn("declare double @cos", llvm_code, "Должно быть объявление функции cos")
        self.assertIn("call double @sqrt", llvm_code, "Должен быть вызов sqrt")
        self.assertIn("call double @sin", llvm_code, "Должен быть вызов sin")
        self.assertIn("call double @cos", llvm_code, "Должен быть вызов cos")
        self.assertIn("double", llvm_code, "Должен использоваться тип double")

    def test_math_functions_llvm(self):
        code = """
PROGRAM MATHFN
    REAL X, Y
    X = 1.0
    Y = SIN(X)
    Y = COS(X)
    Y = TAN(X)
    Y = SQRT(X)
    Y = EXP(X)
    Y = LOG(X)
    Y = LOG10(X)
    Y = ABS(X)
END
"""
        llvm_code = compile_code_llvm(code)
        self.assertIn("declare", llvm_code, "Должны быть объявления функций")
        self.assertIn("call", llvm_code, "Должны быть вызовы функций")


class TestLLVMTypes(unittest.TestCase):
    def test_integer_type_llvm(self):
        code = """
PROGRAM INTEGERS
    INTEGER I, J, K
    I = 10
    J = 20
    K = I + J
END
"""
        llvm_code = compile_code_llvm(code)
        self.assertIn("i32", llvm_code, "Должен использоваться тип i32 для INTEGER")
        self.assertIn("alloca i32", llvm_code, "Должны быть аллокации i32")

    def test_real_type_llvm(self):
        code = """
PROGRAM REALS
    REAL X, Y, Z
    X = 3.14
    Y = 2.71
    Z = X + Y
END
"""
        llvm_code = compile_code_llvm(code)
        self.assertIn("double", llvm_code, "Должен использоваться тип double для REAL")
        self.assertIn("alloca double", llvm_code, "Должны быть аллокации double")

    def test_logical_type_llvm(self):
        code = """
PROGRAM LOGICALS
    LOGICAL L1, L2, L3
    L1 = .TRUE.
    L2 = .FALSE.
    L3 = L1 .AND. L2
END
"""
        llvm_code = compile_code_llvm(code)
        self.assertIn("i1", llvm_code, "Должен использоваться тип i1 для LOGICAL")
        self.assertIn("alloca i1", llvm_code, "Должны быть аллокации i1")


if __name__ == '__main__':
    unittest.main()

