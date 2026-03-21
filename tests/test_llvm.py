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
    return llvm_gen.generate(ast)


class TestLLVMBasic(unittest.TestCase):
    def test_llvm_structure(self):
        code = """
PROGRAM LLVM
    INTEGER X
    X = 42
    PRINT *, X
END
"""
        llvm_code = compile_code_llvm(code)
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
        self.assertIn("define i32 @main()", llvm_code)
        self.assertIn("alloca i32", llvm_code)
        self.assertIn("store", llvm_code)
        self.assertIn("ret i32 0", llvm_code)


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
        self.assertIn("loop_1:", llvm_code)
        self.assertIn("loop_body_1:", llvm_code)
        self.assertIn("loop_end_1:", llvm_code)
        self.assertIn("br i1", llvm_code)
        self.assertIn("add i32", llvm_code)

    def test_nested_loops_llvm(self):
        code = """PROGRAM NESTLP
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
        self.assertIn("define i32 @main()", llvm_code)
        self.assertIn("loop_1:", llvm_code)
        self.assertIn("mul i32", llvm_code)
        self.assertIn("add i32", llvm_code)
        self.assertIn("br label", llvm_code)


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
        self.assertIn("icmp", llvm_code)
        self.assertIn("br i1", llvm_code)
        self.assertIn("if_then", llvm_code)

    def test_complex_if_structure_llvm(self):
        code = """PROGRAM CMPIF
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
        self.assertIn("icmp", llvm_code)
        self.assertIn("br i1", llvm_code)
        self.assertIn("add i32", llvm_code)
        self.assertIn("sub i32", llvm_code)


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
        self.assertIn("add", llvm_code)
        self.assertIn("sub", llvm_code)
        self.assertIn("mul", llvm_code)
        self.assertIn("sdiv", llvm_code)
        self.assertIn("fadd", llvm_code)
        self.assertIn("fsub", llvm_code)
        self.assertIn("fmul", llvm_code)
        self.assertIn("fdiv", llvm_code)

    def test_complex_expressions_llvm(self):
        code = """PROGRAM CMPEX
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
        self.assertIn("add i32", llvm_code)
        self.assertIn("mul i32", llvm_code)
        self.assertIn("sdiv i32", llvm_code)
        self.assertIn("sub i32", llvm_code)


class TestLLVMArrays(unittest.TestCase):
    def test_array_operations_llvm(self):
        code = """PROGRAM ARRTST
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
        self.assertIn("alloca [10 x i32]", llvm_code)
        self.assertIn("getelementptr", llvm_code)
        self.assertIn("store i32", llvm_code)
        self.assertIn("load i32", llvm_code)

    def test_parameter_based_bounds_array_llvm(self):
        code = """PROGRAM BOUND1
        IMPLICIT NONE
        INTEGER LOW, HIGH
        PARAMETER (LOW = 5, HIGH = 7)
        INTEGER A(LOW:HIGH)
        A(LOW) = 10
        A(HIGH) = 20
        PRINT *, A(LOW), A(HIGH)
        END"""
        llvm_code = compile_code_llvm(code)
        self.assertIn("sub i32", llvm_code)
        self.assertIn("getelementptr", llvm_code)

    def test_common_and_exit_llvm(self):
        code = """PROGRAM CMNEXI
        IMPLICIT NONE
        INTEGER A(6), I
        COMMON /BLK/ A
        DO I = 1, 10
            IF (I .GT. 3) EXIT
        ENDDO
        CALL SHOW()
        END

        SUBROUTINE SHOW()
        IMPLICIT NONE
        INTEGER A(6)
        COMMON /BLK/ A
        PRINT *, A(1)
        RETURN
        END"""
        llvm_code = compile_code_llvm(code)
        self.assertIn("@COMMON_BLK_A = global [6 x i32] zeroinitializer", llvm_code)
        self.assertIn("br label %loop_end_", llvm_code)

    def test_parallel_runtime_helper_is_declared(self):
        code = """PROGRAM PARLLP
        IMPLICIT NONE
        INTEGER I, J
        INTEGER A(64,64)
        DO I = 1, 64
            DO J = 1, 64
                A(I,J) = I + J
            ENDDO
        ENDDO
        END"""
        llvm_code = compile_code_llvm(code)
        self.assertIn("declare void @fortran_parallel_for_i32", llvm_code)


class TestLLVMFunctions(unittest.TestCase):
    def test_function_calls_llvm(self):
        code = """PROGRAM FNTEST
        IMPLICIT NONE
        REAL X, Y, Z
        X = 2.5
        Y = SQRT(X)
        Z = SIN(X) + COS(X)
        PRINT *, Y, Z
        END"""
        llvm_code = compile_code_llvm(code)
        self.assertIn("declare double @sqrt", llvm_code)
        self.assertIn("declare double @sin", llvm_code)
        self.assertIn("declare double @cos", llvm_code)
        self.assertIn("call double @sqrt", llvm_code)
        self.assertIn("call double @sin", llvm_code)
        self.assertIn("call double @cos", llvm_code)

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
        self.assertIn("declare", llvm_code)
        self.assertIn("call", llvm_code)


class TestLLVMTypes(unittest.TestCase):
    def test_integer_type_llvm(self):
        code = """
PROGRAM INTS
    INTEGER I, J, K
    I = 10
    J = 20
    K = I + J
END
"""
        llvm_code = compile_code_llvm(code)
        self.assertIn("i32", llvm_code)
        self.assertIn("alloca i32", llvm_code)

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
        self.assertIn("double", llvm_code)
        self.assertIn("alloca double", llvm_code)

    def test_logical_type_llvm(self):
        code = """
PROGRAM LOGICS
    LOGICAL L1, L2, L3
    L1 = .TRUE.
    L2 = .FALSE.
    L3 = L1 .AND. L2
END
"""
        llvm_code = compile_code_llvm(code)
        self.assertIn("i1", llvm_code)
        self.assertIn("alloca i1", llvm_code)


if __name__ == '__main__':
    unittest.main()
