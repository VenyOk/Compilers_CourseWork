import unittest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.core import Lexer, Parser, TokenType
from scripts.main import compile_fortran
class TestAllOperators(unittest.TestCase):
    def test_all_relational_operators_fortran_style(self):
        code = """
        PROGRAM TEST
        INTEGER X, Y
        LOGICAL A, B, C, D, E, F
        X = 5
        Y = 10
        A = X .EQ. Y
        B = X .NE. Y
        C = X .LT. Y
        D = X .LE. Y
        E = X .GT. Y
        F = X .GE. Y
        PRINT *, A, B, C, D, E, F
        END
        """
        result = compile_fortran(code, 'both')
        self.assertIsNone(result.get('errors') or None)
    def test_all_relational_operators_c_style(self):
        pass
    def test_all_logical_operators(self):
        code = """
        PROGRAM TEST
        LOGICAL A, B, C, D, E, F, G
        A = .TRUE.
        B = .FALSE.
        C = A .AND. B
        D = A .OR. B
        E = .NOT. A
        F = A .EQV. B
        G = A .NEQV. B
        PRINT *, C, D, E, F, G
        END
        """
        result = compile_fortran(code, 'both')
        self.assertIsNone(result.get('errors') or None)
    def test_all_arithmetic_operators(self):
        code = """
        PROGRAM TEST
        INTEGER A, B, C, D, E
        REAL X, Y, Z
        A = 10
        B = 3
        C = A + B
        D = A - B
        E = A * B
        X = 10.0
        Y = 3.0
        Z = X / Y
        Z = X ** Y
        PRINT *, C, D, E, Z
        END
        """
        result = compile_fortran(code, 'both')
        self.assertIsNone(result.get('errors') or None)
    def test_unary_operators(self):
        code = """
        PROGRAM TEST
        INTEGER X, Y
        REAL A, B
        X = 5
        Y = -X
        A = 3.14
        B = +A
        B = -A
        PRINT *, Y, B
        END
        """
        result = compile_fortran(code, 'both')
        self.assertIsNone(result.get('errors') or None)
class TestAllDataTypes(unittest.TestCase):
    def test_integer_type(self):
        code = """
        PROGRAM TEST
        INTEGER X, Y, Z
        X = 42
        Y = -100
        Z = 0
        PRINT *, X, Y, Z
        END
        """
        result = compile_fortran(code, 'both')
        self.assertIsNone(result.get('errors') or None)
    def test_real_type(self):
        code = """
        PROGRAM TEST
        REAL X, Y, Z
        X = 3.14
        Y = -2.5
        Z = 0.0
        PRINT *, X, Y, Z
        END
        """
        result = compile_fortran(code, 'both')
        self.assertIsNone(result.get('errors') or None)
    def test_real_double_precision(self):
        code = """
        PROGRAM TEST
        REAL*8 X, Y
        X = 3.141592653589793D0
        Y = 1.5D-10
        PRINT *, X, Y
        END
        """
        result = compile_fortran(code, 'both')
        self.assertIsNone(result.get('errors') or None)
    def test_logical_type(self):
        code = """
        PROGRAM TEST
        LOGICAL A, B
        A = .TRUE.
        B = .FALSE.
        PRINT *, A, B
        END
        """
        result = compile_fortran(code, 'both')
        self.assertIsNone(result.get('errors') or None)
    def test_character_type(self):
        code = """
        PROGRAM TEST
        CHARACTER*10 NAME
        CHARACTER*20 MESSAGE
        NAME = 'Hello'
        MESSAGE = "World"
        PRINT *, NAME, MESSAGE
        END
        """
        result = compile_fortran(code, 'both')
        self.assertIsNone(result.get('errors') or None)
    def test_complex_type(self):
        code = """
        PROGRAM TEST
        COMPLEX Z
        PRINT *, 'Complex type declared'
        END
        """
        result = compile_fortran(code, 'both')
        self.assertIsNone(result.get('errors') or None)
class TestAllBuiltinFunctions(unittest.TestCase):
    def test_trigonometric_functions(self):
        code = """
        PROGRAM TEST
        REAL X, Y, Z, W, V, U
        X = 0.5
        Y = SIN(X)
        Z = COS(X)
        W = TAN(X)
        V = ASIN(0.5)
        U = ACOS(0.5)
        PRINT *, Y, Z, W, V, U
        END
        """
        result = compile_fortran(code, 'both')
        self.assertIsNone(result.get('errors') or None)
    def test_exponential_functions(self):
        code = """
        PROGRAM TEST
        REAL X, Y, Z, W
        X = 2.0
        Y = EXP(X)
        Z = LOG(X)
        W = LOG10(100.0)
        PRINT *, Y, Z, W
        END
        """
        result = compile_fortran(code, 'both')
        self.assertIsNone(result.get('errors') or None)
    def test_sqrt_and_abs(self):
        code = """
        PROGRAM TEST
        REAL X, Y
        INTEGER A, B
        X = 16.0
        Y = SQRT(X)
        A = -5
        B = ABS(A)
        Y = ABS(-3.14)
        PRINT *, Y, B
        END
        """
        result = compile_fortran(code, 'both')
        self.assertIsNone(result.get('errors') or None)
    def test_min_max_functions(self):
        code = """
        PROGRAM TEST
        INTEGER A, B, C, M, N
        REAL X, Y, Z, P, Q
        A = 5
        B = 3
        C = 7
        M = MIN(A, B, C)
        N = MAX(A, B, C)
        X = 1.5
        Y = 2.5
        Z = 0.5
        P = MIN(X, Y, Z)
        Q = MAX(X, Y, Z)
        PRINT *, M, N, P, Q
        END
        """
        result = compile_fortran(code, 'both')
        self.assertIsNone(result.get('errors') or None)
    def test_mod_function(self):
        code = """
        PROGRAM TEST
        INTEGER A, B, C
        REAL X, Y, Z
        A = 17
        B = 5
        C = MOD(A, B)
        X = 17.5
        Y = 5.0
        Z = MOD(X, Y)
        PRINT *, C, Z
        END
        """
        result = compile_fortran(code, 'both')
        self.assertIsNone(result.get('errors') or None)
    def test_type_conversion_functions(self):
        code = """
        PROGRAM TEST
        INTEGER I
        REAL X, Y
        X = 3.7
        I = INT(X)
        I = 10
        Y = FLOAT(I)
        PRINT *, I, Y
        END
        """
        result = compile_fortran(code, 'both')
        self.assertIsNone(result.get('errors') or None)
class TestAllControlStructures(unittest.TestCase):
    def test_if_then_endif(self):
        code = """
        PROGRAM TEST
        INTEGER X
        X = 5
        IF (X .GT. 0) THEN
            PRINT *, 'Positive'
        ENDIF
        END
        """
        result = compile_fortran(code, 'both')
        self.assertIsNone(result.get('errors') or None)
    def test_if_else_endif(self):
        code = """
        PROGRAM TEST
        INTEGER X
        X = -5
        IF (X .GT. 0) THEN
            PRINT *, 'Positive'
        ELSE
            PRINT *, 'Non-positive'
        ENDIF
        END
        """
        result = compile_fortran(code, 'both')
        self.assertIsNone(result.get('errors') or None)
    def test_if_elseif_else_endif(self):
        code = """
        PROGRAM TEST
        INTEGER X
        X = 0
        IF (X .LT. 0) THEN
            PRINT *, 'Negative'
        ELSEIF (X .GT. 0) THEN
            PRINT *, 'Positive'
        ELSE
            PRINT *, 'Zero'
        ENDIF
        END
        """
        result = compile_fortran(code, 'both')
        self.assertIsNone(result.get('errors') or None)
    def test_multiple_elseif(self):
        code = """
        PROGRAM TEST
        INTEGER X
        X = 50
        IF (X .LT. 0) THEN
            PRINT *, 'Negative'
        ELSEIF (X .LT. 10) THEN
            PRINT *, 'Small'
        ELSEIF (X .LT. 100) THEN
            PRINT *, 'Medium'
        ELSE
            PRINT *, 'Large'
        ENDIF
        END
        """
        result = compile_fortran(code, 'both')
        self.assertIsNone(result.get('errors') or None)
    def test_nested_if_statements(self):
        code = """
        PROGRAM TEST
        INTEGER X, Y
        X = 5
        Y = 10
        IF (X .GT. 0) THEN
            IF (Y .GT. 0) THEN
                IF (X .LT. Y) THEN
                    PRINT *, 'All conditions true'
                ENDIF
            ENDIF
        ENDIF
        END
        """
        result = compile_fortran(code, 'both')
        self.assertIsNone(result.get('errors') or None)
    def test_do_loop_basic(self):
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
        result = compile_fortran(code, 'both')
        self.assertIsNone(result.get('errors') or None)
    def test_do_loop_with_step(self):
        code = """
        PROGRAM TEST
        INTEGER I, SUM
        SUM = 0
        DO I = 1, 20, 2
            SUM = SUM + I
        ENDDO
        PRINT *, SUM
        END
        """
        result = compile_fortran(code, 'both')
        self.assertIsNone(result.get('errors') or None)
    def test_do_loop_negative_step(self):
        code = """
        PROGRAM TEST
        INTEGER I
        DO I = 10, 1, -1
            PRINT *, I
        ENDDO
        END
        """
        result = compile_fortran(code, 'both')
        self.assertIsNone(result.get('errors') or None)
    def test_nested_do_loops(self):
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
        result = compile_fortran(code, 'both')
        self.assertIsNone(result.get('errors') or None)
    def test_do_while_loop(self):
        code = """
        PROGRAM TEST
        INTEGER I
        I = 0
        DO WHILE (I .LT. 10)
            I = I + 1
            PRINT *, I
        ENDDO
        END
        """
        result = compile_fortran(code, 'both')
        self.assertIsNone(result.get('errors') or None)
    def test_continue_statement(self):
        code = """
        PROGRAM TEST
        INTEGER I
        DO I = 1, 10
            IF (I .EQ. 5) THEN
                CONTINUE
            ELSE
                PRINT *, I
            ENDIF
        ENDDO
        END
        """
        result = compile_fortran(code, 'both')
        self.assertIsNone(result.get('errors') or None)
    def test_goto_statement(self):
        code = """
        PROGRAM TEST
        INTEGER I
        I = 0
        10 I = I + 1
        PRINT *, I
        IF (I .LT. 5) THEN
            GOTO 10
        ENDIF
        END
        """
        result = compile_fortran(code, 'both')
        self.assertIsNone(result.get('errors') or None)
    def test_stop_statement(self):
        code = """
        PROGRAM TEST
        PRINT *, 'Before stop'
        STOP
        PRINT *, 'After stop'
        END
        """
        result = compile_fortran(code, 'both')
        self.assertIsNone(result.get('errors') or None)
class TestAllIOStatements(unittest.TestCase):
    def test_print_single_value(self):
        code = """
        PROGRAM TEST
        INTEGER X
        X = 42
        PRINT *, X
        END
        """
        result = compile_fortran(code, 'both')
        self.assertIsNone(result.get('errors') or None)
    def test_print_multiple_values(self):
        code = """
        PROGRAM TEST
        INTEGER X, Y, Z
        X = 1
        Y = 2
        Z = 3
        PRINT *, X, Y, Z
        END
        """
        result = compile_fortran(code, 'both')
        self.assertIsNone(result.get('errors') or None)
    def test_print_with_strings(self):
        code = """
        PROGRAM TEST
        INTEGER X
        X = 42
        PRINT *, 'Value:', X
        PRINT *, 'Result is', X * 2
        END
        """
        result = compile_fortran(code, 'both')
        self.assertIsNone(result.get('errors') or None)
    def test_read_statement(self):
        code = """
        PROGRAM TEST
        INTEGER X, Y
        READ *, X, Y
        PRINT *, X + Y
        END
        """
        result = compile_fortran(code, 'both')
        self.assertIsNone(result.get('errors') or None)
    def test_read_alternative_format(self):
        code = """
        PROGRAM TEST
        INTEGER X
        READ (*, *) X
        PRINT *, X
        END
        """
        result = compile_fortran(code, 'both')
        self.assertIsNone(result.get('errors') or None)
    def test_write_statement(self):
        code = """
        PROGRAM TEST
        INTEGER X
        X = 42
        WRITE (*, *) X
        WRITE (*, *) 'Value:', X
        END
        """
        result = compile_fortran(code, 'both')
        self.assertIsNone(result.get('errors') or None)
class TestArrays(unittest.TestCase):
    def test_one_dimensional_array(self):
        code = """
        PROGRAM TEST
        INTEGER A(10)
        INTEGER I
        DO I = 1, 10
            A(I) = I * 2
        ENDDO
        PRINT *, A(1), A(5), A(10)
        END
        """
        result = compile_fortran(code, 'both')
        self.assertIsNone(result.get('errors') or None)
    def test_two_dimensional_array(self):
        code = """
        PROGRAM TEST
        REAL MATRIX(5, 5)
        INTEGER I, J
        DO I = 1, 5
            DO J = 1, 5
                MATRIX(I, J) = I * J
            ENDDO
        ENDDO
        PRINT *, MATRIX(2, 3)
        END
        """
        result = compile_fortran(code, 'both')
        self.assertIsNone(result.get('errors') or None)
    def test_three_dimensional_array(self):
        code = """
        PROGRAM TEST
        INTEGER TENSOR(3, 3, 3)
        INTEGER I, J, K
        DO I = 1, 3
            DO J = 1, 3
                DO K = 1, 3
                    TENSOR(I, J, K) = I + J + K
                ENDDO
            ENDDO
        ENDDO
        PRINT *, TENSOR(1, 1, 1)
        END
        """
        result = compile_fortran(code, 'both')
        self.assertIsNone(result.get('errors') or None)
    def test_character_array(self):
        code = """
        PROGRAM TEST
        CHARACTER*10 WORDS(5)
        INTEGER I
        DO I = 1, 5
            WORDS(I) = 'Word'
        ENDDO
        PRINT *, WORDS(1)
        END
        """
        result = compile_fortran(code, 'both')
        self.assertIsNone(result.get('errors') or None)
    def test_array_in_expressions(self):
        code = """
        PROGRAM TEST
        INTEGER A(10), B(10), C(10)
        INTEGER I
        DO I = 1, 10
            A(I) = I
            B(I) = I * 2
            C(I) = A(I) + B(I)
        ENDDO
        PRINT *, C(5)
        END
        """
        result = compile_fortran(code, 'both')
        self.assertIsNone(result.get('errors') or None)
class TestComplexExpressions(unittest.TestCase):
    def test_operator_precedence(self):
        code = """
        PROGRAM TEST
        INTEGER A, B, C, D, RESULT
        A = 2
        B = 3
        C = 4
        D = 5
        RESULT = A + B * C
        RESULT = (A + B) * C
        RESULT = A ** B ** C
        RESULT = A + B * C / D
        PRINT *, RESULT
        END
        """
        result = compile_fortran(code, 'both')
        self.assertIsNone(result.get('errors') or None)
    def test_mixed_arithmetic_and_logical(self):
        code = """
        PROGRAM TEST
        INTEGER X, Y, Z
        LOGICAL RESULT
        X = 10
        Y = 5
        Z = 15
        RESULT = (X + Y) .GT. Z .AND. X .LT. 20
        RESULT = X .GT. 0 .OR. Y .LT. 0
        PRINT *, RESULT
        END
        """
        result = compile_fortran(code, 'both')
        self.assertIsNone(result.get('errors') or None)
    def test_nested_expressions(self):
        code = """
        PROGRAM TEST
        REAL X, Y, Z, RESULT
        X = 2.0
        Y = 3.0
        Z = 4.0
        RESULT = (X + Y) * (Z - X) / (Y + 1.0)
        RESULT = SQRT(X ** 2 + Y ** 2)
        RESULT = SIN(X) * COS(Y) + TAN(Z)
        PRINT *, RESULT
        END
        """
        result = compile_fortran(code, 'both')
        self.assertIsNone(result.get('errors') or None)
    def test_logical_equivalence_operations(self):
        code = """
        PROGRAM TEST
        LOGICAL A, B, C, D
        A = .TRUE.
        B = .TRUE.
        C = .FALSE.
        D = A .EQV. B
        D = A .NEQV. C
        D = (A .AND. B) .EQV. (.NOT. C)
        PRINT *, D
        END
        """
        result = compile_fortran(code, 'both')
        self.assertIsNone(result.get('errors') or None)
    def test_type_mixing_in_expressions(self):
        code = """
        PROGRAM TEST
        INTEGER I
        REAL X, Y
        I = 5
        X = 3.14
        Y = I + X
        Y = I * X
        Y = FLOAT(I) + X
        I = INT(X) + I
        PRINT *, Y, I
        END
        """
        result = compile_fortran(code, 'both')
        self.assertIsNone(result.get('errors') or None)
class TestComplexPrograms(unittest.TestCase):
    def test_factorial_program(self):
        code = """
        PROGRAM FACTORIAL
        IMPLICIT NONE
        INTEGER N, I, F
        N = 5
        F = 1
        DO I = 1, N
            F = F * I
        ENDDO
        PRINT *, 'Factorial of', N, 'is', F
        END
        """
        result = compile_fortran(code, 'both')
        self.assertIsNone(result.get('errors') or None)
    def test_array_processing_program(self):
        code = """
        PROGRAM ARRAY_PROC
        IMPLICIT NONE
        INTEGER I, N, SUM
        INTEGER A(100)
        N = 10
        SUM = 0
        DO I = 1, N
            A(I) = I * I
            SUM = SUM + A(I)
        ENDDO
        PRINT *, 'Sum of squares:', SUM
        END
        """
        result = compile_fortran(code, 'both')
        self.assertIsNone(result.get('errors') or None)
    def test_matrix_operations(self):
        code = """
        PROGRAM MATRIX
        IMPLICIT NONE
        INTEGER I, J
        REAL A(5, 5), B(5, 5), C(5, 5)
        DO I = 1, 5
            DO J = 1, 5
                A(I, J) = I * J
                B(I, J) = I + J
                C(I, J) = A(I, J) + B(I, J)
            ENDDO
        ENDDO
        PRINT *, 'Matrix C(3,3) =', C(3, 3)
        END
        """
        result = compile_fortran(code, 'both')
        self.assertIsNone(result.get('errors') or None)
    def test_conditional_processing(self):
        code = """
        PROGRAM COND_PROC
        IMPLICIT NONE
        INTEGER I, COUNT_POS, COUNT_NEG, COUNT_ZERO
        INTEGER A(10)
        COUNT_POS = 0
        COUNT_NEG = 0
        COUNT_ZERO = 0
        DO I = 1, 10
            A(I) = I - 5
            IF (A(I) .GT. 0) THEN
                COUNT_POS = COUNT_POS + 1
            ELSEIF (A(I) .LT. 0) THEN
                COUNT_NEG = COUNT_NEG + 1
            ELSE
                COUNT_ZERO = COUNT_ZERO + 1
            ENDIF
        ENDDO
        PRINT *, 'Positive:', COUNT_POS
        PRINT *, 'Negative:', COUNT_NEG
        PRINT *, 'Zero:', COUNT_ZERO
        END
        """
        result = compile_fortran(code, 'both')
        self.assertIsNone(result.get('errors') or None)
    def test_mathematical_calculations(self):
        code = """
        PROGRAM MATH_CALC
        IMPLICIT NONE
        REAL X, Y, Z, PI
        PI = 3.14159265
        X = PI / 4.0
        Y = SIN(X) ** 2 + COS(X) ** 2
        Z = SQRT(ABS(Y - 1.0))
        PRINT *, 'Result:', Z
        END
        """
        result = compile_fortran(code, 'both')
        self.assertIsNone(result.get('errors') or None)
    def test_nested_control_structures(self):
        code = """
        PROGRAM NESTED
        IMPLICIT NONE
        INTEGER I, J, COUNT
        COUNT = 0
        DO I = 1, 10
            IF (I .GT. 5) THEN
                DO J = 1, 5
                    IF (J .EQ. I - 5) THEN
                        COUNT = COUNT + 1
                    ENDIF
                ENDDO
            ENDIF
        ENDDO
        PRINT *, 'Count:', COUNT
        END
        """
        result = compile_fortran(code, 'both')
        self.assertIsNone(result.get('errors') or None)
    def test_all_features_combined(self):
        code = """
        PROGRAM ALL_FEATURES
        IMPLICIT NONE
        INTEGER I, J, N, SUM, MAX_VAL
        REAL X(10), Y(10), AVG, PI
        LOGICAL FLAG
        CHARACTER*20 MSG
        PI = 3.14159265
        N = 10
        SUM = 0
        FLAG = .TRUE.
        MSG = 'Processing'
        DO I = 1, N
            X(I) = FLOAT(I) * PI / 10.0
            Y(I) = SIN(X(I))
            SUM = SUM + INT(Y(I) * 100.0)
            IF (Y(I) .GT. 0.5) THEN
                FLAG = FLAG .AND. .TRUE.
            ELSE
                FLAG = FLAG .OR. .FALSE.
            ENDIF
        ENDDO
        AVG = FLOAT(SUM) / FLOAT(N)
        MAX_VAL = MAX(SUM, N, 100)
        PRINT *, MSG, 'complete'
        PRINT *, 'Average:', AVG
        PRINT *, 'Max:', MAX_VAL
        PRINT *, 'Flag:', FLAG
        END
        """
        result = compile_fortran(code, 'both')
        self.assertIsNone(result.get('errors') or None)
if __name__ == '__main__':
    unittest.main()
