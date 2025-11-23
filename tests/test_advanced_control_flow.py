import unittest
from src.core import Lexer, Parser, TokenType
class TestArithmeticIF(unittest.TestCase):
    def test_arithmetic_if_parsing(self):
        code = """PROGRAM TEST
        INTEGER X
        X = 5
        IF(X) 10, 20, 30
        10 CONTINUE
        20 CONTINUE
        30 CONTINUE
        STOP
        END"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        self.assertIsNotNone(ast)
    def test_arithmetic_if_with_expression(self):
        code = """PROGRAM TEST
        REAL X, Y
        X = 3.0
        Y = 2.0
        IF(X-Y) 10, 20, 30
        10 CONTINUE
        20 CONTINUE
        30 CONTINUE
        END"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        self.assertIsNotNone(ast)
class TestStructuredIFELSEIF(unittest.TestCase):
    def test_simple_if_then(self):
        code = """PROGRAM TEST
        INTEGER X
        X = 5
        IF(X .GT. 0) THEN
            X = X + 1
        END IF
        STOP
        END"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        self.assertIsNotNone(ast)
    def test_if_then_else(self):
        code = """PROGRAM TEST
        INTEGER X
        X = 5
        IF(X .GT. 10) THEN
            X = X + 1
        ELSE
            X = X - 1
        END IF
        STOP
        END"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        self.assertIsNotNone(ast)
    def test_if_then_else_if_then_else(self):
        code = """PROGRAM TEST
        REAL X
        X = 5.0
        IF(X .LT. -5.0) THEN
            X = X + 7.5
        ELSEIF(X .LT. 0.0) THEN
            X = X * 2.0 / 10.0
        ELSE
            X = SIN(X)
        ENDIF
        WRITE(*,*) X
        STOP
        END"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        self.assertIsNotNone(ast)
    def test_multiple_else_if(self):
        code = """PROGRAM TEST
        INTEGER X
        X = 5
        IF(X .LT. 0) THEN
            PRINT *, 'negative'
        ELSEIF(X .LT. 5) THEN
            PRINT *, 'small'
        ELSEIF(X .LT. 10) THEN
            PRINT *, 'medium'
        ELSEIF(X .LT. 20) THEN
            PRINT *, 'large'
        ELSE
            PRINT *, 'huge'
        ENDIF
        STOP
        END"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        self.assertIsNotNone(ast)
class TestDOWithLabel(unittest.TestCase):
    def test_do_loop_with_label_continue(self):
        code = """PROGRAM TEST
        INTEGER I
        DO 10 I=1,10
            PRINT *, I
        10 CONTINUE
        STOP
        END"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        self.assertIsNotNone(ast)
    def test_do_loop_with_step(self):
        code = """PROGRAM TEST
        INTEGER I
        DO 5 I=1,100,2
            PRINT *, I
        5 CONTINUE
        STOP
        END"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        self.assertIsNotNone(ast)
    def test_nested_do_with_labels(self):
        code = """PROGRAM TEST
        INTEGER I, J
        DO 10 I=1,5
            DO 20 J=1,5
                PRINT *, I, J
            20 CONTINUE
        10 CONTINUE
        STOP
        END"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        self.assertIsNotNone(ast)
class TestDOWHILEWithLabel(unittest.TestCase):
    def test_do_while_with_label(self):
        code = """PROGRAM TEST
        INTEGER I
        I = 1
        DO 10 WHILE(I .LE. 5)
            PRINT *, I
            I = I + 1
        10 CONTINUE
        STOP
        END"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        self.assertIsNotNone(ast)
    def test_do_while_complex_condition(self):
        code = """PROGRAM TEST
        REAL X, Y
        X = 0.0
        Y = 1.0
        DO 25 WHILE(X**2 + Y**2 .LE. R**2)
            PRINT *, X, Y
            X = X + 0.1
        25 CONTINUE
        STOP
        END"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        self.assertIsNotNone(ast)
class TestComplexPrograms(unittest.TestCase):
    def test_salary_calculation_program(self):
        code = """PROGRAM SALARY
        IMPLICIT NONE
        REAL RATE, TIME, PAY, TT
        READ(*, *) RATE, TIME
        IF(TIME .GT. 40) THEN
            TT = TIME - 40.0
            PAY = 40.0 * RATE
            PAY = PAY + 1.5 * RATE * TT
        ELSE
            PAY = RATE * TIME
        END IF
        WRITE(*, *) PAY
        STOP
        END"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        self.assertIsNotNone(ast)
    def test_polynomial_sum_program(self):
        code = """PROGRAM POLYSUM
        IMPLICIT NONE
        REAL X, E, F, A
        INTEGER I, N
        READ(*, *) X, E, N
        F = 1.0
        A = -X
        I = 1
        DO 10 WHILE(ABS(A/I) .GE. E)
            F = F + A / I
            I = I + 1
            A = -X * A
        10 CONTINUE
        WRITE(*, *) F
        STOP
        END"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        self.assertIsNotNone(ast)
    def test_piecewise_function(self):
        code = """PROGRAM PIECEWISE
        IMPLICIT NONE
        REAL X, F
        READ(*, *) X
        IF(X .LT. -5.0) THEN
            F = X + 7.5
        ELSEIF(X .LT. 0.0) THEN
            F = X * X / 10.0
        ELSE
            F = SIN(X)
        ENDIF
        WRITE(*, *) F
        STOP
        END"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        self.assertIsNotNone(ast)
if __name__ == "__main__":
    unittest.main(verbosity=2)