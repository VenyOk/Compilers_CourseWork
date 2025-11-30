import unittest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.semantic import SemanticAnalyzer
from src.core import Lexer, Parser


class TestImplicitValidation(unittest.TestCase):
    def test_implicit_none_valid(self):
        code = """      PROGRAM TEST39
      IMPLICIT NONE
      INTEGER I
      REAL X
      I = 5
      X = 3.14
      PRINT *, I, X
      END"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        semantic = SemanticAnalyzer()
        success = semantic.analyze(ast)
        self.assertTrue(success)

    def test_implicit_none_invalid(self):
        code = """      PROGRAM TEST40
      IMPLICIT NONE
      I = 5
      X = 3.14
      PRINT *, I, X
      END"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        semantic = SemanticAnalyzer()
        success = semantic.analyze(ast)
        self.assertFalse(success)
        errors = semantic.get_errors()
        self.assertGreater(len(errors), 0)


class TestArraysValidation(unittest.TestCase):
    def test_arrays_valid(self):
        code = """      PROGRAM TEST11
      INTEGER I
      DIMENSION A(1:10, 1:20)
      DIMENSION B(100)
      INTEGER C(5:15)
      I = 1
      A(1, 1) = 10
      B(1) = 20
      C(5) = 30
      PRINT *, A(1, 1), B(1), C(5)
      END"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        semantic = SemanticAnalyzer()
        success = semantic.analyze(ast)
        self.assertTrue(success)

    def test_arrays_7d_valid(self):
        code = """      PROGRAM TEST38
      DIMENSION A(1:2, 1:2, 1:2, 1:2, 1:2, 1:2, 1:2)
      INTEGER I
      I = 1
      A(1, 1, 1, 1, 1, 1, 1) = 100
      PRINT *, A(1, 1, 1, 1, 1, 1, 1)
      END"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        semantic = SemanticAnalyzer()
        success = semantic.analyze(ast)
        self.assertTrue(success)


class TestParameterValidation(unittest.TestCase):
    def test_parameter_valid(self):
        code = """      PROGRAM TEST14
      PARAMETER (PI = 3.14159, E = 2.71828)
      PARAMETER (TWO = 2, FOUR = TWO * TWO)
      PARAMETER (SUM = PI + E)
      REAL X
      X = PI * 2.0
      PRINT *, PI, E, TWO, FOUR, SUM
      END"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        semantic = SemanticAnalyzer()
        success = semantic.analyze(ast)
        self.assertTrue(success)

    def test_parameter_complex_expr(self):
        code = """      PROGRAM TEST35
      PARAMETER (A = 2, B = 3, C = A + B)
      PARAMETER (D = A * B, E = D - C)
      PARAMETER (F = A ** B)
      INTEGER I
      I = C
      PRINT *, A, B, C, D, E, F
      END"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        semantic = SemanticAnalyzer()
        success = semantic.analyze(ast)
        self.assertTrue(success)

    def test_parameter_string(self):
        code = """      PROGRAM TEST41
      PARAMETER (MSG = 'HELLO')
      CHARACTER*10 STR
      STR = MSG
      PRINT *, STR
      END"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        semantic = SemanticAnalyzer()
        success = semantic.analyze(ast)
        self.assertTrue(success)

    def test_parameter_logical(self):
        code = """      PROGRAM TEST42
      PARAMETER (FLAG = .TRUE.)
      LOGICAL L1
      L1 = FLAG
      PRINT *, L1
      END"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        semantic = SemanticAnalyzer()
        success = semantic.analyze(ast)
        self.assertTrue(success)


class TestIfStatementsValidation(unittest.TestCase):
    def test_if_simple_valid(self):
        code = """      PROGRAM TEST18
      LOGICAL FLAG
      INTEGER I
      FLAG = .TRUE.
      I = 0
      IF (FLAG) I = 1
      PRINT *, I
      END"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        semantic = SemanticAnalyzer()
        success = semantic.analyze(ast)
        self.assertTrue(success)

    def test_if_block_valid(self):
        code = """      PROGRAM TEST19
      LOGICAL FLAG
      INTEGER I, J
      FLAG = .TRUE.
      I = 0
      J = 0
      IF (FLAG) THEN
          I = 1
          J = 2
      ELSE
          I = 3
          J = 4
      ENDIF
      PRINT *, I, J
      END"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        semantic = SemanticAnalyzer()
        success = semantic.analyze(ast)
        self.assertTrue(success)

    def test_if_arithmetic_valid(self):
        code = """      PROGRAM TEST21
      INTEGER I
      REAL X
      I = -5
      X = 0.0
      IF (I) 10, 20, 30
   10 PRINT *, 'NEGATIVE'
      GOTO 40
   20 PRINT *, 'ZERO'
      GOTO 40
   30 PRINT *, 'POSITIVE'
   40 CONTINUE
      END"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        semantic = SemanticAnalyzer()
        success = semantic.analyze(ast)
        self.assertTrue(success)

    def test_if_invalid_type(self):
        code = """      PROGRAM TEST22
      INTEGER I
      I = 5
      IF (I) THEN
          PRINT *, 'ERROR'
      ENDIF
      END"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        semantic = SemanticAnalyzer()
        success = semantic.analyze(ast)
        self.assertFalse(success)
        errors = semantic.get_errors()
        self.assertGreater(len(errors), 0)


class TestDoLoopsValidation(unittest.TestCase):
    def test_do_labeled_valid(self):
        code = """      PROGRAM TEST23
      INTEGER I, SUM
      SUM = 0
   10 DO 20 I = 1, 10, 1
          SUM = SUM + I
   20 CONTINUE
      PRINT *, SUM
      END"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        semantic = SemanticAnalyzer()
        success = semantic.analyze(ast)
        self.assertTrue(success)

    def test_do_enddo_valid(self):
        code = """      PROGRAM TEST24
      INTEGER I, SUM
      SUM = 0
      DO I = 1, 10
          SUM = SUM + I
      END DO
      PRINT *, SUM
      END"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        semantic = SemanticAnalyzer()
        success = semantic.analyze(ast)
        self.assertTrue(success)

    def test_do_default_step(self):
        code = """      PROGRAM TEST25
      INTEGER I, SUM
      SUM = 0
      DO I = 1, 10
          SUM = SUM + I
      END DO
      PRINT *, SUM
      END"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        semantic = SemanticAnalyzer()
        success = semantic.analyze(ast)
        self.assertTrue(success)

    def test_do_invalid_non_constant(self):
        code = """      PROGRAM TEST26
      INTEGER I, J, K
      J = 1
      K = 10
      DO I = J, K
          PRINT *, I
      END DO
      END"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        semantic = SemanticAnalyzer()
        success = semantic.analyze(ast)
        self.assertFalse(success)
        errors = semantic.get_errors()
        self.assertGreater(len(errors), 0)

    def test_do_while_valid(self):
        code = """      PROGRAM TEST37
      INTEGER I
      LOGICAL FLAG
      I = 0
      FLAG = .TRUE.
      DO WHILE (FLAG)
          I = I + 1
          IF (I .GE. 10) FLAG = .FALSE.
      END DO
      PRINT *, I
      END"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        semantic = SemanticAnalyzer()
        success = semantic.analyze(ast)
        self.assertTrue(success)


class TestConcatValidation(unittest.TestCase):
    def test_concat_valid(self):
        code = """      PROGRAM TEST27
      CHARACTER*10 STR1, STR2, STR3
      STR1 = 'HELLO'
      STR2 = 'WORLD'
      STR3 = STR1 // STR2
      PRINT *, STR3
      END"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        semantic = SemanticAnalyzer()
        success = semantic.analyze(ast)
        self.assertTrue(success)

    def test_concat_invalid_type(self):
        code = """      PROGRAM TEST28
      INTEGER I, J
      I = 5
      J = 10
      I = I // J
      END"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        semantic = SemanticAnalyzer()
        success = semantic.analyze(ast)
        self.assertFalse(success)
        errors = semantic.get_errors()
        self.assertGreater(len(errors), 0)


class TestDuplicateValidation(unittest.TestCase):
    def test_duplicate_declaration(self):
        code = """      PROGRAM TEST29
      INTEGER I
      REAL I
      END"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        semantic = SemanticAnalyzer()
        success = semantic.analyze(ast)
        self.assertFalse(success)
        errors = semantic.get_errors()
        self.assertGreater(len(errors), 0)

    def test_duplicate_dimension(self):
        code = """      PROGRAM TEST30
      DIMENSION A(100)
      DIMENSION A(200)
      END"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        semantic = SemanticAnalyzer()
        success = semantic.analyze(ast)
        self.assertFalse(success)
        errors = semantic.get_errors()
        self.assertGreater(len(errors), 0)


class TestComplexPrograms(unittest.TestCase):
    def test_operator_precedence(self):
        code = """      PROGRAM TEST45
      INTEGER I, J, K
      REAL X, Y, Z
      LOGICAL L1, L2, L3
      I = 2
      J = 3
      K = 4
      X = 2.0
      Y = 3.0
      Z = 4.0
      I = I + J * K
      X = X + Y * Z
      L1 = .TRUE.
      L2 = .FALSE.
      L3 = L1 .AND. L2 .OR. L1
      PRINT *, I, X, L3
      END"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        semantic = SemanticAnalyzer()
        success = semantic.analyze(ast)
        self.assertTrue(success)


if __name__ == '__main__':
    unittest.main()

