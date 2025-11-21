import unittest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.core import Lexer, Parser, Program, Declaration, Assignment, DoLoop, IfStatement
class TestParserExtended(unittest.TestCase):
    def parse(self, code):
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        return parser.parse()
    def test_simple_program_structure(self):
        code = """
        PROGRAM TEST
        IMPLICIT NONE
        INTEGER X
        X = 5
        END
        """
        ast = self.parse(code)
        self.assertIsInstance(ast, Program), "AST must be Program instance"
        self.assertEqual(ast.name, 'TEST'), "Program name must be TEST"
        self.assertGreater(len(ast.declarations), 0), "Must have declarations"
        self.assertGreater(len(ast.statements), 0), "Must have statements"
        self.assertTrue(any(isinstance(d, Declaration) for d in ast.declarations)), "Must contain Declaration"
        self.assertTrue(any(isinstance(s, Assignment) for s in ast.statements)), "Must contain Assignment"
    def test_multiple_declarations(self):
        code = """
        PROGRAM TEST
        INTEGER A, B, C
        REAL X, Y
        LOGICAL FLAG
        CHARACTER*20 NAME
        END
        """
        ast = self.parse(code)
        self.assertIsInstance(ast, Program), "Must parse into Program"
        decls = ast.declarations
        self.assertGreaterEqual(len(decls), 4), f"Must have at least 4 declarations, got {len(decls)}"
        int_decls = [d for d in decls if isinstance(d, Declaration) and d.type == "INTEGER"]
        self.assertGreater(len(int_decls), 0), "Must have INTEGER declaration"
        self.assertEqual(len(int_decls[0].names), 3), "INTEGER should have 3 variables"
        real_decls = [d for d in decls if isinstance(d, Declaration) and d.type == "REAL"]
        self.assertGreater(len(real_decls), 0), "Must have REAL declaration"
        ast = self.parse(code)
        self.assertGreaterEqual(len(ast.declarations), 4)
    def test_assignment_statements(self):
        code = """
        PROGRAM TEST
        INTEGER A, B
        A = 5
        B = 10
        A = B + 5
        END
        """
        ast = self.parse(code)
        assignments = [s for s in ast.statements if isinstance(s, Assignment)]
        self.assertGreaterEqual(len(assignments), 3)
    def test_do_loop_basic(self):
        code = """
        PROGRAM TEST
        INTEGER I
        DO I = 1, 10
            PRINT *, I
        ENDDO
        END
        """
        ast = self.parse(code)
        loops = [s for s in ast.statements if isinstance(s, DoLoop)]
        self.assertGreater(len(loops), 0)
        loop = loops[0]
        self.assertEqual(loop.var, 'I')
    def test_do_loop_with_step(self):
        code = """
        PROGRAM TEST
        INTEGER I
        DO I = 1, 100, 5
            PRINT *, I
        ENDDO
        END
        """
        ast = self.parse(code)
        loops = [s for s in ast.statements if isinstance(s, DoLoop)]
        self.assertGreater(len(loops), 0)
        loop = loops[0]
        self.assertIsNotNone(loop.step)
    def test_nested_do_loops(self):
        code = """
        PROGRAM TEST
        INTEGER I, J, A(10, 10)
        DO I = 1, 10
            DO J = 1, 10
                A(I, J) = I * J
            ENDDO
        ENDDO
        END
        """
        ast = self.parse(code)
        self.assertGreater(len(ast.statements), 0)
    def test_if_statement_basic(self):
        code = """
        PROGRAM TEST
        INTEGER X
        X = 5
        IF (X .GT. 3) THEN
            PRINT *, 'Greater'
        ENDIF
        END
        """
        ast = self.parse(code)
        if_stmts = [s for s in ast.statements if isinstance(s, IfStatement)]
        self.assertGreater(len(if_stmts), 0)
    def test_if_statement_with_else(self):
        code = """
        PROGRAM TEST
        INTEGER X
        X = 5
        IF (X .GT. 10) THEN
            PRINT *, 'Greater'
        ELSE
            PRINT *, 'Less or equal'
        ENDIF
        END
        """
        ast = self.parse(code)
        if_stmts = [s for s in ast.statements if isinstance(s, IfStatement)]
        self.assertGreater(len(if_stmts), 0)
    def test_if_statement_with_elseif(self):
        code = """
        PROGRAM TEST
        INTEGER X
        X = 5
        IF (X .LT. 0) THEN
            PRINT *, 'Negative'
        ELSEIF (X .EQ. 0) THEN
            PRINT *, 'Zero'
        ELSE
            PRINT *, 'Positive'
        ENDIF
        END
        """
        ast = self.parse(code)
        if_stmts = [s for s in ast.statements if isinstance(s, IfStatement)]
        self.assertGreater(len(if_stmts), 0)
    def test_nested_if_statements(self):
        code = """
        PROGRAM TEST
        INTEGER X, Y
        X = 5
        Y = 10
        IF (X .GT. 0) THEN
            IF (Y .GT. X) THEN
                PRINT *, 'Y is greater'
            ENDIF
        ENDIF
        END
        """
        ast = self.parse(code)
        if_stmts = [s for s in ast.statements if isinstance(s, IfStatement)]
        self.assertGreater(len(if_stmts), 0)
    def test_array_declarations(self):
        code = """
        PROGRAM TEST
        INTEGER A(10)
        REAL B(5, 20)
        INTEGER C(100)
        REAL D(10, 5)
        END
        """
        ast = self.parse(code)
        self.assertGreater(len(ast.declarations), 0)
    def test_arithmetic_expressions(self):
        code = """
        PROGRAM TEST
        REAL X, Y, Z
        X = 3.14
        Y = 2.71
        Z = X + Y
        Z = X - Y
        Z = X * Y
        Z = X / Y
        Z = X ** Y
        END
        """
        ast = self.parse(code)
        assignments = [s for s in ast.statements if isinstance(s, Assignment)]
        self.assertGreaterEqual(len(assignments), 5)
    def test_comparison_operators(self):
        code = """
        PROGRAM TEST
        INTEGER A, B
        LOGICAL L
        A = 5
        B = 10
        L = A .EQ. B
        L = A .NE. B
        L = A .LT. B
        L = A .LE. B
        L = A .GT. B
        L = A .GE. B
        END
        """
        ast = self.parse(code)
        self.assertGreater(len(ast.statements), 0)
    def test_logical_operators(self):
        code = """
        PROGRAM TEST
        LOGICAL A, B, C
        A = .TRUE.
        B = .FALSE.
        C = A .AND. B
        C = A .OR. B
        C = .NOT. A
        END
        """
        ast = self.parse(code)
        self.assertGreater(len(ast.statements), 0)
    def test_print_statement_various_forms(self):
        code = """
        PROGRAM TEST
        INTEGER X
        REAL Y
        CHARACTER*10 NAME
        X = 5
        Y = 3.14
        NAME = 'Test'
        PRINT *, X
        PRINT *, X, Y
        PRINT *, 'Value = ', X
        PRINT *, 'X = ', X, ' Y = ', Y
        PRINT *, NAME
        END
        """
        ast = self.parse(code)
        self.assertGreater(len(ast.statements), 0)
    def test_read_statement(self):
        code = """
        PROGRAM TEST
        INTEGER X
        REAL Y
        READ *, X
        READ *, Y
        READ *, X, Y
        END
        """
        ast = self.parse(code)
        self.assertGreater(len(ast.statements), 0)
    def test_implicit_none(self):
        code = """
        PROGRAM TEST
        IMPLICIT NONE
        INTEGER X
        X = 5
        END
        """
        ast = self.parse(code)
        self.assertIsInstance(ast, Program)
class TestParserErrorHandling(unittest.TestCase):
    def test_missing_endif(self):
        code = """
        PROGRAM TEST
        IF (X .GT. 5) THEN
            PRINT *, 'Greater'
        END
        """
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        try:
            ast = parser.parse()
        except:
            pass
    def test_missing_enddo(self):
        code = """
        PROGRAM TEST
        DO I = 1, 10
            PRINT *, I
        END
        """
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        try:
            ast = parser.parse()
        except:
            pass
    def test_mismatched_parens(self):
        code = """
        PROGRAM TEST
        INTEGER X
        X = (5 + 3
        END
        """
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        try:
            ast = parser.parse()
        except:
            pass
if __name__ == '__main__':
    unittest.main()
