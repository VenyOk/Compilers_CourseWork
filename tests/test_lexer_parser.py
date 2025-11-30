import unittest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core import Lexer, Parser, TokenType, Program, Declaration, Assignment, DoLoop, IfStatement, PrintStatement


class TestLexer(unittest.TestCase):
    def test_simple_tokens(self):
        code = "PROGRAM TEST"
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        self.assertEqual(tokens[0].type, TokenType.PROGRAM)
        self.assertEqual(tokens[1].type, TokenType.IDENTIFIER)
        self.assertEqual(tokens[2].type, TokenType.EOF)

    def test_integer_literal(self):
        code = "123 456 789"
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        self.assertEqual(tokens[0].type, TokenType.INTEGER_LIT)
        self.assertEqual(tokens[0].value, 123)
        self.assertEqual(tokens[1].type, TokenType.INTEGER_LIT)
        self.assertEqual(tokens[1].value, 456)

    def test_integer_literals_extended(self):
        lexer = Lexer("0 123 -456 +789 999999")
        tokens = lexer.tokenize()
        self.assertEqual(tokens[0].type, TokenType.INTEGER_LIT)
        self.assertEqual(tokens[0].value, 0)
        self.assertEqual(tokens[0].line, 1)
        self.assertEqual(tokens[0].col, 1)
        self.assertEqual(tokens[1].type, TokenType.INTEGER_LIT)
        self.assertEqual(tokens[1].value, 123)

    def test_real_literal(self):
        code = "3.14 2.71 1.0E5"
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        self.assertEqual(tokens[0].type, TokenType.REAL_LIT)
        self.assertEqual(tokens[0].value, 3.14)
        self.assertEqual(tokens[1].type, TokenType.REAL_LIT)

    def test_real_literals_extended(self):
        lexer = Lexer("3.14 0.5 .25 1. 1.0E10 2.5E-5 1.0D+15")
        tokens = lexer.tokenize()
        real_tokens = [t for t in tokens if t.type == TokenType.REAL_LIT]
        self.assertGreaterEqual(len(real_tokens), 5)
        self.assertEqual(real_tokens[0].value, 3.14)
        self.assertIsInstance(real_tokens[0].value, float)

    def test_string_literal(self):
        code = "'hello' \"world\""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        self.assertEqual(tokens[0].type, TokenType.STRING_LIT)
        self.assertEqual(tokens[0].value, "hello")
        self.assertEqual(tokens[1].type, TokenType.STRING_LIT)
        self.assertEqual(tokens[1].value, "world")

    def test_string_literals_extended(self):
        lexer = Lexer("'hello' 'world' 'test string'")
        tokens = lexer.tokenize()
        str_tokens = [t for t in tokens if t.type == TokenType.STRING_LIT]
        self.assertGreaterEqual(len(str_tokens), 3)
        self.assertIn('hello', str_tokens[0].value)

    def test_keywords(self):
        code = "INTEGER REAL LOGICAL DO IF END"
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        self.assertEqual(tokens[0].type, TokenType.INTEGER)
        self.assertEqual(tokens[1].type, TokenType.REAL)
        self.assertEqual(tokens[2].type, TokenType.LOGICAL)
        self.assertEqual(tokens[3].type, TokenType.DO)
        self.assertEqual(tokens[4].type, TokenType.IF)
        self.assertEqual(tokens[5].type, TokenType.END)

    def test_keywords_case_insensitive(self):
        lexer = Lexer("PROGRAM program Program END end End")
        tokens = lexer.tokenize()
        self.assertEqual(tokens[0].type, TokenType.PROGRAM)
        self.assertEqual(tokens[1].type, TokenType.PROGRAM)
        self.assertEqual(tokens[2].type, TokenType.PROGRAM)

    def test_operators(self):
        code = "+ - * / ** = .EQ. .NE. .AND. .OR."
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        self.assertEqual(tokens[0].type, TokenType.PLUS)
        self.assertEqual(tokens[1].type, TokenType.MINUS)
        self.assertEqual(tokens[2].type, TokenType.STAR)
        self.assertEqual(tokens[3].type, TokenType.SLASH)
        self.assertEqual(tokens[4].type, TokenType.POWER)
        self.assertEqual(tokens[5].type, TokenType.ASSIGN_OP)
        self.assertEqual(tokens[6].type, TokenType.EQ)
        self.assertEqual(tokens[7].type, TokenType.NE)
        self.assertEqual(tokens[8].type, TokenType.AND)
        self.assertEqual(tokens[9].type, TokenType.OR)

    def test_comparison_operators_fortran_style(self):
        lexer = Lexer(".EQ. .NE. .LT. .LE. .GT. .GE.")
        tokens = lexer.tokenize()
        self.assertEqual(tokens[0].type, TokenType.EQ)
        self.assertEqual(tokens[1].type, TokenType.NE)
        self.assertEqual(tokens[2].type, TokenType.LT)
        self.assertEqual(tokens[3].type, TokenType.LE)
        self.assertEqual(tokens[4].type, TokenType.GT)
        self.assertEqual(tokens[5].type, TokenType.GE)

    def test_logical_operators_fortran_style(self):
        lexer = Lexer(".AND. .OR. .NOT. .TRUE. .FALSE.")
        tokens = lexer.tokenize()
        self.assertEqual(tokens[0].type, TokenType.AND)
        self.assertEqual(tokens[1].type, TokenType.OR)
        self.assertEqual(tokens[2].type, TokenType.NOT)
        self.assertEqual(tokens[3].type, TokenType.TRUE)
        self.assertEqual(tokens[4].type, TokenType.FALSE)

    def test_identifiers(self):
        lexer = Lexer("X MyVar _test A1B2C3 RESULT")
        tokens = lexer.tokenize()
        ident_tokens = [t for t in tokens if t.type == TokenType.IDENTIFIER]
        self.assertGreaterEqual(len(ident_tokens), 5)

    def test_comments(self):
        code = "INTEGER X ! это комментарий\nREAL Y"
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        self.assertEqual(tokens[0].type, TokenType.INTEGER)
        self.assertEqual(tokens[1].type, TokenType.IDENTIFIER)
        self.assertEqual(tokens[2].type, TokenType.REAL)

    def test_comments_c_style(self):
        code = """C это комментарий
        X = 5
        c ещё комментарий
        Y = 10"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        identifiers = [t.value for t in tokens if t.type == TokenType.IDENTIFIER]
        self.assertIn('X', identifiers)
        self.assertIn('Y', identifiers)

    def test_position_tracking(self):
        code = """X = 5
        Y = 10"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        self.assertEqual(tokens[0].line, 1)
        lines = set(t.line for t in tokens)
        self.assertGreater(len(lines), 1)

    def test_whitespace_handling(self):
        code = "X=5\nY   =    10\nZ\t=\t15"
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        identifiers = [t.value for t in tokens if t.type == TokenType.IDENTIFIER]
        self.assertIn('X', identifiers)
        self.assertIn('Y', identifiers)
        self.assertIn('Z', identifiers)

    def test_empty_input(self):
        lexer = Lexer("")
        tokens = lexer.tokenize()
        self.assertGreater(len(tokens), 0)
        self.assertEqual(tokens[-1].type, TokenType.EOF)


class TestParser(unittest.TestCase):
    def parse(self, code):
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        return parser.parse()

    def test_simple_program(self):
        code = """
PROGRAM TEST
END
"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        self.assertIsInstance(ast, Program)
        self.assertEqual(ast.name, "TEST")

    def test_simple_program_structure(self):
        code = """
        PROGRAM TEST
        IMPLICIT NONE
        INTEGER X
        X = 5
        END
        """
        ast = self.parse(code)
        self.assertIsInstance(ast, Program)
        self.assertEqual(ast.name, 'TEST')
        self.assertGreater(len(ast.declarations), 0)
        self.assertGreater(len(ast.statements), 0)

    def test_program_with_declarations(self):
        code = """
PROGRAM TEST
    INTEGER X, Y
    REAL PI
END
"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        self.assertEqual(len(ast.declarations), 2)
        self.assertEqual(ast.declarations[0].type, "INTEGER")
        self.assertEqual(ast.declarations[1].type, "REAL")

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
        self.assertIsInstance(ast, Program)
        decls = ast.declarations
        self.assertGreaterEqual(len(decls), 4)
        int_decls = [d for d in decls if isinstance(d, Declaration) and d.type == "INTEGER"]
        self.assertGreater(len(int_decls), 0)

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

    def test_do_loop(self):
        code = """
PROGRAM TEST
    INTEGER I, SUM
    SUM = 0
    DO I = 1, 10
        SUM = SUM + I
    ENDDO
END
"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        self.assertEqual(len(ast.statements), 2)
        self.assertIsInstance(ast.statements[1], DoLoop)

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

    def test_if_statement(self):
        code = """
PROGRAM TEST
    INTEGER X
    X = 5
    IF (X .GT. 0) THEN
        PRINT *, 'Positive'
    ENDIF
END
"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        self.assertEqual(len(ast.statements), 2)
        self.assertIsInstance(ast.statements[1], IfStatement)

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

    def test_print_statement(self):
        code = """
PROGRAM TEST
    INTEGER X
    X = 10
    PRINT *, X
END
"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        self.assertEqual(len(ast.statements), 2)
        self.assertIsInstance(ast.statements[1], PrintStatement)
        self.assertEqual(len(ast.statements[1].items), 1)

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


class TestIntegration(unittest.TestCase):
    def test_factorial_program(self):
        code = """
PROGRAM FACTORIAL
    INTEGER N, I, F
    N = 5
    F = 1
    DO I = 1, N
        F = F * I
    END DO
    PRINT *, F
END
"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        self.assertGreater(len(tokens), 0)
        self.assertEqual(tokens[-1].type, TokenType.EOF)
        parser = Parser(tokens)
        ast = parser.parse()
        self.assertIsInstance(ast, Program)

    def test_conditional_program(self):
        code = """
PROGRAM CONDITIONS
    INTEGER X, Y
    X = 15
    Y = 10
    IF (X .GT. Y) THEN
        PRINT *, 'X greater'
    ELSE
        PRINT *, 'Y greater'
    ENDIF
END
"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        self.assertIsInstance(ast, Program)
        self.assertEqual(len(ast.statements), 3)


if __name__ == '__main__':
    unittest.main()

