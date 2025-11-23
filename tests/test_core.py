import unittest
from src.core import Lexer, Parser, TokenType, Program, Declaration, DoLoop, IfStatement, PrintStatement
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
    def test_real_literal(self):
        code = "3.14 2.71 1.0E5"
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        self.assertEqual(tokens[0].type, TokenType.REAL_LIT)
        self.assertEqual(tokens[0].value, 3.14)
        self.assertEqual(tokens[1].type, TokenType.REAL_LIT)
        self.assertEqual(tokens[2].type, TokenType.REAL_LIT)
    def test_string_literal(self):
        code = "'hello' \"world\""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        self.assertEqual(tokens[0].type, TokenType.STRING_LIT)
        self.assertEqual(tokens[0].value, "hello")
        self.assertEqual(tokens[1].type, TokenType.STRING_LIT)
        self.assertEqual(tokens[1].value, "world")
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
    def test_comments(self):
        code = "INTEGER X ! это комментарий\nREAL Y"
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        self.assertEqual(tokens[0].type, TokenType.INTEGER)
        self.assertEqual(tokens[1].type, TokenType.IDENTIFIER)
        self.assertEqual(tokens[2].type, TokenType.REAL)
class TestParser(unittest.TestCase):
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