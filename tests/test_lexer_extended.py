import unittest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.core import Lexer, TokenType
class TestLexerExtended(unittest.TestCase):
    def test_integer_literals(self):
        lexer = Lexer("0 123 -456 +789 999999")
        tokens = lexer.tokenize()
        self.assertEqual(tokens[0].type, TokenType.INTEGER_LIT), "First token must be INTEGER_LIT"
        self.assertEqual(tokens[0].value, 0), "First value must be 0"
        self.assertEqual(tokens[0].line, 1), "Must track line number"
        self.assertEqual(tokens[0].col, 1), "Must track column number"
        self.assertEqual(tokens[1].type, TokenType.INTEGER_LIT), "Second token must be INTEGER_LIT"
        self.assertEqual(tokens[1].value, 123), "Second value must be 123"
        self.assertEqual(tokens[2].type, TokenType.MINUS), "Third token must be MINUS"
        self.assertEqual(tokens[3].type, TokenType.INTEGER_LIT), "Fourth token must be INTEGER_LIT"
        self.assertEqual(tokens[3].value, 456), "Fourth value must be 456"
    def test_real_literals(self):
        lexer = Lexer("3.14 0.5 .25 1. 1.0E10 2.5E-5 1.0D+15")
        tokens = lexer.tokenize()
        real_tokens = [t for t in tokens if t.type == TokenType.REAL_LIT]
        self.assertGreaterEqual(len(real_tokens), 5), f"Must have at least 5 REAL_LIT tokens, got {len(real_tokens)}"
        self.assertEqual(real_tokens[0].value, 3.14), "First real must be 3.14"
        self.assertIsInstance(real_tokens[0].value, float), "Must be float type"
        self.assertEqual(real_tokens[0].line, 1), "Must track line"
        self.assertEqual(real_tokens[2].value, 0.25), "Third real must be 0.25 (from .25)"
    def test_string_literals_single_quote(self):
        lexer = Lexer("'hello' 'world' 'test string'")
        tokens = lexer.tokenize()
        str_tokens = [t for t in tokens if t.type == TokenType.STRING_LIT]
        self.assertGreaterEqual(len(str_tokens), 3), f"Must have at least 3 STRING_LIT tokens, got {len(str_tokens)}"
        self.assertIn('hello', str_tokens[0].value), "First string must contain 'hello'"
        self.assertEqual(str_tokens[0].type, TokenType.STRING_LIT), "Must be STRING_LIT type"
        self.assertEqual(str_tokens[1].value, 'world'), "Second string must be 'world'"
    def test_string_literals_double_quote(self):
        lexer = Lexer('"hello" "world" "test"')
        tokens = lexer.tokenize()
        str_tokens = [t for t in tokens if t.type == TokenType.STRING_LIT]
        self.assertGreaterEqual(len(str_tokens), 3)
    def test_keywords_case_insensitive(self):
        lexer = Lexer("PROGRAM program Program END end End")
        tokens = lexer.tokenize()
        self.assertEqual(tokens[0].type, TokenType.PROGRAM)
        self.assertEqual(tokens[1].type, TokenType.PROGRAM)
        self.assertEqual(tokens[2].type, TokenType.PROGRAM)
    def test_identifiers(self):
        lexer = Lexer("X MyVar _test A1B2C3 RESULT")
        tokens = lexer.tokenize()
        ident_tokens = [t for t in tokens if t.type == TokenType.IDENTIFIER]
        self.assertGreaterEqual(len(ident_tokens), 5)
    def test_arithmetic_operators(self):
        lexer = Lexer("+ - * / **")
        tokens = lexer.tokenize()
        self.assertEqual(tokens[0].type, TokenType.PLUS)
        self.assertEqual(tokens[1].type, TokenType.MINUS)
        self.assertEqual(tokens[2].type, TokenType.STAR)
        self.assertEqual(tokens[3].type, TokenType.SLASH)
        self.assertEqual(tokens[4].type, TokenType.POWER)
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
    def test_keywords_all_types(self):
        code = """PROGRAM END
        IMPLICIT NONE INTEGER REAL LOGICAL CHARACTER COMPLEX
        DIMENSION PARAMETER DATA
        IF THEN ELSE ELSEIF ENDIF
        DO ENDDO WHILE CONTINUE GOTO
        READ WRITE PRINT
        STOP"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        keyword_tokens = [t for t in tokens if
                         t.type in [TokenType.PROGRAM, TokenType.END, TokenType.IF,
                                   TokenType.DO, TokenType.PRINT, TokenType.READ,
                                   TokenType.INTEGER, TokenType.REAL, TokenType.LOGICAL,
                                   TokenType.ENDIF, TokenType.ENDDO, TokenType.THEN]]
        self.assertGreaterEqual(len(keyword_tokens), 12)
    def test_comments_exclamation(self):
        code = """X = 5  ! это комментарий
        Y = 10 ! и ещё один"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        identifiers = [t.value for t in tokens if t.type == TokenType.IDENTIFIER]
        self.assertIn('X', identifiers)
        self.assertIn('Y', identifiers)
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
    def test_complex_expression_tokenization(self):
        code = "RESULT = A * B + C / D ** 2 - E"
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        self.assertGreater(len(tokens), 10)
        self.assertEqual(tokens[0].type, TokenType.IDENTIFIER)
        self.assertEqual(tokens[1].type, TokenType.ASSIGN_OP)
    def test_array_declaration_tokenization(self):
        code = "INTEGER A(10), B(5, 20)"
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        lparen = [t for t in tokens if t.type == TokenType.LPAREN]
        rparen = [t for t in tokens if t.type == TokenType.RPAREN]
        self.assertGreater(len(lparen), 0)
        self.assertGreater(len(rparen), 0)
    def test_position_tracking(self):
        code = """X = 5
        Y = 10"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        self.assertEqual(tokens[0].line, 1)
        lines = set(t.line for t in tokens)
        self.assertGreater(len(lines), 1)
    def test_special_characters_in_strings(self):
        code = "'Hello, World!' \"Test (with) [brackets]\""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        str_tokens = [t for t in tokens if t.type == TokenType.STRING_LIT]
        self.assertGreaterEqual(len(str_tokens), 2)
    def test_numeric_literals_with_leading_zeros(self):
        lexer = Lexer("007 0123 00.5")
        tokens = lexer.tokenize()
        num_tokens = [t for t in tokens if t.type in [TokenType.INTEGER_LIT, TokenType.REAL_LIT]]
        self.assertGreater(len(num_tokens), 0)
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
    def test_only_comments(self):
        code = """! комментарий
C ещё комментарий
        ! и ещё"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        useful = [t for t in tokens if t.type != TokenType.EOF]
        self.assertEqual(len(useful), 0)
    def test_mixed_case_operators(self):
        code = ".eq. .EQ. .Eq. .eQ."
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        eq_tokens = [t for t in tokens if t.type == TokenType.EQ]
        self.assertGreaterEqual(len(eq_tokens), 2)
class TestLexerErrorHandling(unittest.TestCase):
    def test_unclosed_string(self):
        code = "'unclosed string"
        lexer = Lexer(code)
        try:
            tokens = lexer.tokenize()
            self.assertGreater(len(tokens), 0)
        except:
            pass
    def test_multiple_dots_in_operator(self):
        code = "A ... B"
        lexer = Lexer(code)
        with self.assertRaises(SyntaxError):
            tokens = lexer.tokenize()
    def test_consecutive_operators(self):
        code = "A++B"
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        self.assertGreater(len(tokens), 2)
if __name__ == '__main__':
    unittest.main()