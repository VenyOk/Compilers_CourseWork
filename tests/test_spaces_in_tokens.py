import unittest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.core import Lexer, Parser
from src.semantic import SemanticAnalyzer
class TestSpacesInIdentifiers(unittest.TestCase):
    def test_identifier_with_spaces(self):
        code = """      PROGRAM TEST
      INTEGER M Y V A R
      M Y V A R = 1 2 3
      END"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        errors = lexer.get_errors()
        self.assertEqual(len(errors), 0)
        identifiers = [t.value for t in tokens if t.type.name == 'IDENTIFIER']
        self.assertIn('MYVAR', identifiers)
    def test_multiple_identifiers_with_spaces(self):
        code = """      PROGRAM TEST
      INTEGER V A R 1, V A R 2
      V A R 1 = 1 0
      V A R 2 = 2 0
      END"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        errors = lexer.get_errors()
        self.assertEqual(len(errors), 0)
        identifiers = [t.value for t in tokens if t.type.name == 'IDENTIFIER']
        self.assertIn('VAR1', identifiers)
        self.assertIn('VAR2', identifiers)
class TestSpacesInNumbers(unittest.TestCase):
    def test_integer_with_spaces(self):
        code = """      P R O G R A M T E S T
      I N T E G E R I
      I = 1 2 3
      E N D"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        errors = lexer.get_errors()
        self.assertEqual(len(errors), 0)
        integer_lits = [t for t in tokens if t.type.name == 'INTEGER_LIT']
        self.assertGreater(len(integer_lits), 0)
        self.assertEqual(integer_lits[0].value, 123)
    def test_multiple_integers_with_spaces(self):
        code = """      P R O G R A M T E S T
      I N T E G E R I, J
      I = 1 2 3
      J = 4 5 6
      E N D"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        errors = lexer.get_errors()
        self.assertEqual(len(errors), 0)
        integer_lits = [t.value for t in tokens if t.type.name == 'INTEGER_LIT']
        self.assertIn(123, integer_lits)
        self.assertIn(456, integer_lits)
    def test_real_with_spaces_integer_part(self):
        code = """      P R O G R A M T E S T
      R E A L X
      X = 3.14
      E N D"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        errors = lexer.get_errors()
        self.assertEqual(len(errors), 0)
        real_lits = [t for t in tokens if t.type.name == 'REAL_LIT']
        self.assertGreater(len(real_lits), 0)
        self.assertAlmostEqual(real_lits[0].value, 3.14, places=2)
    def test_real_with_spaces_in_exponent(self):
        code = """      P R O G R A M T E S T
      R E A L X
      X = 1.5E+2
      E N D"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        errors = lexer.get_errors()
        self.assertEqual(len(errors), 0)
        real_lits = [t for t in tokens if t.type.name == 'REAL_LIT']
        self.assertGreater(len(real_lits), 0)
        self.assertAlmostEqual(real_lits[0].value, 150.0, places=1)
    def test_real_with_d_exponent(self):
        code = """      P R O G R A M T E S T
      R E A L X
      X = 2.5D-1
      E N D"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        errors = lexer.get_errors()
        self.assertEqual(len(errors), 0)
        real_lits = [t for t in tokens if t.type.name == 'REAL_LIT']
        self.assertGreater(len(real_lits), 0)
        self.assertAlmostEqual(real_lits[0].value, 0.25, places=2)
class TestSpacesInParser(unittest.TestCase):
    def test_parse_program_with_spaces(self):
        code = """      PROGRAM TEST
      INTEGER I
      I = 1 2 3
      PRINT *, I
      END"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        self.assertIsNotNone(ast)
        self.assertEqual(ast.name, "TEST")
        self.assertEqual(len(ast.statements), 2)
    def test_parse_assignment_with_spaces(self):
        code = """      PROGRAM TEST
      INTEGER I, J
      I = 1 2 3
      J = 4 5 6
      END"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        self.assertIsNotNone(ast)
        assignments = [s for s in ast.statements if s.__class__.__name__ == 'Assignment']
        self.assertEqual(len(assignments), 2)
    def test_parse_arithmetic_with_spaces(self):
        code = """      PROGRAM TEST
      INTEGER I
      I = 1 2 3 + 4 5 6
      END"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        self.assertIsNotNone(ast)
        assignments = [s for s in ast.statements if s.__class__.__name__ == 'Assignment']
        self.assertEqual(len(assignments), 1)
    def test_parse_if_with_spaces(self):
        code = """      PROGRAM TEST
      INTEGER I
      IF (I .GT. 0) THEN
      PRINT *, I
      ENDIF
      END"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        self.assertIsNotNone(ast)
        if_statements = [s for s in ast.statements if s.__class__.__name__ == 'IfStatement']
        self.assertEqual(len(if_statements), 1)
    def test_parse_do_loop_with_spaces(self):
        code = """      PROGRAM TEST
      INTEGER I
      DO I = 1, 1 0
      PRINT *, I
      ENDDO
      END"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        self.assertIsNotNone(ast)
        do_loops = [s for s in ast.statements if s.__class__.__name__ == 'DoLoop']
        self.assertEqual(len(do_loops), 1)
class TestSpacesInSemantic(unittest.TestCase):
    def test_semantic_analysis_with_spaces(self):
        code = """      PROGRAM TEST
      INTEGER I, J
      I = 1 2 3
      J = 4 5 6
      PRINT *, I, J
      END"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        analyzer = SemanticAnalyzer()
        analyzer.analyze(ast)
        errors = analyzer.get_errors()
        self.assertEqual(len(errors), 0)
    def test_semantic_implicit_with_spaces(self):
        code = """      PROGRAM TEST
      IMPLICIT NONE
      INTEGER I
      I = 1 2 3
      END"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        analyzer = SemanticAnalyzer()
        analyzer.analyze(ast)
        errors = analyzer.get_errors()
        self.assertEqual(len(errors), 0)
    def test_semantic_array_with_spaces(self):
        code = """      PROGRAM TEST
      INTEGER A(1 0)
      A(1) = 5
      END"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        analyzer = SemanticAnalyzer()
        analyzer.analyze(ast)
        errors = analyzer.get_errors()
        self.assertEqual(len(errors), 0)
class TestSpacesEdgeCases(unittest.TestCase):
    def test_spaces_between_keywords(self):
        code = """      PROGRAM TEST
      INTEGER I
      END"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        errors = lexer.get_errors()
        self.assertEqual(len(errors), 0)
        token_values = [t.value for t in tokens if t.type.name != 'COMMENT' and t.type.name != 'EOF']
        self.assertIn('PROGRAM', token_values)
        self.assertIn('TEST', token_values)
        self.assertIn('INTEGER', token_values)
        self.assertIn('I', token_values)
        self.assertIn('END', token_values)
    def test_mixed_spaces_and_no_spaces(self):
        code = """      PROGRAM TEST
      INTEGER I
      I = 1 2 3
      PRINT *, I
      END"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        errors = lexer.get_errors()
        self.assertEqual(len(errors), 0)
        parser = Parser(tokens)
        ast = parser.parse()
        self.assertIsNotNone(ast)
    def test_spaces_in_function_call(self):
        code = """      PROGRAM TEST
      REAL X
      X = SIN(3.14)
      END"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        errors = lexer.get_errors()
        self.assertEqual(len(errors), 0)
        parser = Parser(tokens)
        ast = parser.parse()
        self.assertIsNotNone(ast)
        function_calls = []
        for stmt in ast.statements:
            if stmt.__class__.__name__ == 'Assignment':
                if stmt.value.__class__.__name__ == 'FunctionCall':
                    function_calls.append(stmt.value.name)
        self.assertIn('SIN', function_calls)
    def test_spaces_in_parameter(self):
        code = """      PROGRAM TEST
      PARAMETER (N = 1 0 0)
      INTEGER I
      I = N
      END"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        errors = lexer.get_errors()
        self.assertEqual(len(errors), 0)
        token_types = [t.type.name for t in tokens if t.type.name != 'COMMENT' and t.type.name != 'EOF']
        self.assertIn('PARAMETER', token_types)
        parser = Parser(tokens)
        ast = parser.parse()
        self.assertIsNotNone(ast)
        parameter_statements = [d for d in ast.declarations if d.__class__.__name__ == 'ParameterStatement']
        self.assertGreater(len(parameter_statements), 0)
if __name__ == '__main__':
    unittest.main()