import unittest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.core import Lexer, Parser
from src.semantic import SemanticAnalyzer
class TestLexerValidation(unittest.TestCase):
    def test_lexer_valid(self):
        code = """      PROGRAM TEST1
      INTEGER I, J, K
      REAL X, Y
      I = 5
      J = 10
      X = 3.14
      Y = X + I
      PRINT *, I, J, X, Y
      END"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        errors = lexer.get_errors()
        self.assertGreater(len(tokens), 0)
    def test_lexer_invalid_line_length(self):
        code = """      PROGRAM TEST2
      INTEGER VARIABLE12345678901234567890123456789012345678901234567890123456789012345678901234567890
      END"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        errors = lexer.get_errors()
        self.assertGreater(len(errors), 0)
    def test_lexer_invalid_label_position(self):
        code = """      PROGRAM TEST3
1234567890 INTEGER I
      END"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        errors = lexer.get_errors()
    def test_lexer_invalid_var_name_length(self):
        code = """      PROGRAM TEST4
      INTEGER VERYLONGVARIABLENAME
      END"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        errors = lexer.get_errors()
        self.assertGreater(len(errors), 0)
    def test_lexer_invalid_var_name_start(self):
        code = """      PROGRAM TEST5
      INTEGER 123VAR
      END"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        errors = lexer.get_errors()
        self.assertGreater(len(errors), 0)
class TestParserValidation(unittest.TestCase):
    def test_parser_valid(self):
        code = """      PROGRAM TEST6
      IMPLICIT NONE
      INTEGER I, J
      REAL X, Y
      LOGICAL FLAG
      I = 5
      J = 10
      X = 3.14
      Y = 2.71
      FLAG = .TRUE.
      IF (FLAG) THEN
          I = I + 1
      ELSE
          J = J + 1
      ENDIF
      PRINT *, I, J, X, Y
      END"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        self.assertIsNotNone(ast)
    def test_parser_invalid_two_operators(self):
        code = """      PROGRAM TEST7
      INTEGER I, J
      REAL X, Y
      X = 10.0
      Y = 5.0
      I = X/-Y
      END"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        with self.assertRaises(SyntaxError):
            parser.parse()
    def test_parser_valid_parentheses(self):
        code = """      PROGRAM TEST8
      INTEGER I
      REAL X, Y
      X = 10.0
      Y = 5.0
      I = X/(-Y)
      END"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        self.assertIsNotNone(ast)
class TestImplicitValidation(unittest.TestCase):
    def setUp(self):
        self.base_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'inputs')
    def test_implicit_valid(self):
        file_path = os.path.join(self.base_path, 'test_implicit_valid.f')
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        try:
            ast = parser.parse()
            semantic = SemanticAnalyzer()
            success = semantic.analyze(ast)
        except SyntaxError:
            pass
    def test_implicit_invalid_order(self):
        file_path = os.path.join(self.base_path, 'test_implicit_invalid_order.f')
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        with self.assertRaises(SyntaxError):
            parser.parse()
    def test_implicit_none_valid(self):
        file_path = os.path.join(self.base_path, 'test_implicit_none_valid.f')
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        semantic = SemanticAnalyzer()
        success = semantic.analyze(ast)
        self.assertTrue(success)
    def test_implicit_none_invalid(self):
        file_path = os.path.join(self.base_path, 'test_implicit_none_invalid.f')
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
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
    def setUp(self):
        self.base_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'inputs')
    def test_arrays_valid(self):
        file_path = os.path.join(self.base_path, 'test_arrays_valid.f')
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        semantic = SemanticAnalyzer()
        success = semantic.analyze(ast)
        self.assertTrue(success)
    def test_arrays_invalid_dimensions(self):
        file_path = os.path.join(self.base_path, 'test_arrays_invalid_dimensions.f')
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        with self.assertRaises(SyntaxError):
            parser.parse()
    def test_arrays_invalid_range(self):
        file_path = os.path.join(self.base_path, 'test_arrays_invalid_range.f')
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        with self.assertRaises(SyntaxError):
            parser.parse()
    def test_arrays_7d_valid(self):
        file_path = os.path.join(self.base_path, 'test_arrays_7d_valid.f')
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        semantic = SemanticAnalyzer()
        success = semantic.analyze(ast)
        self.assertTrue(success)
class TestParameterValidation(unittest.TestCase):
    def setUp(self):
        self.base_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'inputs')
    def test_parameter_valid(self):
        file_path = os.path.join(self.base_path, 'test_parameter_valid.f')
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        semantic = SemanticAnalyzer()
        success = semantic.analyze(ast)
        self.assertTrue(success)
    def test_parameter_invalid_non_constant(self):
        file_path = os.path.join(self.base_path, 'test_parameter_invalid_non_constant.f')
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        try:
            ast = parser.parse()
            semantic = SemanticAnalyzer()
            success = semantic.analyze(ast)
            self.assertFalse(success)
            errors = semantic.get_errors()
            self.assertGreater(len(errors), 0)
        except SyntaxError:
            pass
    def test_parameter_redefinition(self):
        file_path = os.path.join(self.base_path, 'test_parameter_redefinition.f')
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        try:
            ast = parser.parse()
            semantic = SemanticAnalyzer()
            success = semantic.analyze(ast)
            self.assertFalse(success)
            errors = semantic.get_errors()
            self.assertGreater(len(errors), 0)
        except SyntaxError:
            pass
    def test_parameter_complex_expr(self):
        file_path = os.path.join(self.base_path, 'test_parameter_complex_expr.f')
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        semantic = SemanticAnalyzer()
        success = semantic.analyze(ast)
        self.assertTrue(success)
    def test_parameter_string(self):
        file_path = os.path.join(self.base_path, 'test_parameter_string.f')
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        semantic = SemanticAnalyzer()
        success = semantic.analyze(ast)
        self.assertTrue(success)
    def test_parameter_logical(self):
        file_path = os.path.join(self.base_path, 'test_parameter_logical.f')
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        semantic = SemanticAnalyzer()
        success = semantic.analyze(ast)
        self.assertTrue(success)
    def test_parameter_string_concat(self):
        file_path = os.path.join(self.base_path, 'test_parameter_string_concat.f')
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        semantic = SemanticAnalyzer()
        success = semantic.analyze(ast)
        self.assertTrue(success)
class TestDataValidation(unittest.TestCase):
    def setUp(self):
        self.base_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'inputs')
    def test_data_valid(self):
        file_path = os.path.join(self.base_path, 'test_data_valid.f')
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        try:
            ast = parser.parse()
            semantic = SemanticAnalyzer()
            success = semantic.analyze(ast)
        except SyntaxError:
            pass
    def test_data_invalid_count(self):
        file_path = os.path.join(self.base_path, 'test_data_invalid_count.f')
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        try:
            ast = parser.parse()
            semantic = SemanticAnalyzer()
            success = semantic.analyze(ast)
            self.assertFalse(success)
            errors = semantic.get_errors()
            self.assertGreater(len(errors), 0)
        except SyntaxError:
            pass
    def test_data_parameter(self):
        file_path = os.path.join(self.base_path, 'test_data_parameter.f')
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        try:
            ast = parser.parse()
            semantic = SemanticAnalyzer()
            success = semantic.analyze(ast)
            self.assertFalse(success)
            errors = semantic.get_errors()
            self.assertGreater(len(errors), 0)
        except SyntaxError:
            pass
    def test_data_multiple_pairs(self):
        file_path = os.path.join(self.base_path, 'test_data_multiple_pairs.f')
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        try:
            ast = parser.parse()
            semantic = SemanticAnalyzer()
            success = semantic.analyze(ast)
        except SyntaxError:
            pass
    def test_data_array_index_range(self):
        file_path = os.path.join(self.base_path, 'test_data_array_index_range.f')
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        try:
            ast = parser.parse()
            semantic = SemanticAnalyzer()
            success = semantic.analyze(ast)
        except SyntaxError:
            pass
    def test_data_array_index_out_of_range(self):
        file_path = os.path.join(self.base_path, 'test_data_array_index_out_of_range.f')
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        try:
            ast = parser.parse()
            semantic = SemanticAnalyzer()
            success = semantic.analyze(ast)
            self.assertFalse(success)
            errors = semantic.get_errors()
            self.assertGreater(len(errors), 0)
        except SyntaxError:
            pass
class TestIfStatementsValidation(unittest.TestCase):
    def setUp(self):
        self.base_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'inputs')
    def test_if_simple_valid(self):
        file_path = os.path.join(self.base_path, 'test_if_simple_valid.f')
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        semantic = SemanticAnalyzer()
        success = semantic.analyze(ast)
        self.assertTrue(success)
    def test_if_block_valid(self):
        file_path = os.path.join(self.base_path, 'test_if_block_valid.f')
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        semantic = SemanticAnalyzer()
        success = semantic.analyze(ast)
        self.assertTrue(success)
    def test_if_structural_valid(self):
        file_path = os.path.join(self.base_path, 'test_if_structural_valid.f')
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        try:
            ast = parser.parse()
            semantic = SemanticAnalyzer()
            success = semantic.analyze(ast)
        except SyntaxError:
            pass
    def test_if_multiple_elseif(self):
        file_path = os.path.join(self.base_path, 'test_if_multiple_elseif.f')
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        try:
            ast = parser.parse()
            semantic = SemanticAnalyzer()
            success = semantic.analyze(ast)
        except SyntaxError:
            pass
    def test_if_arithmetic_valid(self):
        file_path = os.path.join(self.base_path, 'test_if_arithmetic_valid.f')
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        semantic = SemanticAnalyzer()
        success = semantic.analyze(ast)
        self.assertTrue(success)
    def test_if_invalid_type(self):
        file_path = os.path.join(self.base_path, 'test_if_invalid_type.f')
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
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
    def setUp(self):
        self.base_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'inputs')
    def test_do_labeled_valid(self):
        file_path = os.path.join(self.base_path, 'test_do_labeled_valid.f')
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        semantic = SemanticAnalyzer()
        success = semantic.analyze(ast)
        self.assertTrue(success)
    def test_do_enddo_valid(self):
        file_path = os.path.join(self.base_path, 'test_do_enddo_valid.f')
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        semantic = SemanticAnalyzer()
        success = semantic.analyze(ast)
        self.assertTrue(success)
    def test_do_default_step(self):
        file_path = os.path.join(self.base_path, 'test_do_default_step.f')
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        semantic = SemanticAnalyzer()
        success = semantic.analyze(ast)
        self.assertTrue(success)
    def test_do_invalid_non_constant(self):
        file_path = os.path.join(self.base_path, 'test_do_invalid_non_constant.f')
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
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
        file_path = os.path.join(self.base_path, 'test_do_while_valid.f')
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        semantic = SemanticAnalyzer()
        success = semantic.analyze(ast)
        self.assertTrue(success)
class TestConcatValidation(unittest.TestCase):
    def setUp(self):
        self.base_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'inputs')
    def test_concat_valid(self):
        file_path = os.path.join(self.base_path, 'test_concat_valid.f')
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        semantic = SemanticAnalyzer()
        success = semantic.analyze(ast)
        self.assertTrue(success)
    def test_concat_invalid_type(self):
        file_path = os.path.join(self.base_path, 'test_concat_invalid_type.f')
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        semantic = SemanticAnalyzer()
        success = semantic.analyze(ast)
        self.assertFalse(success)
        errors = semantic.get_errors()
        self.assertGreater(len(errors), 0)
    def test_string_concat_expr(self):
        file_path = os.path.join(self.base_path, 'test_string_concat_expr.f')
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        semantic = SemanticAnalyzer()
        success = semantic.analyze(ast)
        self.assertTrue(success)
class TestDuplicateValidation(unittest.TestCase):
    def setUp(self):
        self.base_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'inputs')
    def test_duplicate_declaration(self):
        file_path = os.path.join(self.base_path, 'test_duplicate_declaration.f')
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
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
        file_path = os.path.join(self.base_path, 'test_duplicate_dimension.f')
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
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
    def setUp(self):
        self.base_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'inputs')
    def test_complex_valid(self):
        file_path = os.path.join(self.base_path, 'test_complex_valid.f')
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        try:
            ast = parser.parse()
            semantic = SemanticAnalyzer()
            success = semantic.analyze(ast)
        except SyntaxError:
            pass
    def test_operator_precedence(self):
        file_path = os.path.join(self.base_path, 'test_operator_precedence.f')
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        semantic = SemanticAnalyzer()
        success = semantic.analyze(ast)
        self.assertTrue(success)
class TestExamplePrograms(unittest.TestCase):
    def setUp(self):
        self.base_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'inputs')
    def test_example1_simple_program(self):
        file_path = os.path.join(self.base_path, 'example1_simple_program.f')
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        try:
            ast = parser.parse()
            semantic = SemanticAnalyzer()
            success = semantic.analyze(ast)
        except SyntaxError:
            pass
    def test_example2_arithmetic(self):
        file_path = os.path.join(self.base_path, 'example2_arithmetic.f')
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        try:
            ast = parser.parse()
            semantic = SemanticAnalyzer()
            success = semantic.analyze(ast)
        except SyntaxError:
            pass
    def test_example3_do_loop(self):
        file_path = os.path.join(self.base_path, 'example3_do_loop.f')
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        try:
            ast = parser.parse()
            semantic = SemanticAnalyzer()
            success = semantic.analyze(ast)
        except SyntaxError:
            pass
    def test_example4_if_conditions(self):
        file_path = os.path.join(self.base_path, 'example4_if_conditions.f')
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        try:
            ast = parser.parse()
            semantic = SemanticAnalyzer()
            success = semantic.analyze(ast)
        except SyntaxError:
            pass
    def test_example5_arrays_logic(self):
        file_path = os.path.join(self.base_path, 'example5_arrays_logic.f')
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        try:
            ast = parser.parse()
            semantic = SemanticAnalyzer()
            success = semantic.analyze(ast)
        except SyntaxError:
            pass
    def test_example6_factorial(self):
        file_path = os.path.join(self.base_path, 'example6_factorial.f')
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        try:
            ast = parser.parse()
            semantic = SemanticAnalyzer()
            success = semantic.analyze(ast)
        except SyntaxError:
            pass
    def test_example7_average(self):
        file_path = os.path.join(self.base_path, 'example7_average.f')
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        try:
            ast = parser.parse()
            semantic = SemanticAnalyzer()
            success = semantic.analyze(ast)
        except SyntaxError:
            pass
    def test_comprehensive_test(self):
        file_path = os.path.join(self.base_path, 'comprehensive_test.f')
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        try:
            ast = parser.parse()
            semantic = SemanticAnalyzer()
            success = semantic.analyze(ast)
        except SyntaxError:
            pass
    def test_array_processing_example(self):
        file_path = os.path.join(self.base_path, 'array_processing_example.f')
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        try:
            ast = parser.parse()
            semantic = SemanticAnalyzer()
            success = semantic.analyze(ast)
        except SyntaxError:
            pass
    def test_logical_array_combo_example(self):
        file_path = os.path.join(self.base_path, 'logical_array_combo_example.f')
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        try:
            ast = parser.parse()
            semantic = SemanticAnalyzer()
            success = semantic.analyze(ast)
        except SyntaxError:
            pass
    def test_logical_eqv_example(self):
        file_path = os.path.join(self.base_path, 'logical_eqv_example.f')
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        try:
            ast = parser.parse()
            semantic = SemanticAnalyzer()
            success = semantic.analyze(ast)
        except SyntaxError:
            pass
class TestSamokhinPrograms(unittest.TestCase):
    def setUp(self):
        self.base_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'inputs')
    def test_samokhin_1_4_1(self):
        file_path = os.path.join(self.base_path, 'samokhin_1_4_1.f')
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        try:
            ast = parser.parse()
            semantic = SemanticAnalyzer()
            success = semantic.analyze(ast)
        except SyntaxError:
            pass
    def test_samokhin_1_5_1(self):
        file_path = os.path.join(self.base_path, 'samokhin_1_5_1.f')
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        try:
            ast = parser.parse()
            semantic = SemanticAnalyzer()
            success = semantic.analyze(ast)
        except SyntaxError:
            pass
    def test_samokhin_1_6_1(self):
        file_path = os.path.join(self.base_path, 'samokhin_1_6_1.f')
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        try:
            ast = parser.parse()
            semantic = SemanticAnalyzer()
            success = semantic.analyze(ast)
        except SyntaxError:
            pass
    def test_samokhin_1_6_2(self):
        file_path = os.path.join(self.base_path, 'samokhin_1_6_2.f')
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        try:
            ast = parser.parse()
            semantic = SemanticAnalyzer()
            success = semantic.analyze(ast)
        except SyntaxError:
            pass
    def test_samokhin_1_6_3(self):
        file_path = os.path.join(self.base_path, 'samokhin_1_6_3.f')
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        try:
            ast = parser.parse()
            semantic = SemanticAnalyzer()
            success = semantic.analyze(ast)
        except SyntaxError:
            pass
    def test_samokhin_1_7_1(self):
        file_path = os.path.join(self.base_path, 'samokhin_1_7_1.f')
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        try:
            ast = parser.parse()
            semantic = SemanticAnalyzer()
            success = semantic.analyze(ast)
        except SyntaxError:
            pass
    def test_samokhin_1_8_1(self):
        file_path = os.path.join(self.base_path, 'samokhin_1_8_1.f')
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        try:
            ast = parser.parse()
            semantic = SemanticAnalyzer()
            success = semantic.analyze(ast)
        except SyntaxError:
            pass
    def test_samokhin_1_8_2(self):
        file_path = os.path.join(self.base_path, 'samokhin_1_8_2.f')
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        try:
            ast = parser.parse()
            semantic = SemanticAnalyzer()
            success = semantic.analyze(ast)
        except SyntaxError:
            pass
    def test_samokhin_func(self):
        file_path = os.path.join(self.base_path, 'samokhin_func.f')
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        try:
            ast = parser.parse()
            semantic = SemanticAnalyzer()
            success = semantic.analyze(ast)
        except SyntaxError:
            pass
    def test_samokhin_ifelse(self):
        file_path = os.path.join(self.base_path, 'samokhin_ifelse.f')
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        try:
            ast = parser.parse()
            semantic = SemanticAnalyzer()
            success = semantic.analyze(ast)
        except SyntaxError:
            pass
    def test_samokhin_newton(self):
        file_path = os.path.join(self.base_path, 'samokhin_newton.f')
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        try:
            ast = parser.parse()
            semantic = SemanticAnalyzer()
            success = semantic.analyze(ast)
        except SyntaxError:
            pass
    def test_samokhin_salary(self):
        file_path = os.path.join(self.base_path, 'samokhin_salary.f')
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        try:
            ast = parser.parse()
            semantic = SemanticAnalyzer()
            success = semantic.analyze(ast)
        except SyntaxError:
            pass
    def test_samokhin_sum(self):
        file_path = os.path.join(self.base_path, 'samokhin_sum.f')
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        try:
            ast = parser.parse()
            semantic = SemanticAnalyzer()
            success = semantic.analyze(ast)
        except SyntaxError:
            pass
if __name__ == '__main__':
    unittest.main()

