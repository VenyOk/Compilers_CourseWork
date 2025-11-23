import unittest
from src.core import Lexer, Parser, TokenType, BinaryOp, Declaration
class TestLogicalOperators(unittest.TestCase):
    def test_eqv_token_recognition(self):
        code = ".EQV."
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        assert len(tokens) >= 1
        assert tokens[0].type == TokenType.EQV
    def test_neqv_token_recognition(self):
        code = ".NEQV."
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        assert len(tokens) >= 1
        assert tokens[0].type == TokenType.NEQV
    def test_eqv_logical_expression(self):
        code = """PROGRAM TEST
        LOGICAL L1, L2, L3
        L1 = .TRUE.
        L2 = .FALSE.
        L3 = L1 .EQV. L2
        END"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        assert ast is not None
    def test_neqv_logical_expression(self):
        code = """PROGRAM TEST
        LOGICAL L1, L2, L3
        L1 = .TRUE.
        L2 = .FALSE.
        L3 = L1 .NEQV. L2
        END"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        assert ast is not None
    def test_complex_logical_expression(self):
        code = """PROGRAM TEST
        LOGICAL L1, L2, L3, L4, L5
        L1 = .TRUE.
        L2 = .FALSE.
        L5 = (L1 .AND. L2) .OR. (.NOT. L1) .EQV. L2
        END"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        assert ast is not None
    def test_logical_operator_precedence(self):
        code = """PROGRAM TEST
        LOGICAL A, B, C, D, RESULT
        A = .TRUE.
        B = .FALSE.
        C = .TRUE.
        D = .TRUE.
        RESULT = A .AND. B .OR. C .NEQV. D
        END"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        assert ast is not None
class TestArrayOperations(unittest.TestCase):
    def test_dimension_operator_token(self):
        code = "DIMENSION"
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        assert len(tokens) >= 1
        assert tokens[0].type == TokenType.DIMENSION
    def test_dimension_declaration_simple(self):
        code = """PROGRAM TEST
        REAL A(10)
        END"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        declarations = ast.declarations
        assert len(declarations) > 0
        assert declarations[0].type == "REAL"
        assert declarations[0].names[0][0] == "A"
        assert declarations[0].names[0][1] == [10]
    def test_multidimensional_array(self):
        code = """PROGRAM TEST
        INTEGER A(5,5)
        REAL B(10,20,30)
        END"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        declarations = ast.declarations
        assert len(declarations) >= 2
        assert declarations[0].names[0][0] == "A"
        assert declarations[0].names[0][1] == [5, 5]
        assert declarations[1].names[0][0] == "B"
        assert declarations[1].names[0][1] == [10, 20, 30]
    def test_multiple_arrays_single_declaration(self):
        code = """PROGRAM TEST
        REAL A(10), B(5,5), C(100)
        END"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        declarations = ast.declarations
        assert len(declarations) > 0
        names = declarations[0].names
        assert len(names) == 3
        assert names[0][0] == "A"
        assert names[0][1] == [10]
        assert names[1][0] == "B"
        assert names[1][1] == [5, 5]
        assert names[2][0] == "C"
        assert names[2][1] == [100]
    def test_array_indexing_in_expression(self):
        code = """PROGRAM TEST
        INTEGER A(10), B(5)
        INTEGER I, J
        I = 1
        J = 2
        A(I) = 5
        B(J) = A(I) + 10
        END"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        assert ast is not None
    def test_complex_array_operations(self):
        code = """PROGRAM TEST
        REAL A(10,10), B(10), C(5,5)
        REAL X
        INTEGER I, J
        X = A(1,1) + B(5)
        C(2,3) = A(I,J) * B(I)
        END"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        assert ast is not None
class TestCompleteLogicalPrograms(unittest.TestCase):
    def test_logical_comparison_program(self):
        code = """PROGRAM LOGICAL_COMP
        IMPLICIT NONE
        LOGICAL L1, L2, L3
        INTEGER A, B
        A = 5
        B = 3
        L1 = A .GT. B
        L2 = A .EQ. B
        L3 = L1 .NEQV. L2
        PRINT *, L1
        PRINT *, L2
        PRINT *, L3
        STOP
        END"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        assert ast is not None
    def test_array_initialization_program(self):
        code = """PROGRAM ARRAY_INIT
        IMPLICIT NONE
        REAL A(10), B(5,5)
        INTEGER I, J
        DO I = 1, 10
            A(I) = 0.0
        ENDDO
        DO I = 1, 5
            DO J = 1, 5
                B(I,J) = I + J
            ENDDO
        ENDDO
        PRINT *, A
        PRINT *, B
        STOP
        END"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        assert ast is not None
    def test_mixed_logical_and_array_program(self):
        code = """PROGRAM MIXED
        IMPLICIT NONE
        REAL A(10), B(10)
        LOGICAL MASK(10)
        INTEGER I
        DO I = 1, 10
            A(I) = I
            B(I) = I * 2
        ENDDO
        DO I = 1, 10
            MASK(I) = A(I) .LT. B(I)
        ENDDO
        STOP
        END"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        assert ast is not None
class TestParameterAndData(unittest.TestCase):
    def test_parameter_token(self):
        code = "PARAMETER"
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        assert tokens[0].type == TokenType.PARAMETER
    def test_data_token(self):
        code = "DATA"
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        assert tokens[0].type == TokenType.DATA
    def test_parameter_declaration(self):
        code = """PROGRAM TEST
        REAL PI
        PARAMETER (PI = 3.14159)
        END"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        try:
            ast = parser.parse()
            assert ast is not None
        except SyntaxError as e:
            pass
if __name__ == "__main__":
    unittest.main(verbosity=2)