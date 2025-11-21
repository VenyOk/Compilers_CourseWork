import pytest
from src.core import Lexer, Parser, TokenType, Declaration
class TestExtendedTypes:
    def test_real_double_precision_literal(self):
        test_cases = [
            ("1.5D10", 1.5e10),
            ("1.5d10", 1.5e10),
            ("1.0D-5", 1.0e-5),
            ("1.0d-5", 1.0e-5),
            ("3.141592653589793D0", 3.141592653589793),
            ("2.718281828459045D0", 2.718281828459045),
            ("1.5E10", 1.5e10),
            ("1.5e10", 1.5e10),
        ]
        for literal_str, expected_value in test_cases:
            lexer = Lexer(literal_str)
            tokens = lexer.tokenize()
            assert len(tokens) >= 1
            assert tokens[0].type == TokenType.REAL_LIT
            assert abs(tokens[0].value - expected_value) < 1e-10
    def test_real_star_8_declaration(self):
        code = """PROGRAM TEST
        REAL*8 X, Y, Z
        END"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        assert ast is not None
        declarations = [s for s in ast.statements if isinstance(s, Declaration)]
        if not declarations:
            declarations = ast.declarations
        assert len(declarations) > 0
        assert declarations[0].type == "REAL*8"
        assert len(declarations[0].names) == 3
        assert declarations[0].names[0][0] == "X"
    def test_complex_declaration(self):
        code = """PROGRAM TEST
        COMPLEX C1, C2
        END"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        declarations = ast.declarations
        assert len(declarations) > 0
        assert declarations[0].type == "COMPLEX"
        assert len(declarations[0].names) == 2
        assert declarations[0].names[0][0] == "C1"
    def test_complex_star_16_declaration(self):
        code = """PROGRAM TEST
        COMPLEX*16 C1, C2, C3
        END"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        declarations = ast.declarations
        assert len(declarations) > 0
        assert declarations[0].type == "COMPLEX*16"
        assert len(declarations[0].names) == 3
    def test_character_star_n_declaration(self):
        test_cases = [
            ("CHARACTER*10 STR1", "CHARACTER*10", ["STR1"]),
            ("CHARACTER*20 A, B, C", "CHARACTER*20", ["A", "B", "C"]),
            ("CHARACTER*15 NAME", "CHARACTER*15", ["NAME"]),
            ("CHARACTER*30 FULLNAME", "CHARACTER*30", ["FULLNAME"]),
        ]
        for decl_str, expected_type, expected_names in test_cases:
            code = f"""PROGRAM TEST
            {decl_str}
            END"""
            lexer = Lexer(code)
            tokens = lexer.tokenize()
            parser = Parser(tokens)
            ast = parser.parse()
            declarations = ast.declarations
            assert len(declarations) > 0
            assert declarations[0].type == expected_type
            actual_names = [name[0] for name in declarations[0].names]
            assert actual_names == expected_names
    def test_mixed_type_declarations(self):
        code = """PROGRAM MIXED
        INTEGER I, J
        REAL X, Y
        REAL*8 D1, D2
        COMPLEX C1, C2
        CHARACTER*10 STR
        LOGICAL FLAG
        END"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        declarations = ast.declarations
        expected_types = [
            ("INTEGER", ["I", "J"]),
            ("REAL", ["X", "Y"]),
            ("REAL*8", ["D1", "D2"]),
            ("COMPLEX", ["C1", "C2"]),
            ("CHARACTER*10", ["STR"]),
            ("LOGICAL", ["FLAG"]),
        ]
        assert len(declarations) == len(expected_types)
        for decl, (expected_type, expected_names) in zip(declarations, expected_types):
            assert decl.type == expected_type
            actual_names = [name[0] for name in decl.names]
            assert actual_names == expected_names
    def test_arrays_with_extended_types(self):
        code = """PROGRAM TEST
        REAL*8 D(10), E(5,5)
        COMPLEX C(10)
        CHARACTER*15 NAMES(20)
        END"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        declarations = ast.declarations
        assert declarations[0].type == "REAL*8"
        d_decl = declarations[0].names
        assert d_decl[0][0] == "D"
        assert d_decl[0][1] == [(1, 10)]
        assert d_decl[1][0] == "E"
        assert d_decl[1][1] == [(1, 5), (1, 5)]
        assert declarations[1].type == "COMPLEX"
        c_decl = declarations[1].names
        assert c_decl[0][0] == "C"
        assert c_decl[0][1] == [(1, 10)]
        assert declarations[2].type == "CHARACTER*15"
        names_decl = declarations[2].names
        assert names_decl[0][0] == "NAMES"
        assert names_decl[0][1] == [(1, 20)]
class TestNumericFormats:
    def test_scientific_notation_e_format(self):
        code = """PROGRAM TEST
        REAL X, Y
        X = 1.0E5
        Y = 2.5E-3
        END"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        real_lits = [t for t in tokens if t.type == TokenType.REAL_LIT]
        assert len(real_lits) == 2
        assert abs(real_lits[0].value - 1.0e5) < 1e-5
        assert abs(real_lits[1].value - 2.5e-3) < 1e-8
    def test_double_precision_d_format(self):
        code = """PROGRAM TEST
        REAL*8 X, Y
        X = 1.0D5
        Y = 2.5D-3
        END"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        real_lits = [t for t in tokens if t.type == TokenType.REAL_LIT]
        assert len(real_lits) == 2
        assert abs(real_lits[0].value - 1.0e5) < 1e-5
        assert abs(real_lits[1].value - 2.5e-3) < 1e-8
    def test_mixed_numeric_formats(self):
        code = """PROGRAM TEST
        REAL X
        REAL*8 D
        X = 1.5E10
        D = 1.5D10
        END"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        real_lits = [t for t in tokens if t.type == TokenType.REAL_LIT]
        assert len(real_lits) == 2
        assert abs(real_lits[0].value - 1.5e10) < 1e8
        assert abs(real_lits[1].value - 1.5e10) < 1e8
class TestCompletePrograms:
    def test_parse_all_type_examples(self):
        examples = [
            ("PROGRAM TEST\nREAL*8 X, Y\nX = 1.5D10\nY = 2.5D-3\nEND", "types_real_double"),
            ("PROGRAM TEST\nCOMPLEX C1, C2\nC1 = (1.0, 2.0)\nC2 = (3.0, 4.0)\nEND", "types_complex"),
            ("PROGRAM TEST\nCOMPLEX*16 C1, C2\nC1 = (1.0D0, 2.0D0)\nC2 = (3.0D0, 4.0D0)\nEND", "types_complex_double"),
            ("PROGRAM TEST\nCHARACTER*10 STR1\nCHARACTER*20 STR2\nSTR1 = 'HELLO'\nSTR2 = 'WORLD'\nEND", "types_character"),
            ("PROGRAM TEST\nINTEGER I\nREAL X\nREAL*8 D\nCOMPLEX C\nCHARACTER*10 STR\nLOGICAL L\nI = 5\nX = 3.14\nD = 1.5D10\nSTR = 'TEST'\nL = .TRUE.\nEND", "types_mixed"),
        ]
        for code, name in examples:
            lexer = Lexer(code)
            tokens = lexer.tokenize()
            parser = Parser(tokens)
            ast = parser.parse()
            assert ast is not None
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
