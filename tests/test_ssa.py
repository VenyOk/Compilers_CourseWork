import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ssa_generator import SSAGenerator
from src.semantic import SemanticAnalyzer
from src.core import Lexer, Parser


def compile_code_ssa(code: str):
    lexer = Lexer(code)
    tokens = lexer.tokenize()
    if lexer.get_errors():
        raise Exception(f"Lexer errors: {lexer.get_errors()}")
    parser = Parser(tokens)
    ast = parser.parse()
    semantic = SemanticAnalyzer()
    if not semantic.analyze(ast):
        raise Exception(f"Semantic errors: {semantic.get_errors()}")
    ssa_gen = SSAGenerator()
    ssa_instructions = ssa_gen.generate(ast)
    ssa_str = ssa_gen.to_string(ssa_instructions)
    return {"ssa": ssa_str, "ssa_instructions": ssa_instructions}


class TestSSABasic(unittest.TestCase):
    def test_simple_assignment_ssa(self):
        code = """
PROGRAM TEST
    INTEGER X, Y
    X = 1
    Y = 2
    X = X + Y
END
"""
        result = compile_code_ssa(code)
        ssa = result["ssa"]
        instructions = result["ssa_instructions"]
        self.assertIsNotNone(ssa)
        self.assertGreater(len(instructions), 0)
        self.assertIn("X_", ssa)
        self.assertIn("Y_", ssa)

    def test_ssa_generation(self):
        code = """
PROGRAM SSA
    INTEGER X, Y
    X = 1
    Y = 2
    X = X + Y
END
"""
        result = compile_code_ssa(code)
        self.assertIn("alloca", result["ssa"])


class TestSSALoops(unittest.TestCase):
    def test_do_loop_ssa(self):
        code = """
PROGRAM DOLOOP
    INTEGER I, SUM
    SUM = 0
    DO I = 1, 10
        SUM = SUM + I
    ENDDO
    PRINT *, SUM
END
"""
        result = compile_code_ssa(code)
        ssa = result["ssa"]
        instructions = result["ssa_instructions"]
        phi_instrs = [i for i in instructions if "phi" in i]
        self.assertGreater(len(phi_instrs), 0)
        self.assertIn("SUM_", ssa)
        self.assertIn("I_", ssa)

    def test_nested_loops_ssa(self):
        code = """PROGRAM NESTED
        IMPLICIT NONE
        INTEGER I, J, SUM
        SUM = 0
        DO I = 1, 10
            DO J = 1, 5
                SUM = SUM + I * J
            ENDDO
        ENDDO
        PRINT *, SUM
        END"""
        result = compile_code_ssa(code)
        ssa = result["ssa"]
        instructions = result["ssa_instructions"]
        phi_instrs = [i for i in instructions if "phi" in i]
        mul_instrs = [i for i in instructions if " = * " in i]
        add_instrs = [i for i in instructions if " = + " in i]
        self.assertGreater(len(phi_instrs), 0)
        self.assertGreater(len(mul_instrs), 0)
        self.assertGreater(len(add_instrs), 0)
        self.assertIn("SUM_", ssa)
        self.assertIn("I_", ssa)
        self.assertIn("J_", ssa)

    def test_do_while_loop_ssa(self):
        code = """PROGRAM DOWHL
        IMPLICIT NONE
        INTEGER I
        I = 0
        DO WHILE (I .LT. 10)
            I = I + 1
        ENDDO
        PRINT *, I
        END"""
        result = compile_code_ssa(code)
        ssa = result["ssa"]
        self.assertIn("I_", ssa)
        self.assertIn("do while", ssa)


class TestSSAControlFlow(unittest.TestCase):
    def test_if_statement_ssa(self):
        code = """
PROGRAM IFTEST
    INTEGER X, Y
    X = 10
    IF (X .GT. 5) THEN
        Y = 1
    ELSE
        Y = 0
    ENDIF
END
"""
        result = compile_code_ssa(code)
        ssa = result["ssa"]
        instructions = result["ssa_instructions"]
        branches = [i for i in instructions if i.startswith("if ") or i == "else" or i == "end if"]
        self.assertIn("Y_", ssa)
        self.assertGreater(len(branches), 0)

    def test_complex_if_nested_ssa(self):
        code = """PROGRAM NESTIF
        IMPLICIT NONE
        INTEGER X, Y, RESULT
        X = 10
        Y = 20
        IF (X .GT. 0) THEN
            IF (Y .GT. 10) THEN
                RESULT = X + Y
            ELSE
                RESULT = X - Y
            ENDIF
        ELSE
            RESULT = 0
        ENDIF
        PRINT *, RESULT
        END"""
        result = compile_code_ssa(code)
        instructions = result["ssa_instructions"]
        branches = [i for i in instructions if i.startswith("if ") or i == "else" or i == "end if"]
        result_versions = [i for i in instructions if "RESULT_" in i]
        self.assertGreater(len(branches), 0)
        self.assertGreater(len(result_versions), 1)


class TestSSAExpressions(unittest.TestCase):
    def test_complex_expressions_ssa(self):
        code = """PROGRAM CMPEX
        IMPLICIT NONE
        INTEGER A, B, C, D, RESULT
        LOGICAL FLAG
        A = 10
        B = 20
        C = 30
        D = 5
        RESULT = (A + B) * C / D - A * 2
        FLAG = (A .GT. B) .AND. (C .LT. D)
        IF (FLAG .OR. (RESULT .GT. 0)) THEN
            RESULT = RESULT + 100
        ENDIF
        PRINT *, RESULT, FLAG
        END"""
        result = compile_code_ssa(code)
        instructions = result["ssa_instructions"]
        mul_instrs = [i for i in instructions if " = * " in i]
        div_instrs = [i for i in instructions if " = / " in i]
        add_instrs = [i for i in instructions if " = + " in i]
        sub_instrs = [i for i in instructions if " = - " in i]
        self.assertGreater(len(mul_instrs), 0)
        self.assertGreater(len(div_instrs), 0)
        self.assertGreater(len(add_instrs), 1)
        self.assertGreater(len(sub_instrs), 0)


class TestSSAArrays(unittest.TestCase):
    def test_array_operations_ssa(self):
        code = """PROGRAM ARROPS
        IMPLICIT NONE
        INTEGER A(10), B(10), C(10)
        INTEGER I
        DO I = 1, 10
            A(I) = I * 2
            B(I) = I * 3
            C(I) = A(I) + B(I)
        ENDDO
        PRINT *, C(5)
        END"""
        result = compile_code_ssa(code)
        ssa = result["ssa"]
        self.assertIn("A = alloca INTEGER", ssa)
        self.assertIn("B = alloca INTEGER", ssa)
        self.assertIn("C = alloca INTEGER", ssa)
        self.assertIn("I_", ssa)
        self.assertIn("load C_2 5", ssa)


class TestSSAFunctions(unittest.TestCase):
    def test_function_calls_ssa(self):
        code = """PROGRAM FNCALL
        IMPLICIT NONE
        REAL X, Y, Z
        INTEGER I
        X = 2.5
        Y = SQRT(X)
        Z = SIN(X) * COS(X) + EXP(X)
        I = INT(Z)
        PRINT *, Y, Z, I
        END"""
        result = compile_code_ssa(code)
        instructions = result["ssa_instructions"]
        call_instrs = [i for i in instructions if " = call " in i]
        self.assertGreaterEqual(len(call_instrs), 5)


if __name__ == "__main__":
    unittest.main()
