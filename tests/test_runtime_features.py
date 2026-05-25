import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.frontend.lexer import Lexer
from src.frontend.parser import Parser
from src.semantic.analyzer import SemanticAnalyzer
from src.ir.llvm import LLVMGenerator
from src.optimizations.pipeline import OptimizationPipeline


ROOT = Path(__file__).resolve().parent.parent


def compile_to_llvm(source: str) -> str:
    lexer = Lexer(source)
    tokens = lexer.tokenize()
    if lexer.get_errors():
        raise AssertionError(f"Lexer errors: {lexer.get_errors()}")
    parser = Parser(tokens)
    ast = parser.parse()
    semantic = SemanticAnalyzer()
    if not semantic.analyze(ast):
        raise AssertionError(f"Semantic errors: {semantic.get_errors()}")
    llvm_gen = LLVMGenerator()
    return llvm_gen.generate(ast)


def compile_to_llvm_optimized(source: str, level: int = 3) -> str:
    lexer = Lexer(source)
    tokens = lexer.tokenize()
    if lexer.get_errors():
        raise AssertionError(f"Lexer errors: {lexer.get_errors()}")
    parser = Parser(tokens)
    ast = parser.parse()
    semantic = SemanticAnalyzer()
    if not semantic.analyze(ast):
        raise AssertionError(f"Semantic errors: {semantic.get_errors()}")
    pipeline = OptimizationPipeline(level=level)
    ast = pipeline.run(ast)
    semantic_after = SemanticAnalyzer()
    if not semantic_after.analyze(ast):
        raise AssertionError(f"Semantic errors after optimization: {semantic_after.get_errors()}")
    llvm_gen = LLVMGenerator()
    return llvm_gen.generate(ast)


def compile_input_file(name: str, optimized: bool = False, level: int = 3) -> str:
    source = (ROOT / "inputs" / name).read_text(encoding="utf-8")
    if optimized:
        return compile_to_llvm_optimized(source, level=level)
    return compile_to_llvm(source)


def run_llvm_ir(ir_code: str, timeout: int = 20) -> str:
    with tempfile.TemporaryDirectory() as tmpdir:
        ll_path = Path(tmpdir) / "runtime.ll"
        ll_path.write_text(ir_code, encoding="utf-8")
        result = subprocess.run(
            [sys.executable, "test_llvmlite_run.py", str(ll_path)],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            raise AssertionError(
                f"Execution failed: code={result.returncode}, stderr={result.stderr}"
            )
        return result.stdout


def assert_outputs_close(testcase: unittest.TestCase, left: str, right: str, places: int = 4) -> None:
    left_lines = [line.strip() for line in left.splitlines() if line.strip()]
    right_lines = [line.strip() for line in right.splitlines() if line.strip()]
    testcase.assertEqual(len(left_lines), len(right_lines))
    for left_value, right_value in zip(left_lines, right_lines):
        try:
            testcase.assertAlmostEqual(float(left_value), float(right_value), places=places)
        except ValueError:
            testcase.assertEqual(left_value, right_value)


class TestRuntimeFeatures(unittest.TestCase):
    def test_parameter_based_lower_bounds_runtime(self):
        source = """      PROGRAM LOWBND
      PARAMETER (LOW = 5, HIGH = LOW + 2)
      INTEGER A(LOW:HIGH)
      A(LOW) = 10
      A(LOW + 1) = 15
      A(HIGH) = 20
      PRINT *, A(LOW)
      PRINT *, A(LOW + 1)
      PRINT *, A(HIGH)
      END"""
        llvm_code = compile_to_llvm(source)
        output = run_llvm_ir(llvm_code)
        self.assertEqual(output.strip().splitlines(), ["10", "15", "20"])

    def test_common_column_major_alias_runtime(self):
        source = """      PROGRAM CMNAL
      INTEGER A(2,3), I, J
      COMMON /BLK/ A
      DO J = 1, 3
          DO I = 1, 2
              A(I,J) = 0
          END DO
      END DO
      A(2,1) = 7
      A(1,2) = 9
      CALL SHOW()
      END

      SUBROUTINE SHOW()
      INTEGER A(6)
      COMMON /BLK/ A
      PRINT *, A(2)
      PRINT *, A(3)
      RETURN
      END"""
        llvm_code = compile_to_llvm(source)
        output = run_llvm_ir(llvm_code)
        self.assertEqual(output.strip().splitlines(), ["7", "9"])

    def test_exit_runtime(self):
        source = """      PROGRAM EXITRT
      INTEGER I, SUM
      SUM = 0
      DO I = 1, 10
          IF (I .GT. 3) EXIT
          SUM = SUM + I
      END DO
      PRINT *, SUM
      END"""
        llvm_code = compile_to_llvm(source)
        output = run_llvm_ir(llvm_code)
        self.assertEqual(output.strip(), "6")

    def test_o3_fill_runtime_stays_sequential(self):
        source = """      PROGRAM PARRT
      INTEGER I, J, S
      INTEGER A(256,256)
      DO I = 1, 256
          DO J = 1, 256
              A(I,J) = I + J
          END DO
      END DO
      S = 0
      DO I = 1, 256
          DO J = 1, 256
              S = S + A(I,J)
          END DO
      END DO
        PRINT *, S
        END"""
        llvm_code = compile_to_llvm_optimized(source, level=3)
        output = run_llvm_ir(llvm_code)
        self.assertEqual(output.strip(), "16842752")

    def test_o3_matches_o0_for_loop_kernels(self):
        cases = {
            "matmul": """      PROGRAM KMM
      IMPLICIT NONE
      INTEGER I, J, K
      REAL A(8,8), B(8,8), C(8,8), S
      DO I = 1, 8
          DO J = 1, 8
              A(I,J) = FLOAT(I + 2 * J)
              B(I,J) = FLOAT(3 * I - J)
              C(I,J) = 0.0
          END DO
      END DO
      DO I = 1, 8
          DO J = 1, 8
              DO K = 1, 8
                  C(I,J) = C(I,J) + A(I,K) * B(K,J)
              END DO
          END DO
      END DO
      S = 0.0
      DO I = 1, 8
          DO J = 1, 8
              S = S + C(I,J)
          END DO
      END DO
      PRINT *, C(1,1)
      PRINT *, C(8,8)
      PRINT *, S
      END""",
            "jacobi2d": """      PROGRAM KJAC
      IMPLICIT NONE
      INTEGER T, I, J
      REAL U(12,12), V(12,12), S
      DO I = 1, 12
          DO J = 1, 12
              U(I,J) = FLOAT(I + J)
              V(I,J) = 0.0
          END DO
      END DO
      DO T = 1, 3
          DO I = 2, 11
              DO J = 2, 11
                  V(I,J) = 0.25*(U(I-1,J)+U(I+1,J)+U(I,J-1)+U(I,J+1))
              END DO
          END DO
          DO I = 2, 11
              DO J = 2, 11
                  U(I,J) = V(I,J)
              END DO
          END DO
      END DO
      S = 0.0
      DO I = 2, 11
          DO J = 2, 11
              S = S + U(I,J)
          END DO
      END DO
      PRINT *, U(2,2)
      PRINT *, U(11,11)
      PRINT *, S
      END""",
            "gs2d": """      PROGRAM KGS2
      IMPLICIT NONE
      INTEGER T, I, J
      REAL U(12,12), S
      DO I = 1, 12
          DO J = 1, 12
              U(I,J) = FLOAT(I * 2 + J)
          END DO
      END DO
      DO T = 1, 3
          DO I = 2, 11
              DO J = 2, 11
                  U(I,J) = 0.25*(U(I-1,J)+U(I+1,J)+U(I,J-1)+U(I,J+1))
              END DO
          END DO
      END DO
      S = 0.0
      DO I = 2, 11
          DO J = 2, 11
              S = S + U(I,J)
          END DO
      END DO
      PRINT *, U(2,2)
      PRINT *, U(11,11)
      PRINT *, S
      END""",
            "dirichlet2d": """      PROGRAM KDIR
      IMPLICIT NONE
      INTEGER T, I, J
      REAL U(10,10), A(10,10), B(10,10)
      REAL C(10,10), D(10,10), Y0(10,10), S
      DO I = 1, 10
          DO J = 1, 10
              U(I,J) = FLOAT(I + J)
              A(I,J) = 0.10
              B(I,J) = 0.20
              C(I,J) = 0.30
              D(I,J) = 0.40
              Y0(I,J) = FLOAT(I - J) * 0.01
          END DO
      END DO
      DO T = 1, 2
          DO I = 2, 9
              DO J = 2, 9
                  S = A(I,J)*U(I-1,J) + B(I,J)*U(I+1,J)
                  S = S + C(I,J)*U(I,J-1) + D(I,J)*U(I,J+1)
                  U(I,J) = S + Y0(I,J)
              END DO
          END DO
      END DO
      S = 0.0
      DO I = 2, 9
          DO J = 2, 9
              S = S + U(I,J)
          END DO
      END DO
      PRINT *, U(2,2)
      PRINT *, U(9,9)
      PRINT *, S
      END""",
            "gs3d": """      PROGRAM KGS3
      IMPLICIT NONE
      INTEGER T, I, J, K
      REAL U(8,8,8), S
      DO I = 1, 8
          DO J = 1, 8
              DO K = 1, 8
                  U(I,J,K) = FLOAT(I + J + K)
              END DO
          END DO
      END DO
      DO T = 1, 2
          DO I = 2, 7
              DO J = 2, 7
                  DO K = 2, 7
                      S = U(I-1,J,K) + U(I+1,J,K) + U(I,J-1,K)
                      S = S + U(I,J+1,K) + U(I,J,K-1) + U(I,J,K+1)
                      U(I,J,K) = S / 6.0
                  END DO
              END DO
          END DO
      END DO
      S = 0.0
      DO I = 2, 7
          DO J = 2, 7
              DO K = 2, 7
                  S = S + U(I,J,K)
              END DO
          END DO
      END DO
      PRINT *, U(2,2,2)
      PRINT *, U(7,7,7)
      PRINT *, S
      END""",
        }
        for name, source in cases.items():
            with self.subTest(name=name):
                out0 = run_llvm_ir(compile_to_llvm(source), timeout=30)
                out3 = run_llvm_ir(compile_to_llvm_optimized(source, level=3), timeout=30)
                assert_outputs_close(self, out0, out3, places=4)


if __name__ == "__main__":
    unittest.main()
