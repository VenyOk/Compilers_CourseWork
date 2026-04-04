import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core import Lexer, Parser
from src.semantic import SemanticAnalyzer
from src.llvm_generator import LLVMGenerator
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


def find_clang() -> str | None:
    candidates = [
        os.environ.get("CLANG"),
        shutil.which("clang"),
        r"C:\Program Files\LLVM\bin\clang.exe",
    ]
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    return None


def find_openmp_runtime_dir(clang_path: str | None) -> str | None:
    explicit = os.environ.get("OPENMP_RUNTIME")
    if explicit and os.path.exists(explicit):
        return str(Path(explicit).resolve().parent)
    candidates = []
    if clang_path:
        candidates.append(str(Path(clang_path).resolve().parent))
    candidates.extend([
        str(ROOT / ".llvm" / "clang+llvm-22.1.2-x86_64-pc-windows-msvc" / "bin"),
        r"C:\Program Files\LLVM\bin",
    ])
    for candidate in candidates:
        if os.path.exists(os.path.join(candidate, "libomp.dll")) or os.path.exists(os.path.join(candidate, "libiomp5md.dll")):
            return candidate
    return None


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

    def test_parallel_fill_runtime(self):
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
        self.assertIn("@__kmpc_fork_call", llvm_code)
        self.assertIn("@__kmpc_for_static_init_4", llvm_code)
        self.assertIn("@__kmpc_for_static_fini", llvm_code)
        self.assertNotIn("@fortran_parallel_for_i32", llvm_code)
        output = run_llvm_ir(llvm_code)
        self.assertEqual(output.strip(), "16842752")

    def test_parallel_wavefront_runtime_matches_unoptimized(self):
        source = """      PROGRAM PWFRT
      INTEGER I, J, K
      REAL U(20,20,20), S
      DO I = 1, 20
          DO J = 1, 20
              DO K = 1, 20
                  U(I,J,K) = I + J + K
              END DO
          END DO
      END DO
      DO I = 2, 19
          DO J = 2, 19
              DO K = 2, 19
                  S = U(I-1,J,K) + U(I+1,J,K) + U(I,J-1,K)
                  S = S + U(I,J+1,K) + U(I,J,K-1) + U(I,J,K+1)
                  U(I,J,K) = S / 6.0
              END DO
          END DO
      END DO
      PRINT *, U(10,10,10)
      END"""
        llvm_base = compile_to_llvm(source)
        llvm_opt = compile_to_llvm_optimized(source, level=3)
        base_output = run_llvm_ir(llvm_base)
        opt_output = run_llvm_ir(llvm_opt)
        self.assertEqual(opt_output.strip(), base_output.strip())

    def test_metelitsa_gs2d_runtime_matches_unoptimized(self):
        llvm_base = compile_input_file("bench_metelitsa_gs2d.f", optimized=False)
        llvm_opt = compile_input_file("bench_metelitsa_gs2d.f", optimized=True, level=3)
        base_output = run_llvm_ir(llvm_base)
        opt_output = run_llvm_ir(llvm_opt)
        self.assertEqual(opt_output.strip(), base_output.strip())

    def test_metelitsa_dirichlet_2d_runtime_matches_unoptimized(self):
        llvm_base = compile_input_file("bench_metelitsa_dir2d.f", optimized=False)
        llvm_opt = compile_input_file("bench_metelitsa_dir2d.f", optimized=True, level=3)
        base_output = run_llvm_ir(llvm_base)
        opt_output = run_llvm_ir(llvm_opt)
        self.assertEqual(opt_output.strip(), base_output.strip())

    def test_parallel_ir_runs_via_native_clang_smoke(self):
        clang = find_clang()
        if not clang or os.name != "nt":
            self.skipTest("native clang smoke is only enabled on Windows with clang installed")
        openmp_dir = find_openmp_runtime_dir(clang)
        if not openmp_dir:
            self.skipTest("OpenMP runtime was not found for native clang smoke")
        source = """      PROGRAM PCLANG
      INTEGER I, J, S
      INTEGER A(128,128)
      DO I = 1, 128
          DO J = 1, 128
              A(I,J) = I + J
          END DO
      END DO
      S = 0
      DO I = 1, 128
          DO J = 1, 128
              S = S + A(I,J)
          END DO
      END DO
      PRINT *, S
      END"""
        llvm_code = compile_to_llvm_optimized(source, level=3)
        with tempfile.TemporaryDirectory() as tmpdir:
            ll_path = Path(tmpdir) / "native_parallel.ll"
            exe_path = Path(tmpdir) / "native_parallel.exe"
            ll_path.write_text(llvm_code, encoding="utf-8")
            result = subprocess.run(
                [clang, "-fopenmp", str(ll_path), "-Wl,/STACK:67108864", "-o", str(exe_path)],
                capture_output=True,
                text=True,
                timeout=60,
            )
            if result.returncode != 0:
                raise AssertionError(f"clang failed: {result.stderr}")
            run = subprocess.run(
                [str(exe_path)],
                capture_output=True,
                text=True,
                timeout=60,
                env={**os.environ, "OMP_NUM_THREADS": "4", "PATH": openmp_dir + os.pathsep + os.environ.get("PATH", "")},
            )
            if run.returncode != 0:
                raise AssertionError(f"native executable failed: {run.stderr}")
            self.assertEqual(run.stdout.strip(), "2113536")


if __name__ == "__main__":
    unittest.main()
