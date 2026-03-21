import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core import Lexer, Parser
from src.semantic import SemanticAnalyzer
from src.llvm_generator import LLVMGenerator


ROOT = Path(__file__).resolve().parent.parent
STRASSEN_PATH = ROOT / "inputs" / "test_strassen.f"


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


def run_llvm_ir(ir_code: str, timeout: int = 20) -> str:
    with tempfile.TemporaryDirectory() as tmpdir:
        ll_path = Path(tmpdir) / "strass.ll"
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


def replace_n(source: str, n: int) -> str:
    return source.replace("N = 8", f"N = {n}", 1)


def gen_matrix(n: int, seed: int):
    mat = []
    for i in range(1, n + 1):
        row = []
        for j in range(1, n + 1):
            val = (i * 17 + j * 23 + seed) * 31
            row.append(float(abs(val) % 10))
        mat.append(row)
    return mat


def matmul(a, b):
    n = len(a)
    c = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for k in range(n):
            aik = a[i][k]
            for j in range(n):
                c[i][j] += aik * b[k][j]
    return c


def parse_result_matrix(output: str, n: int):
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    start = lines.index("Result C = A * B:") + 1
    vals = [float(lines[start + idx]) for idx in range(n * n)]
    return [vals[i * n:(i + 1) * n] for i in range(n)]


def parse_summary(output: str):
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    n_idx = lines.index("N:")
    s_idx = lines.index("Checksum:")
    c_idx = lines.index("Corners:")
    return (
        int(float(lines[n_idx + 1])),
        float(lines[s_idx + 1]),
        float(lines[c_idx + 1]),
        float(lines[c_idx + 2]),
    )


class TestStrassenRuntime(unittest.TestCase):
    def test_small_strassen_path_matches_exact_matrix(self):
        source = replace_n(STRASSEN_PATH.read_text(encoding="utf-8"), 4)
        llvm_code = compile_to_llvm(source)
        output = run_llvm_ir(llvm_code, timeout=20)
        actual = parse_result_matrix(output, 4)
        expected = matmul(gen_matrix(4, 12345), gen_matrix(4, 13345))
        for i in range(4):
            for j in range(4):
                self.assertAlmostEqual(actual[i][j], expected[i][j], places=4)

    def test_blocked_large_path_matches_checksum_and_corners(self):
        source = replace_n(STRASSEN_PATH.read_text(encoding="utf-8"), 64)
        llvm_code = compile_to_llvm(source)
        output = run_llvm_ir(llvm_code, timeout=30)
        n_val, checksum, c11, cnn = parse_summary(output)
        expected = matmul(gen_matrix(64, 12345), gen_matrix(64, 13345))
        expected_sum = sum(sum(row) for row in expected)
        self.assertEqual(n_val, 64)
        self.assertAlmostEqual(checksum, expected_sum, places=3)
        self.assertAlmostEqual(c11, expected[0][0], places=4)
        self.assertAlmostEqual(cnn, expected[-1][-1], places=4)

    def test_scaled_1024_compiles_end_to_end(self):
        source = replace_n(STRASSEN_PATH.read_text(encoding="utf-8"), 1024)
        llvm_code = compile_to_llvm(source)
        self.assertIn("define i32 @main()", llvm_code)
        self.assertIn("define void @STRASN", llvm_code)
        self.assertIn("call i8* @malloc(i64 8388608)", llvm_code)
        self.assertIn("bitcast i8* ", llvm_code)


if __name__ == "__main__":
    unittest.main()
