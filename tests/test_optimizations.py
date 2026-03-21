import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core import Lexer, Parser
from src.semantic import SemanticAnalyzer
from src.llvm_generator import LLVMGenerator
from src.optimizations.pipeline import OptimizationPipeline


def parse_program(code: str):
    lexer = Lexer(code)
    tokens = lexer.tokenize()
    if lexer.get_errors():
        raise AssertionError(f"lexer errors: {lexer.get_errors()}")
    return Parser(tokens).parse()


def analyze_ok(ast):
    analyzer = SemanticAnalyzer()
    ok = analyzer.analyze(ast)
    return ok, analyzer.get_errors(), analyzer.get_warnings()


def optimize_program(code: str, level: int = 3):
    ast = parse_program(code)
    ok, errors, _ = analyze_ok(ast)
    if not ok:
        raise AssertionError(f"semantic errors before optimization: {errors}")
    pipeline = OptimizationPipeline(level=level)
    optimized = pipeline.run(ast)
    return optimized, pipeline.stats


class TestOptimizationPipeline(unittest.TestCase):
    def test_generated_temporaries_are_declared_under_implicit_none(self):
        code = """
PROGRAM TEMPD
    IMPLICIT NONE
    INTEGER I
    REAL S
    S = 0.0
    DO I = 1, 32
        S = S + SQRT(3.14159265 * 3.14159265 + 1.0)
        S = S + SQRT(3.14159265 * 3.14159265 + 1.0)
    ENDDO
END
"""
        optimized, stats = optimize_program(code, level=3)
        ok, errors, _ = analyze_ok(optimized)
        self.assertTrue(ok, errors)
        self.assertGreater(stats["LoopInvariantCodeMotion"]["hoisted"], 0)
        self.assertGreater(stats["CommonSubexpressionElimination"]["cse_vars"], 0)
        declared = {
            name
            for decl in optimized.declarations
            if hasattr(decl, "names")
            for name, _ in decl.names
        }
        self.assertTrue(any(name.startswith("licm_tmp_") for name in declared))
        self.assertTrue(any(name.startswith("cse_tmp_") for name in declared))
        llvm_code = LLVMGenerator().generate(optimized)
        self.assertIn("%licm_tmp_", llvm_code)
        self.assertIn("%cse_tmp_", llvm_code)

    def test_wavefront_is_applied_to_profitable_two_dimensional_stencil(self):
        code = """
PROGRAM GS2D
    IMPLICIT NONE
    INTEGER I, J
    REAL U(12,12)
    DO I = 2, 11
        DO J = 2, 11
            U(I,J) = 0.125*(U(I-1,J)+U(I+1,J)+U(I,J-1)+U(I,J+1)
     &          +U(I-1,J-1)+U(I-1,J+1)+U(I+1,J-1)+U(I+1,J+1))
        ENDDO
    ENDDO
END
"""
        optimized, stats = optimize_program(code, level=3)
        ok, errors, _ = analyze_ok(optimized)
        self.assertTrue(ok, errors)
        self.assertEqual(stats["LoopSkewing"]["skewed"], 1)
        self.assertEqual(stats["LoopTiling"]["tiled"], 1)
        self.assertEqual(stats["LoopWavefront"]["wavefronted"], 1)
        llvm_code = LLVMGenerator().generate(optimized)
        self.assertIn("%wf_h0 = alloca i32", llvm_code)
        self.assertIn("%tile_I = alloca i32", llvm_code)
        self.assertIn("%tile_skew_J = alloca i32", llvm_code)

    def test_wavefront_is_not_applied_to_two_dimensional_five_point_stencil_when_not_profitable(self):
        code = """
PROGRAM GS2SM
    IMPLICIT NONE
    INTEGER I, J
    REAL U(128,128)
    DO I = 2, 127
        DO J = 2, 127
            U(I,J) = 0.25*(U(I-1,J)+U(I+1,J)+U(I,J-1)+U(I,J+1))
        ENDDO
    ENDDO
END
"""
        _, stats = optimize_program(code, level=3)
        self.assertEqual(stats["LoopWavefront"]["wavefronted"], 0)

    def test_parallelization_is_applied_to_three_dimensional_wavefront_band(self):
        code = """
PROGRAM PGS3D
    IMPLICIT NONE
    INTEGER I, J, K
    REAL U(40,40,40), S
    DO I = 2, 39
        DO J = 2, 39
            DO K = 2, 39
                S = U(I-1,J,K) + U(I+1,J,K) + U(I,J-1,K)
                S = S + U(I,J+1,K) + U(I,J,K-1) + U(I,J,K+1)
                U(I,J,K) = S / 6.0
            ENDDO
        ENDDO
    ENDDO
END
"""
        optimized, stats = optimize_program(code, level=3)
        ok, errors, _ = analyze_ok(optimized)
        self.assertTrue(ok, errors)
        self.assertGreaterEqual(stats["LoopParallelization"]["parallelized"], 1)
        llvm_code = LLVMGenerator().generate(optimized)
        self.assertIn("@fortran_parallel_for_i32", llvm_code)
        self.assertIn("@parallel_worker_", llvm_code)

    def test_parallelization_is_not_applied_to_two_dimensional_wavefront_band_when_not_profitable(self):
        code = """
PROGRAM PGS2D
    IMPLICIT NONE
    INTEGER I, J
    REAL U(96,96)
    DO I = 2, 95
        DO J = 2, 95
            U(I,J) = 0.25*(U(I-1,J)+U(I+1,J)+U(I,J-1)+U(I,J+1))
        ENDDO
    ENDDO
END
"""
        _, stats = optimize_program(code, level=3)
        self.assertEqual(stats["LoopParallelization"]["parallelized"], 0)

    def test_wavefront_is_applied_to_three_dimensional_stencil(self):
        code = """
PROGRAM ST3D
    IMPLICIT NONE
    INTEGER I, J, K
    REAL U(8,8,8)
    DO I = 2, 7
        DO J = 2, 7
            DO K = 2, 7
                U(I,J,K) = U(I-1,J,K) + U(I,J-1,K) + U(I,J,K-1)
            ENDDO
        ENDDO
    ENDDO
END
"""
        optimized, stats = optimize_program(code, level=3)
        ok, errors, _ = analyze_ok(optimized)
        self.assertTrue(ok, errors)
        self.assertEqual(stats["LoopSkewing"]["skewed"], 1)
        self.assertEqual(stats["LoopTiling"]["tiled"], 1)
        self.assertEqual(stats["LoopWavefront"]["wavefronted"], 1)
        llvm_code = LLVMGenerator().generate(optimized)
        self.assertIn("%tile_skew_J = alloca i32", llvm_code)
        self.assertIn("%tile_skew_K = alloca i32", llvm_code)
        self.assertIn("%wf_h0 = alloca i32", llvm_code)

    def test_wavefront_is_applied_to_four_dimensional_stencil(self):
        code = """
PROGRAM ST4D
    IMPLICIT NONE
    INTEGER I, J, K, L
    REAL U(6,6,6,6), S
    DO I = 2, 5
        DO J = 2, 5
            DO K = 2, 5
                DO L = 2, 5
                    S = U(I-1,J,K,L) + U(I,J-1,K,L)
                    S = S + U(I,J,K-1,L) + U(I,J,K,L-1)
                    U(I,J,K,L) = S
                ENDDO
            ENDDO
        ENDDO
    ENDDO
END
"""
        optimized, stats = optimize_program(code, level=3)
        ok, errors, _ = analyze_ok(optimized)
        self.assertTrue(ok, errors)
        self.assertEqual(stats["LoopSkewing"]["skewed"], 1)
        self.assertEqual(stats["LoopTiling"]["tiled"], 1)
        self.assertEqual(stats["LoopWavefront"]["wavefronted"], 1)
        llvm_code = LLVMGenerator().generate(optimized)
        self.assertIn("%tile_skew_L = alloca i32", llvm_code)
        self.assertIn("%wf_h0 = alloca i32", llvm_code)

    def test_generated_loop_variables_are_declared_inside_subroutine_for_nd_case(self):
        code = """
PROGRAM WRAP
    IMPLICIT NONE
    CALL STEP3D()
END
SUBROUTINE STEP3D()
    IMPLICIT NONE
    INTEGER I, J, K
    REAL U(8,8,8)
    DO I = 2, 7
        DO J = 2, 7
            DO K = 2, 7
                U(I,J,K) = U(I-1,J,K) + U(I,J-1,K) + U(I,J,K-1)
            ENDDO
        ENDDO
    ENDDO
END
"""
        optimized, _ = optimize_program(code, level=3)
        ok, errors, _ = analyze_ok(optimized)
        self.assertTrue(ok, errors)
        self.assertEqual(len(optimized.subroutines), 1)
        declared = {
            name
            for decl in optimized.subroutines[0].declarations
            if hasattr(decl, "names")
            for name, _ in decl.names
        }
        self.assertTrue(any(name.startswith("tile_") for name in declared))
        self.assertTrue(any(name.startswith("skew_") for name in declared))
        self.assertTrue(any(name.startswith("wf_") for name in declared))

    def test_wavefront_is_not_applied_to_regular_matmul(self):
        code = """
PROGRAM MATMUL
    IMPLICIT NONE
    INTEGER N, I, J, K
    REAL A(4,4), B(4,4), C(4,4)
    N = 4
    DO I = 1, N
        DO J = 1, N
            C(I,J) = 0.0
        ENDDO
    ENDDO
    DO I = 1, N
        DO J = 1, N
            DO K = 1, N
                C(I,J) = C(I,J) + A(I,K) * B(K,J)
            ENDDO
        ENDDO
    ENDDO
END
"""
        optimized, stats = optimize_program(code, level=3)
        ok, errors, _ = analyze_ok(optimized)
        self.assertTrue(ok, errors)
        self.assertEqual(stats["LoopWavefront"]["wavefronted"], 0)
        llvm_code = LLVMGenerator().generate(optimized)
        self.assertIn("define i32 @main()", llvm_code)

    def test_parallelization_is_applied_to_independent_array_fill(self):
        code = """
PROGRAM PFILL
    IMPLICIT NONE
    INTEGER I, J, S
    INTEGER A(256,256)
    DO I = 1, 256
        DO J = 1, 256
            A(I,J) = I + J
        ENDDO
    ENDDO
    S = 0
    DO I = 1, 256
        DO J = 1, 256
            S = S + A(I,J)
        ENDDO
    ENDDO
END
"""
        optimized, stats = optimize_program(code, level=3)
        ok, errors, _ = analyze_ok(optimized)
        self.assertTrue(ok, errors)
        self.assertGreaterEqual(stats["LoopParallelization"]["parallelized"], 1)
        llvm_code = LLVMGenerator().generate(optimized)
        self.assertIn("call void @fortran_parallel_for_i32", llvm_code)

    def test_outer_time_loop_with_inner_stateful_nests_is_not_parallelized(self):
        code = """
PROGRAM STTIM
    IMPLICIT NONE
    INTEGER I, J, T
    REAL U(64,64), V(64,64)
    DO I = 1, 64
        DO J = 1, 64
            U(I,J) = 0.0
            V(I,J) = 0.0
        ENDDO
    ENDDO
    DO T = 1, 4
        DO I = 2, 63
            DO J = 2, 63
                V(I,J) = 0.25*(U(I-1,J)+U(I+1,J)+U(I,J-1)+U(I,J+1))
            ENDDO
        ENDDO
        DO I = 2, 63
            DO J = 2, 63
                U(I,J) = V(I,J)
            ENDDO
        ENDDO
    ENDDO
END
"""
        _, stats = optimize_program(code, level=3)
        self.assertEqual(stats["LoopParallelization"]["parallelized"], 0)

    def test_generated_temporaries_are_declared_inside_function(self):
        code = """
PROGRAM WRAPF
    IMPLICIT NONE
    REAL R
    R = ACC()
END
REAL FUNCTION ACC()
    IMPLICIT NONE
    INTEGER I
    REAL S
    S = 0.0
    DO I = 1, 32
        S = S + SQRT(3.14159265 * 3.14159265 + 1.0)
        S = S + SQRT(3.14159265 * 3.14159265 + 1.0)
    ENDDO
    ACC = S
    RETURN
END
"""
        optimized, stats = optimize_program(code, level=3)
        ok, errors, _ = analyze_ok(optimized)
        self.assertTrue(ok, errors)
        self.assertEqual(len(optimized.functions), 1)
        declared = {
            name
            for decl in optimized.functions[0].declarations
            if hasattr(decl, "names")
            for name, _ in decl.names
        }
        self.assertTrue(any(name.startswith("licm_tmp_") for name in declared))
        self.assertTrue(any(name.startswith("cse_tmp_") for name in declared))
        self.assertGreater(stats["LoopInvariantCodeMotion"]["hoisted"], 0)
        self.assertGreater(stats["CommonSubexpressionElimination"]["cse_vars"], 0)

    def test_small_three_dimensional_nest_is_skipped_by_profitability_gates(self):
        code = """
PROGRAM SM3D
    IMPLICIT NONE
    INTEGER I, J, K
    REAL A(2,2,2)
    DO I = 1, 2
        DO J = 1, 2
            DO K = 1, 2
                A(I,J,K) = I + J + K
            ENDDO
        ENDDO
    ENDDO
END
"""
        _, stats = optimize_program(code, level=3)
        self.assertEqual(stats["LoopInterchange"]["interchanged"], 0)
        self.assertEqual(stats["LoopTiling"]["tiled"], 0)
        self.assertEqual(stats["LoopSkewing"]["skewed"], 0)
        self.assertEqual(stats["LoopWavefront"]["wavefronted"], 0)
        self.assertEqual(stats["LoopParallelization"]["parallelized"], 0)

    def test_non_affine_three_dimensional_case_falls_back_safely(self):
        code = """
PROGRAM NAF3D
    IMPLICIT NONE
    INTEGER I, J, K
    REAL U(8,8,8)
    DO I = 2, 7
        DO J = 2, 7
            DO K = 2, 7
                U(I,J,K) = U(I*J,J,K) + U(I,J,K-1)
            ENDDO
        ENDDO
    ENDDO
END
"""
        optimized, stats = optimize_program(code, level=3)
        ok, errors, _ = analyze_ok(optimized)
        self.assertTrue(ok, errors)
        self.assertEqual(stats["LoopSkewing"]["skewed"], 0)
        self.assertEqual(stats["LoopTiling"]["tiled"], 0)
        self.assertEqual(stats["LoopWavefront"]["wavefronted"], 0)
        llvm_code = LLVMGenerator().generate(optimized)
        self.assertIn("define i32 @main()", llvm_code)


if __name__ == "__main__":
    unittest.main()
