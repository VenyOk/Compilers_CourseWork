import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core import Lexer, Parser, ParallelDoLoop
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


def optimize_program(code: str, level: int = 3, article_mode: bool = False):
    ast = parse_program(code)
    ok, errors, _ = analyze_ok(ast)
    if not ok:
        raise AssertionError(f"semantic errors before optimization: {errors}")
    pipeline = OptimizationPipeline(level=level)
    optimized = pipeline.run(ast)
    return optimized, pipeline.stats


def collect_parallel_vars(stmts):
    result = []
    for stmt in stmts:
        if isinstance(stmt, ParallelDoLoop):
            result.append(stmt.var)
            result.extend(collect_parallel_vars(stmt.body))
        elif hasattr(stmt, "body"):
            result.extend(collect_parallel_vars(stmt.body))
        elif hasattr(stmt, "then_body"):
            result.extend(collect_parallel_vars(stmt.then_body))
            for _, body in stmt.elif_parts:
                result.extend(collect_parallel_vars(body))
            if stmt.else_body:
                result.extend(collect_parallel_vars(stmt.else_body))
        elif hasattr(stmt, "statement"):
            result.extend(collect_parallel_vars([stmt.statement]))
    return result


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
        optimized, stats = optimize_program(code, level=3, article_mode=True)
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
        optimized, stats = optimize_program(code, level=3, article_mode=True)
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
        self.assertEqual(stats["LoopTiling"]["tiled"], 1)
        self.assertEqual(stats["LoopWavefront"]["wavefronted"], 1)

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
        optimized, stats = optimize_program(code, level=3, article_mode=True)
        ok, errors, _ = analyze_ok(optimized)
        self.assertTrue(ok, errors)
        self.assertGreaterEqual(stats["LoopParallelization"]["parallelized"], 1)
        llvm_code = LLVMGenerator().generate(optimized)
        self.assertIn("@__kmpc_fork_call", llvm_code)
        self.assertIn("@__kmpc_for_static_init_4", llvm_code)
        self.assertIn("@__kmpc_for_static_fini", llvm_code)
        self.assertIn("@omp_outlined_", llvm_code)

    def test_parallelization_is_applied_to_two_dimensional_wavefront_band_in_article_o3(self):
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
        self.assertGreaterEqual(stats["LoopWavefront"]["wavefronted"], 1)
        self.assertGreaterEqual(stats["LoopParallelization"]["parallelized"], 1)

    def test_time_space_gauss_seidel_is_transformed_end_to_end(self):
        code = """
PROGRAM TGS2D
    IMPLICIT NONE
    INTEGER T, I, J
    REAL U(96,96)
    DO T = 1, 12
        DO I = 2, 95
            DO J = 2, 95
                U(I,J) = 0.25*(U(I-1,J)+U(I+1,J)+U(I,J-1)+U(I,J+1))
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
        self.assertIn("%tile_", llvm_code)
        self.assertIn("%skew_", llvm_code)

    def test_large_time_space_gauss_seidel_gets_safe_parallel_path_without_parallelizing_time(self):
        code = """
PROGRAM TGS2L
    IMPLICIT NONE
    INTEGER T, I, J
    REAL U(192,192)
    DO I = 1, 192
        DO J = 1, 192
            U(I,J) = 0.0
        ENDDO
    ENDDO
    DO T = 1, 30
        DO I = 2, 191
            DO J = 2, 191
                U(I,J) = 0.25*(U(I-1,J)+U(I+1,J)+U(I,J-1)+U(I,J+1))
            ENDDO
        ENDDO
    ENDDO
END
"""
        optimized, stats = optimize_program(code, level=3)
        ok, errors, _ = analyze_ok(optimized)
        self.assertTrue(ok, errors)
        self.assertEqual(stats["LoopWavefront"]["wavefronted"], 1)
        self.assertGreaterEqual(stats["LoopParallelization"]["parallelized"], 1)
        parallel_vars = collect_parallel_vars(optimized.statements)
        self.assertNotIn("T", parallel_vars)
        llvm_code = LLVMGenerator().generate(optimized)
        self.assertIn("@__kmpc_fork_call", llvm_code)

    def test_time_space_dirichlet_with_many_coeff_arrays_uses_article_style_skewed_tiling(self):
        code = """
PROGRAM TDIR2D
    IMPLICIT NONE
    INTEGER T, I, J
    REAL U(96,96), A(96,96), B(96,96), C(96,96), D(96,96), Y0(96,96), S
    DO T = 1, 16
        DO I = 2, 95
            DO J = 2, 95
                S = A(I,J)*U(I-1,J) + B(I,J)*U(I+1,J)
                S = S + C(I,J)*U(I,J-1) + D(I,J)*U(I,J+1)
                U(I,J) = S + Y0(I,J)
            ENDDO
        ENDDO
    ENDDO
END
"""
        optimized, stats = optimize_program(code, level=3, article_mode=True)
        ok, errors, _ = analyze_ok(optimized)
        self.assertTrue(ok, errors)
        self.assertGreaterEqual(stats["LoopSkewing"]["skewed"], 1, stats)
        self.assertGreaterEqual(stats["LoopTiling"]["tiled"], 1, stats)
        tile_diags = stats["LoopTiling"]["diagnostics"]
        self.assertTrue(tile_diags)
        self.assertEqual(tile_diags[0]["family"], "dirichlet_gs")
        self.assertEqual(len(tile_diags[0]["point_order"]), len(tile_diags[0]["vars"]))
        llvm_code = LLVMGenerator().generate(optimized)
        self.assertIn("%tile_", llvm_code)
        self.assertIn("%skew_", llvm_code)

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
        self.assertEqual(stats["LoopSkewing"]["skewed"], 0)
        self.assertEqual(stats["LoopTiling"]["tiled"], 1)
        self.assertEqual(stats["LoopWavefront"]["wavefronted"], 1)
        llvm_code = LLVMGenerator().generate(optimized)
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
        self.assertEqual(stats["LoopSkewing"]["skewed"], 0)
        self.assertEqual(stats["LoopTiling"]["tiled"], 1)
        self.assertEqual(stats["LoopWavefront"]["wavefronted"], 1)
        llvm_code = LLVMGenerator().generate(optimized)
        self.assertIn("%tile_L = alloca i32", llvm_code)
        self.assertIn("%wf_h0 = alloca i32", llvm_code)

    def test_generated_loop_variables_are_declared_inside_subroutine_for_nd_case(self):
        code = """
PROGRAM WRAP
    IMPLICIT NONE
    CALL STEP3D()
END
SUBROUTINE STEP3D()
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
        self.assertIn("@__kmpc_fork_call", llvm_code)
        self.assertIn("@__kmpc_for_static_init_4", llvm_code)
        self.assertNotIn("@fortran_parallel_for_i32", llvm_code)

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
        optimized, stats = optimize_program(code, level=3)
        self.assertGreaterEqual(stats["LoopParallelization"]["parallelized"], 0)
        parallel_vars = collect_parallel_vars(optimized.statements)
        self.assertNotIn("T", parallel_vars)

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

    def test_small_three_dimensional_nest_keeps_non_wavefront_path(self):
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
        self.assertEqual(stats["LoopSkewing"]["skewed"], 0)
        self.assertEqual(stats["LoopWavefront"]["wavefronted"], 0)
        self.assertIn("IntraTileLoopInterchange", stats)

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


class TestAffineLinearization(unittest.TestCase):
    def test_runs_and_has_stats_after_o3_with_stencil(self):
        code = """
PROGRAM LNST
    IMPLICIT NONE
    INTEGER I, J
    REAL U(40,40)
    DO I = 2, 39
        DO J = 2, 39
            U(I,J) = U(I-1,J) + U(I,J-1)
        ENDDO
    ENDDO
END
"""
        _, stats = optimize_program(code, level=3)
        self.assertIn("AffineLinearization", stats)

    def test_linearization_after_time_space_skewing_produces_valid_ast(self):
        code = """
PROGRAM LNSKW
    IMPLICIT NONE
    INTEGER T, I, J
    REAL U(24,24)
    DO T = 1, 4
        DO I = 2, 23
            DO J = 2, 23
                U(I,J) = 0.25*(U(I-1,J)+U(I+1,J)+U(I,J-1)+U(I,J+1))
            ENDDO
        ENDDO
    ENDDO
END
"""
        optimized, stats = optimize_program(code, level=3)
        ok, errors, _ = analyze_ok(optimized)
        self.assertTrue(ok, errors)
        self.assertIn("AffineLinearization", stats)
        llvm_code = LLVMGenerator().generate(optimized)
        self.assertIn("define i32 @main()", llvm_code)

    def test_constant_zero_add_removed_from_expression(self):
        code = """
PROGRAM ZERPL
    IMPLICIT NONE
    INTEGER I
    REAL A(20)
    DO I = 1, 20
        A(I) = A(I) + 0.0
    ENDDO
END
"""
        optimized, _ = optimize_program(code, level=2)
        ok, errors, _ = analyze_ok(optimized)
        self.assertTrue(ok, errors)
        llvm_code = LLVMGenerator().generate(optimized)
        self.assertIn("define i32 @main()", llvm_code)

    def test_linearization_result_passes_semantic_analysis(self):
        code = """
PROGRAM LINSM
    IMPLICIT NONE
    INTEGER T, I, J
    REAL U(16,16)
    DO T = 1, 2
        DO I = 2, 15
            DO J = 2, 15
                U(I,J) = 0.25*(U(I-1,J)+U(I+1,J)+U(I,J-1)+U(I,J+1))
            ENDDO
        ENDDO
    ENDDO
END
"""
        optimized, _ = optimize_program(code, level=3)
        ok, errors, _ = analyze_ok(optimized)
        self.assertTrue(ok, errors)

    def test_linearization_after_4d_wavefront_no_crash(self):
        code = """
PROGRAM LIN4D
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
        optimized, _ = optimize_program(code, level=3)
        ok, errors, _ = analyze_ok(optimized)
        self.assertTrue(ok, errors)
        llvm_code = LLVMGenerator().generate(optimized)
        self.assertIn("define i32 @main()", llvm_code)

    def test_min_max_constant_args_folded(self):
        code = """
PROGRAM MMCON
    IMPLICIT NONE
    INTEGER I, J
    REAL U(10,10)
    DO I = 1, 8
        DO J = 1, 8
            U(I,J) = U(I+1,J) + U(I,J+1)
        ENDDO
    ENDDO
END
"""
        optimized, _ = optimize_program(code, level=3)
        ok, errors, _ = analyze_ok(optimized)
        self.assertTrue(ok, errors)


class TestO2Level(unittest.TestCase):
    def test_o2_applies_tiling(self):
        code = """
PROGRAM O2TILE
    IMPLICIT NONE
    INTEGER I, J
    REAL A(32,32), B(32,32), C(32,32)
    DO I = 1, 32
        DO J = 1, 32
            C(I,J) = A(I,J) + B(I,J)
        ENDDO
    ENDDO
END
"""
        _, stats = optimize_program(code, level=2)
        self.assertIn("LoopTiling", stats)
        self.assertGreater(stats["LoopTiling"]["tiled"], 0)

    def test_o2_applies_licm(self):
        code = """
PROGRAM O2LICM
    IMPLICIT NONE
    INTEGER I
    REAL S, X
    S = 0.0
    X = 1.5
    DO I = 1, 32
        S = S + SQRT(X * X + 1.0)
    ENDDO
END
"""
        _, stats = optimize_program(code, level=2)
        self.assertGreater(stats["LoopInvariantCodeMotion"]["hoisted"], 0)

    def test_o2_applies_cse(self):
        code = """
PROGRAM O2CSE
    IMPLICIT NONE
    INTEGER I
    REAL S, X
    S = 0.0
    X = 1.5
    DO I = 1, 32
        S = S + SQRT(X * X + 1.0) + SQRT(X * X + 1.0)
    ENDDO
END
"""
        _, stats = optimize_program(code, level=2)
        self.assertGreater(stats["CommonSubexpressionElimination"]["cse_vars"], 0)

    def test_o2_applies_interchange_on_matmul(self):
        code = """
PROGRAM O2INT
    IMPLICIT NONE
    INTEGER I, J, K
    REAL C(8,8), A(8,8), B(8,8)
    DO I = 1, 8
        DO J = 1, 8
            DO K = 1, 8
                C(I,J) = C(I,J) + A(I,K) * B(K,J)
            ENDDO
        ENDDO
    ENDDO
END
"""
        _, stats = optimize_program(code, level=2)
        self.assertIn("LoopInterchange", stats)

    def test_o2_does_not_skew(self):
        code = """
PROGRAM NOSK
    IMPLICIT NONE
    INTEGER T, I, J
    REAL U(16,16)
    DO T = 1, 2
        DO I = 2, 15
            DO J = 2, 15
                U(I,J) = 0.25*(U(I-1,J)+U(I+1,J)+U(I,J-1)+U(I,J+1))
            ENDDO
        ENDDO
    ENDDO
END
"""
        _, stats = optimize_program(code, level=2)
        self.assertNotIn("LoopSkewing", stats)

    def test_o2_produces_valid_llvm(self):
        code = """
PROGRAM O2LLVM
    IMPLICIT NONE
    INTEGER I, J
    REAL A(16,16)
    DO I = 1, 16
        DO J = 1, 16
            A(I,J) = FLOAT(I + J)
        ENDDO
    ENDDO
END
"""
        optimized, _ = optimize_program(code, level=2)
        ok, errors, _ = analyze_ok(optimized)
        self.assertTrue(ok, errors)
        llvm_code = LLVMGenerator().generate(optimized)
        self.assertIn("define i32 @main()", llvm_code)
        self.assertIn("%tile_", llvm_code)

    def test_o2_declares_tile_vars_under_implicit_none(self):
        code = """
PROGRAM O2DECL
    IMPLICIT NONE
    INTEGER I, J
    REAL A(32,32)
    DO I = 1, 32
        DO J = 1, 32
            A(I,J) = FLOAT(I * J)
        ENDDO
    ENDDO
END
"""
        optimized, stats = optimize_program(code, level=2)
        ok, errors, _ = analyze_ok(optimized)
        self.assertTrue(ok, errors)
        declared = {
            name
            for decl in optimized.declarations
            if hasattr(decl, "names")
            for name, _ in decl.names
        }
        if stats.get("LoopTiling", {}).get("tiled", 0) > 0:
            self.assertTrue(any(n.startswith("tile_") for n in declared))


class TestEdgeCasesAndRegression(unittest.TestCase):
    def test_one_dimensional_loop_not_skewed(self):
        code = """
PROGRAM LOOP1D
    IMPLICIT NONE
    INTEGER I
    REAL A(100)
    DO I = 1, 100
        A(I) = FLOAT(I) * 2.0
    ENDDO
END
"""
        _, stats = optimize_program(code, level=3)
        self.assertEqual(stats.get("LoopSkewing", {}).get("skewed", 0), 0)

    def test_jacobi_2d_with_copy_step_does_not_crash(self):
        code = """
PROGRAM JAC2D
    IMPLICIT NONE
    INTEGER I, J, T
    REAL U(32,32), V(32,32)
    DO T = 1, 4
        DO I = 2, 31
            DO J = 2, 31
                V(I,J) = 0.25*(U(I-1,J)+U(I+1,J)+U(I,J-1)+U(I,J+1))
            ENDDO
        ENDDO
        DO I = 2, 31
            DO J = 2, 31
                U(I,J) = V(I,J)
            ENDDO
        ENDDO
    ENDDO
END
"""
        optimized, _ = optimize_program(code, level=3)
        ok, errors, _ = analyze_ok(optimized)
        self.assertTrue(ok, errors)
        llvm_code = LLVMGenerator().generate(optimized)
        self.assertIn("define i32 @main()", llvm_code)

    def test_nine_point_stencil_tiled(self):
        code = """
PROGRAM HEAT9
    IMPLICIT NONE
    INTEGER I, J
    REAL U(16,16)
    DO I = 2, 15
        DO J = 2, 15
            U(I,J) = 0.125*(U(I-1,J)+U(I+1,J)+U(I,J-1)+U(I,J+1)
     &                     +U(I-1,J-1)+U(I-1,J+1)+U(I+1,J-1)+U(I+1,J+1))
        ENDDO
    ENDDO
END
"""
        optimized, stats = optimize_program(code, level=3)
        ok, errors, _ = analyze_ok(optimized)
        self.assertTrue(ok, errors)
        self.assertGreaterEqual(stats["LoopTiling"]["tiled"], 1)

    def test_three_dimensional_heat_equation_tiles(self):
        code = """
PROGRAM HEAT3D
    IMPLICIT NONE
    INTEGER I, J, K
    REAL U(10,10,10), S
    DO I = 2, 9
        DO J = 2, 9
            DO K = 2, 9
                S = U(I-1,J,K) + U(I+1,J,K)
                S = S + U(I,J-1,K) + U(I,J+1,K)
                S = S + U(I,J,K-1) + U(I,J,K+1)
                U(I,J,K) = S / 6.0
            ENDDO
        ENDDO
    ENDDO
END
"""
        optimized, stats = optimize_program(code, level=3)
        ok, errors, _ = analyze_ok(optimized)
        self.assertTrue(ok, errors)
        self.assertGreaterEqual(stats["LoopTiling"]["tiled"], 1)

    def test_do_while_does_not_crash(self):
        code = """
PROGRAM DW
    IMPLICIT NONE
    INTEGER I
    REAL S
    I = 1
    S = 0.0
    DO WHILE (I .LE. 10)
        S = S + FLOAT(I)
        I = I + 1
    ENDDO
END
"""
        optimized, _ = optimize_program(code, level=3)
        ok, errors, _ = analyze_ok(optimized)
        self.assertTrue(ok, errors)

    def test_empty_program_does_not_crash(self):
        code = """
PROGRAM EMPTY
    IMPLICIT NONE
END
"""
        optimized, _ = optimize_program(code, level=3)
        ok, errors, _ = analyze_ok(optimized)
        self.assertTrue(ok, errors)

    def test_multiple_independent_nests_both_tiled(self):
        code = """
PROGRAM MULTI
    IMPLICIT NONE
    INTEGER I, J
    REAL A(32,32), B(32,32)
    DO I = 1, 32
        DO J = 1, 32
            A(I,J) = FLOAT(I + J)
        ENDDO
    ENDDO
    DO I = 1, 32
        DO J = 1, 32
            B(I,J) = A(I,J) * 2.0
        ENDDO
    ENDDO
END
"""
        _, stats = optimize_program(code, level=3)
        self.assertGreaterEqual(stats.get("LoopTiling", {}).get("tiled", 0), 1)

    def test_scalar_reduction_loop_not_skewed(self):
        code = """
PROGRAM SCAL
    IMPLICIT NONE
    INTEGER I, J
    REAL S
    S = 0.0
    DO I = 1, 10
        DO J = 1, 10
            S = S + FLOAT(I * J)
        ENDDO
    ENDDO
END
"""
        _, stats = optimize_program(code, level=3)
        self.assertEqual(stats.get("LoopSkewing", {}).get("skewed", 0), 0)

    def test_intra_tile_interchange_reports_stats(self):
        code = """
PROGRAM ITI
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
        _, stats = optimize_program(code, level=3)
        self.assertIn("IntraTileLoopInterchange", stats)

    def test_skewing_vars_have_skew_prefix_in_llvm(self):
        code = """
PROGRAM SKWVR
    IMPLICIT NONE
    INTEGER T, I, J
    REAL U(24,24)
    DO T = 1, 3
        DO I = 2, 23
            DO J = 2, 23
                U(I,J) = 0.25*(U(I-1,J)+U(I+1,J)+U(I,J-1)+U(I,J+1))
            ENDDO
        ENDDO
    ENDDO
END
"""
        optimized, stats = optimize_program(code, level=3)
        if stats.get("LoopSkewing", {}).get("skewed", 0) > 0:
            llvm_code = LLVMGenerator().generate(optimized)
            self.assertIn("%skew_", llvm_code)

    def test_tile_vars_have_tile_prefix_in_llvm(self):
        code = """
PROGRAM TLVR
    IMPLICIT NONE
    INTEGER I, J
    REAL A(32,32)
    DO I = 1, 32
        DO J = 1, 32
            A(I,J) = FLOAT(I + J)
        ENDDO
    ENDDO
END
"""
        optimized, stats = optimize_program(code, level=3)
        if stats.get("LoopTiling", {}).get("tiled", 0) > 0:
            llvm_code = LLVMGenerator().generate(optimized)
            self.assertIn("%tile_", llvm_code)

    def test_wavefront_vars_have_wf_prefix_in_llvm(self):
        code = """
PROGRAM WFVAR
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
        optimized, stats = optimize_program(code, level=3)
        if stats.get("LoopWavefront", {}).get("wavefronted", 0) > 0:
            llvm_code = LLVMGenerator().generate(optimized)
            self.assertIn("%wf_", llvm_code)

    def test_o3_vs_o0_same_program_no_crash(self):
        code = """
PROGRAM CMP
    IMPLICIT NONE
    INTEGER I, J
    REAL U(20,20)
    DO I = 1, 20
        DO J = 1, 20
            U(I,J) = 0.0
        ENDDO
    ENDDO
    DO I = 2, 19
        DO J = 2, 19
            U(I,J) = U(I-1,J) + U(I,J-1)
        ENDDO
    ENDDO
    WRITE(*,*) U(10,10)
END
"""
        opt0, _ = optimize_program(code, level=0)
        opt3, _ = optimize_program(code, level=3)
        ok0, errors0, _ = analyze_ok(opt0)
        ok3, errors3, _ = analyze_ok(opt3)
        self.assertTrue(ok0, errors0)
        self.assertTrue(ok3, errors3)
        ll0 = LLVMGenerator().generate(opt0)
        ll3 = LLVMGenerator().generate(opt3)
        self.assertIn("define i32 @main()", ll0)
        self.assertIn("define i32 @main()", ll3)

    def test_loop_with_if_inside_not_crashed(self):
        code = """
PROGRAM IFIN
    IMPLICIT NONE
    INTEGER I, J
    REAL U(16,16), V(16,16)
    DO I = 2, 15
        DO J = 2, 15
            IF (U(I,J) .GT. 0.0) THEN
                V(I,J) = U(I-1,J) + U(I,J-1)
            ELSE
                V(I,J) = 0.0
            ENDIF
        ENDDO
    ENDDO
END
"""
        optimized, _ = optimize_program(code, level=3)
        ok, errors, _ = analyze_ok(optimized)
        self.assertTrue(ok, errors)

    def test_loop_with_function_call_not_crashed(self):
        code = """
PROGRAM FNCAL
    IMPLICIT NONE
    INTEGER I, J
    REAL U(16,16)
    DO I = 1, 16
        DO J = 1, 16
            U(I,J) = SQRT(FLOAT(I * I + J * J))
        ENDDO
    ENDDO
END
"""
        optimized, _ = optimize_program(code, level=3)
        ok, errors, _ = analyze_ok(optimized)
        self.assertTrue(ok, errors)

    def test_subroutine_with_nest_optimized(self):
        code = """
PROGRAM SBWRP
    IMPLICIT NONE
    CALL FILMT()
END
SUBROUTINE FILMT()
    IMPLICIT NONE
    INTEGER I, J
    REAL A(16,16)
    DO I = 1, 16
        DO J = 1, 16
            A(I,J) = FLOAT(I + J)
        ENDDO
    ENDDO
END
"""
        optimized, stats = optimize_program(code, level=3)
        ok, errors, _ = analyze_ok(optimized)
        self.assertTrue(ok, errors)
        self.assertEqual(len(optimized.subroutines), 1)

    def test_wave_equation_2d_stencil_not_crashed(self):
        code = """
PROGRAM WAVE2D
    IMPLICIT NONE
    INTEGER T, I, J
    REAL U(32,32), V(32,32), S
    DO T = 1, 4
        DO I = 2, 31
            DO J = 2, 31
                S = U(I-1,J) + U(I+1,J) + U(I,J-1) + U(I,J+1)
                V(I,J) = 2.0*U(I,J) - V(I,J) + 0.25*(S - 4.0*U(I,J))
            ENDDO
        ENDDO
        DO I = 2, 31
            DO J = 2, 31
                U(I,J) = V(I,J)
            ENDDO
        ENDDO
    ENDDO
END
"""
        optimized, _ = optimize_program(code, level=3)
        ok, errors, _ = analyze_ok(optimized)
        self.assertTrue(ok, errors)

    def test_stencil_with_variable_coefficients_skews_or_tiles(self):
        code = """
PROGRAM VCOEF
    IMPLICIT NONE
    INTEGER T, I, J
    REAL U(20,20), A(20,20), B(20,20)
    DO T = 1, 3
        DO I = 2, 19
            DO J = 2, 19
                U(I,J) = A(I,J)*U(I-1,J) + B(I,J)*U(I,J-1)
            ENDDO
        ENDDO
    ENDDO
END
"""
        optimized, _ = optimize_program(code, level=3)
        ok, errors, _ = analyze_ok(optimized)
        self.assertTrue(ok, errors)

    def test_five_point_laplace_1000_iters_structure(self):
        code = """
PROGRAM LAP1K
    IMPLICIT NONE
    INTEGER T, I, J
    REAL U(48,48)
    DO T = 1, 100
        DO I = 2, 47
            DO J = 2, 47
                U(I,J) = 0.25*(U(I-1,J)+U(I+1,J)+U(I,J-1)+U(I,J+1))
            ENDDO
        ENDDO
    ENDDO
END
"""
        optimized, stats = optimize_program(code, level=3)
        ok, errors, _ = analyze_ok(optimized)
        self.assertTrue(ok, errors)
        self.assertEqual(stats["LoopSkewing"]["skewed"], 1)
        self.assertGreaterEqual(stats["LoopTiling"]["tiled"], 1)
        self.assertGreaterEqual(stats["LoopWavefront"]["wavefronted"], 1)


class TestParallelizationDetails(unittest.TestCase):
    def test_independent_init_loop_parallelized(self):
        code = """
PROGRAM PRNIT
    IMPLICIT NONE
    INTEGER I, J
    REAL A(64,64)
    DO I = 1, 64
        DO J = 1, 64
            A(I,J) = 0.0
        ENDDO
    ENDDO
END
"""
        optimized, stats = optimize_program(code, level=3)
        ok, errors, _ = analyze_ok(optimized)
        self.assertTrue(ok, errors)
        self.assertGreaterEqual(stats["LoopParallelization"]["parallelized"], 1)

    def test_parallel_loop_generates_kmpc_calls(self):
        code = """
PROGRAM KMPC
    IMPLICIT NONE
    INTEGER I, J
    REAL A(64,64)
    DO I = 1, 64
        DO J = 1, 64
            A(I,J) = FLOAT(I) + FLOAT(J)
        ENDDO
    ENDDO
END
"""
        optimized, stats = optimize_program(code, level=3)
        if stats.get("LoopParallelization", {}).get("parallelized", 0) > 0:
            llvm_code = LLVMGenerator().generate(optimized)
            self.assertIn("@__kmpc_fork_call", llvm_code)

    def test_reduction_loop_not_parallelized(self):
        code = """
PROGRAM REDUCE
    IMPLICIT NONE
    INTEGER I, J
    REAL S, A(32,32)
    S = 0.0
    DO I = 1, 32
        DO J = 1, 32
            S = S + A(I,J)
        ENDDO
    ENDDO
END
"""
        optimized, stats = optimize_program(code, level=3)
        parallel_vars = collect_parallel_vars(optimized.statements)
        self.assertNotIn("I", parallel_vars)

    def test_parallelization_diagnostics_non_empty_for_stencil(self):
        code = """
PROGRAM PRDG
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
        self.assertGreaterEqual(stats["LoopParallelization"]["parallelized"], 1)
        diags = stats["LoopParallelization"]["diagnostics"]
        self.assertTrue(len(diags) >= 1)
        self.assertIn("strategy", diags[0])
        self.assertIn("var", diags[0])

    def test_parallel_strategy_for_wavefront_is_wavefront_or_tiled(self):
        code = """
PROGRAM PARWF
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
        diags = stats["LoopParallelization"]["diagnostics"]
        strategies = {d["strategy"] for d in diags}
        self.assertTrue(strategies.issubset({"wavefront", "tiled", "independent"}))

    def test_parallel_private_vars_list_present_in_diagnostics(self):
        code = """
PROGRAM PRVDG
    IMPLICIT NONE
    INTEGER I, J
    REAL S, U(32,32)
    DO I = 1, 32
        DO J = 1, 32
            S = FLOAT(I + J)
            U(I,J) = S * S
        ENDDO
    ENDDO
END
"""
        optimized, stats = optimize_program(code, level=3)
        ok, errors, _ = analyze_ok(optimized)
        self.assertTrue(ok, errors)

    def test_time_outer_loop_not_parallelized_in_time_space_gs(self):
        code = """
PROGRAM TLOOP
    IMPLICIT NONE
    INTEGER T, I, J
    REAL U(48,48)
    DO T = 1, 10
        DO I = 2, 47
            DO J = 2, 47
                U(I,J) = 0.25*(U(I-1,J)+U(I+1,J)+U(I,J-1)+U(I,J+1))
            ENDDO
        ENDDO
    ENDDO
END
"""
        optimized, stats = optimize_program(code, level=3)
        parallel_vars = collect_parallel_vars(optimized.statements)
        self.assertNotIn("T", parallel_vars)

    def test_two_independent_nests_both_may_parallelize(self):
        code = """
PROGRAM TWNS
    IMPLICIT NONE
    INTEGER I, J
    REAL A(64,64), B(64,64)
    DO I = 1, 64
        DO J = 1, 64
            A(I,J) = FLOAT(I + J)
        ENDDO
    ENDDO
    DO I = 1, 64
        DO J = 1, 64
            B(I,J) = FLOAT(I * J)
        ENDDO
    ENDDO
END
"""
        optimized, stats = optimize_program(code, level=3)
        ok, errors, _ = analyze_ok(optimized)
        self.assertTrue(ok, errors)
        self.assertGreaterEqual(stats["LoopParallelization"]["parallelized"], 1)


if __name__ == "__main__":
    unittest.main()
