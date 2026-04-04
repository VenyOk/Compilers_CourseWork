import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench_runner import (
    ARTICLE_CORE_FILES,
    ARTICLE_LIKE_FILES,
    buildDiagnosticRows,
    buildMarkdownReport,
    buildSpeedupSvg,
    buildSummarySvg,
    csvPathFromJsonPath,
    LONG_BENCH_FILES,
    defaultThreadSweep,
    defaultTileSweep,
    reportPathFromJsonPath,
    selectedBenchmarkFiles,
    speedup,
    speedupSvgPathFromJsonPath,
    summarizeResults,
    summarizeExperiments,
    summarySvgPathFromJsonPath,
)


class TestBenchRunner(unittest.TestCase):
    def test_speedup_helper(self):
        self.assertEqual(speedup(2.0, 1.0), 2.0)
        self.assertIsNone(speedup(None, 1.0))
        self.assertIsNone(speedup(1.0, None))
        self.assertIsNone(speedup(1.0, 0.0))

    def test_summary_and_report_generation(self):
        results = [
            {
                "file": "bench_matmul.f",
                "label": "Matmul 100x100",
                "times": {"O0": 2.0, "O2": 1.0, "O3": 0.5},
            },
            {
                "file": "bench_gs2d.f",
                "label": "2D Gauss-Seidel in-place",
                "times": {"O0": 4.0, "O2": 2.0, "O3": 1.0},
            },
            {
                "file": "bench_heavy_sum.f",
                "label": "[Heavy] Sum",
                "times": {"O0": 3.0, "O2": 3.0, "O3": 2.0},
            },
        ]
        summary = summarizeResults(results, [0, 2, 3])
        self.assertAlmostEqual(summary["levels"]["O2"]["best_speedup"], 2.0)
        self.assertAlmostEqual(summary["levels"]["O3"]["best_speedup"], 4.0)
        self.assertGreater(summary["article_core"]["O3"]["geomean_speedup"], 1.0)
        report = buildMarkdownReport(results, [0, 2, 3], summary)
        self.assertIn("# Benchmark Report", report)
        self.assertIn("## Interpretation", report)
        self.assertIn("Metelitsa Focus", report)
        self.assertIn("Baseline / O2 / O3 Comparison", report)
        self.assertIn("Matmul 100x100", report)
        self.assertIn("2D Gauss-Seidel in-place", report)

    def test_report_generation_with_custom_article_set(self):
        results = [
            {
                "file": "bench_metelitsa_gs2d.f",
                "label": "Metelitsa: GS 2D Laplace",
                "times": {"O0": 10.0, "O2": 8.0, "O3": 5.0},
            },
            {
                "file": "bench_metelitsa_dir2d.f",
                "label": "Metelitsa: Dirichlet 2D",
                "times": {"O0": 12.0, "O2": 9.0, "O3": 6.0},
            },
        ]
        summary = summarizeResults(results, [0, 2, 3], article_core_files={"bench_metelitsa_gs2d.f", "bench_metelitsa_dir2d.f"})
        report = buildMarkdownReport(
            results,
            [0, 2, 3],
            summary,
            article_core_files={"bench_metelitsa_gs2d.f", "bench_metelitsa_dir2d.f"},
            focus_title="Article-Like Focus",
            repeat=5,
            threads=8,
            autotune_enabled=False,
            summary_chart_name="summary.svg",
            speedup_chart_name="speedups.svg",
        )
        self.assertIn("Article-Like Focus", report)
        self.assertIn("Parallel threads: `8`", report)
        self.assertIn("Autotuning: `off`", report)
        self.assertIn("Metelitsa: Dirichlet 2D", report)
        self.assertIn("Charts", report)

    def test_report_generation_includes_article_style_sections_and_diagnostics(self):
        results = [
            {
                "file": "bench_metelitsa_dir2d.f",
                "label": "Metelitsa: Dirichlet 2D",
                "times": {"O0": 12.0, "O2": 9.0, "O3": 6.0},
                "metadata": {
                    "O3": {
                        "optimizer_stats": {
                            "LoopSkewing": {"diagnostics": [{"family": "dirichlet_gs", "reason": "need skew"}]},
                            "LoopTiling": {"diagnostics": [{"family": "dirichlet_gs", "vars": ["T", "I", "J"], "tile_sizes": [64, 50, 50], "point_order": ["T", "J", "I"]}]},
                            "LoopWavefront": {"diagnostics": [{"family": "dirichlet_gs", "reason": "wavefront"}]},
                            "LoopParallelization": {"diagnostics": [{"family": "dirichlet_gs", "strategy": "wavefront", "grain": 4, "reason": "parallel"}]},
                        },
                        "source_loop_diagnostics": [{"family": "dirichlet_gs", "accesses": 9, "depth": 3}],
                        "optimized_loop_diagnostics": [],
                    }
                },
            }
        ]
        summary = summarizeResults(results, [0, 2, 3], article_core_files={"bench_metelitsa_dir2d.f"})
        experiments = [
            {
                "file": "bench_metelitsa_dir2d.f",
                "label": "Metelitsa: Dirichlet 2D",
                "baseline": 12.0,
                "o2": 9.0,
                "o3_sequential": 8.0,
                "o3_parallel_default": 6.5,
                "best": {"threads": 8, "tile_sizes": [64, 50, 50], "time": 6.0},
                "candidates": [{"threads": 8, "tile_sizes": [64, 50, 50], "time": 6.0}],
            }
        ]
        report = buildMarkdownReport(results, [0, 2, 3], summary, experiments=experiments)
        self.assertIn("Article-Style Experiments", report)
        self.assertIn("Article-Style Tile Sweep", report)
        self.assertIn("Transformation Diagnostics", report)
        rows = buildDiagnosticRows(results)
        self.assertEqual(rows[0][1], "dirichlet_gs")
        self.assertIn("64x50x50", report)

    def test_report_generation_with_runtime_lowering_comparison(self):
        results = [
            {
                "file": "bench_gs3d.f",
                "label": "3D Gauss-Seidel in-place",
                "times": {"O0": 10.0, "O2": 8.0, "O3": 4.0},
            }
        ]
        summary = summarizeResults(results, [0, 2, 3], article_core_files={"bench_gs3d.f"})
        compare_payload = {
            "results": [
                {
                    "file": "bench_gs3d.f",
                    "label": "3D Gauss-Seidel in-place",
                    "times": {"O0": 10.0, "O2": 8.5, "O3": 5.0},
                }
            ],
            "summary": {
                "levels": {"O3": {"geomean_speedup": 2.0}},
                "article_core": {"O3": {"geomean_speedup": 2.0}},
            },
        }
        report = buildMarkdownReport(
            results,
            [0, 2, 3],
            summary,
            compare_payload=compare_payload,
        )
        self.assertIn("Before runtime lowering", report)
        self.assertIn("After runtime lowering", report)
        self.assertIn("Observed impact on article-core", report)
        self.assertIn("Old O3/O0", report)

    def test_default_article_core_set_contains_main_loop_benchmarks(self):
        self.assertIn("bench_matmul.f", ARTICLE_CORE_FILES)
        self.assertIn("bench_stencil_2d.f", ARTICLE_CORE_FILES)
        self.assertIn("bench_gs2d.f", ARTICLE_CORE_FILES)
        self.assertIn("bench_gs3d.f", ARTICLE_CORE_FILES)
        self.assertIn("bench_metelitsa_gs2d.f", ARTICLE_CORE_FILES)
        self.assertIn("bench_metelitsa_dir2d.f", ARTICLE_LIKE_FILES)
        self.assertIn("bench_long_gs2d.f", ARTICLE_CORE_FILES)
        self.assertIn("bench_long_matmul.f", ARTICLE_CORE_FILES)
        self.assertIn("bench_long_dir2d.f", ARTICLE_LIKE_FILES)

    def test_long_benchmarks_are_opt_in(self):
        default_files = selectedBenchmarkFiles()
        long_files = selectedBenchmarkFiles(include_long=True)
        long_only_files = selectedBenchmarkFiles(long_only=True)
        self.assertFalse(any("bench_long_" in path for path in default_files))
        self.assertTrue(any("bench_long_" in path for path in long_files))
        self.assertEqual(len(long_only_files), len(LONG_BENCH_FILES))
        self.assertTrue(all("bench_long_" in path for path in long_only_files))

    def test_report_and_artifact_paths_use_expected_suffixes(self):
        self.assertEqual(reportPathFromJsonPath("outputs/results.json"), "outputs\\results.md" if os.name == "nt" else "outputs/results.md")
        self.assertEqual(csvPathFromJsonPath("outputs/results.json"), "outputs\\results.csv" if os.name == "nt" else "outputs/results.csv")
        self.assertEqual(summarySvgPathFromJsonPath("outputs/results.json"), "outputs\\results_summary.svg" if os.name == "nt" else "outputs/results_summary.svg")
        self.assertEqual(speedupSvgPathFromJsonPath("outputs/results.json"), "outputs\\results_speedups.svg" if os.name == "nt" else "outputs/results_speedups.svg")

    def test_svg_generators_produce_svg_markup(self):
        results = [
            {
                "file": "bench_metelitsa_gs2d.f",
                "label": "Metelitsa: GS 2D Laplace",
                "times": {"O0": 8.0, "O2": 6.0, "O3": 4.0},
            },
            {
                "file": "bench_gs3d.f",
                "label": "3D Gauss-Seidel in-place",
                "times": {"O0": 4.0, "O2": 3.0, "O3": 1.5},
            },
        ]
        summary = summarizeResults(results, [0, 2, 3], article_core_files={"bench_metelitsa_gs2d.f", "bench_gs3d.f"})
        summary_svg = buildSummarySvg(summary, [0, 2, 3], "Geomean Speedups")
        speedup_svg = buildSpeedupSvg(results, [0, 2, 3], "Baseline / O2 / O3 Speedups")
        self.assertIn("<svg", summary_svg)
        self.assertIn("Geomean Speedups", summary_svg)
        self.assertIn("<svg", speedup_svg)
        self.assertIn("Baseline / O2 / O3 Speedups", speedup_svg)

    def test_autotuned_experiment_helpers(self):
        self.assertEqual(defaultThreadSweep(0), [1, 2, 4, 8])
        self.assertEqual(defaultThreadSweep(4), [1, 2, 4])
        self.assertTrue(defaultTileSweep("bench_metelitsa_gs2d.f"))
        experiments = [
            {
                "file": "bench_metelitsa_gs2d.f",
                "label": "Metelitsa: GS 2D Laplace",
                "baseline": 10.0,
                "o2": 8.0,
                "o3_sequential": 7.0,
                "o3_parallel_default": 6.0,
                "best": {"threads": 4, "tile_sizes": [24, 50, 50], "time": 5.0},
                "candidates": [],
            }
        ]
        summary = summarizeExperiments(experiments)
        self.assertAlmostEqual(summary["sequential_geomean"], 10.0 / 7.0)
        self.assertAlmostEqual(summary["default_parallel_geomean"], 10.0 / 6.0)
        self.assertAlmostEqual(summary["tuned_geomean"], 2.0)
        report = buildMarkdownReport(
            [
                {
                    "file": "bench_metelitsa_gs2d.f",
                    "label": "Metelitsa: GS 2D Laplace",
                    "times": {"O0": 10.0, "O2": 8.0, "O3": 6.0},
                }
            ],
            [0, 2, 3],
            summarizeResults(
                [
                    {
                        "file": "bench_metelitsa_gs2d.f",
                        "label": "Metelitsa: GS 2D Laplace",
                        "times": {"O0": 10.0, "O2": 8.0, "O3": 6.0},
                    }
                ],
                [0, 2, 3],
                article_core_files={"bench_metelitsa_gs2d.f"},
            ),
            experiments=experiments,
        )
        self.assertIn("Autotuned Article-Core Experiments", report)
        self.assertIn("24x50x50", report)


if __name__ == "__main__":
    unittest.main()
