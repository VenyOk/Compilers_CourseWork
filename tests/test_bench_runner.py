import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench_runner import ARTICLE_CORE_FILES, buildMarkdownReport, reportPathFromJsonPath, speedup, summarizeResults


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
        self.assertIn("Stencil And Gauss-Seidel Focus", report)
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
        )
        self.assertIn("Article-Like Focus", report)
        self.assertIn("Parallel threads: `8`", report)
        self.assertIn("Metelitsa: Dirichlet 2D", report)

    def test_default_article_core_set_contains_main_loop_benchmarks(self):
        self.assertIn("bench_matmul.f", ARTICLE_CORE_FILES)
        self.assertIn("bench_stencil_2d.f", ARTICLE_CORE_FILES)
        self.assertIn("bench_gs2d.f", ARTICLE_CORE_FILES)
        self.assertIn("bench_gs3d.f", ARTICLE_CORE_FILES)

    def test_report_path_uses_markdown_suffix(self):
        self.assertEqual(reportPathFromJsonPath("outputs/results.json"), "outputs\\results.md" if os.name == "nt" else "outputs/results.md")


if __name__ == "__main__":
    unittest.main()
