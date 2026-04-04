import argparse
import csv
import json
import math
import os
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

BENCH_FILES = [
    "inputs/bench_lcinv.f",
    "inputs/bench_csesin.f",
    "inputs/bench_matmul.f",
    "inputs/bench_stencil_2d.f",
    "inputs/bench_gs2d.f",
    "inputs/bench_gs3d.f",
    "inputs/bench_metelitsa_gs2d.f",
    "inputs/bench_metelitsa_dir2d.f",
    "inputs/bench_heavy_licm.f",
    "inputs/bench_heavy_sum.f",
    "inputs/bench_heavy_power.f",
    "inputs/bench_heavy_pi.f",
]

LONG_BENCH_FILES = [
    "perf_inputs/bench_long_gs2d.f",
    "perf_inputs/bench_long_gs3d.f",
    "perf_inputs/bench_long_dir2d.f",
    "perf_inputs/bench_long_matmul.f",
    "perf_inputs/bench_long_csesin.f",
]

BENCH_LABELS = {
    "bench_lcinv.f": "LICM: SQRT invariant",
    "bench_csesin.f": "CSE: SIN/COS repeated",
    "bench_matmul.f": "Matmul 100x100",
    "bench_stencil_2d.f": "2D stencil double-buffer",
    "bench_gs2d.f": "2D Gauss-Seidel in-place",
    "bench_gs3d.f": "3D Gauss-Seidel in-place",
    "bench_metelitsa_gs2d.f": "Metelitsa: GS 2D Laplace",
    "bench_metelitsa_dir2d.f": "Metelitsa: Dirichlet 2D",
    "bench_heavy_licm.f": "[Heavy] LICM",
    "bench_heavy_sum.f": "[Heavy] Sum",
    "bench_heavy_power.f": "[Heavy] Power",
    "bench_heavy_pi.f": "[Heavy] Pi",
    "bench_long_gs2d.f": "[Long] 2D Gauss-Seidel in-place",
    "bench_long_gs3d.f": "[Long] 3D Gauss-Seidel in-place",
    "bench_long_dir2d.f": "[Long] Dirichlet 2D",
    "bench_long_matmul.f": "[Long] Matmul",
    "bench_long_csesin.f": "[Long] CSE SIN/COS",
}

ARTICLE_CORE_FILES = {
    "bench_matmul.f",
    "bench_stencil_2d.f",
    "bench_gs2d.f",
    "bench_gs3d.f",
    "bench_metelitsa_gs2d.f",
    "bench_metelitsa_dir2d.f",
    "bench_long_gs2d.f",
    "bench_long_gs3d.f",
    "bench_long_dir2d.f",
    "bench_long_matmul.f",
}

ARTICLE_LIKE_FILES = {
    "bench_metelitsa_gs2d.f",
    "bench_metelitsa_dir2d.f",
    "bench_long_gs2d.f",
    "bench_long_dir2d.f",
}

FOCUS_TITLE = "Metelitsa Focus"

ARTICLE_TILE_SWEEPS: Dict[str, List[List[int]]] = {
    "bench_gs2d.f": [[24, 32, 32], [24, 40, 40], [24, 50, 50]],
    "bench_gs3d.f": [[8, 12, 12, 12], [8, 16, 16, 16], [12, 16, 16, 16]],
    "bench_metelitsa_gs2d.f": [[24, 32, 32], [24, 40, 40], [24, 50, 50], [32, 50, 50]],
    "bench_metelitsa_dir2d.f": [[16, 24, 24], [24, 32, 32], [24, 40, 40], [24, 50, 50]],
    "bench_long_gs2d.f": [[32, 48, 48], [48, 64, 64], [64, 64, 64]],
    "bench_long_gs3d.f": [[8, 16, 16, 16], [12, 20, 20, 20], [16, 24, 24, 24]],
    "bench_long_dir2d.f": [[24, 32, 32], [32, 48, 48], [48, 64, 64]],
}


def selectedBenchmarkFiles(filter_text: Optional[str] = None, include_long: bool = False, long_only: bool = False) -> List[str]:
    relative_paths = list(LONG_BENCH_FILES) if long_only else list(BENCH_FILES)
    if include_long and not long_only:
        relative_paths.extend(LONG_BENCH_FILES)
    files = [str(ROOT / path) for path in relative_paths]
    if filter_text:
        files = [path for path in files if filter_text in path]
    return [path for path in files if os.path.exists(path)]


@contextmanager
def temporaryEnv(name: str, value: Optional[str]):
    old = os.environ.get(name)
    if value is None or value == "":
        os.environ.pop(name, None)
    else:
        os.environ[name] = value
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = old


def tileKey(tile_sizes: Optional[Sequence[int]]) -> str:
    if not tile_sizes:
        return "default"
    return "-".join(str(value) for value in tile_sizes)


def compileArtifact(fortranFile: str, optLevel: int, tile_sizes: Optional[Sequence[int]] = None) -> Optional[Dict]:
    from src.core import Lexer, Parser
    from src.semantic import SemanticAnalyzer
    from src.llvm_generator import LLVMGenerator
    from src.optimizations.loop_analysis import describeNest, extractLoopNests

    suffix = f"O{optLevel}"
    if tile_sizes:
        suffix += f"_tiles_{tileKey(tile_sizes)}"
    llPath = ROOT / "outputs" / f"{Path(fortranFile).stem}_{suffix}.ll"
    llPath.parent.mkdir(exist_ok=True)

    try:
        with open(fortranFile, encoding="utf-8") as f:
            source = f.read()
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        if lexer.get_errors():
            return None
        parser = Parser(tokens)
        ast = parser.parse()
        source_diagnostics = [describeNest(nest) for _, nest, _ in extractLoopNests(ast.statements)]
        sem = SemanticAnalyzer()
        if not sem.analyze(ast):
            return None
        pipeline_stats = {}
        if optLevel > 0:
            from src.optimizations.pipeline import OptimizationPipeline

            tile_override = ",".join(str(value) for value in tile_sizes) if tile_sizes else None
            with temporaryEnv("FORTRAN_TILE_SIZES", tile_override):
                pipeline = OptimizationPipeline(level=optLevel)
                ast = pipeline.run(ast)
                pipeline_stats = pipeline.stats
            sem_after = SemanticAnalyzer()
            if not sem_after.analyze(ast):
                return None
        optimized_diagnostics = [describeNest(nest) for _, nest, _ in extractLoopNests(ast.statements)]
        gen = LLVMGenerator()
        llCode = gen.generate(ast)
        with open(llPath, "w", encoding="utf-8") as f:
            f.write(llCode)
        return {
            "ll_path": str(llPath),
            "optimizer_stats": pipeline_stats,
            "source_loop_diagnostics": source_diagnostics,
            "optimized_loop_diagnostics": optimized_diagnostics,
        }
    except Exception:
        return None


def compileToLl(fortranFile: str, optLevel: int, tile_sizes: Optional[Sequence[int]] = None) -> Optional[str]:
    artifact = compileArtifact(fortranFile, optLevel, tile_sizes=tile_sizes)
    if artifact is None:
        return None
    return artifact["ll_path"]


def runLl(llPath: str, repeat: int = 3, threads: int = 0) -> Optional[float]:
    runner = str(ROOT / "test_llvmlite_run.py")
    try:
        env = os.environ.copy()
        if threads > 0:
            env["OMP_NUM_THREADS"] = str(threads)
        bench_runs = max(int(repeat), 1)
        result = subprocess.run(
            [sys.executable, runner, llPath, "--bench", str(bench_runs)],
            capture_output=True,
            text=True,
            timeout=120,
            env=env,
        )
        if result.returncode != 0:
            return None
        for line in result.stdout.splitlines():
            if line.startswith("Median:"):
                value = float(line.split()[1])
                return value / 1000.0
        return None
    except Exception:
        return None


def defaultThreadSweep(max_threads: int) -> List[int]:
    if max_threads > 0:
        values = {1, max_threads}
        if max_threads >= 2:
            values.add(2)
        if max_threads >= 4:
            values.add(4)
        if max_threads >= 8:
            values.add(8)
        return sorted(values)
    return [1, 2, 4, 8]


def defaultTileSweep(file_name: str) -> List[List[int]]:
    return ARTICLE_TILE_SWEEPS.get(file_name, [])


def benchmarkFile(fortranFile: str, levels: List[int], repeat: int = 3, threads: int = 0) -> Dict:
    label = BENCH_LABELS.get(Path(fortranFile).name, Path(fortranFile).stem)
    result = {"file": Path(fortranFile).name, "label": label, "times": {}, "metadata": {}}
    for level in levels:
        artifact = compileArtifact(fortranFile, level)
        if artifact is None:
            result["times"][f"O{level}"] = None
            result["metadata"][f"O{level}"] = None
            continue
        result["metadata"][f"O{level}"] = {
            "optimizer_stats": artifact["optimizer_stats"],
            "source_loop_diagnostics": artifact["source_loop_diagnostics"],
            "optimized_loop_diagnostics": artifact["optimized_loop_diagnostics"],
        }
        result["times"][f"O{level}"] = runLl(artifact["ll_path"], repeat=repeat, threads=threads)
    return result


def runAutotunedExperiment(fortranFile: str, repeat: int, threads: int) -> Optional[Dict]:
    file_name = Path(fortranFile).name
    if file_name not in ARTICLE_CORE_FILES:
        return None
    base_artifact = compileArtifact(fortranFile, 0)
    o2_artifact = compileArtifact(fortranFile, 2)
    o3_default_artifact = compileArtifact(fortranFile, 3)
    if base_artifact is None or o2_artifact is None or o3_default_artifact is None:
        return None
    baseline = runLl(base_artifact["ll_path"], repeat=repeat, threads=1)
    o2_time = runLl(o2_artifact["ll_path"], repeat=repeat, threads=1)
    o3_seq = runLl(o3_default_artifact["ll_path"], repeat=repeat, threads=1)
    o3_par = runLl(o3_default_artifact["ll_path"], repeat=repeat, threads=threads)
    if baseline is None or o2_time is None or o3_seq is None or o3_par is None:
        return None
    if o3_seq <= o3_par:
        best = {"threads": 1, "tile_sizes": None, "time": o3_seq}
    else:
        best = {"threads": threads if threads > 0 else defaultThreadSweep(threads)[-1], "tile_sizes": None, "time": o3_par}
    candidates = []
    for tile_sizes in defaultTileSweep(file_name):
        artifact = compileArtifact(fortranFile, 3, tile_sizes=tile_sizes)
        if artifact is None:
            continue
        for thread_count in defaultThreadSweep(threads):
            tuned_time = runLl(artifact["ll_path"], repeat=repeat, threads=thread_count)
            if tuned_time is None:
                continue
            candidate = {"threads": thread_count, "tile_sizes": list(tile_sizes), "time": tuned_time}
            candidates.append(candidate)
            if tuned_time < best["time"]:
                best = candidate
    return {
        "file": file_name,
        "label": BENCH_LABELS.get(file_name, file_name),
        "baseline": baseline,
        "o2": o2_time,
        "o3_sequential": o3_seq,
        "o3_parallel_default": o3_par,
        "metadata": {
            "O2": {
                "optimizer_stats": o2_artifact["optimizer_stats"],
                "source_loop_diagnostics": o2_artifact["source_loop_diagnostics"],
                "optimized_loop_diagnostics": o2_artifact["optimized_loop_diagnostics"],
            },
            "O3": {
                "optimizer_stats": o3_default_artifact["optimizer_stats"],
                "source_loop_diagnostics": o3_default_artifact["source_loop_diagnostics"],
                "optimized_loop_diagnostics": o3_default_artifact["optimized_loop_diagnostics"],
            },
        },
        "best": {
            "threads": best["threads"],
            "tile_sizes": list(best["tile_sizes"]) if best["tile_sizes"] else None,
            "time": best["time"],
        },
        "candidates": candidates,
    }


def buildAutotunedExperiments(files: Sequence[str], repeat: int, threads: int) -> List[Dict]:
    experiments = []
    for path in files:
        experiment = runAutotunedExperiment(path, repeat=repeat, threads=threads)
        if experiment is not None:
            experiments.append(experiment)
    return experiments


def speedup(base: Optional[float], optimized: Optional[float]) -> Optional[float]:
    if base is None or optimized is None or optimized <= 0:
        return None
    return base / optimized


def geometricMean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return math.exp(sum(math.log(value) for value in values) / len(values))


def arithmeticMean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / len(values)


def summarizeResults(results: List[Dict], levels: List[int], article_core_files: Optional[Set[str]] = None) -> Dict:
    core_files = article_core_files or ARTICLE_CORE_FILES
    article_results = [item for item in results if item["file"] in core_files]
    summary = {"levels": {}, "article_core": {}}
    for level in levels:
        if level == 0:
            continue
        key = f"O{level}"
        values = []
        best_item = None
        best_speed = None
        for result in results:
            ratio = speedup(result["times"].get("O0"), result["times"].get(key))
            if ratio is None:
                continue
            values.append(ratio)
            if best_speed is None or ratio > best_speed:
                best_speed = ratio
                best_item = result
        article_values = [
            ratio
            for ratio in (
                speedup(result["times"].get("O0"), result["times"].get(key))
                for result in article_results
            )
            if ratio is not None
        ]
        summary["levels"][key] = {
            "count": len(values),
            "geomean_speedup": geometricMean(values),
            "mean_speedup": arithmeticMean(values),
            "best_file": best_item["file"] if best_item else None,
            "best_label": best_item["label"] if best_item else None,
            "best_speedup": best_speed,
        }
        summary["article_core"][key] = {
            "count": len(article_values),
            "geomean_speedup": geometricMean(article_values),
            "mean_speedup": arithmeticMean(article_values),
        }
    return summary


def summarizeExperiments(experiments: List[Dict]) -> Dict:
    tuned = [speedup(item["baseline"], item["best"]["time"]) for item in experiments if speedup(item["baseline"], item["best"]["time"]) is not None]
    default_parallel = [speedup(item["baseline"], item["o3_parallel_default"]) for item in experiments if speedup(item["baseline"], item["o3_parallel_default"]) is not None]
    sequential = [speedup(item["baseline"], item["o3_sequential"]) for item in experiments if speedup(item["baseline"], item["o3_sequential"]) is not None]
    return {
        "count": len(experiments),
        "tuned_geomean": geometricMean(tuned),
        "default_parallel_geomean": geometricMean(default_parallel),
        "sequential_geomean": geometricMean(sequential),
    }


def formatMs(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return f"{value * 1000:.2f} ms"


def formatSpeedup(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return f"{value:.2f}x"


def formatTiles(tile_sizes: Optional[Sequence[int]]) -> str:
    if not tile_sizes:
        return "default"
    return "x".join(str(value) for value in tile_sizes)


def comparisonColumns(levels: List[int]) -> Tuple[List[str], List[str]]:
    cols = [f"O{level}" for level in levels]
    speed_cols = [f"O{level}/O0" for level in levels if level > 0]
    return cols, speed_cols


def buildRows(results: List[Dict], levels: List[int]) -> List[List[str]]:
    cols, _ = comparisonColumns(levels)
    rows = []
    for result in results:
        row = [result["label"]]
        for col in cols:
            row.append(formatMs(result["times"].get(col)))
        base = result["times"].get("O0")
        for level in levels:
            if level == 0:
                continue
            row.append(formatSpeedup(speedup(base, result["times"].get(f"O{level}"))))
        rows.append(row)
    return rows


def buildExperimentRows(experiments: List[Dict]) -> List[List[str]]:
    rows = []
    for item in experiments:
        best = item["best"]
        rows.append([
            item["label"],
            formatMs(item["baseline"]),
            formatMs(item["o2"]),
            formatMs(item["o3_sequential"]),
            formatMs(item["o3_parallel_default"]),
            formatMs(best["time"]),
            str(best["threads"]),
            formatTiles(best["tile_sizes"]),
            formatSpeedup(speedup(item["baseline"], best["time"])),
            formatSpeedup(speedup(item["o3_parallel_default"], best["time"])),
        ])
    return rows


def buildArticleStyleRows(experiments: List[Dict]) -> List[List[str]]:
    rows = []
    for item in experiments:
        best = item["best"]
        rows.append([
            item["label"],
            formatMs(item["baseline"]),
            formatMs(item["o3_sequential"]),
            formatMs(item["o3_parallel_default"]),
            formatMs(best["time"]),
            formatTiles(best["tile_sizes"]),
            str(best["threads"]),
            formatSpeedup(speedup(item["baseline"], item["o3_sequential"])),
            formatSpeedup(speedup(item["baseline"], item["o3_parallel_default"])),
            formatSpeedup(speedup(item["baseline"], best["time"])),
        ])
    return rows


def familyPriority(name: Optional[str]) -> int:
    if name == "dirichlet_gs":
        return 5
    if name == "gauss_seidel":
        return 4
    if name == "coefficient_stencil":
        return 3
    if name == "single_array_stencil":
        return 2
    if name == "stencil":
        return 1
    return 0


def bestLoopDiagnostic(entries: Optional[List[Dict]]) -> Optional[Dict]:
    if not entries:
        return None
    ranked = sorted(
        entries,
        key=lambda item: (
            familyPriority(item.get("family")),
            int(item.get("accesses") or 0),
            int(item.get("depth") or len(item.get("vars") or [])),
        ),
    )
    return ranked[-1]


def firstPassDiagnostic(stats: Optional[Dict], pass_name: str) -> Optional[Dict]:
    if not isinstance(stats, dict):
        return None
    pass_stats = stats.get(pass_name)
    if not isinstance(pass_stats, dict):
        return None
    entries = pass_stats.get("diagnostics")
    if not isinstance(entries, list) or not entries:
        return None
    ranked = sorted(
        entries,
        key=lambda item: (
            familyPriority(item.get("family")),
            int(item.get("accesses") or 0),
            len(item.get("vars") or []),
        ),
    )
    return ranked[-1]


def buildDiagnosticRows(results: List[Dict]) -> List[List[str]]:
    rows = []
    for result in results:
        meta = result.get("metadata", {}).get("O3")
        if not isinstance(meta, dict):
            continue
        source_diag = bestLoopDiagnostic(meta.get("source_loop_diagnostics"))
        stats = meta.get("optimizer_stats", {})
        skew_diag = firstPassDiagnostic(stats, "LoopSkewing")
        tile_diag = firstPassDiagnostic(stats, "LoopTiling")
        wavefront_diag = firstPassDiagnostic(stats, "LoopWavefront")
        parallel_diag = firstPassDiagnostic(stats, "LoopParallelization")
        if source_diag is None and not any([skew_diag, tile_diag, wavefront_diag, parallel_diag]):
            continue
        family = source_diag.get("family") if source_diag else "-"
        skew_value = "no"
        if skew_diag:
            skew_value = f"yes ({skew_diag.get('reason', '-')})"
        tile_value = "no"
        if tile_diag:
            tile_value = f"{formatTiles(tile_diag.get('tile_sizes'))}"
        point_order = "-"
        if tile_diag and tile_diag.get("point_order"):
            point_order = " -> ".join(tile_diag["point_order"])
        wavefront_value = "no"
        if wavefront_diag:
            wavefront_value = f"yes ({wavefront_diag.get('reason', '-')})"
        parallel_value = "no"
        if parallel_diag:
            parallel_value = f"{parallel_diag.get('strategy', 'parallel')} ({parallel_diag.get('grain', '-')})"
        rows.append([
            result["label"],
            family,
            skew_value,
            tile_value,
            point_order,
            wavefront_value,
            parallel_value,
        ])
    return rows


def buildTileSweepRows(experiments: List[Dict], limit: int = 3) -> List[List[str]]:
    rows = []
    for item in experiments:
        baseline = item["baseline"]
        candidates = sorted(item.get("candidates", []), key=lambda candidate: candidate["time"])
        for candidate in candidates[:limit]:
            rows.append([
                item["label"],
                formatTiles(candidate.get("tile_sizes")),
                str(candidate.get("threads")),
                formatMs(candidate.get("time")),
                formatSpeedup(speedup(baseline, candidate.get("time"))),
            ])
    return rows


def renderMarkdownTable(headers: List[str], rows: List[List[str]]) -> List[str]:
    lines = [
        "|" + " | ".join(headers) + " |",
        "|" + " | ".join(["---"] + ["---:" for _ in headers[1:]]) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return lines


def printTable(results: List[Dict], levels: List[int]) -> None:
    cols, speed_cols = comparisonColumns(levels)
    col_width = 12
    header = f"{'Benchmark':<32}" + "".join(f"{col:>{col_width}}" for col in cols)
    header += "".join(f"{col:>{col_width}}" for col in speed_cols)
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    for result in results:
        label = result["label"][:31]
        base = result["times"].get("O0")
        line = f"{label:<32}"
        for col in cols:
            value = result["times"].get(col)
            line += f"{'-':>{col_width}}" if value is None else f"{value * 1000:>{col_width - 2}.2f}ms"
        for level in levels:
            if level == 0:
                continue
            line += f"{formatSpeedup(speedup(base, result['times'].get(f'O{level}'))):>{col_width}}"
        print(line)
    print("=" * len(header))


def printSummary(summary: Dict, levels: List[int], repeat: int, threads: int, experiments: List[Dict]) -> None:
    print("\nSpeedup summary")
    print("-" * 80)
    print(f"Median repeats: {repeat}")
    print(f"Parallel threads: {'auto' if threads <= 0 else threads}")
    for level in levels:
        if level == 0:
            continue
        key = f"O{level}"
        overall = summary["levels"].get(key, {})
        article = summary["article_core"].get(key, {})
        print(
            f"{key}: geomean={formatSpeedup(overall.get('geomean_speedup'))}, "
            f"mean={formatSpeedup(overall.get('mean_speedup'))}, "
            f"best={overall.get('best_file') or '-'} ({formatSpeedup(overall.get('best_speedup'))})"
        )
        print(
            f"    article-core: geomean={formatSpeedup(article.get('geomean_speedup'))}, "
            f"mean={formatSpeedup(article.get('mean_speedup'))}, "
            f"count={article.get('count', 0)}"
        )
    if experiments:
        tuned = summarizeExperiments(experiments)
        print(
            f"Autotuned article-core: sequential={formatSpeedup(tuned.get('sequential_geomean'))}, "
            f"default-parallel={formatSpeedup(tuned.get('default_parallel_geomean'))}, "
            f"tuned={formatSpeedup(tuned.get('tuned_geomean'))}"
        )


def buildSpeedupChartData(results: List[Dict], levels: List[int]) -> List[Tuple[str, Dict[str, float]]]:
    data = []
    for result in results:
        base = result["times"].get("O0")
        values = {}
        for level in levels:
            if level == 0:
                continue
            ratio = speedup(base, result["times"].get(f"O{level}"))
            if ratio is not None:
                values[f"O{level}"] = ratio
        data.append((result["label"], values))
    return data


def buildSpeedupSvg(results: List[Dict], levels: List[int], title: str) -> str:
    chart_data = buildSpeedupChartData(results, levels)
    width = 980
    row_height = 28
    left_margin = 280
    top_margin = 50
    legend_height = 40
    inner_width = width - left_margin - 40
    max_value = 1.0
    for _, values in chart_data:
        for value in values.values():
            max_value = max(max_value, value)
    height = top_margin + legend_height + max(1, len(chart_data)) * row_height + 40
    colors = {"O2": "#3b82f6", "O3": "#ef4444"}
    level_names = [f"O{level}" for level in levels if level > 0]
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width / 2}" y="28" text-anchor="middle" font-family="Segoe UI, Arial" font-size="20" fill="#111827">{title}</text>',
    ]
    legend_x = left_margin
    for level_name in level_names:
        color = colors.get(level_name, "#6b7280")
        lines.append(f'<rect x="{legend_x}" y="40" width="14" height="14" fill="{color}"/>')
        lines.append(f'<text x="{legend_x + 20}" y="52" font-family="Segoe UI, Arial" font-size="12" fill="#111827">{level_name}/O0</text>')
        legend_x += 90
    for tick in range(0, int(math.ceil(max_value)) + 1):
        x = left_margin + (tick / max_value) * inner_width if max_value > 0 else left_margin
        lines.append(f'<line x1="{x}" y1="{top_margin + legend_height}" x2="{x}" y2="{height - 20}" stroke="#e5e7eb" stroke-width="1"/>')
        lines.append(f'<text x="{x}" y="{top_margin + legend_height - 6}" text-anchor="middle" font-family="Segoe UI, Arial" font-size="11" fill="#6b7280">{tick:.0f}x</text>')
    for index, (label, values) in enumerate(chart_data):
        y = top_margin + legend_height + index * row_height + 18
        lines.append(f'<text x="{left_margin - 10}" y="{y}" text-anchor="end" font-family="Segoe UI, Arial" font-size="12" fill="#111827">{label}</text>')
        offset = -6
        for level_name in level_names:
            value = values.get(level_name)
            if value is None:
                offset += 12
                continue
            bar_width = (value / max_value) * inner_width if max_value > 0 else 0
            color = colors.get(level_name, "#6b7280")
            lines.append(f'<rect x="{left_margin}" y="{y + offset - 8}" width="{bar_width}" height="10" fill="{color}" opacity="0.85"/>')
            lines.append(f'<text x="{left_margin + bar_width + 6}" y="{y + offset}" font-family="Segoe UI, Arial" font-size="11" fill="#111827">{value:.2f}x</text>')
            offset += 12
    lines.append("</svg>")
    return "\n".join(lines)


def topSpeedups(results: List[Dict], level: int, limit: int = 3) -> List[Tuple[str, float]]:
    key = f"O{level}"
    values = []
    for result in results:
        ratio = speedup(result["times"].get("O0"), result["times"].get(key))
        if ratio is None:
            continue
        values.append((result["label"], ratio))
    values.sort(key=lambda item: item[1], reverse=True)
    return values[:limit]


def weakBenchmarks(results: List[Dict], level: int) -> List[Tuple[str, float]]:
    key = f"O{level}"
    values = []
    for result in results:
        ratio = speedup(result["times"].get("O0"), result["times"].get(key))
        if ratio is None or ratio >= 1.0:
            continue
        values.append((result["label"], ratio))
    values.sort(key=lambda item: item[1])
    return values


def buildInterpretationLines(results: List[Dict], levels: List[int], summary: Dict, experiments: List[Dict]) -> List[str]:
    lines = ["## Interpretation", ""]
    if 3 in levels:
        o3 = summary["levels"].get("O3", {})
        article = summary["article_core"].get("O3", {})
        lines.append(
            f"- `O3` overall geomean is `{formatSpeedup(o3.get('geomean_speedup'))}`, "
            f"while article-core geomean is `{formatSpeedup(article.get('geomean_speedup'))}`."
        )
        best_file = o3.get("best_file")
        best_speed = formatSpeedup(o3.get("best_speedup"))
        if best_file:
            lines.append(f"- The strongest `O3` result is `{best_file}` with speedup `{best_speed}`.")
        best_o3 = topSpeedups(results, 3)
        if best_o3:
            lines.append(
                "- The most profitable benchmarks are: "
                + ", ".join(f"`{label}` ({value:.2f}x)" for label, value in best_o3)
                + "."
            )
        weak_o3 = weakBenchmarks(results, 3)
        if weak_o3:
            lines.append(
                "- Remaining weak spots are: "
                + ", ".join(f"`{label}` ({value:.2f}x)" for label, value in weak_o3[:3])
                + "."
            )
    if experiments:
        tuned = summarizeExperiments(experiments)
        lines.append(
            f"- In autotuned article-core experiments, transformed sequential execution reaches "
            f"`{formatSpeedup(tuned.get('sequential_geomean'))}`, default parallel reaches "
            f"`{formatSpeedup(tuned.get('default_parallel_geomean'))}`, and the best tuned mode reaches "
            f"`{formatSpeedup(tuned.get('tuned_geomean'))}`."
        )
    lines.append(
        "- For the diploma text, the cleanest storyline is: baseline `O0`, scalar wins on `CSE`/`LICM`, "
        "loop wins on `GS`/`stencil`, and conservative handling of coefficient-heavy `Dirichlet` kernels."
    )
    lines.append("")
    return lines


def buildRuntimeLoweringRows(previous_payload: Dict, results: List[Dict]) -> List[List[str]]:
    previous_results = {
        item["file"]: item
        for item in previous_payload.get("results", [])
        if isinstance(item, dict) and "file" in item
    }
    rows = []
    for result in results:
        previous = previous_results.get(result["file"])
        if previous is None:
            continue
        old_speedup = speedup(previous.get("times", {}).get("O0"), previous.get("times", {}).get("O3"))
        new_speedup = speedup(result["times"].get("O0"), result["times"].get("O3"))
        delta = None if old_speedup is None or new_speedup is None else new_speedup - old_speedup
        rows.append([
            result["label"],
            formatSpeedup(old_speedup),
            formatSpeedup(new_speedup),
            "-" if delta is None else f"{delta:+.2f}x",
        ])
    return rows


def buildRuntimeLoweringLines(previous_payload: Dict, summary: Dict, results: List[Dict]) -> List[str]:
    previous_summary = previous_payload.get("summary", {})
    old_overall = previous_summary.get("levels", {}).get("O3", {}).get("geomean_speedup")
    old_article = previous_summary.get("article_core", {}).get("O3", {}).get("geomean_speedup")
    new_overall = summary.get("levels", {}).get("O3", {}).get("geomean_speedup")
    new_article = summary.get("article_core", {}).get("O3", {}).get("geomean_speedup")
    lines = [
        "## Before runtime lowering",
        "",
        f"- Previous `O3` overall geomean: `{formatSpeedup(old_overall)}`",
        f"- Previous article-core `O3` geomean: `{formatSpeedup(old_article)}`",
        "",
        "## After runtime lowering",
        "",
        f"- Current `O3` overall geomean: `{formatSpeedup(new_overall)}`",
        f"- Current article-core `O3` geomean: `{formatSpeedup(new_article)}`",
        "",
        "## Observed impact on article-core",
        "",
    ]
    rows = buildRuntimeLoweringRows(previous_payload, results)
    if rows:
        lines.extend(renderMarkdownTable(["Benchmark", "Old O3/O0", "New O3/O0", "Delta"], rows))
        lines.append("")
    return lines


def buildSummarySvg(summary: Dict, levels: List[int], title: str) -> str:
    metrics = []
    for level in levels:
        if level == 0:
            continue
        key = f"O{level}"
        metrics.append((f"{key} overall", summary["levels"].get(key, {}).get("geomean_speedup") or 0.0, "#2563eb"))
        metrics.append((f"{key} article-core", summary["article_core"].get(key, {}).get("geomean_speedup") or 0.0, "#dc2626"))
    width = 760
    height = 340
    left_margin = 90
    bottom_margin = 60
    top_margin = 50
    inner_width = width - left_margin - 40
    inner_height = height - top_margin - bottom_margin
    max_value = max([1.0] + [value for _, value, _ in metrics])
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width / 2}" y="28" text-anchor="middle" font-family="Segoe UI, Arial" font-size="20" fill="#111827">{title}</text>',
    ]
    for tick in range(0, int(math.ceil(max_value)) + 1):
        y = top_margin + inner_height - (tick / max_value) * inner_height if max_value > 0 else top_margin + inner_height
        lines.append(f'<line x1="{left_margin}" y1="{y}" x2="{width - 20}" y2="{y}" stroke="#e5e7eb" stroke-width="1"/>')
        lines.append(f'<text x="{left_margin - 10}" y="{y + 4}" text-anchor="end" font-family="Segoe UI, Arial" font-size="11" fill="#6b7280">{tick:.0f}x</text>')
    bar_width = inner_width / max(len(metrics), 1) * 0.55
    for index, (label, value, color) in enumerate(metrics):
        slot_center = left_margin + ((index + 0.5) / max(len(metrics), 1)) * inner_width
        bar_height = (value / max_value) * inner_height if max_value > 0 else 0
        x = slot_center - bar_width / 2
        y = top_margin + inner_height - bar_height
        lines.append(f'<rect x="{x}" y="{y}" width="{bar_width}" height="{bar_height}" fill="{color}" opacity="0.9"/>')
        lines.append(f'<text x="{slot_center}" y="{y - 6}" text-anchor="middle" font-family="Segoe UI, Arial" font-size="11" fill="#111827">{value:.2f}x</text>')
        lines.append(f'<text x="{slot_center}" y="{height - 26}" text-anchor="middle" font-family="Segoe UI, Arial" font-size="11" fill="#111827">{label}</text>')
    lines.append("</svg>")
    return "\n".join(lines)


def buildMarkdownReport(
    results: List[Dict],
    levels: List[int],
    summary: Dict,
    experiments: Optional[List[Dict]] = None,
    article_core_files: Optional[Set[str]] = None,
    focus_title: str = FOCUS_TITLE,
    repeat: int = 3,
    threads: int = 0,
    autotune_enabled: bool = True,
    speedup_chart_name: Optional[str] = None,
    summary_chart_name: Optional[str] = None,
    compare_payload: Optional[Dict] = None,
) -> str:
    experiments = experiments or []
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cols, speed_cols = comparisonColumns(levels)
    core_files = article_core_files or ARTICLE_CORE_FILES
    article_like_results = [result for result in results if result["file"] in ARTICLE_LIKE_FILES]
    all_rows = buildRows(results, levels)
    focus_rows = buildRows([result for result in results if result["file"] in core_files], levels)
    article_like_rows = buildRows(article_like_results, levels)
    experiment_rows = buildExperimentRows(experiments)
    article_style_rows = buildArticleStyleRows(experiments)
    diagnostic_rows = buildDiagnosticRows([result for result in results if result["file"] in core_files or result["file"] in ARTICLE_LIKE_FILES])
    tile_sweep_rows = buildTileSweepRows([item for item in experiments if item["file"] in ARTICLE_LIKE_FILES])
    tuned_summary = summarizeExperiments(experiments) if experiments else {}
    lines = [
        "# Benchmark Report",
        "",
        f"Generated: {generated_at}",
        "",
        "## Run Parameters",
        "",
        f"- Levels: `{' '.join(f'O{level}' for level in levels)}`",
        f"- Median repeats: `{repeat}`",
        f"- Parallel threads: `{'auto' if threads <= 0 else threads}`",
        f"- Autotuning: `{'on' if autotune_enabled else 'off'}`",
        "",
        "## Summary",
        "",
    ]
    lines.extend(renderMarkdownTable(
        ["Level", "Geomean", "Mean", "Best case", "Article-core geomean"],
        [
            [
                f"O{level}",
                formatSpeedup(summary["levels"].get(f"O{level}", {}).get("geomean_speedup")),
                formatSpeedup(summary["levels"].get(f"O{level}", {}).get("mean_speedup")),
                (
                    f"{summary['levels'][f'O{level}']['best_file']} "
                    f"({formatSpeedup(summary['levels'][f'O{level}']['best_speedup'])})"
                    if summary["levels"].get(f"O{level}", {}).get("best_file")
                    else "-"
                ),
                formatSpeedup(summary["article_core"].get(f"O{level}", {}).get("geomean_speedup")),
            ]
            for level in levels
            if level > 0
        ],
    ))
    lines.extend([""])
    lines.extend(buildInterpretationLines(results, levels, summary, experiments))
    if compare_payload:
        lines.extend(buildRuntimeLoweringLines(compare_payload, summary, results))
    lines.extend(["", "## Baseline / O2 / O3 Comparison", ""])
    lines.extend(renderMarkdownTable(["Benchmark"] + cols + speed_cols, all_rows))
    lines.extend(["", f"## {focus_title}", ""])
    lines.extend(renderMarkdownTable(["Benchmark"] + cols + speed_cols, focus_rows))
    if article_like_rows:
        lines.extend(["", "## Article-Like Inputs", ""])
        lines.extend(renderMarkdownTable(["Benchmark"] + cols + speed_cols, article_like_rows))
    if article_style_rows:
        lines.extend(["", "## Article-Style Experiments", ""])
        lines.extend(renderMarkdownTable(
            ["Benchmark", "O0", "O3 seq", "O3 par", "Best tuned", "Tiles", "Threads", "Seq/O0", "Par/O0", "Best/O0"],
            article_style_rows,
        ))
    if experiment_rows:
        lines.extend([
            "",
            "## Autotuned Article-Core Experiments",
            "",
            f"- Sequential transformed geomean: `{formatSpeedup(tuned_summary.get('sequential_geomean'))}`",
            f"- Default parallel geomean: `{formatSpeedup(tuned_summary.get('default_parallel_geomean'))}`",
            f"- Tuned parallel geomean: `{formatSpeedup(tuned_summary.get('tuned_geomean'))}`",
            "",
        ])
        lines.extend(renderMarkdownTable(
            ["Benchmark", "O0", "O2", "O3 seq", "O3 par", "Best tuned", "Threads", "Tiles", "Best/O0", "Best/O3 par"],
            experiment_rows,
        ))
    if tile_sweep_rows:
        lines.extend(["", "## Article-Style Tile Sweep", ""])
        lines.extend(renderMarkdownTable(
            ["Benchmark", "Tiles", "Threads", "Time", "Speedup"],
            tile_sweep_rows,
        ))
    if diagnostic_rows:
        lines.extend(["", "## Transformation Diagnostics", ""])
        lines.extend(renderMarkdownTable(
            ["Benchmark", "Family", "Skew", "Tiles", "Point order", "Wavefront", "Parallel"],
            diagnostic_rows,
        ))
    if summary_chart_name or speedup_chart_name:
        lines.extend(["", "## Charts", ""])
        if summary_chart_name:
            lines.append(f"![Summary chart]({summary_chart_name})")
        if speedup_chart_name:
            lines.append(f"![Speedup chart]({speedup_chart_name})")
        lines.append("")
    return "\n".join(lines)


def saveJson(payload: Dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\nSaved JSON: {path}")


def saveMarkdownReport(report: str, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Saved Markdown report: {path}")


def saveCsv(results: List[Dict], levels: List[int], experiments: List[Dict], path: str) -> None:
    experiment_by_file = {item["file"]: item for item in experiments}
    cols, _ = comparisonColumns(levels)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        header = ["file", "label"] + cols + [f"O{level}/O0" for level in levels if level > 0] + [
            "O3_seq",
            "O3_par",
            "O3_best",
            "best_threads",
            "best_tiles",
            "best/O0",
            "best/O3_par",
        ]
        writer.writerow(header)
        for result in results:
            base = result["times"].get("O0")
            row = [result["file"], result["label"]]
            for col in cols:
                row.append(result["times"].get(col))
            for level in levels:
                if level == 0:
                    continue
                row.append(speedup(base, result["times"].get(f"O{level}")))
            experiment = experiment_by_file.get(result["file"])
            if experiment is None:
                row.extend(["", "", "", "", "", "", ""])
            else:
                row.extend([
                    experiment["o3_sequential"],
                    experiment["o3_parallel_default"],
                    experiment["best"]["time"],
                    experiment["best"]["threads"],
                    formatTiles(experiment["best"]["tile_sizes"]),
                    speedup(experiment["baseline"], experiment["best"]["time"]),
                    speedup(experiment["o3_parallel_default"], experiment["best"]["time"]),
                ])
            writer.writerow(row)
    print(f"Saved CSV: {path}")


def saveTextArtifact(text: str, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Saved artifact: {path}")


def reportPathFromJsonPath(jsonPath: str) -> str:
    return str(Path(jsonPath).with_suffix(".md"))


def csvPathFromJsonPath(jsonPath: str) -> str:
    return str(Path(jsonPath).with_suffix(".csv"))


def summarySvgPathFromJsonPath(jsonPath: str) -> str:
    path = Path(jsonPath)
    return str(path.with_name(f"{path.stem}_summary.svg"))


def speedupSvgPathFromJsonPath(jsonPath: str) -> str:
    path = Path(jsonPath)
    return str(path.with_name(f"{path.stem}_speedups.svg"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark runner for the Fortran optimizer")
    parser.add_argument("--filter", "-f", default=None)
    parser.add_argument("--levels", "-l", nargs="+", type=int, default=[0, 2, 3])
    parser.add_argument("--repeat", "-r", type=int, default=3)
    parser.add_argument("--threads", "-t", type=int, default=0)
    parser.add_argument("--json", "-j", default=None)
    parser.add_argument("--include-long", action="store_true")
    parser.add_argument("--long-only", action="store_true")
    parser.add_argument("--no-autotune", action="store_true")
    parser.add_argument("--no-run", action="store_true")
    parser.add_argument("--compare-json", default=None)
    args = parser.parse_args()

    files = selectedBenchmarkFiles(
        filter_text=args.filter,
        include_long=args.include_long,
        long_only=args.long_only,
    )
    if not files:
        print("No benchmark files matched.")
        return

    print(f"\nBenchmark set: {len(files)} files, levels {args.levels}")
    print(f"Median repeats: {args.repeat}")
    print(f"Parallel threads: {'auto' if args.threads <= 0 else args.threads}\n")

    results = []
    for path in files:
        name = Path(path).name
        print(f"  {name}...", end=" ", flush=True)
        if args.no_run:
            result = {"file": name, "label": BENCH_LABELS.get(name, name), "times": {}}
            for level in args.levels:
                ll = compileToLl(path, level)
                result["times"][f"O{level}"] = "OK" if ll else "ERR"
            results.append(result)
            print(" ".join(f"O{level}:{result['times'][f'O{level}']}" for level in args.levels))
        else:
            result = benchmarkFile(path, args.levels, repeat=args.repeat, threads=args.threads)
            results.append(result)
            parts = []
            for level in args.levels:
                value = result["times"].get(f"O{level}")
                parts.append(f"O{level}:{value * 1000:.1f}ms" if value is not None else f"O{level}:ERR")
            print("  ".join(parts))

    outPath = args.json or str(ROOT / "outputs" / "bench_results.json")
    if args.no_run:
        saveJson({"results": results}, outPath)
        return

    experiments = [] if args.no_autotune else buildAutotunedExperiments(files, repeat=args.repeat, threads=args.threads)
    summary = summarizeResults(results, args.levels, article_core_files=ARTICLE_CORE_FILES)
    compare_payload = None
    if args.compare_json and os.path.exists(args.compare_json):
        with open(args.compare_json, encoding="utf-8") as f:
            compare_payload = json.load(f)
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "levels": args.levels,
        "repeat": args.repeat,
        "threads": args.threads,
        "autotune_enabled": not args.no_autotune,
        "results": results,
        "summary": summary,
        "experiments": experiments,
        "experiment_summary": summarizeExperiments(experiments),
        "compare_json": args.compare_json,
    }
    saveJson(payload, outPath)
    printTable(results, args.levels)
    printSummary(summary, args.levels, args.repeat, args.threads, experiments)

    markdown_path = reportPathFromJsonPath(outPath)
    csv_path = csvPathFromJsonPath(outPath)
    summary_svg_path = summarySvgPathFromJsonPath(outPath)
    speedup_svg_path = speedupSvgPathFromJsonPath(outPath)
    report = buildMarkdownReport(
        results,
        args.levels,
        summary,
        experiments,
        article_core_files=ARTICLE_CORE_FILES,
        focus_title=FOCUS_TITLE,
        repeat=args.repeat,
        threads=args.threads,
        autotune_enabled=not args.no_autotune,
        summary_chart_name=Path(summary_svg_path).name,
        speedup_chart_name=Path(speedup_svg_path).name,
        compare_payload=compare_payload,
    )
    saveMarkdownReport(report, markdown_path)
    saveCsv(results, args.levels, experiments, csv_path)
    saveTextArtifact(buildSummarySvg(summary, args.levels, "Geomean Speedups"), summary_svg_path)
    saveTextArtifact(buildSpeedupSvg(results, args.levels, "Baseline / O2 / O3 Speedups"), speedup_svg_path)


if __name__ == "__main__":
    main()
