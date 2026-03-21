import argparse
import json
import math
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

BENCH_FILES = [
    "inputs/bench_lcinv.f",
    "inputs/bench_csesin.f",
    "inputs/bench_matmul.f",
    "inputs/bench_stencil_2d.f",
    "inputs/bench_gs2d.f",
    "inputs/bench_gs3d.f",
    "inputs/bench_heavy_licm.f",
    "inputs/bench_heavy_sum.f",
    "inputs/bench_heavy_power.f",
    "inputs/bench_heavy_pi.f",
]

BENCH_LABELS = {
    "bench_lcinv.f": "LICM: SQRT invariant",
    "bench_csesin.f": "CSE: SIN/COS repeated",
    "bench_matmul.f": "Matmul 100x100",
    "bench_stencil_2d.f": "2D stencil double-buffer",
    "bench_gs2d.f": "2D Gauss-Seidel in-place",
    "bench_gs3d.f": "3D Gauss-Seidel in-place",
    "bench_heavy_licm.f": "[Heavy] LICM",
    "bench_heavy_sum.f": "[Heavy] Sum",
    "bench_heavy_power.f": "[Heavy] Power",
    "bench_heavy_pi.f": "[Heavy] Pi",
    "bench_metelitsa_gs2d.f": "Metelitsa: GS 2D Laplace",
    "bench_metelitsa_dir2d.f": "Metelitsa: Dirichlet 2D",
}

ARTICLE_CORE_FILES = {
    "bench_matmul.f",
    "bench_stencil_2d.f",
    "bench_gs2d.f",
    "bench_gs3d.f",
}

FOCUS_TITLE = "Stencil And Gauss-Seidel Focus"


def compileToLl(fortranFile: str, optLevel: int) -> Optional[str]:
    from src.core import Lexer, Parser
    from src.semantic import SemanticAnalyzer
    from src.llvm_generator import LLVMGenerator

    llPath = ROOT / "outputs" / f"{Path(fortranFile).stem}_O{optLevel}.ll"
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
        sem = SemanticAnalyzer()
        if not sem.analyze(ast):
            return None
        if optLevel > 0:
            from src.optimizations.pipeline import OptimizationPipeline

            pipeline = OptimizationPipeline(level=optLevel)
            ast = pipeline.run(ast)
            sem_after = SemanticAnalyzer()
            if not sem_after.analyze(ast):
                return None
        gen = LLVMGenerator()
        llCode = gen.generate(ast)
        with open(llPath, "w", encoding="utf-8") as f:
            f.write(llCode)
        return str(llPath)
    except Exception:
        return None


def runLl(llPath: str, repeat: int = 3, threads: int = 0) -> Optional[float]:
    runner = str(ROOT / "test_llvmlite_run.py")
    times = []
    for _ in range(repeat):
        try:
            env = os.environ.copy()
            if threads > 0:
                env["FORTRAN_PARALLEL_THREADS"] = str(threads)
            result = subprocess.run(
                [sys.executable, runner, llPath],
                capture_output=True,
                text=True,
                timeout=60,
                env=env,
            )
            if result.returncode != 0:
                return None
            for line in result.stderr.splitlines() + result.stdout.splitlines():
                if line.startswith("[Time:"):
                    ms = float(line.split()[1])
                    times.append(ms / 1000.0)
                    break
            else:
                return None
        except Exception:
            return None
    if not times:
        return None
    return sorted(times)[len(times) // 2]


def benchmarkFile(fortranFile: str, levels: List[int], repeat: int = 3, threads: int = 0) -> Dict:
    label = BENCH_LABELS.get(Path(fortranFile).name, Path(fortranFile).stem)
    result = {"file": Path(fortranFile).name, "label": label, "times": {}}
    for level in levels:
        ll = compileToLl(fortranFile, level)
        if ll is None:
            result["times"][f"O{level}"] = None
            continue
        result["times"][f"O{level}"] = runLl(ll, repeat=repeat, threads=threads)
    return result


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
    summary = {"levels": {}, "article_core": {}}
    article_results = [item for item in results if item["file"] in core_files]
    for level in levels:
        if level == 0:
            continue
        key = f"O{level}"
        values = []
        best_item = None
        best_speedup = None
        for result in results:
            current_speedup = speedup(result["times"].get("O0"), result["times"].get(key))
            if current_speedup is None:
                continue
            values.append(current_speedup)
            if best_speedup is None or current_speedup > best_speedup:
                best_speedup = current_speedup
                best_item = result
        article_values = [
            current_speedup
            for current_speedup in (
                speedup(result["times"].get("O0"), result["times"].get(key))
                for result in article_results
            )
            if current_speedup is not None
        ]
        summary["levels"][key] = {
            "count": len(values),
            "geomean_speedup": geometricMean(values),
            "mean_speedup": arithmeticMean(values),
            "best_file": best_item["file"] if best_item else None,
            "best_label": best_item["label"] if best_item else None,
            "best_speedup": best_speedup,
        }
        summary["article_core"][key] = {
            "count": len(article_values),
            "geomean_speedup": geometricMean(article_values),
            "mean_speedup": arithmeticMean(article_values),
        }
    return summary


def formatMs(value: Optional[float]) -> str:
    if value is None:
        return "—"
    return f"{value * 1000:.2f} ms"


def formatSpeedup(value: Optional[float]) -> str:
    if value is None:
        return "—"
    return f"{value:.2f}x"


def printTable(results: List[Dict], levels: List[int]) -> None:
    cols = [f"O{level}" for level in levels]
    speedup_cols = [f"O{level}/O0" for level in levels if level > 0]
    col_width = 12
    header = f"{'Benchmark':<30}" + "".join(f"{col:>{col_width}}" for col in cols)
    if 0 in levels:
        header += "".join(f"{col:>{col_width}}" for col in speedup_cols)
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    for result in results:
        label = result["label"][:29]
        times = result["times"]
        base = times.get("O0")
        line = f"{label:<30}"
        for col in cols:
            value = times.get(col)
            if value is None:
                line += f"{'—':>{col_width}}"
            else:
                line += f"{value * 1000:>{col_width - 2}.2f}ms"
        if 0 in levels and base is not None:
            for level in levels:
                if level == 0:
                    continue
                line += f"{formatSpeedup(speedup(base, times.get(f'O{level}'))):>{col_width}}"
        print(line)
    print("=" * len(header))


def printSummary(summary: Dict, levels: List[int], repeat: int, threads: int) -> None:
    print("\nСводка ускорений")
    print("-" * 80)
    print(f"Повторений: {repeat}")
    print(f"Потоки: {'auto' if threads <= 0 else threads}")
    for level in levels:
        if level == 0:
            continue
        key = f"O{level}"
        overall = summary["levels"].get(key, {})
        article = summary["article_core"].get(key, {})
        print(
            f"{key}: geomean={formatSpeedup(overall.get('geomean_speedup'))}, "
            f"mean={formatSpeedup(overall.get('mean_speedup'))}, "
            f"best={overall.get('best_file') or '—'} ({formatSpeedup(overall.get('best_speedup'))})"
        )
        print(
            f"    article-core: geomean={formatSpeedup(article.get('geomean_speedup'))}, "
            f"mean={formatSpeedup(article.get('mean_speedup'))}, "
            f"count={article.get('count', 0)}"
        )


def buildMarkdownReport(
    results: List[Dict],
    levels: List[int],
    summary: Dict,
    article_core_files: Optional[Set[str]] = None,
    focus_title: str = FOCUS_TITLE,
    repeat: int = 3,
    threads: int = 0,
) -> str:
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cols = [f"O{level}" for level in levels]
    speedup_cols = [f"O{level}/O0" for level in levels if level > 0]
    core_files = article_core_files or ARTICLE_CORE_FILES
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
        "",
        "## Summary",
        "",
        "| Level | Geomean | Mean | Best case | Article-core geomean |",
        "| --- | ---: | ---: | --- | ---: |",
    ]
    for level in levels:
        if level == 0:
            continue
        key = f"O{level}"
        overall = summary["levels"].get(key, {})
        article = summary["article_core"].get(key, {})
        best_case = "—"
        if overall.get("best_file"):
            best_case = f"{overall['best_file']} ({formatSpeedup(overall.get('best_speedup'))})"
        lines.append(
            f"| {key} | {formatSpeedup(overall.get('geomean_speedup'))} | {formatSpeedup(overall.get('mean_speedup'))} | "
            f"{best_case} | {formatSpeedup(article.get('geomean_speedup'))} |"
        )
    lines.extend([
        "",
        "## All Benchmarks",
        "",
        "|" + " | ".join(["Benchmark"] + cols + speedup_cols) + " |",
        "|" + " | ".join(["---"] + ["---:" for _ in cols + speedup_cols]) + " |",
    ])
    for result in results:
        row = [result["label"]]
        for col in cols:
            row.append(formatMs(result["times"].get(col)))
        base = result["times"].get("O0")
        for level in levels:
            if level == 0:
                continue
            row.append(formatSpeedup(speedup(base, result["times"].get(f"O{level}"))))
        lines.append("| " + " | ".join(row) + " |")
    lines.extend([
        "",
        f"## {focus_title}",
        "",
        "|" + " | ".join(["Benchmark"] + cols + speedup_cols) + " |",
        "|" + " | ".join(["---"] + ["---:" for _ in cols + speedup_cols]) + " |",
    ])
    for result in results:
        if result["file"] not in core_files:
            continue
        row = [result["label"]]
        for col in cols:
            row.append(formatMs(result["times"].get(col)))
        base = result["times"].get("O0")
        for level in levels:
            if level == 0:
                continue
            row.append(formatSpeedup(speedup(base, result["times"].get(f"O{level}"))))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return "\n".join(lines)


def saveJson(results: List[Dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nРезультаты сохранены: {path}")


def saveMarkdownReport(report: str, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Отчет сохранен: {path}")


def reportPathFromJsonPath(jsonPath: str) -> str:
    json_path = Path(jsonPath)
    return str(json_path.with_suffix(".md"))


def main():
    parser = argparse.ArgumentParser(description="Бенчмарк-раннер оптимизирующего компилятора Fortran")
    parser.add_argument("--filter", "-f", default=None)
    parser.add_argument("--levels", "-l", nargs="+", type=int, default=[0, 2, 3])
    parser.add_argument("--repeat", "-r", type=int, default=3)
    parser.add_argument("--threads", "-t", type=int, default=0)
    parser.add_argument("--json", "-j", default=None)
    parser.add_argument("--no-run", action="store_true")
    args = parser.parse_args()

    files = [str(ROOT / path) for path in BENCH_FILES]
    if args.filter:
        files = [path for path in files if args.filter in path]
    files = [path for path in files if os.path.exists(path)]
    if not files:
        print("Нет файлов для бенчмарка.")
        return

    print(f"\nБенчмарк: {len(files)} файлов, уровни {args.levels}")
    print(f"Повторений: {args.repeat}")
    print(f"Потоки: {'auto' if args.threads <= 0 else args.threads}\n")

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
    saveJson(results, outPath)

    if args.no_run:
        return

    printTable(results, args.levels)
    summary = summarizeResults(results, args.levels, article_core_files=ARTICLE_CORE_FILES)
    printSummary(summary, args.levels, args.repeat, args.threads)
    report = buildMarkdownReport(
        results,
        args.levels,
        summary,
        article_core_files=ARTICLE_CORE_FILES,
        focus_title=FOCUS_TITLE,
        repeat=args.repeat,
        threads=args.threads,
    )
    saveMarkdownReport(report, reportPathFromJsonPath(outPath))


if __name__ == "__main__":
    main()
