import argparse
import os
import sys
import json
import subprocess
from pathlib import Path
from typing import List, Dict, Optional

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

BENCH_FILES = [
    "inputs/bench_licm.f",
    "inputs/bench_cse.f",
    "inputs/bench_stencil_2d.f",
    "inputs/bench_heavy_licm.f",
    "inputs/bench_heavy_sum.f",
    "inputs/bench_heavy_power.f",
    "inputs/bench_heavy_pi.f",
]

BENCH_LABELS = {
    "bench_licm.f":        "LICM",
    "bench_cse.f":         "CSE",
    "bench_stencil_2d.f":  "2D Stencil (tiling)",
    "bench_heavy_licm.f":  "[Heavy] LICM",
    "bench_heavy_sum.f":   "[Heavy] Sum",
    "bench_heavy_power.f": "[Heavy] Power",
    "bench_heavy_pi.f":    "[Heavy] Pi",
}

REPEAT = 3


def compileToLl(fortranFile: str, optLevel: int) -> Optional[str]:
    from src.core import Lexer, Parser
    from src.semantic import SemanticAnalyzer
    from src.llvm_generator import LLVMGenerator

    llPath = ROOT / "outputs" / f"{Path(fortranFile).stem}_O{optLevel}.ll"
    llPath.parent.mkdir(exist_ok=True)

    try:
        with open(fortranFile) as f:
            source = f.read()
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        sem = SemanticAnalyzer()
        ok = sem.analyze(ast)
        if not ok:
            return None
        if optLevel > 0:
            from src.optimizations.pipeline import OptimizationPipeline
            pipeline = OptimizationPipeline(level=optLevel)
            ast = pipeline.run(ast)
        gen = LLVMGenerator()
        llCode = gen.generate(ast)
        with open(llPath, 'w') as f:
            f.write(llCode)
        return str(llPath)
    except Exception:
        return None


def runLl(llPath: str) -> Optional[float]:
    runner = str(ROOT / "test_llvmlite_run.py")
    times = []
    for _ in range(REPEAT):
        try:
            r = subprocess.run(
                [sys.executable, runner, llPath],
                capture_output=True, text=True, timeout=30,
            )
            if r.returncode != 0:
                return None
            for line in r.stderr.splitlines() + r.stdout.splitlines():
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


def benchmarkFile(fortranFile: str, levels: List[int]) -> Dict:
    label = BENCH_LABELS.get(Path(fortranFile).name, Path(fortranFile).stem)
    result = {"file": Path(fortranFile).name, "label": label, "times": {}}
    for level in levels:
        ll = compileToLl(fortranFile, level)
        if ll is None:
            result["times"][f"O{level}"] = None
            continue
        t = runLl(ll)
        result["times"][f"O{level}"] = t
    return result


def printTable(results: List[Dict], levels: List[int]) -> None:
    cols = [f"O{l}" for l in levels]
    speedupCols = [f"O{l}/O0" for l in levels if l > 0]
    colW = 12
    header = f"{'Benchmark':<28}" + "".join(f"{c:>{colW}}" for c in cols)
    if 0 in levels:
        header += "".join(f"{c:>{colW}}" for c in speedupCols)
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    for r in results:
        label = r["label"][:27]
        times = r["times"]
        t0 = times.get("O0")
        line = f"{label:<28}"
        for col in cols:
            t = times.get(col)
            if t is None:
                line += f"{'—':>{colW}}"
            else:
                line += f"{t*1000:>{colW-2}.2f}ms"
        if 0 in levels and t0 is not None:
            for l in levels:
                if l == 0:
                    continue
                tl = times.get(f"O{l}")
                if tl is not None and tl > 0:
                    speedup = t0 / tl
                    line += f"{speedup:>{colW-1}.2f}x"
                else:
                    line += f"{'—':>{colW}}"
        print(line)
    print("=" * len(header))
    print(f"\nВсе времена — среднее по {REPEAT} запускам.")
    print("Столбцы OX/O0 — коэффициент ускорения (> 1.0 = быстрее).")


def saveJson(results: List[Dict], path: str) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nРезультаты сохранены: {path}")


def main():
    parser = argparse.ArgumentParser(description="Бенчмарк-раннер оптимизирующего компилятора Fortran")
    parser.add_argument('--filter', '-f', default=None, help='Фильтр по имени файла')
    parser.add_argument('--levels', '-l', nargs='+', type=int, default=[0, 2, 3],
                        help='Уровни оптимизации (по умолчанию: 0 2 3)')
    parser.add_argument('--json', '-j', default=None, help='Сохранить результаты в JSON')
    parser.add_argument('--no-run', action='store_true', help='Только компилировать, без JIT')
    args = parser.parse_args()

    files = [str(ROOT / f) for f in BENCH_FILES]
    if args.filter:
        files = [f for f in files if args.filter in f]
    files = [f for f in files if os.path.exists(f)]
    if not files:
        print("Нет файлов для бенчмарка.")
        return

    print(f"\nБенчмарк: {len(files)} файлов, уровни {args.levels}")
    print(f"Повторений: {REPEAT}\n")

    results = []
    for fpath in files:
        name = Path(fpath).name
        print(f"  {name}...", end=' ', flush=True)
        if args.no_run:
            r = {"file": name, "label": BENCH_LABELS.get(name, name), "times": {}}
            for level in args.levels:
                ll = compileToLl(fpath, level)
                r["times"][f"O{level}"] = "OK" if ll else "ERR"
            results.append(r)
            print(" ".join(f"O{l}:{r['times'][f'O{l}']}" for l in args.levels))
        else:
            r = benchmarkFile(fpath, args.levels)
            results.append(r)
            parts = []
            for l in args.levels:
                t = r["times"].get(f"O{l}")
                parts.append(f"O{l}:{t*1000:.1f}ms" if t is not None else f"O{l}:ERR")
            print("  ".join(parts))

    if not args.no_run:
        printTable(results, args.levels)

    outPath = args.json or str(ROOT / "outputs" / "bench_results.json")
    saveJson(results, outPath)


if __name__ == '__main__':
    main()
