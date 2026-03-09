"""Бенчмарк-раннер для оптимизирующего компилятора Fortran.

Сравнивает время JIT-исполнения программ при уровнях оптимизации O0/O1/O2/O3.
Формирует таблицу ускорений по образцу из статьи Метелицы (2024).

Использование:
    python bench_runner.py
    python bench_runner.py --filter licm
    python bench_runner.py --levels 0 1 2
    python bench_runner.py --csv results.csv
"""
import argparse
import time
import os
import sys
import json
import traceback
from pathlib import Path
from typing import List, Dict, Optional

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

BENCH_FILES = [
    "inputs/bench_const_fold.f",
    "inputs/bench_const_prop.f",
    "inputs/bench_licm.f",
    "inputs/bench_cse.f",
    "inputs/bench_dead_code.f",
    "inputs/bench_strength_red.f",
    "inputs/bench_combined.f",
    "inputs/bench_stencil_2d.f",
    "inputs/bench_heavy_fold.f",
    "inputs/bench_heavy_licm.f",
    "inputs/bench_heavy_strength.f",
    "inputs/bench_heavy_sum.f",
    "inputs/bench_heavy_power.f",
    "inputs/bench_heavy_pi.f",
    "inputs/bench_heavy_combined.f",
    "inputs/bench_heavy_allopt.f",
]

BENCH_LABELS = {
    "bench_const_fold.f":      "Const Folding",
    "bench_const_prop.f":      "Const Propagation",
    "bench_licm.f":            "LICM",
    "bench_cse.f":             "CSE",
    "bench_dead_code.f":       "Dead Code Elim",
    "bench_strength_red.f":    "Strength Reduction",
    "bench_combined.f":        "Combined O1",
    "bench_stencil_2d.f":      "2D Stencil (tiling)",
    "bench_heavy_fold.f":      "[Heavy] Const Folding",
    "bench_heavy_licm.f":      "[Heavy] LICM",
    "bench_heavy_strength.f":  "[Heavy] Strength Red",
    "bench_heavy_sum.f":       "[Heavy] Sum",
    "bench_heavy_power.f":     "[Heavy] Power",
    "bench_heavy_pi.f":        "[Heavy] Pi",
    "bench_heavy_combined.f":  "[Heavy] Combined",
    "bench_heavy_allopt.f":    "[Heavy] All Opts",
}

REPEAT = 3  # повторений для усреднения


def compile_to_ll(fortran_file: str, opt_level: int) -> Optional[str]:
    """Компилировать .f → .ll с заданным уровнем оптимизации.

    Возвращает путь к .ll файлу или None при ошибке.
    """
    from src.core import Lexer, Parser
    from src.semantic import SemanticAnalyzer
    from src.llvm_generator import LLVMGenerator

    ll_path = ROOT / "outputs" / f"{Path(fortran_file).stem}_O{opt_level}.ll"
    ll_path.parent.mkdir(exist_ok=True)

    try:
        with open(fortran_file) as f:
            source = f.read()

        lexer = Lexer(source)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()

        sem = SemanticAnalyzer()
        ok = sem.analyze(ast)
        if not ok:
            return None

        if opt_level > 0:
            from src.optimizations.pipeline import OptimizationPipeline
            pipeline = OptimizationPipeline(level=opt_level)
            ast = pipeline.run(ast)

        gen = LLVMGenerator()
        ll_code = gen.generate(ast)

        with open(ll_path, 'w') as f:
            f.write(ll_code)

        return str(ll_path)

    except Exception:
        return None


def _init_llvm():
    """Инициализировать llvmlite (безопасно для разных версий)."""
    import llvmlite.binding as llvm
    for fn in [llvm.initialize_native_target, llvm.initialize_native_asmprinter]:
        try:
            fn()
        except Exception:
            pass


def run_ll(ll_path: str) -> Optional[float]:
    """JIT-исполнить .ll файл через subprocess, вернуть медианное время (сек)."""
    import subprocess
    runner = str(ROOT / "test_llvmlite_run.py")
    times = []
    for _ in range(REPEAT):
        try:
            r = subprocess.run(
                [sys.executable, runner, ll_path],
                capture_output=True, text=True, timeout=30,
            )
            if r.returncode != 0:
                return None
            # test_llvmlite_run.py prints "[Time: X.XXX ms]" at the end
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


def benchmark_file(fortran_file: str, levels: List[int]) -> Dict:
    """Запустить бенчмарк для одного файла, вернуть результаты."""
    label = BENCH_LABELS.get(Path(fortran_file).name,
                              Path(fortran_file).stem)
    result = {"file": Path(fortran_file).name, "label": label, "times": {}}

    for level in levels:
        ll = compile_to_ll(fortran_file, level)
        if ll is None:
            result["times"][f"O{level}"] = None
            continue

        t = run_ll(ll)
        result["times"][f"O{level}"] = t

    return result


def print_table(results: List[Dict], levels: List[int]) -> None:
    """Вывести таблицу ускорений."""
    cols = [f"O{l}" for l in levels]
    speedup_cols = [f"O{l}/O0" for l in levels if l > 0]

    header_label = "Benchmark"
    col_w = 12

    # Заголовок
    header = f"{'Benchmark':<28}" + "".join(f"{c:>{col_w}}" for c in cols)
    if 0 in levels:
        header += "".join(f"{c:>{col_w}}" for c in speedup_cols)
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
                line += f"{'—':>{col_w}}"
            else:
                line += f"{t*1000:>{col_w-2}.2f}ms"

        if 0 in levels and t0 is not None:
            for l in levels:
                if l == 0:
                    continue
                tl = times.get(f"O{l}")
                if tl is not None and tl > 0:
                    speedup = t0 / tl
                    line += f"{speedup:>{col_w-1}.2f}x"
                else:
                    line += f"{'—':>{col_w}}"

        print(line)

    print("=" * len(header))
    print("\nВсе времена — среднее по", REPEAT, "запускам.")
    print("Столбцы O1/O0, O2/O0, O3/O0 — коэффициент ускорения.")
    print("(> 1.0 = быстрее; < 1.0 = медленнее — возможно из-за накладных расходов JIT)")


def save_json(results: List[Dict], path: str) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nРезультаты сохранены: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Бенчмарк-раннер оптимизирующего компилятора Fortran"
    )
    parser.add_argument(
        '--filter', '-f', default=None,
        help='Фильтр по имени файла (подстрока)'
    )
    parser.add_argument(
        '--levels', '-l', nargs='+', type=int, default=[0, 1, 2],
        help='Уровни оптимизации для сравнения (по умолчанию: 0 1 2)'
    )
    parser.add_argument(
        '--json', '-j', default=None,
        help='Сохранить результаты в JSON-файл'
    )
    parser.add_argument(
        '--no-run', action='store_true',
        help='Только компилировать, без JIT-исполнения'
    )
    args = parser.parse_args()

    files = [str(ROOT / f) for f in BENCH_FILES]
    if args.filter:
        files = [f for f in files if args.filter in f]

    files = [f for f in files if os.path.exists(f)]
    if not files:
        print("Нет файлов для бенчмарка.")
        return

    print(f"\nБенчмарк: {len(files)} файлов, уровни O{args.levels}")
    print(f"Каждый запуск повторяется {REPEAT} раз\n")

    results = []
    for fpath in files:
        name = Path(fpath).name
        print(f"  {name}...", end=' ', flush=True)

        if args.no_run:
            # Только компиляция
            r = {"file": name, "label": BENCH_LABELS.get(name, name), "times": {}}
            for level in args.levels:
                ll = compile_to_ll(fpath, level)
                status = "OK" if ll else "ERR"
                r["times"][f"O{level}"] = status
            results.append(r)
            print(" ".join(f"O{l}:{r['times'][f'O{l}']}" for l in args.levels))
        else:
            r = benchmark_file(fpath, args.levels)
            results.append(r)
            parts = []
            for l in args.levels:
                t = r["times"].get(f"O{l}")
                if t is None:
                    parts.append(f"O{l}:ERR")
                else:
                    parts.append(f"O{l}:{t*1000:.1f}ms")
            print("  ".join(parts))

    if not args.no_run:
        print_table(results, args.levels)

    if args.json:
        save_json(results, args.json)
    else:
        save_json(results, str(ROOT / "outputs" / "bench_results.json"))


if __name__ == '__main__':
    main()
