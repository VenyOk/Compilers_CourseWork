"""Benchmark test_strassen.f at various matrix sizes via clang native exe."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

STRASSEN_SRC = ROOT / "inputs" / "test_strassen.f"
OUT_DIR = ROOT / "outputs" / "strassen_bench"
EXE_DIR = OUT_DIR / "exe"
DEFAULT_SIZES = [8, 16, 32, 64, 128, 256, 512]
LEVELS = [0, 2, 3]


def replace_n(source: str, n: int) -> str:
    return source.replace("N = 8", f"N = {n}", 1)


def gen_matrix(n: int, seed: int) -> List[List[float]]:
    mat = []
    for i in range(1, n + 1):
        row = []
        for j in range(1, n + 1):
            val = (i * 17 + j * 23 + seed) * 31
            row.append(float(abs(val) % 10))
        mat.append(row)
    return mat


def matmul_ref(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    n = len(a)
    c = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for k in range(n):
            aik = a[i][k]
            for j in range(n):
                c[i][j] += aik * b[k][j]
    return c


def expected_summary(n: int):
    expected = matmul_ref(gen_matrix(n, 12345), gen_matrix(n, 13345))
    checksum = sum(sum(row) for row in expected)
    return checksum, expected[0][0], expected[-1][-1]


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


def compile_strassen(n: int, opt_level: int) -> Optional[str]:
    from src.frontend.lexer import Lexer
    from src.frontend.parser import Parser
    from src.semantic.analyzer import SemanticAnalyzer
    from src.ir.llvm import LLVMGenerator

    source = replace_n(STRASSEN_SRC.read_text(encoding="utf-8"), n)
    lexer = Lexer(source)
    tokens = lexer.tokenize()
    if lexer.get_errors():
        return None
    parser = Parser(tokens)
    ast = parser.parse()
    sem = SemanticAnalyzer()
    if not sem.analyze(ast):
        return None
    if opt_level > 0:
        from src.optimizations.pipeline import OptimizationPipeline

        pipeline = OptimizationPipeline(level=opt_level)
        ast = pipeline.run(ast)
        sem_after = SemanticAnalyzer()
        if not sem_after.analyze(ast):
            return None
    ll_path = OUT_DIR / f"strassen_N{n}_O{opt_level}.ll"
    ll_path.parent.mkdir(parents=True, exist_ok=True)
    ll_code = LLVMGenerator().generate(ast)
    ll_path.write_text(ll_code, encoding="utf-8")
    return str(ll_path)


def resolve_clang() -> str:
    candidates = [
        Path(r"C:\Program Files\LLVM\bin\clang.exe"),
        ROOT / ".llvm" / "clang+llvm-22.1.2-x86_64-pc-windows-msvc" / "bin" / "clang.exe",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    import shutil

    found = shutil.which("clang")
    if found:
        return found
    raise RuntimeError("clang.exe not found")


def resolve_vcvars() -> str:
    pf86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
    for edition in ("18", "2022", "2019"):
        path = Path(pf86) / "Microsoft Visual Studio" / edition / "BuildTools" / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
        if path.exists():
            return str(path)
        path = Path(pf86) / "Microsoft Visual Studio" / edition / "Community" / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
        if path.exists():
            return str(path)
    raise RuntimeError("vcvars64.bat not found")


def compile_exe(clang: str, vcvars: str, ll_path: str, exe_path: str, stack_bytes: int = 134217728) -> tuple[bool, str]:
    cmd = (
        f'"{vcvars}" >nul && "{clang}" "{ll_path}" '
        f'-Wl,/STACK:{stack_bytes} -o "{exe_path}" 2>&1'
    )
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    detail = (result.stdout + result.stderr).strip()
    return result.returncode == 0 and Path(exe_path).exists(), detail


def run_exe(exe_path: str, timeout_s: int = 600) -> tuple[int, str, float]:
    exe = Path(exe_path)
    start = time.perf_counter()
    result = subprocess.run(
        [str(exe)],
        capture_output=True,
        text=True,
        timeout=timeout_s,
        cwd=str(exe.parent),
    )
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    output = (result.stdout + result.stderr).strip()
    return result.returncode, output, elapsed_ms


def median(values: List[float]) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    return float(ordered[len(ordered) // 2])


def speedup(base: Optional[float], optimized: Optional[float]) -> Optional[float]:
    if base is None or optimized is None or optimized <= 0:
        return None
    return base / optimized


def bench_size(n: int, repeat: int, clang: str, vcvars: str) -> Dict:
    exp_checksum, exp_c11, exp_cnn = expected_summary(n)
    algo = "MATSTD" if n < 64 else "MATBLK"
    item: Dict = {
        "n": n,
        "algorithm": algo,
        "expected_checksum": exp_checksum,
        "levels": {},
    }
    for level in LEVELS:
        level_key = f"O{level}"
        ll_path = compile_strassen(n, level)
        if ll_path is None:
            item["levels"][level_key] = {"compile_ok": False, "detail": "compile to ll failed"}
            continue
        exe_path = EXE_DIR / f"strassen_N{n}_O{level}.exe"
        EXE_DIR.mkdir(parents=True, exist_ok=True)
        ok, detail = compile_exe(clang, vcvars, ll_path, str(exe_path))
        if not ok:
            item["levels"][level_key] = {"compile_ok": False, "detail": detail[:300]}
            continue
        warmup_code, warmup_out, _ = run_exe(str(exe_path))
        if warmup_code != 0:
            item["levels"][level_key] = {
                "compile_ok": True,
                "run_ok": False,
                "detail": f"warmup exit {warmup_code}",
            }
            continue
        try:
            n_val, checksum, c11, cnn = parse_summary(warmup_out)
        except Exception as exc:
            item["levels"][level_key] = {"compile_ok": True, "run_ok": False, "detail": str(exc)}
            continue
        output_ok = (
            n_val == n
            and abs(checksum - exp_checksum) < 0.5
            and abs(c11 - exp_c11) < 1e-3
            and abs(cnn - exp_cnn) < 1e-3
        )
        samples = []
        for _ in range(repeat):
            code, _, elapsed_ms = run_exe(str(exe_path))
            if code != 0:
                samples = []
                break
            samples.append(elapsed_ms)
        if not samples:
            item["levels"][level_key] = {
                "compile_ok": True,
                "run_ok": False,
                "output_ok": output_ok,
                "detail": "benchmark run failed",
            }
            continue
        item["levels"][level_key] = {
            "compile_ok": True,
            "run_ok": True,
            "output_ok": output_ok,
            "median_ms": median(samples),
            "samples_ms": samples,
            "checksum": checksum,
        }
    base = item["levels"].get("O0", {}).get("median_ms")
    for level in (2, 3):
        opt = item["levels"].get(f"O{level}", {}).get("median_ms")
        item["levels"][f"O{level}"]["speedup_vs_O0"] = speedup(base, opt)
    return item


def print_table(results: List[Dict]) -> None:
    print()
    print(f"{'N':>5}  {'Algo':<7}  {'O0 ms':>10}  {'O2 ms':>10}  {'O3 ms':>10}  {'O2/O0':>8}  {'O3/O0':>8}  {'OK':>4}")
    print("-" * 72)
    for row in results:
        lv = row["levels"]
        o0 = lv.get("O0", {})
        o2 = lv.get("O2", {})
        o3 = lv.get("O3", {})
        if not (o0.get("run_ok") and o2.get("run_ok") and o3.get("run_ok")):
            print(f"{row['n']:>5}  {row['algorithm']:<7}  {'ERR':>10}  {'ERR':>10}  {'ERR':>10}")
            continue
        ok = o0.get("output_ok") and o2.get("output_ok") and o3.get("output_ok")
        print(
            f"{row['n']:>5}  {row['algorithm']:<7}  "
            f"{o0['median_ms']:>10.1f}  {o2['median_ms']:>10.1f}  {o3['median_ms']:>10.1f}  "
            f"{o2.get('speedup_vs_O0', 0):>7.2f}x  {o3.get('speedup_vs_O0', 0):>7.2f}x  "
            f"{'yes' if ok else 'no':>4}"
        )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark Strassen/blocked matmul Fortran program")
    parser.add_argument("--sizes", "-s", nargs="+", type=int, default=DEFAULT_SIZES)
    parser.add_argument("--repeat", "-r", type=int, default=3)
    parser.add_argument("--json", "-j", default=str(OUT_DIR / "strassen_results.json"))
    args = parser.parse_args()

    clang = resolve_clang()
    vcvars = resolve_vcvars()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"clang: {clang}")
    print(f"sizes: {args.sizes}, repeat: {args.repeat}")
    print("note: arrays are declared 1024x1024; N controls multiply work only")

    results = []
    for n in args.sizes:
        print(f"  N={n}...", end=" ", flush=True)
        row = bench_size(n, args.repeat, clang, vcvars)
        results.append(row)
        lv = row["levels"]
        parts = []
        for level in LEVELS:
            key = f"O{level}"
            entry = lv.get(key, {})
            if entry.get("run_ok"):
                sp = entry.get("speedup_vs_O0")
                sp_txt = f"{sp:.2f}x" if sp is not None else "-"
                parts.append(f"{key}:{entry['median_ms']:.0f}ms({sp_txt})")
            else:
                parts.append(f"{key}:ERR")
        print("  ".join(parts))

    print_table(results)
    payload = {
        "sizes": args.sizes,
        "repeat": args.repeat,
        "results": results,
    }
    Path(args.json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nJSON: {args.json}")


if __name__ == "__main__":
    main()
