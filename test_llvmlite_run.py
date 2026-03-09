import llvmlite.binding as llvm
import ctypes
import sys
import os
import time

for fn in [llvm.initialize_native_target, llvm.initialize_native_asmprinter,
           llvm.initialize_all_targets, llvm.initialize_all_asmprinters]:
    try:
        fn()
    except Exception:
        pass


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_llvmlite_run.py <path_to_ll_file> [--bench N]", file=sys.stderr)
        sys.exit(1)

    ll_path = sys.argv[1]
    bench_runs = 0
    if '--bench' in sys.argv:
        idx = sys.argv.index('--bench')
        bench_runs = int(sys.argv[idx + 1]) if idx + 1 < len(sys.argv) else 50

    with open(ll_path, 'r', encoding='utf-8') as f:
        ir_code = f.read()

    mod = llvm.parse_assembly(ir_code)
    mod.verify()

    no_opt = '--no-opt' in sys.argv

    target = llvm.Target.from_default_triple()
    if no_opt:
        target_machine = target.create_target_machine(opt=0)
    else:
        target_machine = target.create_target_machine()
    backing_mod = llvm.parse_assembly("")
    engine = llvm.create_mcjit_compiler(backing_mod, target_machine)
    engine.add_module(mod)
    engine.finalize_object()
    engine.run_static_constructors()

    func_ptr = engine.get_function_address('main')
    if not func_ptr:
        print("ERROR: Could not find 'main' function", file=sys.stderr)
        sys.exit(2)

    cfunc = ctypes.CFUNCTYPE(ctypes.c_int)(func_ptr)

    if bench_runs > 0:
        cfunc()
        if sys.platform == 'win32':
            ctypes.cdll.msvcrt.fflush(None)
        sys.stdout.flush()

        old_stdout = os.dup(1)
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 1)

        times = []
        for _ in range(bench_runs):
            start = time.perf_counter_ns()
            cfunc()
            elapsed = time.perf_counter_ns() - start
            times.append(elapsed)
            if sys.platform == 'win32':
                ctypes.cdll.msvcrt.fflush(None)

        os.dup2(old_stdout, 1)
        os.close(devnull)
        os.close(old_stdout)

        times.sort()
        median_ns = times[len(times) // 2]
        mean_ns = sum(times) // len(times)
        min_ns = times[0]
        max_ns = times[-1]

        print(f"Runs:   {bench_runs}")
        print(f"Median: {median_ns / 1_000_000:.3f} ms")
        print(f"Mean:   {mean_ns / 1_000_000:.3f} ms")
        print(f"Min:    {min_ns / 1_000_000:.3f} ms")
        print(f"Max:    {max_ns / 1_000_000:.3f} ms")
        sys.exit(0)

    sys.stdout.flush()

    start = time.perf_counter_ns()
    result = cfunc()
    elapsed = time.perf_counter_ns() - start

    if sys.platform == 'win32':
        ctypes.cdll.msvcrt.fflush(None)

    print(f"[Time: {elapsed / 1_000_000:.3f} ms]", file=sys.stderr)

    sys.exit(result)


if __name__ == '__main__':
    main()
