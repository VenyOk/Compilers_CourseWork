import os
import sys
import subprocess
import time

sys.path.insert(0, os.path.dirname(__file__))

from src.core import Lexer, Parser
from src.semantic import SemanticAnalyzer
from src.ssa_generator import SSAGenerator
from src.llvm_generator import LLVMGenerator

EXPECTED_OUTPUTS = {
    "test_simple_arithmetic.f": "15\n5\n",
    "test_real_numbers.f": "5.85\n",
    "test_arithmetic_mixed.f": "8.5\n",
    "test_multiplication.f": "180\n",
    "test_exponential.f": "150.25\n",
    "test_power.f": "8\n",
    "test_mod.f": "2\n",
    "test_min_max.f": "5\n15\n",
    "test_operator_precedence.f": "14\n26\n-2\n",
    "test_all_operators.f": "37\n1\n1\n0\n",
    "test_complex_expressions.f": "44\n6.25\n0\n0\n0\n",
    "test_mixed_types.f": "22\n0\n",
    "test_logical_ops.f": "1\n",
    "test_comparison_ops.f": "0\n1\n1\n0\n",
    "test_eqv_neqv.f": "0\n1\n",
    "test_logical_complex.f": "1\n0\n1\n1\n1\n",
    "test_unary_ops.f": "-10\n10\n-5.5\n",
    "test_if_statement.f": "1\n",
    "test_do_loop.f": "55\n",
    "test_do_while.f": "55\n",
    "test_labeled_do.f": "15\n",
    "test_goto.f": "55\n",
    "test_continue.f": "50\n",
    "test_conditional_loops.f": "40\n120\n",
    "test_nested_loops.f": "90\n",
    "test_fibonacci.f": "0\n1\n1\n2\n3\n5\n8\n13\n21\n34\n",
    "test_prime_check.f": "2\n3\n5\n7\n11\n13\n17\n19\n8\n",
    "test_stop.f": "10\n",
    "test_array.f": "10\n",
    "test_array_operations.f": "20\n2\n20\n",
    "test_mixed_arrays.f": "3\n6\n2.5\n",
    "test_matrix_multiply.f": "52\n",
    "test_character.f": "Hello\n",
    "test_string_operations.f": "Hello\nWorld\nTest\n",
    "test_string_concat.f": "HelloWorld\n",
    "test_complex_type.f": "(1, 2)\n",
    "test_write.f": "10\n20\n30\n",
    "test_read.f": "30\n",
    "test_function_def.f": "15\n",
    "test_int_float.f": "3\n2\n3\n2\n",
    "test_functions.f": "2\n-1.41045\n",
    "test_all_math.f": "1.90331\n2.03444\n",
    "test_math_functions.f": "2\n-1.41045\n20\n",
    "test_subroutine.f": "10\n5\n",
    "test_multiple_subroutines.f": "11\n22\n8\n242\n242\n",
    "test_strassen_simple.f": "Matrix A:\n1\n2\n3\n4\nMatrix B:\n5\n6\n7\n8\nResult C = A * B (expected: 19, 22, 43, 50):\n19\n22\n43\n50\n",
    "test_strassen.f": None,
    "test_arithmetic_if.f": "0\n",
    "test_dimension.f": "10\n100\n",
    "test_nested_arrays.f": "1\n12\n",
    "test_parameter.f": "75\n",
    "test_implicit.f": "30\n6\n",
    "test_bubble_sort.f": "1\n2\n3\n4\n5\n6\n7\n8\n9\n10\n",
    "test_newton_sqrt.f": "1.41421\n3\n",
    "test_factorial.f": "3628800\n479001600\n",
    "test_matrix_transpose.f": "1\n4\n7\n2\n5\n8\n3\n6\n9\n",
    "test_collatz.f": "111\n16\n",
    "test_gcd.f": "6\n25\n1\n",
    "test_number_classify.f": "10\n10\n6\n8\n",
    "test_quadratic.f": "3\n2\n1\n",
    "test_complex_arithmetic.f": "(4, 2)\n(2, 6)\n(11, -2)\n(-3, -4)\n(8, 4)\n(-1, 0)\n",
    "test_double_precision.f": "3.14159\n11.1111\n1e-012\n0.333333\n5.18738\n1.41421\n1024\n",
    "test_func_const_arg.f": "10\n12\n23\n",
}


def compile_fortran(fpath, output_dir):
    base = os.path.splitext(os.path.basename(fpath))[0]
    with open(fpath, 'r', encoding='utf-8') as f:
        code = f.read()

    lexer = Lexer(code)
    tokens = lexer.tokenize()

    parser = Parser(tokens)
    ast = parser.parse()

    semantic = SemanticAnalyzer()
    success = semantic.analyze(ast)
    if not success:
        return (None, None, f"Semantic errors: {semantic.get_errors()}")

    ssa_gen = SSAGenerator()
    ssa_instructions = ssa_gen.generate(ast)
    ssa_str = ssa_gen.to_string(ssa_instructions)
    ssa_path = os.path.join(output_dir, f"{base}.ssa")
    with open(ssa_path, 'w', encoding='utf-8') as f:
        f.write(ssa_str)

    llvm_gen = LLVMGenerator()
    ir_code = llvm_gen.generate(ast)
    ll_path = os.path.join(output_dir, f"{base}.ll")
    with open(ll_path, 'w', encoding='utf-8') as f:
        f.write(ir_code)

    return (ir_code, ll_path, None)


def run_llvm_ir(ll_path, timeout=15):
    try:
        result = subprocess.run(
            [sys.executable, 'test_llvmlite_run.py', ll_path],
            capture_output=True, text=True, timeout=timeout,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        return (result.returncode, result.stdout, result.stderr)
    except subprocess.TimeoutExpired:
        return (-1, "", "TIMEOUT")


def compare_output(actual, expected):
    if expected is None:
        return (True, "unchecked")

    actual_clean = actual.strip()
    expected_clean = expected.strip()

    if actual_clean == expected_clean:
        return (True, "exact match")

    actual_lines = actual_clean.split('\n')
    expected_lines = expected_clean.split('\n')

    if len(actual_lines) != len(expected_lines):
        return (False, f"line count: got {len(actual_lines)}, expected {len(expected_lines)}")

    for i, (a, e) in enumerate(zip(actual_lines, expected_lines)):
        if a.strip() == e.strip():
            continue
        try:
            if abs(float(a.strip()) - float(e.strip())) < 1e-4:
                continue
        except ValueError:
            pass
        return (False, f"line {i+1}: got '{a.strip()}', expected '{e.strip()}'")

    return (True, "match (float tolerance)")


def main():
    inputs_dir = 'inputs'
    output_dir = 'outputs'
    os.makedirs(output_dir, exist_ok=True)

    files = sorted([f for f in os.listdir(inputs_dir) if f.endswith('.f')])

    print("=" * 100)
    print(f"FULL-CYCLE TESTING: {len(files)} Fortran files")
    print("Pipeline: .f -> Lexer -> Parser -> Semantic -> SSA (.ssa) -> LLVM IR (.ll) -> JIT -> Verify")
    print(f"Output directory: {os.path.abspath(output_dir)}")
    print("=" * 100)
    print()

    stats = {
        'compile_ok': 0, 'compile_fail': 0,
        'run_ok': 0, 'run_fail': 0,
        'output_match': 0, 'output_mismatch': 0, 'output_unchecked': 0
    }
    failures = []
    start_time = time.time()

    for fname in files:
        fpath = os.path.join(inputs_dir, fname)
        expected = EXPECTED_OUTPUTS.get(fname)
        line = f"  {fname:<40}"

        try:
            ir_code, ll_path, error = compile_fortran(fpath, output_dir)
        except Exception as e:
            stats['compile_fail'] += 1
            failures.append(f"[COMPILE FAIL] {fname}: {type(e).__name__}: {e}")
            print(f"{line} COMPILE FAIL  ({type(e).__name__}: {str(e)[:60]})")
            continue

        if ir_code is None:
            stats['compile_fail'] += 1
            failures.append(f"[COMPILE FAIL] {fname}: {error}")
            print(f"{line} COMPILE FAIL  ({str(error)[:60]})")
            continue

        stats['compile_ok'] += 1

        try:
            exit_code, stdout, stderr = run_llvm_ir(ll_path)
        except Exception as e:
            stats['run_fail'] += 1
            failures.append(f"[RUN FAIL] {fname}: {type(e).__name__}: {e}")
            print(f"{line} COMPILE OK -> RUN FAIL  ({type(e).__name__}: {str(e)[:50]})")
            continue

        if stderr and ("error" in stderr.lower() or "traceback" in stderr.lower()):
            stats['run_fail'] += 1
            stderr_short = stderr.strip().split('\n')[-1][:80]
            failures.append(f"[RUN FAIL] {fname}: {stderr_short}")
            print(f"{line} COMPILE OK -> RUN FAIL  ({stderr_short[:50]})")
            continue

        if exit_code == -1:
            stats['run_fail'] += 1
            failures.append(f"[TIMEOUT] {fname}")
            print(f"{line} COMPILE OK -> TIMEOUT")
            continue

        stats['run_ok'] += 1

        match, details = compare_output(stdout, expected)
        if expected is None:
            stats['output_unchecked'] += 1
            out_preview = stdout.strip().replace('\n', ' | ')[:40]
            print(f"{line} COMPILE OK -> RUN OK    (output: {out_preview})")
        elif match:
            stats['output_match'] += 1
            print(f"{line} COMPILE OK -> RUN OK -> OUTPUT MATCH")
        else:
            stats['output_mismatch'] += 1
            failures.append(f"[OUTPUT MISMATCH] {fname}: {details}")
            actual_preview = stdout.strip().replace('\n', ' | ')[:40]
            expected_preview = expected.strip().replace('\n', ' | ')[:40]
            print(f"{line} COMPILE OK -> RUN OK -> OUTPUT MISMATCH")
            print(f"{'':44}expected: {expected_preview}")
            print(f"{'':44}actual:   {actual_preview}")

    elapsed = time.time() - start_time

    print()
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"  Total files:         {len(files)}")
    print(f"  Compile OK:          {stats['compile_ok']}")
    print(f"  Compile FAIL:        {stats['compile_fail']}")
    print(f"  Run OK:              {stats['run_ok']}")
    print(f"  Run FAIL:            {stats['run_fail']}")
    print(f"  Output MATCH:        {stats['output_match']}")
    print(f"  Output MISMATCH:     {stats['output_mismatch']}")
    print(f"  Output unchecked:    {stats['output_unchecked']}")
    print(f"  Time:                {elapsed:.1f}s")
    print(f"  Output files:        {output_dir}/*.ll, {output_dir}/*.ssa")
    print()

    if failures:
        print("FAILURES:")
        print("-" * 100)
        for f in failures:
            print(f"  {f}")
        print()

    return 0 if not failures else 1


if __name__ == '__main__':
    sys.exit(main())
