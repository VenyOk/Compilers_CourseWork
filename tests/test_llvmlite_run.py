import sys
import ctypes
import os
from pathlib import Path

if len(sys.argv) != 2:
    print("Usage: python test_llvmlite_run.py <llvm_ir_file>", file=sys.stderr)
    sys.exit(1)

ll_path = Path(sys.argv[1])

try:
    from llvmlite import binding as llvm
except ImportError:
    print("llvmlite not installed, using fallback runtime", file=sys.stderr)
    sys.exit(1)

llvm.initialize()
llvm.initialize_all_targets()
llvm.initialize_native_asmprinter()

ir_code = ll_path.read_text(encoding="utf-8")

try:
    mod = llvm.parse_assembly(ir_code)
    mod.verify()
except RuntimeError as e:
    print(f"LLVM parse error: {e}", file=sys.stderr)
    sys.exit(1)

engine = llvm.create_mcjit_compiler(mod, llvm.Target.from_triple(llvm.get_process_triple()))

runtime_path = Path(__file__).parent.parent / "src" / "runtime" / "libfortran_runtime.so"
if runtime_path.exists():
    ctypes.CDLL(str(runtime_path))

engine.finalize_object()

main_ptr = engine.get_function_address("main")
if not main_ptr:
    print("main function not found", file=sys.stderr)
    sys.exit(1)

main_func = ctypes.CFUNCTYPE(ctypes.c_int32)(main_ptr)
exit_code = main_func()

sys.exit(exit_code)
