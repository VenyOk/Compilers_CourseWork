import llvmlite.binding as llvm
import ctypes
import sys
import os

for fn in [llvm.initialize_native_target, llvm.initialize_native_asmprinter,
           llvm.initialize_all_targets, llvm.initialize_all_asmprinters]:
    try:
        fn()
    except Exception:
        pass


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_llvmlite_run.py <path_to_ll_file>", file=sys.stderr)
        sys.exit(1)

    ll_path = sys.argv[1]
    with open(ll_path, 'r', encoding='utf-8') as f:
        ir_code = f.read()

    mod = llvm.parse_assembly(ir_code)
    mod.verify()

    target = llvm.Target.from_default_triple()
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
    
    sys.stdout.flush()
    
    result = cfunc()
    
    if sys.platform == 'win32':
        ctypes.cdll.msvcrt.fflush(None)
    
    sys.exit(result)


if __name__ == '__main__':
    main()
