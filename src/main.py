import argparse
import sys
from src.core import Lexer, Parser, pretty_print_ast
from src.semantic import SemanticAnalyzer
from src.ssa_generator import SSAGenerator
from src.llvm_generator import LLVMGenerator


def analyze_file(file_path: str, ssa_output: str = None, llvm_output: str = None, show_ast: bool = False) -> int:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
    except FileNotFoundError:
        print(f"Ошибка: файл '{file_path}' не найден", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Ошибка чтения файла: {e}", file=sys.stderr)
        return 1
    try:
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        lexer_errors = lexer.get_errors()
        if lexer_errors:
            print("=" * 80)
            print("Ошибки лексера:")
            print("=" * 80)
            for i, error in enumerate(lexer_errors, 1):
                print(f"  {i}. {error}")
            print()
        parser = Parser(tokens)
        ast = parser.parse()
        if show_ast:
            print("AST:")
            print("=" * 80)
            print(pretty_print_ast(ast))
            print()
        print("Семантический анализ:")
        print("=" * 80)
        semantic = SemanticAnalyzer()
        success = semantic.analyze(ast)
        errors = semantic.get_errors()
        warnings = semantic.get_warnings()
        if errors:
            print("ОШИБКИ:")
            for i, error in enumerate(errors, 1):
                print(f"  {i}. {error}")
            print()
        if warnings:
            print("ПРЕДУПРЕЖДЕНИЯ:")
            for i, warning in enumerate(warnings, 1):
                print(f"  {i}. {warning}")
            print()
        if success and not warnings:
            print("[OK] Семантический анализ завершен успешно, ошибок и предупреждений нет.")
        elif success:
            print("[OK] Семантический анализ завершен успешно, но есть предупреждения.")
        else:
            print("[ОШИБКА] Семантический анализ выявил ошибки.")
        if success:
            if ssa_output:
                try:
                    ssa_gen = SSAGenerator()
                    ssa_instructions = ssa_gen.generate(ast)
                    ssa_str = ssa_gen.to_string(ssa_instructions)
                    with open(ssa_output, 'w', encoding='utf-8') as f:
                        f.write(ssa_str)
                    print(f"[OK] SSA форма записана в '{ssa_output}'")
                except Exception as e:
                    print(f"Ошибка при генерации SSA: {e}", file=sys.stderr)
                    return 1
            if llvm_output:
                try:
                    llvm_gen = LLVMGenerator()
                    llvm_code = llvm_gen.generate(ast)
                    with open(llvm_output, 'w', encoding='utf-8') as f:
                        f.write(llvm_code)
                    print(f"[OK] LLVM IR записан в '{llvm_output}'")
                except Exception as e:
                    print(
                        f"Ошибка при генерации LLVM IR: {e}", file=sys.stderr)
                    return 1
        return 0 if success else 1
    except SyntaxError as e:
        print(f"Синтаксическая ошибка: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Ошибка компиляции: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


def main():
    parser = argparse.ArgumentParser(
        description="Анализатор Fortran кода - выводит AST и результаты семантической проверки",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  %(prog)s -f program.f
  %(prog)s -f inputs/example1_simple_program.f
        """
    )
    parser.add_argument(
        '-f', '--file',
        required=True,
        help='Путь к .f файлу для анализа'
    )
    parser.add_argument(
        '-s', '--ssa',
        dest='ssa_output',
        help='Путь к выходному файлу для SSA формы (.s)'
    )
    parser.add_argument(
        '-l', '--llvm',
        dest='llvm_output',
        help='Путь к выходному файлу для LLVM IR (.ll)'
    )
    parser.add_argument(
        '-a', '--ast',
        dest='show_ast',
        action='store_true',
        help='Выводить AST дерево'
    )
    args = parser.parse_args()
    if not args.file.endswith('.f'):
        print(
            f"Предупреждение: файл '{args.file}' не имеет расширения .f", file=sys.stderr)
    return analyze_file(args.file, args.ssa_output, args.llvm_output, args.show_ast)


if __name__ == '__main__':
    sys.exit(main())
