import argparse
import sys
from src.core import Lexer, Parser, pretty_print_ast
from src.semantic import SemanticAnalyzer
def analyze_file(file_path: str) -> int:
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
        print("=" * 80)
        print("AST (Abstract Syntax Tree):")
        print("=" * 80)
        print(pretty_print_ast(ast))
        print()
        print("=" * 80)
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
            print("[ERROR] Семантический анализ выявил ошибки.")
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
    args = parser.parse_args()
    if not args.file.endswith('.f'):
        print(f"Предупреждение: файл '{args.file}' не имеет расширения .f", file=sys.stderr)
    return analyze_file(args.file)
if __name__ == '__main__':
    sys.exit(main())
