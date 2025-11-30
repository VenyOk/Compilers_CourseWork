# Компилятор Fortran

Запуск компилятора:

```bash
python -m src.main -f <путь_к_файлу.f>
```

Примеры:
```bash
python -m src.main -f inputs/input.f
python -m src.main -f program.f
```

Генерация SSA формы:
```bash
python -m src.main -f program.f -s output.ssa
```

Генерация LLVM IR:
```bash
python -m src.main -f program.f -l output.ll
```

Генерация SSA и LLVM одновременно:
```bash
python -m src.main -f program.f -s output.ssa -l output.ll
```
