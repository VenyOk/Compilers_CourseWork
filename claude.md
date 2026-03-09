# Claude.md — База знаний проекта: Оптимизирующий компилятор Fortran

## Описание проекта

**Тема диплома**: Оптимизирующий компилятор языка Fortran 77 с реализацией оптимизаций гнёзд циклов итерационного типа.

**Научная база**: Статья Метелицы Е.А. «Обоснование методов ускорения гнёзд циклов итерационного типа» // Программные системы: теория и приложения. 2024. Т.15. №1. С.63–94. DOI: 10.25209/2079-3316-2024-15-1-63-94

**Язык реализации**: Python 3.x  
**Бэкенд**: LLVM IR через llvmlite  
**Цель**: Fortran 77 → AST → (оптимизации) → LLVM IR → JIT-исполнение

---

## Структура проекта

```
Compilers_CourseWork/
├── src/
│   ├── core.py                  # Лексер + Парсер + AST-узлы
│   ├── semantic.py              # Семантический анализ
│   ├── ssa_generator.py         # Генерация текстового SSA IR (для отладки)
│   ├── llvm_generator.py        # Генерация LLVM IR из AST
│   ├── main.py                  # CLI точка входа
│   ├── __init__.py
│   └── optimizations/           # Все оптимизационные проходы
│       ├── __init__.py
│       ├── base.py              # Абстрактный ASTOptimizationPass
│       ├── pipeline.py          # OptimizationPipeline (O0/O1/O2)
│       ├── loop_analysis.py     # Анализ зависимостей и гнёзд циклов
│       ├── constant_folding.py  # Свёртка констант в выражениях
│       ├── constant_propagation.py  # Распространение констант
│       ├── algebraic_simplification.py  # Алгебраические упрощения
│       ├── dead_code_elimination.py  # Удаление мёртвого кода
│       ├── strength_reduction.py  # Снижение стоимости операций
│       ├── cse.py               # Устранение общих подвыражений
│       ├── licm.py              # Вынос инвариантов из циклов
│       ├── loop_interchange.py  # Перестановка циклов
│       ├── loop_tiling.py       # Тайлинг (блокировка) циклов
│       └── loop_skewing.py      # Скашивание циклов
├── inputs/                      # Тестовые .f программы
├── outputs/                     # Сгенерированные .ll и .ssa файлы
├── tests/                       # pytest тесты
│   ├── test_lexer_parser.py
│   ├── test_semantic.py
│   ├── test_ssa.py
│   └── test_llvm.py
├── run_all_tests.py             # Полный тест-раннер с JIT-исполнением
├── test_llvmlite_run.py         # JIT-запускатор .ll файлов
├── bench_runner.py              # Бенчмарк-раннер (O0 vs O1 vs O2)
├── requirements.txt
├── GRAMMAR.txt                  # Формальная грамматика Fortran 77
└── claude.md                    # Этот файл
```

---

## Пайплайн компилятора

```
Fortran .f
    │
    ▼
[Lexer] core.py → tokens
    │
    ▼
[Parser] core.py → Program AST
    │
    ▼
[SemanticAnalyzer] semantic.py → проверка типов, объявлений
    │
    ▼
[OptimizationPipeline] src/optimizations/ → трансформированный AST
    │  (только если -O1 или -O2)
    ▼
[LLVMGenerator] llvm_generator.py → .ll LLVM IR текст
    │
    ▼
[llvmlite JIT] test_llvmlite_run.py → исполнение
```

**Важно**: `SSAGenerator` генерирует отдельный текстовый SSA IR для отладки/отображения, но **не используется** LLVMGenerator. Оба работают независимо от AST.

---

## Справочник AST-узлов (src/core.py)

### Корневой узел
```python
@dataclass
class Program(ASTNode):
    name: str
    declarations: List[Declaration]   # объявления переменных
    statements: List[Statement]       # тело программы
    statement_functions: List         # функции-операторы
    subroutines: List[Subroutine]    # подпрограммы
    functions: List[FunctionDef]     # функции
```

### Ключевые узлы для оптимизаций

#### Циклы
```python
@dataclass
class DoLoop(Statement):
    var: str            # имя переменной цикла (напр. "I")
    start: Expression   # начало (напр. IntegerLiteral(1))
    end: Expression     # конец (напр. Variable("N"))
    step: Expression    # шаг (IntegerLiteral(1) по умолчанию)
    body: List[Statement]

@dataclass
class LabeledDoLoop(Statement):
    label: str          # метка для DO 10 I = 1, N
    var: str
    start: Expression
    end: Expression
    step: Expression
    body: List[Statement]

@dataclass
class DoWhile(Statement):
    condition: Expression
    body: List[Statement]
```

#### Присваивание
```python
@dataclass
class Assignment(Statement):
    target: str              # имя переменной-цели
    value: Expression        # правая часть
    indices: List[Expression]  # индексы массива (пусто для скаляров)
    # A(I,J) = expr → target="A", indices=[Variable("I"), Variable("J")]
```

#### Выражения
```python
@dataclass
class BinaryOp(Expression):
    left: Expression
    op: str    # '+', '-', '*', '/', '**', '.EQ.', '.AND.', '//', ...
    right: Expression

@dataclass
class UnaryOp(Expression):
    op: str    # '-', '+', '.NOT.'
    operand: Expression

@dataclass
class Variable(Expression):
    name: str   # имя переменной (как в исходнике)

@dataclass
class ArrayRef(Expression):
    name: str              # имя массива
    indices: List[Expression]

@dataclass
class IntegerLiteral(Expression):
    value: int

@dataclass
class RealLiteral(Expression):
    value: float

@dataclass
class FunctionCall(Expression):
    name: str
    args: List[Expression]
```

#### Условия и управление
```python
@dataclass
class IfStatement(Statement):
    condition: Expression
    then_body: List[Statement]
    elif_parts: List[Tuple[Expression, List[Statement]]]
    else_body: List[Statement]

@dataclass
class GotoStatement(Statement):
    label: str

@dataclass
class ContinueStatement(Statement):
    label: Optional[str]
```

### Объявления
```python
@dataclass
class Declaration(ASTNode):
    type: str       # 'INTEGER', 'REAL', 'LOGICAL', 'CHARACTER', 'COMPLEX', 'DOUBLEPRECISION'
    names: List[Tuple[str, Optional[List]]]  # [(имя, [размерности])]
    type_size: Optional[str]

@dataclass
class DimensionStatement(ASTNode):
    names: List[Tuple[str, List]]   # [(имя, [размерности])]
    # размерность: int или (lo, hi) tuple
```

---

## Оптимизационные проходы

### Базовый класс (src/optimizations/base.py)

```python
class ASTOptimizationPass(ABC):
    name: str           # имя для отчётов
    stats: dict         # статистика применений

    @abstractmethod
    def run(self, program: Program) -> Program: ...

    def transform_expr(self, expr: Expression) -> Expression:
        """Рекурсивный обход выражения — переопределить при необходимости."""
        ...

    def transform_stmt(self, stmt: Statement) -> Statement:
        """Рекурсивный обход оператора."""
        ...

    def transform_stmts(self, stmts: List[Statement]) -> List[Statement]:
        """Обход списка операторов с возможностью вставки/удаления."""
        ...
```

### Уровни оптимизации (pipeline.py)

| Флаг | Проходы |
|------|---------|
| `-O0` | без оптимизаций |
| `-O1` | ConstantFolding, ConstantPropagation, AlgebraicSimplification, DeadCodeElimination, StrengthReduction |
| `-O2` | O1 + CSE, LICM, LoopInterchange, LoopTiling |
| `-O3` | O2 + LoopSkewing (для стенсильных гнёзд) |

---

## Алгоритмы оптимизации гнёзд циклов (по Метелице 2024)

### 1. Вынос инвариантных выражений (LICM)
**Файл**: `licm.py`

Выражение является **инвариантом цикла**, если ни одна переменная в нём не изменяется в теле цикла. Такое выражение можно вычислить один раз перед циклом.

```fortran
! До LICM:
DO I = 1, 1000
    S = S + (A * B + C) * FLOAT(I)   ! A*B+C — инвариант
ENDDO

! После LICM:
_inv1 = A * B + C
DO I = 1, 1000
    S = S + _inv1 * FLOAT(I)
ENDDO
```

**Алгоритм**:
1. Собрать множество `modified_vars` — переменные, присваиваемые в теле
2. Для каждого выражения в теле: если все переменные ∉ `modified_vars` → вынести
3. Рекурсивно для вложенных циклов

### 2. Устранение общих подвыражений (CSE)
**Файл**: `cse.py`

Если одно и то же выражение вычисляется несколько раз в блоке (между переопределениями переменных), заменить повторные вычисления на временную переменную.

```fortran
! До CSE:
S1 = S1 + (A * B + C)
S2 = S2 + (A * B + C) * 2.0  ! A*B+C вычислен дважды

! После CSE:
_cse1 = A * B + C
S1 = S1 + _cse1
S2 = S2 + _cse1 * 2.0
```

### 3. Перестановка циклов (Loop Interchange)
**Файл**: `loop_interchange.py`

В Fortran массивы хранятся в **column-major** порядке (первый индекс меняется быстрее). Для максимальной кэш-локальности самый внутренний цикл должен проходить по первому индексу.

```fortran
! До (плохой порядок для cache):
DO I = 1, N
    DO J = 1, M
        A(I, J) = B(I, J) + 1.0   ! каждый шаг J — страничный промах
    ENDDO
ENDDO

! После interchange (хорошо для cache):
DO J = 1, M
    DO I = 1, N
        A(I, J) = B(I, J) + 1.0   ! каждый шаг I — соседняя ячейка
    ENDDO
ENDDO
```

**Условие корректности**: перестановка допустима если векторы расстояний зависимостей не имеют отрицательных компонент после перестановки.

### 4. Тайлинг (Loop Tiling / Blocking)
**Файл**: `loop_tiling.py`

Делит итерационное пространство на блоки (тайлы) размером `T×T`, чтобы данные тайла помещались в L1 кэш.

```fortran
! До тайлинга:
DO I = 1, N
    DO J = 1, N
        u(I, J) = ...
    ENDDO
ENDDO

! После тайлинга с размером T:
DO IT = 1, N, T
    DO JT = 1, N, T
        DO I = IT, MIN(IT+T-1, N)
            DO J = JT, MIN(JT+T-1, N)
                u(I, J) = ...
            ENDDO
        ENDDO
    ENDDO
ENDDO
```

**Формула оптимального T** (из статьи Метелицы, стр. 86):
```
d1 = d2 ≤ √(|L1| + 4) − 2
```
Для L1 = 32 КБ = 4096 чисел double: `T ≤ 62`

Реализуется через атрибут прохода `tile_size` (по умолчанию 32 или 64).

### 5. Скашивание (Loop Skewing)
**Файл**: `loop_skewing.py`

Применяется к **итерационным стенсилам** (алгоритм Гаусса-Зейделя), где зависимости имеют отрицательные компоненты вектора расстояний. Скашивание делает все векторы неотрицательными, что разрешает тайлинг.

```fortran
! Стенсил (зависимость u[i+1][j] → отрицательная!):
DO I = 1, N
    DO J = 1, N
        u(I,J) = (u(I-1,J) + u(I+1,J) + u(I,J-1) + u(I,J+1)) / 4.0
    ENDDO
ENDDO

! После скашивания с матрицей skew=(1,0; 1,1):
! Замена: J' = I + J  →  J = J' - I
DO I = 1, N
    DO Jp = I+1, I+N    ! J' = I + J
        u(I, Jp-I) = (u(I-1,Jp-I) + u(I+1,Jp-I) + u(I,Jp-I-1) + u(I,Jp-I+1)) / 4.0
    ENDDO
ENDDO
```

**Теорема 1 (Метелица)**: После скашивания с фактором 1 и применения метода гиперплоскостей с вектором нормали (1,1,...,1), распараллеливание тайлов на одной гиперплоскости эквивалентно исходной программе.

---

## Ключевые концепции теории зависимостей

### Информационная зависимость
Между вхождениями A и B существует зависимость, если они обращаются к **одной ячейке памяти** и хотя бы одно — генератор (запись).

### Вектор расстояний (Distance Vector)
Для двух обращений `A[f(i)]` и `A[g(i)]` в цикле с переменной `i`: вектор расстояний `d = g(i) - f(i)`.
- `d > 0` — прямая (forward) зависимость: допускает тайлинг
- `d = 0` — цикло-независимая зависимость
- `d < 0` — обратная (backward) зависимость: **требует скашивания** перед тайлингом

### Носитель зависимости (Carrier)
Цикл, создающий циклически-порождённую зависимость. Нумерация от внешнего (0) к внутреннему.

---

## Команды запуска

```bash
# Компиляция без оптимизаций (O0)
python -m src.main -f inputs/bench_combined.f -l outputs/bench_combined.ll

# Компиляция с базовыми оптимизациями (O1)
python -m src.main -f inputs/bench_licm.f -l outputs/bench_licm.ll -O 1

# Компиляция с полными оптимизациями (O2)
python -m src.main -f inputs/bench_licm.f -l outputs/bench_licm.ll -O 2

# Запуск всех тестов
python run_all_tests.py

# Запуск конкретного .ll файла
python test_llvmlite_run.py outputs/bench_licm.ll

# Бенчмарк (O0 vs O1 vs O2)
python bench_runner.py

# pytest тесты
pytest tests/
```

---

## Как добавить новый оптимизационный проход

1. Создать файл `src/optimizations/my_pass.py`
2. Унаследовать от `ASTOptimizationPass`
3. Реализовать `run(self, program: Program) -> Program`
4. Добавить в `pipeline.py` в нужный уровень (`O1_PASSES` или `O2_PASSES`)

```python
from src.optimizations.base import ASTOptimizationPass
from src.core import Program

class MyOptimization(ASTOptimizationPass):
    name = "MyOptimization"

    def run(self, program: Program) -> Program:
        # трансформация AST...
        self.stats = {"applied": count}
        return program
```

---

## Тестирование оптимизаций

Каждый проход должен:
1. **Не менять поведение**: все 79 тестов из `run_all_tests.py` должны проходить с `-O1` и `-O2`
2. **Корректно считать статистику**: заполнять `self.stats` для отчёта pipeline
3. **Быть идемпотентным**: двойное применение = однократное (для большинства проходов)

Тест-файлы специально для оптимизаций находятся в `inputs/bench_*.f`.

---

## Зависимости

```
llvmlite>=0.46.0    # JIT-исполнение LLVM IR
pytest              # тестирование
```

---

## Git-стратегия

```
main        — стабильный baseline (только компилятор без оптимизаций)
develop     — интеграционная ветка всех оптимизаций
  feature/opt-infrastructure      — инфраструктура (base, pipeline, -O флаг)
  feature/basic-optimizations     — O1: folding, propagation, DCE, strength
  feature/licm-cse                — O2 базовые: LICM и CSE
  feature/loop-analysis           — анализ зависимостей
  feature/loop-interchange-tiling — перестановка и тайлинг
  feature/loop-skewing            — скашивание для стенсилов
  feature/benchmarks              — бенчмарки и таблицы ускорений
```

Теги: `v0.1-basic-opt`, `v0.2-licm-cse`, `v1.0-loop-nests`
