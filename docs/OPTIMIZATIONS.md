# Оптимизации компилятора (`src/optimizations/`)

Документ описывает, как устроен pipeline оптимизаций, за что отвечает каждый модуль и что делает каждая функция.

## Общая идея

Оптимизации работают **на AST** (абстрактном синтаксическом дереве), а не на LLVM IR. Каждый проход — класс, наследующий `ASTOptimizationPass`: получает `Program`, возвращает изменённый `Program`, пишет статистику в `self.stats`.

Запуск из `src/main.py`:

```bash
python -m src.main -f program.f -O2 -l out.ll
python -m src.main -f program.f -O3 -l out.ll
```

```mermaid
flowchart LR
    AST[Program AST] --> P[OptimizationPipeline]
    P --> AST2[Изменённый AST]
    AST2 --> LLVM[LLVMGenerator]
```

---

## Pipeline: порядок проходов

Файл: `pipeline.py`

### O0

Оптимизации не применяются.

### O2

```
StrengthReduction
→ LoopInvariantCodeMotion
→ CommonSubexpressionElimination
→ LoopInterchange
→ LoopTiling
→ AffineLinearization
→ DeadCodeElimination
→ GeneratedVariableDeclarations
```

### O3 (pipeline по статье Метелицы)

```
StrengthReduction
→ LoopInterchange
→ LoopSkewing
→ LoopTiling
→ IntraTileLoopInterchange
→ AffineLinearization
→ LoopHeaderPeel
→ LoopInvariantCodeMotion
→ CommonSubexpressionElimination
→ DeadCodeElimination
→ GeneratedVariableDeclarations
```

| Уровень | Акцент |
|---------|--------|
| O2 | Классические локальные и цикловые оптимизации + тайлинг |
| O3 | Skewing, side slice, intra-tile interchange, peel — для stencil/GS/matmul |

---

## Краткие примеры трансформаций

Ниже — упрощённые фрагменты «до → после» для каждого прохода. В реальном AST могут появляться дополнительные служебные переменные и `MIN`/`MAX` в границах.

| Проход | Что меняется |
|--------|--------------|
| StrengthReduction | `**` → умножения |
| LICM | инвариантное выражение выносится перед циклом |
| CSE | повторяющееся выражение вычисляется один раз |
| DCE | удаляется неиспользуемый tmp |
| LoopInterchange | меняется порядок двух `DO` |
| LoopSkewing | новые счётчики `skew_*`, сдвинутые индексы |
| LoopTiling | добавляются `tile_*` и point-циклы |
| IntraTileLoopInterchange | меняется порядок осей внутри тайла |
| LoopHeaderPeel | цикл делится на steady + epilog |
| AffineLinearization | упрощаются аффинные индексы |
| GeneratedVariableDeclarations | добавляются `INTEGER`/`REAL` для tmp |

---

## `base.py` — базовый класс прохода

### Класс `ASTOptimizationPass`

| Метод | Назначение |
|-------|------------|
| `run(program)` | Абстрактный метод: применить проход к программе |
| `transformExpr(expr)` | Рекурсивно обходит выражение (`BinaryOp`, `UnaryOp`, `FunctionCall`, `ArrayRef`) |
| `transformStmt(stmt)` | Рекурсивно обходит оператор (присваивание, циклы, `IF`) |
| `transformStmts(stmts)` | Применяет `transformStmt` к списку операторов |

---

## `pipeline.py` — оркестратор

| Функция / класс | Назначение |
|-----------------|------------|
| `buildPasses(level)` | Возвращает список классов проходов для уровня 0, 2 или 3 |
| `OptimizationPipeline.__init__` | Сохраняет уровень и список классов проходов |
| `OptimizationPipeline.run(program)` | Последовательно создаёт экземпляры проходов и прогоняет AST |
| `OptimizationPipeline.recordStats(p)` | Накапливает `stats` каждого прохода |
| `OptimizationPipeline.report()` | Формирует текстовый отчёт для консоли |

---

## `strength_reduction.py` — упрощение силы операций

**Проход:** `StrengthReduction`

Заменяет `X ** 2`, `X ** 3`, `X ** 4` на цепочку умножений.

**Пример:**

```fortran
! До
Y = X ** 3

! После
Y = X * X * X
```

| Функция | Назначение |
|---------|------------|
| `isSmallPosInt(expr)` | Проверяет, что степень — целое от 2 до 4 |
| `expandPower(base, n, line, col)` | Строит `base * base * ...` (n раз) |
| `reduceExpr(expr, counter)` | Рекурсивно упрощает выражение |
| `reduceStmt(stmt, counter)` | Упрощает выражения внутри оператора |
| `processStmts(stmts, counter)` | Обрабатывает список операторов |
| `StrengthReduction.run` | Прогоняет по program, subroutines, functions; stats: `reduced` |

---

## `licm.py` — Loop-Invariant Code Motion

**Проход:** `LoopInvariantCodeMotion`

Выносит из тела цикла выражения, не зависящие от счётчика (особенно `REAL`-арифметику и вызовы функций). Создаёт временные `licm_tmp_N`.

**Пример:**

```fortran
! До
DO I = 1, N
    Y(I) = A * SIN(X) + FLOAT(I)
ENDDO

! После
licm_tmp_1 = A * SIN(X)
DO I = 1, N
    Y(I) = licm_tmp_1 + FLOAT(I)
ENDDO
```

| Функция | Назначение |
|---------|------------|
| `collectModified(stmts, result)` | Собирает переменные, изменяемые в теле (включая счётчик цикла) |
| `usesModified(expr, modified)` | Проверяет, использует ли выражение изменяемую переменную |
| `isTrivialExpr(expr)` | Литерал или простая переменная |
| `containsReal(expr)` | Есть ли в выражении вещественная семантика |
| `worthHoisting(expr)` | Стоит ли выносить (операции над `REAL`, вызовы функций) |
| `exprRepr(expr)` | Каноническая строка выражения для кэша |
| `ExprHoister.hoistExpr` | Заменяет инвариантное выражение на `licm_tmp_N` |
| `ExprHoister.hoistStmt` | Обходит оператор, вынося инварианты |
| `processLoop(loop, counter)` | Формирует preheader с присваиваниями tmp и новое тело цикла |
| `processStmts(stmts, counter)` | Рекурсивно обрабатывает вложенные циклы и `IF` |
| `LoopInvariantCodeMotion.run` | stats: `hoisted` |

---

## `cse.py` — Common Subexpression Elimination

**Проход:** `CommonSubexpressionElimination`

Находит повторяющиеся чистые подвыражения в линейном блоке, заменяет на `cse_tmp_N`.

**Пример:**

```fortran
! До
A = X + Y
B = X + Y
C = A + 1

! После
cse_tmp_1 = X + Y
A = cse_tmp_1
B = cse_tmp_1
C = A + 1
```

| Функция | Назначение |
|---------|------------|
| `exprKey(expr)` | Ключ для сравнения выражений (коммутативность для `+`, `*`) |
| `isPure(expr)` | Выражение без побочных эффектов |
| `isTrivial(expr)` | Литерал или переменная |
| `varsInExpr(expr, result)` | Собирает имена переменных в выражении |
| `containsArrayRef(expr)` | Есть ли обращение к массиву |
| `CSEBlock.invalidate(varName)` | Удаляет из кэша записи, зависящие от изменённой переменной |
| `CSEBlock.subst(expr, allow_cache)` | Подставляет кэшированное tmp или создаёт новое |
| `CSEBlock.process(stmt)` | Обрабатывает один оператор, возвращает `[новые assign] + [stmt]` |
| `applyCseToStmts(stmts, counter)` | Прогоняет CSE по списку операторов |
| `CommonSubexpressionElimination.run` | stats: `cse_vars` |

---

## `dce.py` — Dead Code Elimination

**Проход:** `DeadCodeElimination`

Удаляет неиспользуемые `cse_tmp_*` и `licm_tmp_*`.

**Пример:**

```fortran
! До
cse_tmp_1 = SIN(X)
A = B + C

! После (cse_tmp_1 нигде не читается)
A = B + C
```

| Функция | Назначение |
|---------|------------|
| `gatherExpr(expr, out)` | Собирает используемые имена из выражения |
| `gatherStmts(stmts, out)` | Собирает live-переменные по всем операторам |
| `isDeadGenerated(stmt, live)` | Присваивание в generated tmp, которое нигде не читается |
| `filterStmts(stmts, live, counter)` | Удаляет мёртвые generated-присваивания |
| `processUnit(stmts, counter)` | Сначала собирает live-set, потом фильтрует |
| `DeadCodeElimination.run` | stats: `eliminated` |

---

## `loop_interchange.py` — перестановка двух циклов

**Проход:** `LoopInterchange`

Меняет местами два идеально вложенных `DO`, если это легально и выгодно.

**Пример (matmul, stride-1 по внутреннему циклу):**

```fortran
! До
DO I = 1, N
    DO J = 1, N
        C(I,J) = C(I,J) + A(I,K) * B(K,J)
    ENDDO
ENDDO

! После (I и J поменяны местами, если зависимости позволяют)
DO J = 1, N
    DO I = 1, N
        C(I,J) = C(I,J) + A(I,K) * B(K,J)
    ENDDO
ENDDO
```

| Функция | Назначение |
|---------|------------|
| `interchangeNest(outer, inner)` | Физически меняет outer и inner цикл местами в AST |
| `tryInterchange(loop)` | Пробует interchange для гнезда; иначе рекурсия внутрь |
| `processStmts(stmts, counter)` | Обходит все циклы верхнего уровня |
| `LoopInterchange.run` | stats: `interchanged` |

---

## `loop_skewing.py` — скашивание циклов (O3)

**Проход:** `LoopSkewing`

При отрицательных зависимостях переписывает индексы и вводит счётчики `skew_I`.

**Пример (идея, упрощённо):**

```fortran
! До: GS-зависимость мешает переставить циклы
DO I = 2, N
    DO J = 2, N
        U(I,J) = 0.25 * (U(I-1,J) + U(I,J-1) + ...)
    ENDDO
ENDDO

! После: новый счётчик skew_J, индексы переписаны
DO skew_I = 2, N
    DO skew_J = 2, N
        U(skew_I, skew_J - (skew_I-2)) = ...
    ENDDO
ENDDO
```

| Функция | Назначение |
|---------|------------|
| `skewVarName(var)` | Имя skew-счётчика: `skew_I` |
| `isSkewVar(var)` | Проверка префикса `skew_` |
| `intExpr`, `addExpr`, `subExpr`, `mulExprByInt` | Построение AST-выражений |
| `substituteExpr(expr, substitutions)` | Подстановка новых индексов в выражение |
| `substituteStmt(stmt, substitutions)` | Подстановка в оператор |
| `buildSubstitutions(nest, matrix)` | Строит отображение `I → skew_I - f*J - ...` |
| `shiftedBound(expr, nest, matrix, index)` | Сдвигает границы цикла после skew |
| `skewNest(nest, matrix)` | Перестраивает всё гнездо с новыми переменными и границами |
| `trySkew(loop, counter, diagnostics)` | Решает, нужен ли skew; вызывает `skewNest` |
| `processStmts(stmts, counter, diagnostics)` | Рекурсивный обход |
| `LoopSkewing.run` | stats: `skewed`, `diagnostics` |

---

## `loop_tiling.py` — тайлинг

**Проход:** `LoopTiling`

Разбивает каждый цикл на tile-уровень (`tile_I`) и point-уровень (`I`).

**Пример (тайл 32×32):**

```fortran
! До
DO I = 1, 256
    DO J = 1, 256
        C(I,J) = ...
    ENDDO
ENDDO

! После
DO tile_I = 1, 256, 32
    DO tile_J = 1, 256, 32
        DO I = tile_I, MIN(tile_I + 31, 256)
            DO J = tile_J, MIN(tile_J + 31, 256)
                C(I,J) = ...
            ENDDO
        ENDDO
    ENDDO
ENDDO
```

| Функция | Назначение |
|---------|------------|
| `tileVarName(var)` | `tile_I` |
| `isTileVar(var)` | Проверка префикса `tile_` |
| `optimalTileSize(depth, l1Bytes, elemSize)` | Эвристика размера тайла из объёма L1 |
| `workingSetTileSize(depth, nArrays, ...)` | Размер с учётом числа массивов |
| `tileSizesFromEnv()` | Читает `FORTRAN_TILE_SIZES=32,32,...` |
| `intExpr`, `addExpr`, `subExpr`, `negExpr`, `mulExprByInt`, `addInt` | AST-арифметика |
| `minExpr`, `maxExpr` | Обёртки над `MIN`/`MAX` |
| `substituteExpr(expr, substitutions)` | Подстановка в выражения границ |
| `tileSizesForNest(nest, ...)` | Подбор размеров тайла по семейству stencil и trip count |
| `buildBounds(nest, tileSizes)` | Строит границы tile- и point-циклов |
| `tileAffineNest(nest, tileSizes)` | Полная трансформация гнезда; может применить side slice |
| `tileDiagnostic(nest, tile_sizes, side_sliced)` | Запись для отладки/bench |
| `tryTile(loop, ...)` | Решает, тайлить ли гнездо; вызывает `tileAffineNest` |
| `processStmts(stmts, ...)` | Обход всех циклов |
| `LoopTiling.run` | stats: `tiled`, `tile_size`, `diagnostics` |

---

## `loop_intra_tile_interchange.py` — перестановка внутри тайла (O3)

**Проход:** `IntraTileLoopInterchange`

Меняет порядок point-циклов после тайлинга/skewing.

**Пример (после тайлинга, J сделали внешним point-циклом):**

```fortran
! До (внутри тайла)
DO I = tile_I, Iend
    DO J = tile_J, Jend
        C(I,J) = ...
    ENDDO
ENDDO

! После
DO J = tile_J, Jend
    DO I = tile_I, Iend
        C(I,J) = ...
    ENDDO
ENDDO
```

Границы могут стать зависимыми (`MAX`/`MIN`) — это side slice.

| Функция | Назначение |
|---------|------------|
| `wrapLoop(loop_info, body)` | Оборачивает тело в `DoLoop` с заданными границами |
| `pointSubNest(nest, point_start, point_depth)` | Выделяет подгнездо point-циклов |
| `rebuildNest(nest, point_start, point_order)` | Пересобирает гнездо в новом порядке (+ side slice) |
| `tryInterchange(loop, counter, diagnostics)` | Пробует перестановку point-циклов |
| `processStatements(statements, counter, diagnostics)` | Обход операторов |
| `IntraTileLoopInterchange.run` | stats: `interchanged`, `diagnostics` |

---

## `loop_header_peel.py` — peeling хвоста тайла (O3)

**Проход:** `LoopHeaderPeel`

Делит point-цикл на steady-state (полные тайлы) и epilog (остаток).

**Пример (N=100, тайл=32, остаток 4):**

```fortran
! До
DO tile_I = 1, 100, 32
    DO I = tile_I, MIN(tile_I + 31, 100)
        ...
    ENDDO
ENDDO

! После
DO tile_I = 1, 96, 32          ! steady: 96 = 3 полных тайла
    DO I = tile_I, tile_I + 31
        ...
    ENDDO
ENDDO
DO I = 97, 100                  ! epilog: хвост
    ...
ENDDO
```

| Функция | Назначение |
|---------|------------|
| `intValue(expr)` | Извлекает целочисленный литерал |
| `intExpr(value, template)` | Создаёт `IntegerLiteral` |
| `addInt(expr, value)` | Прибавляет константу к выражению |
| `tileSpan(inner, tile_var)` | Ширина point-цикла внутри тайла |
| `literalLower(inner, tile_var)` | Нижняя граница point-цикла (константа) |
| `literalUpper(inner, tile_var)` | Верхняя граница point-цикла |
| `firstDirectPointLoop(body)` | Первый point-цикл (не `tile_`) в теле |
| `peelTilePointPair(outer, inner)` | Разбивает пару tile/point на steady + epilog |
| `peelStmt(stmt, counter)` | Пробует peel для одного цикла |
| `peelStmts(stmts, counter)` | Обход списка операторов |
| `LoopHeaderPeel.run` | stats: `peeled` |

---

## `affine_linearization.py` — упрощение аффинных выражений

**Проход:** `AffineLinearization`

Упрощает индексы и границы после tiling/skewing (`I + J - I` → `J`, `2*I + 3*I` → `5*I`).

**Пример:**

```fortran
! До (после skew/tiling)
DO I = tile_I, MIN(tile_I + 31, I + J - I + 5)

! После
DO I = tile_I, MIN(tile_I + 31, J + 5)
```

Ещё пример в теле:

```fortran
! До
A(I, 2*K + 3*K)

! После
A(I, 5*K)
```

| Функция | Назначение |
|---------|------------|
| `exprKey(expr)` | Ключ для сравнения выражений |
| `intValue`, `realValue` | Извлечение констант |
| `makeInt`, `makeReal` | Создание литералов с сохранением позиции |
| `isZero`, `isOne` | Проверки констант |
| `negate(expr)` | Унарный минус с свёрткой констант |
| `flattenAdd(expr)` | Разворачивает сумму/разность в список слагаемых |
| `combineLikeTerms(terms, const_value, template)` | Объединяет одинаковые переменные |
| `linearizeAddExpr(expr)` | Линеаризует сумму |
| `rebuildAdd(terms, const_value, use_real, template)` | Собирает упрощённую сумму обратно |
| `linearizeExpr(expr)` | Рекурсивное упрощение всего выражения |
| `simplifyFunction(expr)` | Свёртка `MIN`/`MAX` с одинаковыми аргументами |
| `simplifyBinary(expr)` | Свёртка `*`, `/`, `**` на константах |
| `AffineLinearization.hasTransformedLoops` | Работает только если есть `tile_`/`skew_` |
| `AffineLinearization.transformExpr` | Применяет `linearizeExpr` |
| `AffineLinearization.run` | stats: `linearized` |

---

## `generated_declarations.py` — объявления служебных переменных

**Проход:** `GeneratedVariableDeclarations`

Добавляет `INTEGER tile_I, skew_J, cse_tmp_1, ...` в declarations.

**Пример:**

```fortran
! До (в declarations)
IMPLICIT NONE
INTEGER I, J

! После (оптимизатор добавил служебные имена)
IMPLICIT NONE
INTEGER I, J
INTEGER tile_I, tile_J, cse_tmp_1, licm_tmp_1
```

| Функция | Назначение |
|---------|------------|
| `isGeneratedName(name)` | Regex: `cse_tmp_N`, `licm_tmp_N`, `tile_*`, `skew_*` |
| `normalizeType(typeName)` | Приводит тип к INTEGER/REAL/... |
| `buildTypeEnv(declarations)` | Карта имя → тип из объявлений |
| `mergeNumericTypes(left, right)` | Объединение типов для inference |
| `inferExprType(expr, env)` | Вывод типа выражения |
| `collectGeneratedLoopVars(stmt, out)` | Собирает generated счётчики циклов |
| `iterStatements(stmts)` | Итератор по всем операторам рекурсивно |
| `collectGeneratedAssignments(stmts)` | Присваивания в generated-переменные |
| `inferGeneratedTypes(declarations, statements)` | Fixpoint-вывод типов generated vars |
| `existingDeclaredNames(declarations)` | Уже объявленные имена |
| `declarationInsertionIndex(declarations)` | Куда вставлять (после IMPLICIT) |
| `addGeneratedDeclarations(declarations, statements)` | Добавляет `Declaration` блоки |
| `GeneratedVariableDeclarations.run` | stats: `declared_generated` |

---

## `side_slice.py` — боковой срез (helper)

Не проход pipeline. Используется в `loop_tiling` и `loop_intra_tile_interchange`.

При перестановке point-циклов пересчитывает границы через `MAX`/`MIN`, чтобы сохранить корректность.

| Функция | Назначение |
|---------|------------|
| `intExpr`, `addExpr`, `subExpr` | AST-арифметика |
| `maxExpr`, `minExpr` | `FunctionCall` для MIN/MAX |
| `boundCoreOffset(expr, var, loop_vars)` | Извлекает константное смещение в границе |
| `replaceBoundCore(expr, var, new_core, ...)` | Заменяет ядро границы |
| `sideSlicePair(point_infos, ...)` | Side slice для пары циклов |
| `sideSlicePointInfos(point_infos, point_order, ...)` | Перестановка двух point-циклов с новыми границами |
| `applyArticleSideSlice(...)` | Вариант для 2+ циклов (режим статьи) |
| `applySideSliceToPointInfos(...)` | Точка входа: article или обычный режим |
| `buildPointLoops(point_infos, body, line, col)` | Собирает вложенные `DoLoop` из описания границ |
| `desiredPointVarOrder(nest, point_start, point_depth)` | Желаемый порядок переменных |
| `sideSliceOrderApplied(nest, ...)` | Уже применён ли нужный порядок |

---

## `article_core.py` — формулы из статьи (helper)

Реализация алгоритмов Метелицы для O3.

| Функция | Назначение |
|---------|------------|
| `maxStencilOffset(nest)` | Максимальное смещение в stencil-индексах |
| `countInnerAssignments(stmts)` | Число присваиваний в массив в теле |
| `isCanonicalIterativeNest(nest)` | Каноническое iterative-type гнездо |
| `articleSliceVolume(sizes)` | Объём рабочего набора с halo (+2 по оси) |
| `fitTileSize(count, side)` | Подгоняет размер тайла под trip count |
| `articleSkewMatrix(nest, needs_skew)` | Матрица skew по зависимостям |
| `articleTileSizesForNest(nest, ...)` | Подбор размеров тайла под L1 и тип задачи |
| `articlePointOrder(prefix_depth, spatial_depth)` | Рекомендуемый порядок point-циклов |

---

## `loop_analysis.py` — анализ гнёзд циклов (helper)

Самый большой модуль. Не проход, а библиотека решений для interchange/skew/tiling.

### Структуры данных

| Имя | Назначение |
|-----|------------|
| `LoopInfo` | Один цикл: var, start, end, step, ссылка на AST-узел |
| `LoopNest` | Список `LoopInfo` + тело гнезда |
| `AffineExpr` | Аффинное выражение: coeffs + const |
| `ArrayAccess` | Обращение к массиву: имя, индексы, read/write |
| `DependenceVector` | Вектор расстояний между двумя обращениями |

### Парсинг и сбор информации

| Функция | Назначение |
|---------|------------|
| `articleModeEnabled()` | Включён ли режим алгоритмов из статьи |
| `constantInt(expr)` | Целочисленное значение литерала или None |
| `mergeCoeffs(left, right, sign)` | Сложение словарей коэффициентов |
| `parseAffine(expr, loop_vars)` | Разбор `a*I + b*J + c` |
| `isAffineNest(nest)` | Все индексы в гнезде аффинны |
| `collectInExpr` / `collectInStmt` | Сбор обращений к массивам |
| `collectAccesses(stmts, loop_vars)` | Список `ArrayAccess` |
| `exprAffineStatus` / `stmtAffineStatus` | Есть ли доступы, все ли аффинны |
| `accessStatus(stmts, loop_vars)` | Агрегат по телу |
| `hasObservableSideEffects(stmts)` | PRINT, READ, CALL и т.д. |

### Зависимости

| Функция | Назначение |
|---------|------------|
| `coefficientMatrix(access, loop_vars)` | Матрица коэффициентов индексов |
| `solveIntegerSystem(matrix, rhs, nvars)` | Решение системы для distance vector |
| `computeDistances(source, sink, loop_vars)` | Вектор расстояний между двумя access |
| `computeDependenceVectors(nest)` | Все зависимости в гнезде |

### Построение и оценка гнезда

| Функция | Назначение |
|---------|------------|
| `buildNest(loop)` | Из вложенного DO строит `LoopNest` |
| `extractLoopNests(stmts)` | Все гнёзда в списке операторов |
| `estimateTripCount(loop_info)` | Число итераций при константных границах |
| `estimateNestVolume(nest, limit_depth)` | Произведение trip counts |
| `hasArrayAccesses(nest)` | Есть ли обращения к массивам |
| `countArrayAccesses(nest)` | Число обращений |
| `uniqueArrayCount(nest)` | Число уникальных массивов |
| `estimateWorkingSet(nest, bytes_per_element)` | Оценка рабочего набора |
| `estimateTileFootprint(nest)` | Объём данных в тайле |
| `describeNest(nest, ...)` | Сводка для диагностики |

### Классификация stencil

| Функция | Назначение |
|---------|------------|
| `isGeneratedLoopVar(var)` | `tile_` или `skew_` |
| `baseActiveLoopVars(nest)` | Счётчики, реально участвующие в индексах |
| `selfDependentArrays(nest)` | Массивы с read+write in-place |
| `stateCarriedPrefixDepth(nest)` | Глубина «временного» префикса |
| `effectiveStateCarriedPrefixDepth(nest)` | Уточнённая версия с учётом skew |
| `activeLoopVars(nest)` | Все активные счётчики |
| `isStencilLikeNest(nest)` | Похоже на stencil (GS, Laplace, ...) |
| `isCoefficientHeavyStencil(nest)` | Много массивов и access |
| `isSimpleSingleArrayStencil(nest)` | Один массив, in-place |
| `stencilFamily(nest)` | `gauss_seidel`, `dirichlet_gs`, `matmul`, ... |
| `localityScore(accesses, var)` | Оценка locality для interchange |
| `stencilReuseScore(nest)` | Переиспользование данных |
| `axisDependenceScore(nest, var)` | Зависимости по оси |
| `referencedLoopVars(expr, loop_vars)` | Какие счётчики в выражении |

### Решения: interchange, tile, skew

| Функция | Назначение |
|---------|------------|
| `chooseIntraTileLoopOrder(nest)` | Оптимальный порядок point-циклов |
| `preferInterchange(nest)` | Выгодно ли менять два outer цикла |
| `canInterchange(nest)` | Легально ли interchange (зависимости ≥ 0) |
| `tileDecision(nest, tile_size, min_depth)` | (bool, reason) — тайлить или нет |
| `shouldTileNest(nest, tile_size, min_depth)` | Обёртка над tileDecision |
| `needsSkewing(nest)` | Нужен ли skew |
| `skewDecision(nest)` | (bool, reason) |
| `shouldSkewNest(nest)` | Обёртка |
| `getSkewMatrix(nest)` | Матрица коэффициентов skew |
| `getSkewFactors(nest)` | Список факторов |
| `postSkewDependencesLegal(nest)` | Зависимости после skew легальны |
| `transformedDependencesLegal(nest, matrix)` | Проверка для конкретной матрицы |
| `canInterchangePointOrder(nest, point_start, point_order)` | Легальна ли перестановка point-циклов |
| `shouldApplySideSliceInterchange(nest, point_start)` | Нужен ли side slice |

### Структура после трансформаций

| Функция | Назначение |
|---------|------------|
| `dependenceBandDepth(nest)` | Глубина band зависимостей |
| `spatialCarrierDepth(dep, prefix_depth)` | Носитель пространственной зависимости |
| `spatialDependenceBandDepth(nest, prefix_depth)` | Глубина spatial band |
| `prefixLoopDepth(nest, prefix)` | Глубина префикса по имени |
| `isTileLoopVar(var)` | Проверка `tile_` |
| `temporalPrefixDepth(nest)` | Глубина temporal-циклов |
| `isIterativeTypeNest(nest)` | Iterative-type (GS с time-loop) |
| `tileBandStart(nest)` | Индекс начала tile-band |
| `sideSliceBandStart(nest)` | Индекс начала side-slice band |
| `tileBandSpan(nest)` | (start, count) tile-циклов |
| `pointBandSpan(nest)` | (start, depth) point-циклов |
| `pointSkewDepth(nest)` | Число skew-переменных в point-band |
| `estimatePrefixVolume(nest, depth)` | Объём префикса |
| `estimateTransformedWorkingSet(nest, ...)` | WS после трансформаций |
| `articleOptimalTileSide(...)` | Оптимальная сторона тайла из L1 |

---

## Служебные имена, создаваемые оптимизациями

| Префикс | Кто создаёт | Пример |
|---------|-------------|--------|
| `licm_tmp_N` | LICM | `licm_tmp_1 = SIN(X)` |
| `cse_tmp_N` | CSE | `cse_tmp_1 = A + B` |
| `tile_I` | LoopTiling | Внешний счётчик тайла |
| `skew_I` | LoopSkewing | Скашенный счётчик |

Все они объявляются проходом `GeneratedVariableDeclarations`.

---

## Переменные окружения

| Переменная | Модуль | Назначение |
|------------|--------|------------|
| `FORTRAN_TILE_SIZES` | `loop_tiling.py` | Принудительные размеры тайлов, напр. `32,32,32` |

---

## Пример: что происходит с `bench_gs2d.f` на O3

1. **StrengthReduction** — мелкие замены `**`.
2. **LoopInterchange** — возможная перестановка осей.
3. **LoopSkewing** — skew из-за GS-зависимостей → `skew_J`, `skew_I`.
4. **LoopTiling** — `tile_*` + point-циклы, размер ~32.
5. **IntraTileLoopInterchange** — порядок осей внутри тайла + side slice.
6. **AffineLinearization** — упрощение `skew_J - 2*skew_I`.
7. **LoopHeaderPeel** — steady + epilog.
8. **LICM** — вынос инвариантов из point-циклов.
9. **CSE** — общие подвыражения в индексах.
10. **DCE** — удаление лишних tmp.
11. **GeneratedVariableDeclarations** — `INTEGER tile_I, skew_J, ...`.

---

## Связанные файлы вне `optimizations/`

| Файл | Роль |
|------|------|
| `src/main.py` | Запуск pipeline при `-O2`/`-O3` |
| `src/ir/llvm.py` | Генерация LLVM из уже оптимизированного AST |
| `tests/test_optimizations.py` | Unit-тесты каждого прохода |
| `bench_runner.py` | Замеры speedup на `bench_*.f` |
