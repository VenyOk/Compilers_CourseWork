"""Анализ гнёзд циклов и информационных зависимостей.

Реализует базовые алгоритмы из теории оптимизации гнёзд циклов
(Метелица 2024, Wolf & Lam 1991, Banerjee 1987).

Основные сущности:
    LoopNest — описание гнезда циклов (список петель с их параметрами)
    ArrayAccess — обращение к массиву внутри гнезда
    DependenceVector — вектор расстояний между двумя обращениями
    LoopNestAnalyzer — анализатор, извлекающий гнёзда из AST

Используется в: loop_interchange.py, loop_tiling.py, loop_skewing.py
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Set, Dict
from src.core import (
    Statement, Expression, Assignment,
    DoLoop, LabeledDoLoop,
    Variable, BinaryOp, UnaryOp, IntegerLiteral, ArrayRef,
)


# ---------------------------------------------------------------------------
# Структуры данных
# ---------------------------------------------------------------------------

@dataclass
class LoopInfo:
    """Информация об одном цикле в гнезде."""
    var: str                     # переменная цикла (напр. 'I')
    start: Expression            # нижняя граница
    end: Expression              # верхняя граница
    step: Expression             # шаг
    node: Statement              # исходный AST-узел


@dataclass
class LoopNest:
    """Гнездо вложенных циклов.

    loops[0] — самый внешний цикл
    loops[-1] — самый внутренний цикл
    body — операторы самого внутреннего цикла
    """
    loops: List[LoopInfo]
    body: List[Statement]
    depth: int = field(init=False)

    def __post_init__(self):
        self.depth = len(self.loops)

    @property
    def vars(self) -> List[str]:
        return [li.var for li in self.loops]


@dataclass
class LinearIndex:
    """Линейная форма индексного выражения: coeff * var + const.

    Для многомерных массивов у каждого измерения своя LinearIndex.
    """
    var: Optional[str]   # переменная цикла (None если константа)
    coeff: int           # коэффициент при переменной
    const: int           # константная часть


@dataclass
class ArrayAccess:
    """Обращение к массиву внутри гнезда."""
    array_name: str
    indices: List[LinearIndex]
    is_write: bool       # True = запись (генератор), False = чтение


@dataclass
class DependenceVector:
    """Вектор расстояний информационной зависимости между двумя обращениями.

    distances[i] — расстояние по i-й переменной гнезда (может быть None = ∞/unknown)
    Положительное расстояние — прямая зависимость (допускает тайлинг).
    Отрицательное — обратная зависимость (требует скашивания).
    """
    distances: List[Optional[int]]
    source: ArrayAccess
    sink: ArrayAccess
    is_loop_independent: bool = False

    def has_negative(self) -> bool:
        """Есть ли отрицательные компоненты (препятствующие тайлингу)?"""
        return any(d is not None and d < 0 for d in self.distances)

    def all_non_negative(self) -> bool:
        """Все компоненты ≥ 0 (тайлинг допустим)?"""
        return all(d is None or d >= 0 for d in self.distances)


# ---------------------------------------------------------------------------
# Вспомогательные функции: разбор индексных выражений
# ---------------------------------------------------------------------------

def _parse_linear(expr: Expression, loop_vars: Set[str]) -> Optional[LinearIndex]:
    """Попытаться разобрать выражение как a*var + b.

    Возвращает LinearIndex или None если не удалось.
    """
    if isinstance(expr, IntegerLiteral):
        return LinearIndex(var=None, coeff=0, const=expr.value)

    if isinstance(expr, Variable):
        if expr.name in loop_vars:
            return LinearIndex(var=expr.name, coeff=1, const=0)
        return None  # внешняя переменная — неизвестна

    if isinstance(expr, BinaryOp):
        op = expr.op

        if op == '+':
            left = _parse_linear(expr.left, loop_vars)
            right = _parse_linear(expr.right, loop_vars)
            if left is not None and right is not None:
                if left.var is None and right.var is None:
                    return LinearIndex(var=None, coeff=0, const=left.const + right.const)
                if left.var is not None and right.var is None:
                    return LinearIndex(var=left.var, coeff=left.coeff, const=left.const + right.const)
                if left.var is None and right.var is not None:
                    return LinearIndex(var=right.var, coeff=right.coeff, const=left.const + right.const)

        if op == '-':
            left = _parse_linear(expr.left, loop_vars)
            right = _parse_linear(expr.right, loop_vars)
            if left is not None and right is not None:
                if left.var is None and right.var is None:
                    return LinearIndex(var=None, coeff=0, const=left.const - right.const)
                if right.var is None:
                    return LinearIndex(var=left.var, coeff=left.coeff, const=left.const - right.const)
                if left.var is None:
                    return LinearIndex(var=right.var, coeff=-right.coeff, const=left.const - right.const)

        if op == '*':
            left = _parse_linear(expr.left, loop_vars)
            right = _parse_linear(expr.right, loop_vars)
            if left is not None and right is not None:
                if left.var is None and right.var is None:
                    return LinearIndex(var=None, coeff=0, const=left.const * right.const)
                # c * var  или  var * c
                if left.var is None and right.var is not None:
                    return LinearIndex(var=right.var, coeff=left.const * right.coeff, const=0)
                if left.var is not None and right.var is None:
                    return LinearIndex(var=left.var, coeff=left.coeff * right.const, const=0)

    if isinstance(expr, UnaryOp) and expr.op == '-':
        inner = _parse_linear(expr.operand, loop_vars)
        if inner is not None:
            return LinearIndex(var=inner.var, coeff=-inner.coeff, const=-inner.const)

    return None


# ---------------------------------------------------------------------------
# Извлечение обращений к массивам из AST
# ---------------------------------------------------------------------------

def _collect_accesses(stmts: List[Statement], loop_vars: Set[str]) -> List[ArrayAccess]:
    """Собрать все обращения к массивам в теле цикла."""
    accesses: List[ArrayAccess] = []
    for stmt in stmts:
        _collect_in_stmt(stmt, loop_vars, accesses)
    return accesses


def _collect_in_expr(expr: Expression, loop_vars: Set[str],
                     accesses: List[ArrayAccess], is_write: bool = False) -> None:
    if isinstance(expr, ArrayRef):
        parsed = [_parse_linear(i, loop_vars) for i in expr.indices]
        if all(p is not None for p in parsed):
            accesses.append(ArrayAccess(
                array_name=expr.name,
                indices=parsed,
                is_write=is_write
            ))
        # Рекурсивно для вложенных выражений в индексах
        for i in expr.indices:
            _collect_in_expr(i, loop_vars, accesses, False)
    elif isinstance(expr, BinaryOp):
        _collect_in_expr(expr.left, loop_vars, accesses)
        _collect_in_expr(expr.right, loop_vars, accesses)
    elif isinstance(expr, UnaryOp):
        _collect_in_expr(expr.operand, loop_vars, accesses)
    elif isinstance(expr, Variable):
        pass


def _collect_in_stmt(stmt: Statement, loop_vars: Set[str],
                     accesses: List[ArrayAccess]) -> None:
    if isinstance(stmt, Assignment):
        if stmt.indices:
            # LHS — запись в массив
            parsed = [_parse_linear(i, loop_vars) for i in stmt.indices]
            if all(p is not None for p in parsed):
                accesses.append(ArrayAccess(
                    array_name=stmt.target,
                    indices=parsed,
                    is_write=True
                ))
        # RHS — всегда чтение
        _collect_in_expr(stmt.value, loop_vars, accesses, False)
        for i in stmt.indices:
            _collect_in_expr(i, loop_vars, accesses, False)
    elif isinstance(stmt, (DoLoop, LabeledDoLoop)):
        new_vars = loop_vars | {stmt.var}
        _collect_in_stmt_list(stmt.body, new_vars, accesses)
    elif isinstance(stmt, DoLoop):
        pass


def _collect_in_stmt_list(stmts, loop_vars, accesses):
    for s in stmts:
        _collect_in_stmt(s, loop_vars, accesses)


# ---------------------------------------------------------------------------
# Анализ зависимостей (тест Банержи — упрощённый)
# ---------------------------------------------------------------------------

def compute_dependence_vectors(nest: LoopNest) -> List[DependenceVector]:
    """Вычислить векторы расстояний зависимостей для гнезда циклов.

    Использует упрощённый тест: для двух линейных обращений A[a*i+c] и A[b*i+d]
    расстояние = (d - c) / a  (если a == b и делится нацело).
    """
    loop_vars = set(nest.vars)
    accesses = _collect_accesses(nest.body, loop_vars)

    deps: List[DependenceVector] = []

    for i, src in enumerate(accesses):
        for j, snk in enumerate(accesses):
            if i == j:
                continue
            # Зависимость только если хотя бы одно — запись
            if not src.is_write and not snk.is_write:
                continue
            if src.array_name != snk.array_name:
                continue
            if len(src.indices) != len(snk.indices):
                continue

            distances = _compute_distances(src, snk, nest.vars)
            if distances is not None:
                is_indep = all(d == 0 for d in distances)
                deps.append(DependenceVector(
                    distances=distances,
                    source=src,
                    sink=snk,
                    is_loop_independent=is_indep,
                ))
    return deps


def _compute_distances(src: ArrayAccess, snk: ArrayAccess,
                        loop_vars: List[str]) -> Optional[List[Optional[int]]]:
    """Вычислить вектор расстояний между двумя обращениями.

    Возвращает список расстояний по каждой переменной цикла или None.
    """
    n_dims = len(src.indices)
    if n_dims != len(snk.indices):
        return None

    var_to_idx = {v: i for i, v in enumerate(loop_vars)}
    distances = [None] * len(loop_vars)

    for dim in range(n_dims):
        li_src = src.indices[dim]
        li_snk = snk.indices[dim]

        # Оба — константы: расстояние 0 (независимость по данному измерению)
        if li_src.var is None and li_snk.var is None:
            continue

        # Оба используют одну переменную цикла с одинаковым коэффициентом
        if (li_src.var is not None and li_src.var == li_snk.var
                and li_src.coeff == li_snk.coeff and li_src.coeff != 0):
            d = li_snk.const - li_src.const
            var_idx = var_to_idx.get(li_src.var)
            if var_idx is not None:
                if distances[var_idx] is None:
                    distances[var_idx] = d
                elif distances[var_idx] != d:
                    return None  # противоречие

        # Разные переменные или коэффициенты — не можем точно определить
    return distances


# ---------------------------------------------------------------------------
# Извлечение гнёзд из AST
# ---------------------------------------------------------------------------

def extract_loop_nests(stmts: List[Statement]) -> List[Tuple[List[Statement], LoopNest, int]]:
    """Найти все «совершенные» гнёзда циклов в списке операторов.

    Возвращает список (prefix_stmts, nest, stmt_index) где:
        prefix_stmts — операторы перед гнездом
        nest         — описание гнезда
        stmt_index   — индекс цикла-корня в stmts
    """
    results = []
    for i, stmt in enumerate(stmts):
        if isinstance(stmt, (DoLoop, LabeledDoLoop)):
            nest = _build_nest(stmt)
            if nest.depth >= 1:
                results.append((stmts[:i], nest, i))
        # Рекурсия в тела условий
        elif isinstance(stmt, DoLoop):
            pass
    return results


def _build_nest(loop: Statement) -> LoopNest:
    """Построить LoopNest, разворачивая цепочку вложенных DO-циклов."""
    loops: List[LoopInfo] = []
    current = loop

    while isinstance(current, (DoLoop, LabeledDoLoop)):
        li = LoopInfo(
            var=current.var,
            start=current.start,
            end=current.end,
            step=current.step,
            node=current,
        )
        loops.append(li)
        body = current.body
        # Цикл является «совершенным» (perfect nest) если в теле ровно один цикл
        inner_loops = [s for s in body if isinstance(s, (DoLoop, LabeledDoLoop))]
        non_loop = [s for s in body if not isinstance(s, (DoLoop, LabeledDoLoop))]

        if len(inner_loops) == 1 and len(non_loop) == 0:
            current = inner_loops[0]
        else:
            # Достигли «дна» гнезда
            return LoopNest(loops=loops, body=body)

    return LoopNest(loops=loops, body=[])


# ---------------------------------------------------------------------------
# Утилиты для проверки условий применимости преобразований
# ---------------------------------------------------------------------------

def can_interchange(nest: LoopNest) -> bool:
    """Можно ли переставить два внешних цикла гнезда без нарушения зависимостей?

    Перестановка допустима, если у всех зависимостей расстояния по обеим
    переменным ≥ 0 после перестановки. По теореме: перестановка I,J → J,I
    допустима, если все векторы вида (d_i, d_j) имеют d_j ≥ 0 или
    (d_j == 0 и d_i > 0).

    Упрощённо: допустима если все векторы расстояний ≥ 0 по обоим направлениям.
    """
    if nest.depth < 2:
        return False
    deps = compute_dependence_vectors(nest)
    for dep in deps:
        if dep.has_negative():
            return False
    return True


def needs_skewing(nest: LoopNest) -> bool:
    """Нужно ли скашивание (есть ли отрицательные зависимости)?"""
    deps = compute_dependence_vectors(nest)
    return any(dep.has_negative() for dep in deps)


def get_skew_factors(nest: LoopNest) -> List[int]:
    """Вычислить минимальные коэффициенты скашивания для каждой переменной.

    Возвращает список коэффициентов [s_0, s_1, ..., s_{n-1}] такой, что
    после скашивания J' = J + s_i * I все векторы расстояний станут
    неотрицательными.

    Стратегия: для каждой пары (i, j) выбрать s = max(0, -d_j/d_i) + 1
    если d_j < 0.
    """
    deps = compute_dependence_vectors(nest)
    n = nest.depth
    factors = [0] * n

    for dep in deps:
        for k in range(1, n):
            d_k = dep.distances[k]
            if d_k is not None and d_k < 0:
                # Ищем минимальный s такой, что d_k + s * d_{k-1} >= 0
                d_prev = dep.distances[k - 1]
                if d_prev is not None and d_prev > 0:
                    s = (-d_k + d_prev - 1) // d_prev
                else:
                    s = abs(d_k) + 1
                factors[k] = max(factors[k], s)

    return factors
