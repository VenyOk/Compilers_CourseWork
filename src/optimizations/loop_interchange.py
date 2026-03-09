"""Перестановка циклов (Loop Interchange).

Меняет порядок двух внешних циклов в гнезде для улучшения кэш-локальности.

В Fortran массивы хранятся в column-major порядке (первый индекс меняется
быстрее всего). Оптимальный порядок для A(I,J): самый внутренний цикл —
по I (первый индекс).

Пример:
    ! Плохо для кэша (каждый шаг J — страничный промах):
    DO I = 1, N
        DO J = 1, M
            A(I, J) = B(I, J) + 1.0
        ENDDO
    ENDDO

    →  (после interchange)

    DO J = 1, M
        DO I = 1, N
            A(I, J) = B(I, J) + 1.0   ! соседние ячейки памяти
        ENDDO
    ENDDO

Условие применимости (теорема Wolf & Lam 1991):
    Перестановка допустима если все векторы расстояний зависимостей
    неотрицательны по обоим индексам.

Примечание: проход консервативен — пропускает гнёзда с отрицательными
зависимостями (они требуют сначала скашивания).
"""
from __future__ import annotations
from typing import List
from dataclasses import replace as dc_replace

from src.core import (
    Program, Statement,
    DoLoop, LabeledDoLoop,
)
from src.optimizations.base import ASTOptimizationPass
from src.optimizations.loop_analysis import (
    LoopNest, LoopInfo, can_interchange,
)


def _interchange_nest(outer: Statement, inner: Statement) -> Statement:
    """Переставить два вложенных цикла: внешний ↔ внутренний.

    Возвращает новый внешний цикл с переставленными параметрами.
    """
    # Построить новое гнездо: заголовок внешнего берём из inner, inner берём из outer
    # Тело остаётся прежним (самого внутреннего)
    inner_cls = type(inner)
    outer_cls = type(outer)

    # Тело внутреннего цикла (не является самим inner)
    if isinstance(inner, (DoLoop, LabeledDoLoop)):
        inner_body = inner.body
        inner_step = inner.step
    else:
        return outer  # не DO-цикл — не переставляем

    if isinstance(outer, (DoLoop, LabeledDoLoop)):
        outer_step = outer.step
    else:
        return outer

    # Новый внутренний цикл: заголовок старого outer, тело inner_body
    new_inner = dc_replace(outer,
                           start=outer.start, end=outer.end, step=outer_step,
                           body=inner_body)

    # Новый внешний цикл: заголовок старого inner, тело = [new_inner]
    new_outer = dc_replace(inner,
                           start=inner.start, end=inner.end, step=inner_step,
                           body=[new_inner])
    return new_outer


def _try_interchange(loop: Statement) -> Statement:
    """Попытаться переставить цикл с его первым вложенным циклом."""
    if not isinstance(loop, (DoLoop, LabeledDoLoop)):
        return loop

    body = loop.body
    # «Совершенное» гнездо: ровно один вложенный цикл, остальных операторов нет
    inner_loops = [s for s in body if isinstance(s, (DoLoop, LabeledDoLoop))]
    non_loops = [s for s in body if not isinstance(s, (DoLoop, LabeledDoLoop))]

    if len(inner_loops) != 1 or len(non_loops) != 0:
        # Не совершенное гнездо — обработать тело рекурсивно
        new_body = [_try_interchange(s) for s in body]
        if new_body != body:
            return dc_replace(loop, body=new_body)
        return loop

    inner = inner_loops[0]

    # Построить LoopNest для анализа зависимостей
    from src.optimizations.loop_analysis import _build_nest
    nest = _build_nest(loop)

    if nest.depth >= 2 and can_interchange(nest):
        interchanged = _interchange_nest(loop, inner)
        # Рекурсивно обработать внутренние уровни
        if isinstance(interchanged, (DoLoop, LabeledDoLoop)):
            new_body = [_try_interchange(s) for s in interchanged.body]
            return dc_replace(interchanged, body=new_body)
        return interchanged

    # Зависимости не позволяют — обработать рекурсивно внутрь
    new_body = [_try_interchange(s) for s in body]
    if any(nb is not b for nb, b in zip(new_body, body)):
        return dc_replace(loop, body=new_body)
    return loop


def _process_stmts(stmts: List[Statement], counter: List[int]) -> List[Statement]:
    result = []
    changed = False
    for stmt in stmts:
        if isinstance(stmt, (DoLoop, LabeledDoLoop)):
            new_stmt = _try_interchange(stmt)
            if new_stmt is not stmt:
                counter[0] += 1
                changed = True
            result.append(new_stmt)
        else:
            result.append(stmt)
    return result if changed else stmts


class LoopInterchange(ASTOptimizationPass):
    """Переставляет циклы в гнезде для улучшения кэш-локальности.

    Применяет перестановку только если анализ зависимостей подтверждает
    её корректность (все векторы расстояний неотрицательны).
    """

    name = "LoopInterchange"

    def run(self, program: Program) -> Program:
        counter = [0]
        new_stmts = _process_stmts(program.statements, counter)
        new_subs = [
            dc_replace(s, statements=_process_stmts(s.statements, counter))
            for s in program.subroutines
        ]
        new_funcs = [
            dc_replace(f, statements=_process_stmts(f.statements, counter))
            for f in program.functions
        ]
        self.stats = {"interchanged": counter[0]}
        return dc_replace(program, statements=new_stmts,
                          subroutines=new_subs, functions=new_funcs)
