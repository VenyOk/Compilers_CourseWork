"""Тайлинг (блокировка) гнёзд циклов (Loop Tiling / Loop Blocking).

Делит итерационное пространство на прямоугольные блоки (тайлы) размером T,
чтобы рабочий набор данных для одного тайла помещался в L1-кэш.

Ключевая идея (Метелица 2024, стр. 83–86):
    Не завершая вычисления итерации t, начинать t+1 пока в кэше
    находятся нужные данные.

Трансформация для 2D-гнезда DO I / DO J с размером тайла T:

    ! До:
    DO I = 1, N
        DO J = 1, N
            u(I, J) = ...
        ENDDO
    ENDDO

    ! После (T = tile_size):
    DO IT = 1, N, T
        DO JT = 1, N, T
            DO I = IT, MIN(IT+T-1, N)
                DO J = JT, MIN(JT+T-1, N)
                    u(I, J) = ...
                ENDDO
            ENDDO
        ENDDO
    ENDDO

Формула оптимального T (Метелица 2024, стр. 86):
    d1 = d2 ≤ √(|L1| + 4) − 2
    Для L1 = 32 КБ, double (8 байт): T_opt ≤ 62

Параметры:
    tile_size — размер тайла (по умолчанию 32, проверен в статье для i7-9700)
    min_depth — минимальная глубина гнезда для тайлинга (по умолчанию 2)
"""
from __future__ import annotations
import math
from typing import List, Optional
from dataclasses import replace as dc_replace

from src.core import (
    Program, Statement, Expression,
    DoLoop, LabeledDoLoop,
    Variable, IntegerLiteral, FunctionCall, BinaryOp,
)
from src.optimizations.base import ASTOptimizationPass
from src.optimizations.loop_analysis import LoopNest, LoopInfo, _build_nest


# ---------------------------------------------------------------------------
# Формула оптимального размера тайла (Метелица 2024)
# ---------------------------------------------------------------------------

def optimal_tile_size(l1_bytes: int = 32 * 1024, elem_size: int = 8) -> int:
    """Вычислить оптимальный размер тайла по формуле из статьи.

    d1 = d2 ≤ √(|L1| / elem_size + 4) − 2

    Args:
        l1_bytes:  объём L1-кэша в байтах (по умолчанию 32 КБ)
        elem_size: размер элемента в байтах (по умолчанию 8 для double)
    """
    n_elems = l1_bytes // elem_size
    t = int(math.sqrt(n_elems + 4)) - 2
    return max(t, 2)


# ---------------------------------------------------------------------------
# AST-узлы для выражений MIN и MAX (Fortran intrinsic)
# ---------------------------------------------------------------------------

def _min_expr(a: Expression, b: Expression) -> Expression:
    """Построить MIN(a, b) как FunctionCall."""
    return FunctionCall(name='MIN', args=[a, b],
                        line=a.line, col=a.col)


def _add_expr(a: Expression, b: int) -> Expression:
    """a + b (целые)."""
    if b == 0:
        return a
    return BinaryOp(left=a, op='+',
                    right=IntegerLiteral(value=b, line=a.line, col=a.col),
                    line=a.line, col=a.col)


def _sub_expr(a: Expression, b: int) -> Expression:
    """a - b (целые)."""
    if b == 0:
        return a
    return BinaryOp(left=a, op='-',
                    right=IntegerLiteral(value=b, line=a.line, col=a.col),
                    line=a.line, col=a.col)


# ---------------------------------------------------------------------------
# Трансформация гнезда
# ---------------------------------------------------------------------------

def _tile_perfect_nest(nest: LoopNest, tile_size: int) -> Statement:
    """Применить тайлинг к совершенному гнезду, вернуть новый внешний цикл.

    Результат: 2*depth вложенных циклов (сначала все «tile», потом все «point»).
    """
    loops = nest.loops
    body = nest.body
    n = len(loops)
    line = loops[0].node.line
    col = loops[0].node.col

    T = IntegerLiteral(value=tile_size, line=line, col=col)

    # Строим изнутри наружу
    # «Point» циклы: DO I = IT, MIN(IT+T-1, N), 1
    point_body = body
    for k in range(n - 1, -1, -1):
        li = loops[k]
        tile_var = f"{li.var}T"
        tile_var_expr = Variable(name=tile_var, line=line, col=col)

        upper = _min_expr(
            _add_expr(tile_var_expr, tile_size - 1),
            li.end
        )
        inner_loop = DoLoop(
            var=li.var,
            start=tile_var_expr,
            end=upper,
            step=li.step,
            body=point_body,
            stmt_label=None,
            line=line,
            col=col,
        )
        point_body = [inner_loop]

    # «Tile» циклы: DO IT = 1, N, T
    tile_body = point_body
    for k in range(n - 1, -1, -1):
        li = loops[k]
        tile_var = f"{li.var}T"
        outer_loop = DoLoop(
            var=tile_var,
            start=li.start,
            end=li.end,
            step=T,
            body=tile_body,
            stmt_label=None,
            line=line,
            col=col,
        )
        tile_body = [outer_loop]

    return tile_body[0]


def _try_tile(loop: Statement, tile_size: int, min_depth: int,
              counter: List[int]) -> Statement:
    """Попытаться применить тайлинг к циклу и его вложенным циклам."""
    if not isinstance(loop, (DoLoop, LabeledDoLoop)):
        return loop

    nest = _build_nest(loop)

    if nest.depth >= min_depth:
        # Применяем тайлинг к верхним min_depth уровням
        # (если глубже — тайлинг уже применяется рекурсивно)
        tiled = _tile_perfect_nest(nest, tile_size)
        counter[0] += 1
        return tiled

    # Гнездо недостаточно глубокое — спуститься внутрь
    new_body = [_try_tile(s, tile_size, min_depth, counter) for s in loop.body]
    return dc_replace(loop, body=new_body)


def _collect_tile_vars(stmts: List[Statement], result: set) -> None:
    """Собрать имена переменных-тайлов (заканчивающихся на T) из DoLoop."""
    for stmt in stmts:
        if isinstance(stmt, (DoLoop, LabeledDoLoop)):
            if stmt.var.endswith('T') and len(stmt.var) >= 2:
                result.add(stmt.var)
            _collect_tile_vars(stmt.body, result)


def _process_stmts(stmts: List[Statement], tile_size: int,
                   min_depth: int, counter: List[int]) -> List[Statement]:
    result = []
    for stmt in stmts:
        if isinstance(stmt, (DoLoop, LabeledDoLoop)):
            result.append(_try_tile(stmt, tile_size, min_depth, counter))
        else:
            result.append(stmt)
    return result


# ---------------------------------------------------------------------------
# Оптимизационный проход
# ---------------------------------------------------------------------------

class LoopTiling(ASTOptimizationPass):
    """Разбивает гнёзда циклов на тайлы для улучшения кэш-локальности.

    Применяется к «совершенным» гнёздам глубиной ≥ min_depth.
    Размер тайла вычисляется по формуле Метелицы или задаётся явно.

    Параметры конструктора:
        tile_size (int): размер тайла; если None — вычисляется автоматически
        min_depth (int): минимальная глубина гнезда (по умолчанию 2)
        l1_bytes (int): объём L1-кэша для формулы (по умолчанию 32 КБ)
    """

    name = "LoopTiling"

    def __init__(self, tile_size: Optional[int] = None,
                 min_depth: int = 2, l1_bytes: int = 32 * 1024):
        super().__init__()
        if tile_size is None:
            self.tile_size = optimal_tile_size(l1_bytes)
        else:
            self.tile_size = tile_size
        self.min_depth = min_depth

    def run(self, program: Program) -> Program:
        from src.core import Declaration
        counter = [0]
        new_stmts = _process_stmts(program.statements, self.tile_size,
                                    self.min_depth, counter)
        new_subs = [
            dc_replace(s, statements=_process_stmts(
                s.statements, self.tile_size, self.min_depth, counter))
            for s in program.subroutines
        ]
        new_funcs = [
            dc_replace(f, statements=_process_stmts(
                f.statements, self.tile_size, self.min_depth, counter))
            for f in program.functions
        ]

        # Добавить объявления для новых переменных тайлов (IT, JT, ...)
        tile_vars: set = set()
        _collect_tile_vars(new_stmts, tile_vars)
        existing_names: set = {
            name for decl in program.declarations
            if isinstance(decl, Declaration)
            for name, _ in decl.names
        }
        new_tile_vars = [v for v in sorted(tile_vars) if v not in existing_names]
        new_decls = list(program.declarations)
        if new_tile_vars:
            tile_decl = Declaration(
                type='INTEGER',
                names=[(v, None) for v in new_tile_vars],
                line=0, col=0,
            )
            new_decls = [tile_decl] + new_decls

        self.stats = {"tiled": counter[0], "tile_size": self.tile_size}
        return dc_replace(program, statements=new_stmts,
                          declarations=new_decls,
                          subroutines=new_subs, functions=new_funcs)
