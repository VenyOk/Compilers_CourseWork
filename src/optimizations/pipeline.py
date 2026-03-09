from __future__ import annotations
from typing import List, Type, TYPE_CHECKING

if TYPE_CHECKING:
    from src.core import Program
    from src.optimizations.base import ASTOptimizationPass


def _build_passes(level: int) -> List[Type["ASTOptimizationPass"]]:
    from src.optimizations.constant_folding import ConstantFolding
    from src.optimizations.constant_propagation import ConstantPropagation
    from src.optimizations.algebraic_simplification import AlgebraicSimplification
    from src.optimizations.dead_code_elimination import DeadCodeElimination
    from src.optimizations.strength_reduction import StrengthReduction
    from src.optimizations.cse import CommonSubexpressionElimination
    from src.optimizations.licm import LoopInvariantCodeMotion
    from src.optimizations.loop_interchange import LoopInterchange
    from src.optimizations.loop_tiling import LoopTiling
    from src.optimizations.loop_skewing import LoopSkewing

    O1 = [
        ConstantFolding,
        ConstantPropagation,
        AlgebraicSimplification,
        StrengthReduction,
        DeadCodeElimination,
    ]

    O2 = O1 + [
        CommonSubexpressionElimination,
        LoopInvariantCodeMotion,
        LoopInterchange,
        LoopTiling,
    ]

    O3 = O2 + [
        LoopSkewing,
        LoopTiling,   # повторный тайлинг после скашивания
    ]

    if level >= 3:
        return O3
    if level >= 2:
        return O2
    if level >= 1:
        return O1
    return []


_REPEATABLE = {
    "ConstantFolding",
    "ConstantPropagation",
    "AlgebraicSimplification",
    "StrengthReduction",
    "DeadCodeElimination",
}
_ONE_SHOT = {
    "CommonSubexpressionElimination",
    "LoopInvariantCodeMotion",
    "LoopInterchange",
    "LoopTiling",
    "LoopSkewing",
}


class OptimizationPipeline:
    """Запускает цепочку оптимизационных проходов над AST программы.

    Уровни:
        0 — без оптимизаций
        1 — базовые (constant folding/propagation, algebraic simp, strength reduction, DCE)
        2 — O1 + CSE, LICM, loop interchange, loop tiling
        3 — O2 + loop skewing + повторный тайлинг (для стенсильных гнёзд)

    Стратегия выполнения:
        - Повторяемые проходы (folding/propagation/simplification) запускаются
          до 3 раз, пока есть изменения.
        - Одноразовые проходы (CSE, LICM, tiling) запускаются ровно один раз.
    """

    MAX_ITERS = 3

    def __init__(self, level: int = 0):
        self.level = level
        self._pass_classes = _build_passes(level)
        self.stats: dict = {}

    def _record(self, p) -> bool:
        """Записать статистику прохода. Вернуть True если есть изменения."""
        changed = False
        if p.stats:
            if p.name not in self.stats:
                self.stats[p.name] = {}
            for k, v in p.stats.items():
                prev = self.stats[p.name].get(k, 0)
                self.stats[p.name][k] = prev + v
                if v > 0:
                    changed = True
        return changed

    def run(self, program: "Program") -> "Program":
        self.stats = {}

        repeatable = [cls for cls in self._pass_classes
                      if cls.name in _REPEATABLE]
        one_shot = [cls for cls in self._pass_classes
                    if cls.name in _ONE_SHOT]

        # Повторяемые проходы — несколько итераций
        for _iteration in range(self.MAX_ITERS):
            changed = False
            for cls in repeatable:
                p = cls()
                program = p.run(program)
                if self._record(p):
                    changed = True
            if not changed:
                break

        # Одноразовые проходы — строго по одному разу
        for cls in one_shot:
            p = cls()
            program = p.run(program)
            self._record(p)

        # Повторный прогон повторяемых после одноразовых
        # (например, константы могут появиться после LICM)
        for _iteration in range(self.MAX_ITERS):
            changed = False
            for cls in repeatable:
                p = cls()
                program = p.run(program)
                if self._record(p):
                    changed = True
            if not changed:
                break

        return program

    def report(self) -> str:
        """Вернуть строку с отчётом о применённых оптимизациях."""
        lines = []
        for name, s in self.stats.items():
            if s:
                detail = ", ".join(f"{k}: {v}" for k, v in s.items() if v > 0)
                if detail:
                    lines.append(f"  {name}: {detail}")
        return "\n".join(lines) if lines else "  (no optimizations applied)"
