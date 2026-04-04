from __future__ import annotations
from typing import List, Type
from src.optimizations.base import ASTOptimizationPass
from src.optimizations.cse import CommonSubexpressionElimination
from src.optimizations.licm import LoopInvariantCodeMotion
from src.optimizations.loop_interchange import LoopInterchange
from src.optimizations.loop_intra_tile_interchange import IntraTileLoopInterchange
from src.optimizations.loop_tiling import LoopTiling
from src.optimizations.loop_skewing import LoopSkewing
from src.optimizations.loop_wavefront import LoopWavefront
from src.optimizations.affine_linearization import AffineLinearization
from src.optimizations.generated_declarations import GeneratedVariableDeclarations
from src.optimizations.loop_parallelization import LoopParallelization


def buildPasses(level: int) -> List[Type[ASTOptimizationPass]]:
    if level == 0:
        return []
    if level == 2:
        return [
            LoopInvariantCodeMotion,
            CommonSubexpressionElimination,
            LoopInterchange,
            LoopTiling,
            AffineLinearization,
            GeneratedVariableDeclarations,
        ]
    if level == 3:
        return [
            LoopSkewing,
            LoopTiling,
            IntraTileLoopInterchange,
            LoopWavefront,
            LoopParallelization,
            AffineLinearization,
            LoopInvariantCodeMotion,
            CommonSubexpressionElimination,
            GeneratedVariableDeclarations,
        ]
    return []


class OptimizationPipeline:
    def __init__(self, level: int = 0):
        self.level = level
        self.passClasses = buildPasses(level)
        self.stats: dict = {}

    def recordStats(self, p) -> None:
        if p.stats:
            if p.name not in self.stats:
                self.stats[p.name] = {}
            for k, v in p.stats.items():
                prev = self.stats[p.name].get(k)
                if isinstance(v, (int, float)) and isinstance(prev, (int, float)):
                    self.stats[p.name][k] = prev + v
                elif isinstance(v, (int, float)) and prev is None:
                    self.stats[p.name][k] = v
                else:
                    self.stats[p.name][k] = v

    def run(self, program):
        self.stats = {}
        for cls in self.passClasses:
            p = cls()
            program = p.run(program)
            self.recordStats(p)
        return program

    def report(self) -> str:
        if not self.stats:
            return "  (no optimizations applied)"
        lines = []
        for name, kv in self.stats.items():
            parts = ", ".join(f"{k}: {v}" for k, v in kv.items())
            lines.append(f"  {name}: {parts}")
        return "\n".join(lines)
