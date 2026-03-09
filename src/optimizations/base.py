from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.core import (
        Program, Statement, Expression, BinaryOp, UnaryOp,
        Variable, IntegerLiteral, RealLiteral, FunctionCall,
        ArrayRef, LogicalLiteral, StringLiteral, ComplexLiteral,
        Assignment, DoLoop, LabeledDoLoop, DoWhile, LabeledDoWhile,
        IfStatement, SimpleIfStatement,
    )


class ASTOptimizationPass(ABC):
    """Базовый класс для всех оптимизационных проходов на уровне AST.

    Каждый проход принимает Program и возвращает (возможно изменённый) Program.
    Статистика применений хранится в self.stats.
    """

    name: str = "UnnamedPass"

    def __init__(self):
        self.stats: dict = {}

    @abstractmethod
    def run(self, program: "Program") -> "Program":
        """Выполнить проход оптимизации над всей программой."""
        ...

    # ------------------------------------------------------------------
    # Вспомогательные методы для рекурсивного обхода AST
    # ------------------------------------------------------------------

    def transform_expr(self, expr: "Expression") -> "Expression":
        """Рекурсивно обойти выражение и применить трансформацию.

        По умолчанию возвращает expr без изменений.
        Переопределить в подклассах для работы с конкретными узлами.
        """
        from src.core import (
            BinaryOp, UnaryOp, FunctionCall, ArrayRef,
        )
        if isinstance(expr, BinaryOp):
            new_left = self.transform_expr(expr.left)
            new_right = self.transform_expr(expr.right)
            if new_left is not expr.left or new_right is not expr.right:
                from dataclasses import replace
                expr = replace(expr, left=new_left, right=new_right)
        elif isinstance(expr, UnaryOp):
            new_operand = self.transform_expr(expr.operand)
            if new_operand is not expr.operand:
                from dataclasses import replace
                expr = replace(expr, operand=new_operand)
        elif isinstance(expr, FunctionCall):
            new_args = [self.transform_expr(a) for a in expr.args]
            if any(na is not a for na, a in zip(new_args, expr.args)):
                from dataclasses import replace
                expr = replace(expr, args=new_args)
        elif isinstance(expr, ArrayRef):
            new_indices = [self.transform_expr(i) for i in expr.indices]
            if any(ni is not i for ni, i in zip(new_indices, expr.indices)):
                from dataclasses import replace
                expr = replace(expr, indices=new_indices)
        return expr

    def transform_stmt(self, stmt: "Statement") -> "Statement":
        """Рекурсивно обойти оператор и применить трансформацию.

        По умолчанию обходит вложенные операторы и выражения.
        Переопределить для специфичной трансформации.
        """
        from src.core import (
            Assignment, DoLoop, LabeledDoLoop, DoWhile, LabeledDoWhile,
            IfStatement, SimpleIfStatement,
        )
        from dataclasses import replace

        if isinstance(stmt, Assignment):
            new_val = self.transform_expr(stmt.value)
            new_idx = [self.transform_expr(i) for i in stmt.indices]
            changed = new_val is not stmt.value or any(
                ni is not i for ni, i in zip(new_idx, stmt.indices)
            )
            if changed:
                stmt = replace(stmt, value=new_val, indices=new_idx)

        elif isinstance(stmt, (DoLoop, LabeledDoLoop)):
            new_start = self.transform_expr(stmt.start)
            new_end = self.transform_expr(stmt.end)
            new_step = self.transform_expr(stmt.step) if stmt.step else stmt.step
            new_body = self.transform_stmts(stmt.body)
            if (new_start is not stmt.start or new_end is not stmt.end
                    or new_step is not stmt.step or new_body is not stmt.body):
                stmt = replace(stmt, start=new_start, end=new_end,
                               step=new_step, body=new_body)

        elif isinstance(stmt, (DoWhile, LabeledDoWhile)):
            new_cond = self.transform_expr(stmt.condition)
            new_body = self.transform_stmts(stmt.body)
            if new_cond is not stmt.condition or new_body is not stmt.body:
                stmt = replace(stmt, condition=new_cond, body=new_body)

        elif isinstance(stmt, IfStatement):
            new_cond = self.transform_expr(stmt.condition)
            new_then = self.transform_stmts(stmt.then_body)
            new_elif = [
                (self.transform_expr(c), self.transform_stmts(b))
                for c, b in stmt.elif_parts
            ]
            new_else = self.transform_stmts(stmt.else_body) if stmt.else_body else stmt.else_body
            stmt = replace(stmt,
                           condition=new_cond,
                           then_body=new_then,
                           elif_parts=new_elif,
                           else_body=new_else)

        elif isinstance(stmt, SimpleIfStatement):
            new_cond = self.transform_expr(stmt.condition)
            new_s = self.transform_stmt(stmt.statement)
            if new_cond is not stmt.condition or new_s is not stmt.statement:
                stmt = replace(stmt, condition=new_cond, statement=new_s)

        return stmt

    def transform_stmts(self, stmts: List["Statement"]) -> List["Statement"]:
        """Обойти список операторов. Возвращает (возможно новый) список."""
        result = []
        changed = False
        for s in stmts:
            new_s = self.transform_stmt(s)
            result.append(new_s)
            if new_s is not s:
                changed = True
        return result if changed else stmts
