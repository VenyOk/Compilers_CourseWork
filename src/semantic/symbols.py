from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List
from src.semantic.types import TypeKind

@dataclass
class VariableInfo:
    name: str
    type_kind: TypeKind
    is_array: bool = False
    dimensions: List[Tuple[int, int]] = None
    is_parameter: bool = False
    value: Optional[object] = None
    explicitly_declared: bool = False

    def __post_init__(self):
        if self.dimensions is None:
            self.dimensions = []

    def get_dimension_size(self, dim_index: int) -> int:
        if dim_index < len(self.dimensions):
            k, l = self.dimensions[dim_index]
            return l - k + 1
        return 0
