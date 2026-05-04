from enum import Enum

class TypeKind(Enum):
    INTEGER = "INTEGER"
    REAL = "REAL"
    LOGICAL = "LOGICAL"
    CHARACTER = "CHARACTER"
    COMPLEX = "COMPLEX"
    UNKNOWN = "UNKNOWN"

class TypeRules:
    @staticmethod
    def are_compatible(target: TypeKind, source: TypeKind) -> bool:
        if target == TypeKind.INTEGER and source == TypeKind.INTEGER:
            return True
        if target == TypeKind.REAL and source in {TypeKind.REAL, TypeKind.INTEGER}:
            return True
        if target == TypeKind.LOGICAL and source == TypeKind.LOGICAL:
            return True
        if target == TypeKind.CHARACTER and source == TypeKind.CHARACTER:
            return True
        if target == TypeKind.COMPLEX:
            return True
        return False

    @staticmethod
    def are_comparable(left: TypeKind, right: TypeKind) -> bool:
        if left == TypeKind.UNKNOWN or right == TypeKind.UNKNOWN:
            return True
        if left in {TypeKind.INTEGER, TypeKind.REAL} and right in {TypeKind.INTEGER, TypeKind.REAL}:
            return True
        if left == TypeKind.LOGICAL and right == TypeKind.LOGICAL:
            return True
        if left == TypeKind.CHARACTER and right == TypeKind.CHARACTER:
            return True
        return False
