class SemanticError(Exception):
    def __init__(self, message: str, node=None, suggestion=None):
        self.message = message
        self.node = node
        self.suggestion = suggestion
        self.line = node.line if node and hasattr(node, 'line') else None
        self.column = node.col if node and hasattr(node, 'col') else None

    def __str__(self):
        loc = f"({self.line}:{self.column}) " if self.line else ""
        sug = f"\n  Подсказка: {self.suggestion}" if self.suggestion else ""
        return f"{loc}Semantic Error: {self.message}{sug}"

class TypeMismatchError(SemanticError):
    pass

class UndefinedSymbolError(SemanticError):
    pass

class DimensionError(SemanticError):
    pass

class TypeError(SemanticError):
    pass
