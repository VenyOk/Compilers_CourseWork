from dataclasses import dataclass

MAX_DIMENSIONS = 7
MAX_IDENTIFIER_LENGTH = 6
MAX_LINE_LENGTH = 80
MAX_STATEMENT_LENGTH = 72
DEFAULT_OPT_LEVEL = 0
DEFAULT_TARGET_TRIPLE = "x86_64-apple-darwin"
DEFAULT_LOG_LEVEL = "INFO"

@dataclass
class CompilerConfig:
    max_dimensions: int = MAX_DIMENSIONS
    max_identifier_length: int = MAX_IDENTIFIER_LENGTH
    max_line_length: int = MAX_LINE_LENGTH
    max_statement_length: int = MAX_STATEMENT_LENGTH
    default_opt_level: int = DEFAULT_OPT_LEVEL
    target_triple: str = DEFAULT_TARGET_TRIPLE
    log_level: str = DEFAULT_LOG_LEVEL
    stop_on_error: bool = False
    allow_implicit_typing: bool = True
