from enum import Enum, auto
from typing import List, Optional, Any
from dataclasses import dataclass
from src.config import MAX_IDENTIFIER_LENGTH, MAX_LINE_LENGTH, MAX_STATEMENT_LENGTH
from src.utils.logger import debug, info, warning, error

class TokenType(Enum):
    INTEGER_LIT = auto()
    REAL_LIT = auto()
    STRING_LIT = auto()
    PROGRAM = auto()
    END = auto()
    IMPLICIT = auto()
    NONE = auto()
    INTEGER = auto()
    REAL = auto()
    COMPLEX = auto()
    DOUBLEPRECISION = auto()
    LOGICAL = auto()
    CHARACTER = auto()
    DIMENSION = auto()
    PARAMETER = auto()
    DATA = auto()
    IF = auto()
    THEN = auto()
    ELSE = auto()
    ELSEIF = auto()
    ENDIF = auto()
    DO = auto()
    ENDDO = auto()
    WHILE = auto()
    CONTINUE = auto()
    GOTO = auto()
    STOP = auto()
    PRINT = auto()
    READ = auto()
    WRITE = auto()
    SIN = auto()
    COS = auto()
    TAN = auto()
    ASIN = auto()
    ACOS = auto()
    ATAN = auto()
    EXP = auto()
    LOG = auto()
    LOG10 = auto()
    SQRT = auto()
    ABS = auto()
    MIN = auto()
    MAX = auto()
    MOD = auto()
    POW = auto()
    INT_FUNC = auto()
    REAL_FUNC = auto()
    FLOAT = auto()
    LPAREN = auto()
    RPAREN = auto()
    COMMA = auto()
    COLON = auto()
    ASSIGN_OP = auto()
    PLUS = auto()
    MINUS = auto()
    STAR = auto()
    SLASH = auto()
    CONCAT = auto()
    POWER = auto()
    EQ = auto()
    NE = auto()
    LT = auto()
    LE = auto()
    GT = auto()
    GE = auto()
    AND = auto()
    OR = auto()
    NOT = auto()
    EQV = auto()
    NEQV = auto()
    TRUE = auto()
    FALSE = auto()
    CALL = auto()
    RETURN = auto()
    SUBROUTINE = auto()
    FUNCTION = auto()
    EXTERNAL = auto()
    COMMON = auto()
    EXIT = auto()
    IDENTIFIER = auto()
    COMMENT = auto()
    EOF = auto()

@dataclass
class Token:
    type: TokenType = TokenType.EOF
    value: Any = None
    line: int = 0
    col: int = 0

    def __str__(self):
        return f"{self.type.name}({self.value})"

class Lexer:
    def __init__(self, text: str):
        self.text = text
        self.len = len(text)
        self.pos = 0
        self.line = 1
        self.col = 1
        self.errors = []
        self.keywords = {
            "PROGRAM": TokenType.PROGRAM,
            "END": TokenType.END,
            "IMPLICIT": TokenType.IMPLICIT,
            "NONE": TokenType.NONE,
            "INTEGER": TokenType.INTEGER,
            "REAL": TokenType.REAL,
            "COMPLEX": TokenType.COMPLEX,
            "DOUBLEPRECISION": TokenType.DOUBLEPRECISION,
            "LOGICAL": TokenType.LOGICAL,
            "CHARACTER": TokenType.CHARACTER,
            "DIMENSION": TokenType.DIMENSION,
            "PARAMETER": TokenType.PARAMETER,
            "DATA": TokenType.DATA,
            "IF": TokenType.IF,
            "THEN": TokenType.THEN,
            "ELSE": TokenType.ELSE,
            "ELSEIF": TokenType.ELSEIF,
            "ENDIF": TokenType.ENDIF,
            "DO": TokenType.DO,
            "ENDDO": TokenType.ENDDO,
            "WHILE": TokenType.WHILE,
            "CONTINUE": TokenType.CONTINUE,
            "GOTO": TokenType.GOTO,
            "STOP": TokenType.STOP,
            "PRINT": TokenType.PRINT,
            "READ": TokenType.READ,
            "WRITE": TokenType.WRITE,
            "SIN": TokenType.SIN,
            "COS": TokenType.COS,
            "TAN": TokenType.TAN,
            "ASIN": TokenType.ASIN,
            "ACOS": TokenType.ACOS,
            "ATAN": TokenType.ATAN,
            "EXP": TokenType.EXP,
            "LOG": TokenType.LOG,
            "LOG10": TokenType.LOG10,
            "SQRT": TokenType.SQRT,
            "ABS": TokenType.ABS,
            "MIN": TokenType.MIN,
            "MAX": TokenType.MAX,
            "MOD": TokenType.MOD,
            "POW": TokenType.POW,
            "FLOAT": TokenType.FLOAT,
            "CALL": TokenType.CALL,
            "RETURN": TokenType.RETURN,
            "SUBROUTINE": TokenType.SUBROUTINE,
            "FUNCTION": TokenType.FUNCTION,
            "EXTERNAL": TokenType.EXTERNAL,
            "COMMON": TokenType.COMMON,
            "EXIT": TokenType.EXIT,
            ".TRUE.": TokenType.TRUE,
            ".FALSE.": TokenType.FALSE,
            ".EQ.": TokenType.EQ,
            ".NE.": TokenType.NE,
            ".LT.": TokenType.LT,
            ".LE.": TokenType.LE,
            ".GT.": TokenType.GT,
            ".GE.": TokenType.GE,
            ".AND.": TokenType.AND,
            ".OR.": TokenType.OR,
            ".NOT.": TokenType.NOT,
            ".EQV.": TokenType.EQV,
            ".NEQV.": TokenType.NEQV,
        }

    def peek(self, offset: int = 0) -> str:
        pos = self.pos + offset
        if pos >= self.len:
            return ''
        return self.text[pos]

    def advance(self) -> str:
        if self.pos >= self.len:
            return ''
        ch = self.text[self.pos]
        self.pos += 1
        if ch == '\n':
            self.line += 1
            self.col = 1
        else:
            self.col += 1
        return ch

    def skip_whitespace(self):
        while self.peek() and self.peek() in ' \t\f\r':
            self.advance()

    def read_comment(self) -> Token:
        start_line = self.line
        start_col = self.col
        comment_text = ""
        if self.peek() == '!':
            self.advance()
            while self.peek() and self.peek() != '\n':
                comment_text += self.advance()
        elif self.peek() == 'C' and self.col == 1:
            self.advance()
            while self.peek() and self.peek() != '\n':
                comment_text += self.advance()
        if self.peek() == '\n':
            self.advance()
        return Token(type=TokenType.COMMENT, value=comment_text, line=start_line, col=start_col)

    def read_number(self) -> Token:
        start_line = self.line
        start_col = self.col
        num_str = ""
        is_real = False
        if self.peek() == '.':
            is_real = True
            num_str += self.advance()
            if not self.peek() or not self.peek().isdigit():
                raise SyntaxError(
                    f"[строка {start_line}, колонка {start_col}] Неверный формат числа. "
                    f"После точки в вещественном литерале ожидается цифра."
                )
        else:
            while self.peek() and self.peek().isdigit():
                num_str += self.advance()
        if self.peek() == '.':
            is_real = True
            num_str += self.advance()
        if is_real:
            while self.peek() and self.peek().isdigit():
                num_str += self.advance()
        exp_char = None
        if self.peek() and self.peek().upper() in {'E', 'D'}:
            is_real = True
            exp_char = self.advance()
            num_str += exp_char
            if self.peek() and self.peek() in {'+', '-'}:
                num_str += self.advance()
            if not self.peek() or not self.peek().isdigit():
                raise SyntaxError(
                    f"[строка {start_line}, колонка {start_col}] Неполная экспонента в числе '{num_str}'. "
                    f"После символа '{exp_char}' ожидается целое число."
                )
            while self.peek() and self.peek().isdigit():
                num_str += self.advance()
        if is_real:
            float_str = num_str.replace('D', 'E').replace('d', 'E')
            if float_str.startswith('.'):
                float_str = '0' + float_str
            if float_str.endswith('.'):
                float_str += '0'
            try:
                return Token(type=TokenType.REAL_LIT, value=float(float_str), line=start_line, col=start_col)
            except ValueError:
                raise SyntaxError(
                    f"[строка {start_line}, колонка {start_col}] Неверный формат вещественного числа '{num_str}'."
                )
        return Token(type=TokenType.INTEGER_LIT, value=int(num_str), line=start_line, col=start_col)

    def read_string(self, quote_char: str) -> Token:
        start_line = self.line
        start_col = self.col
        self.advance()
        string_val = ""
        while self.peek() and self.peek() != quote_char:
            if self.peek() == '\\':
                self.advance()
                string_val += self.advance()
            else:
                string_val += self.advance()
        if self.peek() == quote_char:
            self.advance()
        else:
            raise SyntaxError(
                f"[строка {start_line}, колонка {start_col}] Незавершенная строковая константа. "
                f"Ожидалась закрывающая кавычка '{quote_char}'."
            )
        return Token(type=TokenType.STRING_LIT, value=string_val, line=start_line, col=start_col)

    def read_identifier_or_keyword(self) -> Token:
        start_line = self.line
        start_col = self.col
        ident = ""
        while self.peek() and (self.peek().isalnum() or self.peek() == '_'):
            ident += self.advance()
        if not ident:
            return None
        upper_ident = ident.upper()
        if upper_ident == "DOUBLE":
            saved_pos2 = self.pos
            saved_line2 = self.line
            saved_col2 = self.col
            while self.peek() and self.peek() in ' \t':
                self.advance()
            next_word = ""
            while self.peek() and (self.peek().isalnum() or self.peek() == '_'):
                next_word += self.advance()
            if next_word.upper() == "PRECISION":
                return Token(type=TokenType.DOUBLEPRECISION, value="DOUBLEPRECISION", line=start_line, col=start_col)
            self.pos = saved_pos2
            self.line = saved_line2
            self.col = saved_col2
        if upper_ident in self.keywords:
            if ident and not ident[0].isalpha():
                self.errors.append(
                    f"[строка {start_line}, колонка {start_col}] Имя '{ident}' должно начинаться с буквы."
                )
            return Token(type=self.keywords[upper_ident], value=ident, line=start_line, col=start_col)
        if ident and not ident[0].isalpha():
            self.errors.append(
                f"[строка {start_line}, колонка {start_col}] Имя переменной '{ident}' должно начинаться с буквы."
            )
        upper_ident = ident.upper()
        if upper_ident == "END":
            saved_pos = self.pos
            saved_line = self.line
            saved_col = self.col
            while self.peek() and self.peek() in ' \t':
                self.advance()
            next_word = ""
            while self.peek() and (self.peek().isalnum() or self.peek() == '_'):
                next_word += self.advance()
            next_upper = next_word.upper()
            if next_upper == "IF":
                return Token(type=TokenType.ENDIF, value=ident + next_word, line=start_line, col=start_col)
            if next_upper == "DO":
                return Token(type=TokenType.ENDDO, value=ident + next_word, line=start_line, col=start_col)
            self.pos = saved_pos
            self.line = saved_line
            self.col = saved_col
        if len(ident) > 6:
            self.errors.append(
                f"[строка {start_line}, колонка {start_col}] Имя переменной '{ident}' слишком длинное. "
                f"В Fortran 77 допустимо не более 6 символов."
            )
        return Token(type=TokenType.IDENTIFIER, value=ident, line=start_line, col=start_col)

    def read_operator_or_delimiter(self) -> Optional[Token]:
        start_line = self.line
        start_col = self.col
        ch = self.peek()
        if ch == '*' and self.peek(1) == '*':
            self.advance()
            self.advance()
            return Token(type=TokenType.POWER, value='**', line=start_line, col=start_col)
        if ch == '/' and self.peek(1) == '/':
            self.advance()
            self.advance()
            return Token(type=TokenType.CONCAT, value='//', line=start_line, col=start_col)
        if ch == '.':
            dot_op = ""
            pos_save = self.pos
            col_save = self.col
            self.advance()
            while self.peek() and self.peek() in ' \t':
                self.advance()
            if not self.peek() or not self.peek().isalpha():
                self.pos = pos_save
                self.line = start_line
                self.col = start_col
                return None
            while self.peek() and self.peek() != '.':
                if self.peek() in ' \t':
                    self.advance()
                elif self.peek().isalpha():
                    dot_op += self.advance()
                else:
                    break
            while self.peek() and self.peek() in ' \t':
                self.advance()
            if not dot_op:
                self.pos = pos_save
                self.line = start_line
                self.col = start_col
                return None
            if self.peek() and self.peek() == '.':
                self.advance()
                dot_op_upper = dot_op.upper()
                if dot_op_upper in {"EQ", "NE", "LT", "LE", "GT", "GE", "AND", "OR", "NOT", "EQV", "NEQV", "TRUE", "FALSE"}:
                    full_op = f".{dot_op_upper}."
                    if full_op in self.keywords:
                        return Token(type=self.keywords[full_op], value=full_op, line=start_line, col=start_col)
                self.pos = pos_save
                self.line = start_line
                self.col = start_col
                return None
            self.pos = pos_save
            self.line = start_line
            self.col = start_col
            if self.peek(1) and self.peek(1).isdigit():
                return None
            return None
        single_ops = {
            '(': TokenType.LPAREN,
            ')': TokenType.RPAREN,
            ',': TokenType.COMMA,
            ':': TokenType.COLON,
            '=': TokenType.ASSIGN_OP,
            '+': TokenType.PLUS,
            '-': TokenType.MINUS,
            '*': TokenType.STAR,
            '/': TokenType.SLASH,
        }
        if ch in single_ops:
            self.advance()
            return Token(type=single_ops[ch], value=ch, line=start_line, col=start_col)
        return None

    def next_token(self) -> Token:
        while True:
            if self.peek() == '\n':
                self.advance()
                continue
            if self.col == 1 and self.peek() == 'C':
                return self.read_comment()
            self.skip_whitespace()
            if self.peek() == '\n':
                self.advance()
                continue
            if self.peek() == '!':
                return self.read_comment()
            break
        if self.pos >= self.len:
            return Token(type=TokenType.EOF, value=None, line=self.line, col=self.col)
        start_line = self.line
        start_col = self.col
        ch = self.peek()
        if ch.isalpha() or ch == '_':
            return self.read_identifier_or_keyword()
        if ch.isdigit():
            return self.read_number()
        if ch == '.':
            op_token = self.read_operator_or_delimiter()
            if op_token:
                return op_token
            if self.peek(1) and self.peek(1).isdigit():
                return self.read_number()
            char = self.peek()
            char_repr = repr(char)
            raise SyntaxError(
                f"[строка {start_line}, колонка {start_col}] Неожиданный символ {char_repr}."
            )
        if ch in {"'", '"'}:
            return self.read_string(ch)
        op_token = self.read_operator_or_delimiter()
        if op_token:
            return op_token
        char = self.peek()
        char_repr = repr(char)
        raise SyntaxError(
            f"[строка {start_line}, колонка {start_col}] Неожиданный символ {char_repr}."
        )

    def check_fortran_line_format(self, line_text: str, line_num: int):
        line_text = line_text.rstrip('\n\r')
        if len(line_text) > 80:
            self.errors.append(
                f" {line_num}  80  (: {len(line_text)})"
            )
        if len(line_text) > 80:
            line_text = line_text[:80]
        label_area = line_text[:5].strip()
        if label_area and not label_area.isdigit() and label_area != 'C' and not line_text.strip().startswith('!'):
            pass
        if line_text.strip() and not line_text.strip().startswith('C') and not line_text.strip().startswith('!'):
            if len(line_text) > 6:
                statement_area = line_text[6:72] if len(
                    line_text) > 72 else line_text[6:]
                if len(line_text) > 72 and line_text[72:80].strip():
                    pass

    def tokenize(self) -> List[Token]:
        lines = self.text.split('\n')
        processed_lines = []
        current_line = None
        for i, line in enumerate(lines, 1):
            self.check_fortran_line_format(line, i)
            line_for_processing = line.rstrip('\n\r')
            if len(line_for_processing) > 72:
                line_for_processing = line_for_processing[:72]
            stripped = line_for_processing.strip()
            label_area = line_for_processing[:5] if len(line_for_processing) >= 5 else line_for_processing
            fixed_form_layout = len(line_for_processing) > 5 and all(ch == ' ' or ch.isdigit() for ch in label_area)
            is_comment = bool(
                stripped and (
                    stripped.startswith('!') or
                    (line_for_processing and line_for_processing[0].upper() == 'C' and line_for_processing[:1].strip() == 'C')
                )
            )
            is_continuation = (
                fixed_form_layout and
                not is_comment and
                len(line_for_processing) > 5 and
                line_for_processing[5] not in {' ', '0'}
            )
            if is_comment:
                if current_line is not None:
                    processed_lines.append(current_line + '\n')
                    current_line = None
                processed_lines.append(line_for_processing + '\n')
                continue
            if not stripped:
                if current_line is not None:
                    processed_lines.append(current_line + '\n')
                    current_line = None
                processed_lines.append('\n')
                continue
            if is_continuation and current_line is not None:
                current_line = current_line.rstrip('\n\r') + line_for_processing[6:]
                continue
            if current_line is not None:
                processed_lines.append(current_line + '\n')
            current_line = line_for_processing
        if current_line is not None:
            processed_lines.append(current_line + '\n')
        self.text = ''.join(processed_lines)
        self.len = len(self.text)
        self.pos = 0
        self.line = 1
        self.col = 1
        tokens = []
        try:
            while True:
                token = self.next_token()
                if token is None:
                    char = self.peek()
                    raise SyntaxError(
                        f"[строка {self.line}, колонка {self.col}] Неожиданный символ '{char}'. "
                        f"Не удалось распознать токен."
                    )
                if token.type == TokenType.COMMENT:
                    continue
                tokens.append(token)
                if token.type == TokenType.EOF:
                    break
        except SyntaxError as e:
            raise
        return tokens

    def get_errors(self) -> List[str]:
        return self.errors
