import sys
import re
from enum import Enum, auto
from typing import List, Dict, Optional, Any, Tuple, Set
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
    IDENTIFIER = auto()
    COMMENT = auto()
    EOF = auto()
class Token:
    def __init__(self, type_: TokenType, value: Any, line: int, col: int):
        self.type = type_
        self.value = value
        self.line = line
        self.col = col
    def __repr__(self):
        return f"Token({self.type.name}, {repr(self.value)}, {self.line}:{self.col})"
    def __str__(self):
        return f"{self.type.name}({self.value})"
class Lexer:
    def __init__(self, text: str):
        self.text = text
        self.pos = 0
        self.line = 1
        self.col = 1
        self.len = len(text)
        self.errors = []
        self.keywords = {
            "PROGRAM": TokenType.PROGRAM,
            "END": TokenType.END,
            "IMPLICIT": TokenType.IMPLICIT,
            "NONE": TokenType.NONE,
            "INTEGER": TokenType.INTEGER,
            "REAL": TokenType.REAL,
            "COMPLEX": TokenType.COMPLEX,
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
            "INT": TokenType.INT_FUNC,
            "FLOAT": TokenType.FLOAT,
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
        return Token(TokenType.COMMENT, comment_text, start_line, start_col)
    def read_number(self) -> Token:
        start_line = self.line
        start_col = self.col
        num_str = ""
        while self.peek() and self.peek().isdigit():
            num_str += self.advance()
        if self.peek() == '.':
            num_str += self.advance()
            while self.peek() and self.peek().isdigit():
                num_str += self.advance()
        exp_char = None
        if self.peek() and self.peek().upper() in {'E', 'D'}:
            exp_char = self.peek().upper()
            num_str += self.advance()
            if self.peek() and self.peek() in {'+', '-'}:
                num_str += self.advance()
            while self.peek() and self.peek().isdigit():
                num_str += self.advance()
        if '.' in num_str or exp_char is not None:
            float_str = num_str.replace('D', 'E').replace('d', 'E')
            return Token(TokenType.REAL_LIT, float(float_str), start_line, start_col)
        else:
            return Token(TokenType.INTEGER_LIT, int(num_str), start_line, start_col)
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
            raise SyntaxError(f"Unterminated string at {start_line}:{start_col}")
        return Token(TokenType.STRING_LIT, string_val, start_line, start_col)
    def read_identifier_or_keyword(self) -> Token:
        start_line = self.line
        start_col = self.col
        ident = ""
        while self.peek() and (self.peek().isalnum() or self.peek() == '_'):
            ident += self.advance()
        if len(ident) > 6:
            self.errors.append(
                f"Имя переменной '{ident}' слишком длинное (максимум 6 символов) на строке {start_line}:{start_col}"
            )
        if ident and not ident[0].isalpha():
            self.errors.append(
                f"Имя переменной '{ident}' должно начинаться с буквы на строке {start_line}:{start_col}"
            )
        upper_ident = ident.upper()
        if upper_ident == "END":
            saved_pos = self.pos
            saved_line = self.line
            saved_col = self.col
            while self.peek() and self.peek() in ' \t':
                self.advance()
            next_ident = ""
            while self.peek() and (self.peek().isalnum() or self.peek() == '_'):
                next_ident += self.peek()
                break
            if next_ident and next_ident.upper() in ('I', 'D'):
                temp_pos = self.pos
                full_word = ""
                while self.peek() and (self.peek().isalnum() or self.peek() == '_'):
                    full_word += self.advance()
                full_word_upper = full_word.upper()
                if full_word_upper in ('IF', 'DO'):
                    combined = f"{upper_ident}{full_word_upper}"
                    if combined == "ENDIF":
                        return Token(TokenType.ENDIF, ident + full_word, start_line, start_col)
                    elif combined == "ENDDO":
                        return Token(TokenType.ENDDO, ident + full_word, start_line, start_col)
                else:
                    self.pos = temp_pos
                    self.line = saved_line
                    self.col = saved_col
            else:
                self.pos = saved_pos
                self.line = saved_line
                self.col = saved_col
        if upper_ident in self.keywords:
            return Token(self.keywords[upper_ident], ident, start_line, start_col)
        return Token(TokenType.IDENTIFIER, ident, start_line, start_col)
    def read_operator_or_delimiter(self) -> Optional[Token]:
        start_line = self.line
        start_col = self.col
        ch = self.peek()
        if ch == '*' and self.peek(1) == '*':
            self.advance()
            self.advance()
            return Token(TokenType.POWER, '**', start_line, start_col)
        if ch == '/' and self.peek(1) == '/':
            self.advance()
            self.advance()
            return Token(TokenType.CONCAT, '//', start_line, start_col)
        if ch == '.':
            dot_op = ""
            pos_save = self.pos
            col_save = self.col
            self.advance()
            while self.peek() and self.peek() != '.':
                dot_op += self.advance()
            if self.peek() == '.':
                self.advance()
                dot_op_upper = dot_op.upper()
                if dot_op_upper in {"EQ", "NE", "LT", "LE", "GT", "GE", "AND", "OR", "NOT", "EQV", "NEQV", "TRUE", "FALSE"}:
                    full_op = f".{dot_op_upper}."
                    if full_op in self.keywords:
                        return Token(self.keywords[full_op], full_op, start_line, start_col)
            raise SyntaxError(f"Invalid operator at {start_line}:{start_col}")
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
            return Token(single_ops[ch], ch, start_line, start_col)
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
            return Token(TokenType.EOF, None, self.line, self.col)
        start_line = self.line
        start_col = self.col
        ch = self.peek()
        if ch.isalpha() or ch == '_':
            return self.read_identifier_or_keyword()
        if ch.isdigit():
            return self.read_number()
        if ch == '.' and self.peek(1) and self.peek(1).isdigit():
            return self.read_number()
        if ch in {"'", '"'}:
            return self.read_string(ch)
        op_token = self.read_operator_or_delimiter()
        if op_token:
            return op_token
        char_repr = repr(self.peek())
        raise SyntaxError(f"Unexpected character {char_repr} at {start_line}:{start_col}")
    def check_fortran_line_format(self, line_text: str, line_num: int):
        line_text = line_text.rstrip('\n\r')
        if len(line_text) > 80:
            self.errors.append(
                f"Строка {line_num} превышает 80 символов (длина: {len(line_text)})"
            )
        label_area = line_text[:5].strip()
        if label_area and not label_area.isdigit() and label_area != 'C' and not line_text.strip().startswith('!'):
            pass
        if line_text.strip() and not line_text.strip().startswith('C') and not line_text.strip().startswith('!'):
            if len(line_text) > 6:
                statement_area = line_text[6:72] if len(line_text) > 72 else line_text[6:]
                if len(line_text) > 72 and line_text[72:].strip():
                    self.errors.append(
                        f"Строка {line_num}: операторы должны находиться в позициях 7-72, "
                        f"обнаружен код после позиции 72"
                    )
    def tokenize(self) -> List[Token]:
        lines = self.text.split('\n')
        for i, line in enumerate(lines, 1):
            self.check_fortran_line_format(line, i)
        self.pos = 0
        self.line = 1
        self.col = 1
        tokens = []
        try:
            while True:
                token = self.next_token()
                tokens.append(token)
                if token.type == TokenType.EOF:
                    break
        except SyntaxError as e:
            raise SyntaxError(f"Lexer error: {e}")
        return tokens
    def get_errors(self) -> List[str]:
        return self.errors
class ASTNode:
    def __init__(self, line: int = 0, col: int = 0):
        self.line = line
        self.col = col
class Program(ASTNode):
    def __init__(self, name: str, declarations: List['Declaration'], statements: List['Statement']):
        self.name = name
        self.declarations = declarations
        self.statements = statements
    def __repr__(self):
        return f"Program({self.name}, {len(self.declarations)} decls, {len(self.statements)} stmts)"
class Subroutine(ASTNode):
    def __init__(self, name: str, params: List[str], declarations: List['Declaration'],
                 statements: List['Statement']):
        self.name = name
        self.params = params
        self.declarations = declarations
        self.statements = statements
    def __repr__(self):
        return f"Subroutine({self.name})"
class FunctionDef(ASTNode):
    def __init__(self, name: str, return_type: str, params: List[str],
                 declarations: List['Declaration'], statements: List['Statement']):
        self.name = name
        self.return_type = return_type
        self.params = params
        self.declarations = declarations
        self.statements = statements
    def __repr__(self):
        return f"Function({self.name}: {self.return_type})"
class Declaration(ASTNode):
    def __init__(self, type_: str, names: List[Tuple[str, Optional[List[Tuple[int, int]]]]], type_size: Optional[int] = None):
        self.type = type_
        self.names = names
        self.type_size = type_size
    def __repr__(self):
        names_str = ", ".join(n[0] for n in self.names)
        return f"{self.type} {names_str}"
class ImplicitNone(ASTNode):
    def __repr__(self):
        return "IMPLICIT NONE"
class ImplicitRule(ASTNode):
    def __init__(self, type_name: str, type_size: Optional[int], letters: List[str]):
        self.type_name = type_name
        self.type_size = type_size
        self.letters = letters                                         
    def __repr__(self):
        size_str = f"*{self.type_size}" if self.type_size else ""
        letters_str = ", ".join(self.letters)
        return f"IMPLICIT {self.type_name}{size_str}({letters_str})"
    def get_letters(self) -> Set[str]:
        result = set()
        for letter_spec in self.letters:
            if '-' in letter_spec:
                parts = letter_spec.split('-')
                if len(parts) == 2:
                    start = parts[0].strip().upper()
                    end = parts[1].strip().upper()
                    if len(start) == 1 and len(end) == 1 and start.isalpha() and end.isalpha():
                        start_ord = ord(start)
                        end_ord = ord(end)
                        if start_ord <= end_ord:
                            for i in range(start_ord, end_ord + 1):
                                result.add(chr(i))
            else:
                letter = letter_spec.strip().upper()
                if letter and letter.isalpha():
                    result.add(letter)
        return result
class ImplicitStatement(ASTNode):
    def __init__(self, rules: List[ImplicitRule]):
        self.rules = rules
    def __repr__(self):
        rules_str = ", ".join(str(rule) for rule in self.rules)
        return f"IMPLICIT {rules_str}"
class DimensionStatement(ASTNode):
    def __init__(self, names: List[Tuple[str, List[Tuple[int, int]]]]):
        self.names = names
    def __repr__(self):
        def format_dim(dim_range: Tuple[int, int]) -> str:
            k, l = dim_range
            if k == 1:
                return str(l)                      
            return f"{k}:{l}"
        names_str = ", ".join(f"{n[0]}({','.join(format_dim(d) for d in n[1])})" for n in self.names)
        return f"DIMENSION {names_str}"
class ParameterStatement(ASTNode):
    def __init__(self, params: List[Tuple[str, 'Expression']]):
        self.params = params
    def __repr__(self):
        params_str = ", ".join(f"{name}={expr}" for name, expr in self.params)
        return f"PARAMETER ({params_str})"
class Statement(ASTNode):
    pass
class DataItem(ASTNode):
    def __init__(self, name: str, indices: List['Expression'] = None):
        super().__init__()
        self.name = name
        self.indices = indices or []
    def __repr__(self):
        if self.indices:
            indices_str = "(" + ", ".join(str(idx) for idx in self.indices) + ")"
            return f"{self.name}{indices_str}"
        return self.name
class DataStatement(Statement):
    def __init__(self, items: List[Tuple[List[DataItem], List['Expression']]]):
        super().__init__()
        self.items = items
    def __repr__(self):
        items_str = ", ".join(
            f"{','.join(str(item) for item in vars)} / {','.join(str(v) for v in vals)} /"
            for vars, vals in self.items
        )
        return f"DATA {items_str}"
class Assignment(Statement):
    def __init__(self, target: str, value: 'Expression', indices: List['Expression'] = None, line: int = 0, col: int = 0):
        super().__init__(line, col)
        self.target = target
        self.indices = indices or []
        self.value = value
    def __repr__(self):
        return f"Assign({self.target} = ...)"
class DoLoop(Statement):
    def __init__(self, var: str, start: 'Expression', end: 'Expression',
                 step: Optional['Expression'], body: List[Statement]):
        self.var = var
        self.start = start
        self.end = end
        self.step = step
        self.body = body
    def __repr__(self):
        return f"DO {self.var} = ... END DO"
class DoWhile(Statement):
    def __init__(self, condition: 'Expression', body: List[Statement]):
        self.condition = condition
        self.body = body
    def __repr__(self):
        return f"DO WHILE (...) END DO"
class SimpleIfStatement(Statement):
    def __init__(self, condition: 'Expression', statement: 'Statement'):
        self.condition = condition
        self.statement = statement
    def __repr__(self):
        return f"IF (...) S"
class IfStatement(Statement):
    def __init__(self, condition: 'Expression', then_body: List[Statement],
                 elif_parts: List[Tuple['Expression', List[Statement]]] = None,
                 else_body: List[Statement] = None):
        self.condition = condition
        self.then_body = then_body
        self.elif_parts = elif_parts or []
        self.else_body = else_body
    def __repr__(self):
        return f"IF (...) THEN ... END IF"
class PrintStatement(Statement):
    def __init__(self, items: List['Expression']):
        self.items = items
    def __repr__(self):
        return f"PRINT {len(self.items)} items"
class ReadStatement(Statement):
    def __init__(self, unit: str, format_: str, items: List[str]):
        self.unit = unit
        self.format = format_
        self.items = items
    def __repr__(self):
        return f"READ ({self.unit}, {self.format}) {len(self.items)} items"
class WriteStatement(Statement):
    def __init__(self, unit: str, format_: str, items: List['Expression']):
        self.unit = unit
        self.format = format_
        self.items = items
    def __repr__(self):
        return f"WRITE ({self.unit}, {self.format}) {len(self.items)} items"
class CallStatement(Statement):
    def __init__(self, name: str, args: List['Expression']):
        self.name = name
        self.args = args
    def __repr__(self):
        return f"CALL {self.name}"
class ReturnStatement(Statement):
    def __repr__(self):
        return "RETURN"
class StopStatement(Statement):
    def __repr__(self):
        return "STOP"
class GotoStatement(Statement):
    def __init__(self, label: str):
        self.label = label
    def __repr__(self):
        return f"GOTO {self.label}"
class ContinueStatement(Statement):
    def __init__(self, label: Optional[str] = None):
        self.label = label
    def __repr__(self):
        if self.label:
            return f"CONTINUE ({self.label})"
        return "CONTINUE"
class ArithmeticIfStatement(Statement):
    def __init__(self, condition: 'Expression', label_neg: str, label_zero: str, label_pos: str):
        self.condition = condition
        self.label_neg = label_neg
        self.label_zero = label_zero
        self.label_pos = label_pos
    def __repr__(self):
        return f"IF({self.condition}) {self.label_neg}, {self.label_zero}, {self.label_pos}"
class LabeledDoLoop(Statement):
    def __init__(self, label: str, var: str, start: 'Expression', end: 'Expression',
                 step: Optional['Expression'], body: List[Statement]):
        self.label = label
        self.var = var
        self.start = start
        self.end = end
        self.step = step
        self.body = body
    def __repr__(self):
        return f"DO {self.label} {self.var} = ... END DO"
class LabeledDoWhile(Statement):
    def __init__(self, label: str, condition: 'Expression', body: List[Statement]):
        self.label = label
        self.condition = condition
        self.body = body
    def __repr__(self):
        return f"DO {self.label} WHILE(...) END DO"
class ExitStatement(Statement):
    def __repr__(self):
        return "EXIT"
class Expression(ASTNode):
    pass
class BinaryOp(Expression):
    def __init__(self, left: Expression, op: str, right: Expression, line: int = 0, col: int = 0):
        super().__init__(line, col)
        self.left = left
        self.op = op
        self.right = right
    def __repr__(self):
        return f"({self.op})"
class UnaryOp(Expression):
    def __init__(self, op: str, operand: Expression):
        self.op = op
        self.operand = operand
    def __repr__(self):
        return f"({self.op} ...)"
class FunctionCall(Expression):
    def __init__(self, name: str, args: List[Expression]):
        self.name = name
        self.args = args
    def __repr__(self):
        return f"{self.name}(...)"
class ArrayRef(Expression):
    def __init__(self, name: str, indices: List['Expression'], line: int = 0, col: int = 0):
        super().__init__(line, col)
        self.name = name
        self.indices = indices
    def __repr__(self):
        return f"{self.name}[...]"
class Variable(Expression):
    def __init__(self, name: str, line: int = 0, col: int = 0):
        super().__init__(line, col)
        self.name = name
    def __repr__(self):
        return f"{self.name}"
class IntegerLiteral(Expression):
    def __init__(self, value: int, line: int = 0, col: int = 0):
        super().__init__(line, col)
        self.value = value
    def __repr__(self):
        return str(self.value)
class RealLiteral(Expression):
    def __init__(self, value: float, line: int = 0, col: int = 0):
        super().__init__(line, col)
        self.value = value
    def __repr__(self):
        return str(self.value)
class StringLiteral(Expression):
    def __init__(self, value: str, line: int = 0, col: int = 0):
        super().__init__(line, col)
        self.value = value
    def __repr__(self):
        return repr(self.value)
class LogicalLiteral(Expression):
    def __init__(self, value: bool, line: int = 0, col: int = 0):
        super().__init__(line, col)
        self.value = value
    def __repr__(self):
        return ".TRUE." if self.value else ".FALSE."
class Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0
    def current(self) -> Token:
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return Token(TokenType.EOF, None, 0, 0)
    def peek(self, offset: int = 1) -> Token:
        pos = self.pos + offset
        if pos < len(self.tokens):
            return self.tokens[pos]
        return Token(TokenType.EOF, None, 0, 0)
    def advance(self) -> Token:
        token = self.current()
        self.pos += 1
        return token
    def expect(self, token_type: TokenType) -> Token:
        if self.current().type != token_type:
            raise SyntaxError(
                f"Expected {token_type.name}, got {self.current().type.name} "
                f"at {self.current().line}:{self.current().col}: {self.current().value}"
            )
        return self.advance()
    def match(self, *token_types: TokenType) -> bool:
        return self.current().type in token_types
    def skip_comments(self):
        while self.match(TokenType.COMMENT):
            self.advance()
    def parse(self) -> Program:
        return self.parse_program()
    def parse_program(self) -> Program:
        self.skip_comments()
        name = "MAIN"
        if self.match(TokenType.PROGRAM):
            self.advance()
            name_token = self.expect(TokenType.IDENTIFIER)
            name = name_token.value
        declarations = []
        implicit_found = False
        other_declarations_found = False
        while self.match(TokenType.IMPLICIT):
            if other_declarations_found:
                raise SyntaxError(
                    f"IMPLICIT должен предшествовать всем другим невыполняемым операторам на строке {self.current().line}:{self.current().col}"
                )
            declarations.append(self.parse_implicit_statement())
            implicit_found = True
        while self.match(TokenType.INTEGER, TokenType.REAL, TokenType.LOGICAL, TokenType.COMPLEX, TokenType.CHARACTER,
                         TokenType.DIMENSION, TokenType.PARAMETER, TokenType.DATA):
            other_declarations_found = True
            if self.match(TokenType.DIMENSION):
                declarations.append(self.parse_dimension_statement())
            elif self.match(TokenType.PARAMETER):
                declarations.append(self.parse_parameter_statement())
            elif self.match(TokenType.DATA):
                declarations.append(self.parse_data_statement())
            else:
                declarations.extend(self.parse_declaration())
            self.skip_comments()
        statements = []
        while not self.match(TokenType.END, TokenType.EOF):
            self.skip_comments()
            if self.match(TokenType.END, TokenType.EOF):
                break
            stmt = self.parse_statement()
            statements.append(stmt)
        self.expect(TokenType.END)
        return Program(name, declarations, statements)
    def parse_declaration(self) -> List[Declaration]:
        decls = []
        if self.match(TokenType.INTEGER, TokenType.REAL, TokenType.LOGICAL, TokenType.COMPLEX, TokenType.CHARACTER):
            type_token = self.advance()
            type_name = type_token.value.upper()
            type_size = None
            if self.match(TokenType.STAR):
                self.advance()
                if self.match(TokenType.INTEGER_LIT):
                    size = self.advance().value
                    type_size = size
                    type_name = f"{type_name}*{size}"
            names = []
            while True:
                name_token = self.expect(TokenType.IDENTIFIER)
                name = name_token.value
                dim_ranges = None
                if self.match(TokenType.LPAREN):
                    self.advance()
                    dim_ranges = []
                    while True:
                        if self.match(TokenType.INTEGER_LIT):
                            first_token = self.advance()
                            first_value = first_token.value
                            if self.match(TokenType.COLON):
                                self.advance()                
                                if not self.match(TokenType.INTEGER_LIT):
                                    raise SyntaxError(
                                        f"Ожидается целое число после ':' в диапазоне индексов на строке {self.current().line}:{self.current().col}"
                                    )
                                second_token = self.advance()
                                second_value = second_token.value
                                dim_ranges.append((first_value, second_value))
                            elif self.match(TokenType.COMMA, TokenType.RPAREN):
                                dim_ranges.append((1, first_value))
                            else:
                                raise SyntaxError(
                                    f"Ожидается ':' или ',' после размера массива на строке {self.current().line}:{self.current().col}"
                                )
                        else:
                            raise SyntaxError(
                                f"Ожидается целое число для размера массива на строке {self.current().line}:{self.current().col}"
                            )
                        if not self.match(TokenType.COMMA):
                            break
                        self.advance()
                    if len(dim_ranges) > 7:
                        raise SyntaxError(
                            f"Массив '{name}' имеет {len(dim_ranges)} измерений, максимум допускается 7 на строке {name_token.line}:{name_token.col}"
                        )
                    for i, (k, l) in enumerate(dim_ranges):
                        if k > l:
                            raise SyntaxError(
                                f"Неверный диапазон для измерения {i+1} массива '{name}': "
                                f"начальное значение ({k}) больше конечного ({l}) на строке {name_token.line}:{name_token.col}"
                            )
                    self.expect(TokenType.RPAREN)
                names.append((name, dim_ranges))
                if not self.match(TokenType.COMMA):
                    break
                self.advance()
            decls.append(Declaration(type_name, names, type_size))
        else:
            raise SyntaxError(f"Expected type declaration (INTEGER/REAL/COMPLEX/CHARACTER/LOGICAL) at {self.current().line}:{self.current().col}")
        return decls
    def parse_implicit_statement(self):
        self.expect(TokenType.IMPLICIT)
        if self.match(TokenType.NONE):
            self.advance()
            return ImplicitNone()
        rules = []
        while True:
            if not self.match(TokenType.INTEGER, TokenType.REAL, TokenType.LOGICAL, TokenType.COMPLEX, TokenType.CHARACTER):
                raise SyntaxError(
                    f"Ожидается тип (INTEGER/REAL/LOGICAL/COMPLEX/CHARACTER) в IMPLICIT на строке {self.current().line}:{self.current().col}"
                )
            type_token = self.advance()
            type_name = type_token.value.upper()
            type_size = None
            if self.match(TokenType.STAR):
                self.advance()
                if self.match(TokenType.INTEGER_LIT):
                    type_size = self.advance().value
                else:
                    raise SyntaxError(
                        f"Ожидается целое число после '*' в IMPLICIT на строке {self.current().line}:{self.current().col}"
                    )
            self.expect(TokenType.LPAREN)
            letters = []
            while True:
                letter_spec = ""
                if self.match(TokenType.IDENTIFIER):
                    letter_token = self.advance()
                    letter_spec = letter_token.value.upper()
                    if len(letter_spec) == 1 and letter_spec.isalpha():
                        letters.append(letter_spec)
                    else:
                        raise SyntaxError(
                            f"Ожидается одна буква в IMPLICIT на строке {letter_token.line}:{letter_token.col}, получено '{letter_spec}'"
                        )
                elif self.match(TokenType.INTEGER_LIT):
                    raise SyntaxError(
                        f"Ожидается буква в IMPLICIT на строке {self.current().line}:{self.current().col}"
                    )
                else:
                    raise SyntaxError(
                        f"Ожидается буква в IMPLICIT на строке {self.current().line}:{self.current().col}"
                    )
                if self.match(TokenType.MINUS):
                    self.advance()
                    if self.match(TokenType.IDENTIFIER):
                        end_letter_token = self.advance()
                        end_letter = end_letter_token.value.upper()
                        if len(end_letter) == 1 and end_letter.isalpha():
                            letters[-1] = f"{letters[-1]}-{end_letter}"
                        else:
                            raise SyntaxError(
                                f"Ожидается одна буква в диапазоне IMPLICIT на строке {end_letter_token.line}:{end_letter_token.col}"
                            )
                    else:
                        raise SyntaxError(
                            f"Ожидается буква после '-' в диапазоне IMPLICIT на строке {self.current().line}:{self.current().col}"
                        )
                if not self.match(TokenType.COMMA):
                    break
                self.advance()
            self.expect(TokenType.RPAREN)
            rule = ImplicitRule(type_name, type_size, letters)
            rules.append(rule)
            if not self.match(TokenType.COMMA):
                break
            self.advance()
        return ImplicitStatement(rules)
    def parse_dimension_statement(self) -> 'DimensionStatement':
        self.expect(TokenType.DIMENSION)
        names = []
        while True:
            name_token = self.expect(TokenType.IDENTIFIER)
            name = name_token.value
            self.expect(TokenType.LPAREN)
            dim_ranges = []
            while True:
                if self.match(TokenType.INTEGER_LIT):
                    first_token = self.advance()
                    first_value = first_token.value
                    if self.match(TokenType.COLON):
                        self.advance()                
                        if not self.match(TokenType.INTEGER_LIT):
                            raise SyntaxError(
                                f"Ожидается целое число после ':' в диапазоне индексов на строке {self.current().line}:{self.current().col}"
                            )
                        second_token = self.advance()
                        second_value = second_token.value
                        dim_ranges.append((first_value, second_value))
                    elif self.match(TokenType.COMMA, TokenType.RPAREN):
                        dim_ranges.append((1, first_value))
                    else:
                        raise SyntaxError(
                            f"Ожидается ':' или ',' после размера массива на строке {self.current().line}:{self.current().col}"
                        )
                else:
                    raise SyntaxError(
                        f"Ожидается целое число для размера массива на строке {self.current().line}:{self.current().col}"
                    )
                if not self.match(TokenType.COMMA):
                    break
                self.advance()
            if len(dim_ranges) > 7:
                raise SyntaxError(
                    f"Массив '{name}' имеет {len(dim_ranges)} измерений, максимум допускается 7 на строке {name_token.line}:{name_token.col}"
                )
            for i, (k, l) in enumerate(dim_ranges):
                if k > l:
                    raise SyntaxError(
                        f"Неверный диапазон для измерения {i+1} массива '{name}': "
                        f"начальное значение ({k}) больше конечного ({l}) на строке {name_token.line}:{name_token.col}"
                    )
            self.expect(TokenType.RPAREN)
            names.append((name, dim_ranges))
            if not self.match(TokenType.COMMA):
                break
            self.advance()
        return DimensionStatement(names)
    def parse_parameter_statement(self) -> 'ParameterStatement':
        self.expect(TokenType.PARAMETER)
        self.expect(TokenType.LPAREN)
        params = []
        while True:
            name_token = self.expect(TokenType.IDENTIFIER)
            name = name_token.value
            self.expect(TokenType.ASSIGN_OP)
            value = self.parse_expression()
            params.append((name, value))
            if not self.match(TokenType.COMMA):
                break
            self.advance()
        self.expect(TokenType.RPAREN)
        return ParameterStatement(params)
    def parse_data_statement(self) -> 'DataStatement':
        self.expect(TokenType.DATA)
        items = []
        while True:
            vars_list = []
            while True:
                var_token = self.expect(TokenType.IDENTIFIER)
                var_name = var_token.value
                indices = []
                if self.match(TokenType.LPAREN):
                    self.advance()
                    while True:
                        indices.append(self.parse_expression())
                        if not self.match(TokenType.COMMA):
                            break
                        self.advance()
                    self.expect(TokenType.RPAREN)
                data_item = DataItem(var_name, indices)
                vars_list.append(data_item)
                if not self.match(TokenType.COMMA):
                    break
                self.advance()
            self.expect(TokenType.SLASH)
            values = []
            while True:
                values.append(self.parse_expression())
                if not self.match(TokenType.COMMA):
                    break
                self.advance()
            self.expect(TokenType.SLASH)
            items.append((vars_list, values))
            if not self.match(TokenType.COMMA):
                break
            self.advance()
        return DataStatement(items)
    def parse_statement(self) -> Statement:
        self.skip_comments()
        label = None
        if self.match(TokenType.INTEGER_LIT):
            saved_pos = self.pos
            label_token = self.advance()
            label = str(label_token.value)
            if not self.match(TokenType.IF, TokenType.DO, TokenType.PRINT, TokenType.READ,
                            TokenType.WRITE, TokenType.STOP, TokenType.GOTO, TokenType.CONTINUE,
                            TokenType.DATA, TokenType.IDENTIFIER):
                self.pos = saved_pos
                label = None
        if self.match(TokenType.IF):
            return self.parse_if_statement()
        elif self.match(TokenType.DO):
            return self.parse_do_loop()
        elif self.match(TokenType.PRINT):
            return self.parse_print_statement()
        elif self.match(TokenType.READ):
            return self.parse_read_statement()
        elif self.match(TokenType.WRITE):
            return self.parse_write_statement()
        elif self.match(TokenType.STOP):
            self.advance()
            return StopStatement()
        elif self.match(TokenType.GOTO):
            return self.parse_goto_statement()
        elif self.match(TokenType.CONTINUE):
            self.advance()
            return ContinueStatement(label)
        elif self.match(TokenType.DATA):
            data_stmt = self.parse_data_statement()
            return data_stmt
        elif self.match(TokenType.IDENTIFIER):
            return self.parse_assignment_or_label()
        else:
            raise SyntaxError(
                f"Unexpected token {self.current().type.name} "
                f"at {self.current().line}:{self.current().col}: {self.current().value}"
            )
    def parse_assignment_or_label(self) -> Statement:
        name_token = self.expect(TokenType.IDENTIFIER)
        name = name_token.value
        indices = []
        if self.match(TokenType.LPAREN):
            self.advance()
            while True:
                indices.append(self.parse_expression())
                if not self.match(TokenType.COMMA):
                    break
                self.advance()
            self.expect(TokenType.RPAREN)
        if self.match(TokenType.ASSIGN_OP):
            assign_token = self.advance()
            value = self.parse_expression()
            return Assignment(name, value, indices, name_token.line, name_token.col)
        else:
            raise SyntaxError(
                f"Expected assignment operator '=' after {name} at {self.current().line}:{self.current().col}"
            )
    def parse_assignment(self) -> Assignment:
        name_token = self.expect(TokenType.IDENTIFIER)
        name = name_token.value
        indices = []
        if self.match(TokenType.LPAREN):
            self.advance()
            while True:
                indices.append(self.parse_expression())
                if not self.match(TokenType.COMMA):
                    break
                self.advance()
            self.expect(TokenType.RPAREN)
        assign_token = self.expect(TokenType.ASSIGN_OP)
        value = self.parse_expression()
        return Assignment(name, value, indices, name_token.line, name_token.col)
    def parse_if_statement(self) -> Statement:
        self.expect(TokenType.IF)
        self.expect(TokenType.LPAREN)
        condition = self.parse_expression()
        self.expect(TokenType.RPAREN)
        if self.match(TokenType.INTEGER_LIT):
            label1_token = self.expect(TokenType.INTEGER_LIT)
            label1 = str(label1_token.value)
            self.expect(TokenType.COMMA)
            label2_token = self.expect(TokenType.INTEGER_LIT)
            label2 = str(label2_token.value)
            self.expect(TokenType.COMMA)
            label3_token = self.expect(TokenType.INTEGER_LIT)
            label3 = str(label3_token.value)
            return ArithmeticIfStatement(condition, label1, label2, label3)
        elif self.match(TokenType.THEN):
            self.advance()
            then_body = []
            while not self.is_if_terminator():
                then_body.append(self.parse_statement())
            elif_parts = []
            while self.is_else_if():
                self.advance()
                self.expect(TokenType.IF)
                self.expect(TokenType.LPAREN)
                elif_cond = self.parse_expression()
                self.expect(TokenType.RPAREN)
                self.expect(TokenType.THEN)
                elif_body = []
                while not self.is_if_terminator():
                    elif_body.append(self.parse_statement())
                elif_parts.append((elif_cond, elif_body))
            else_body = None
            if self.match(TokenType.ELSE):
                self.advance()
                if not self.is_else_if():
                    else_body = []
                    while not self.is_if_terminator():
                        else_body.append(self.parse_statement())
                else:
                    while self.is_else_if():
                        self.advance()
                        self.expect(TokenType.IF)
                        self.expect(TokenType.LPAREN)
                        elif_cond = self.parse_expression()
                        self.expect(TokenType.RPAREN)
                        self.expect(TokenType.THEN)
                        elif_body = []
                        while not self.is_if_terminator():
                            elif_body.append(self.parse_statement())
                        elif_parts.append((elif_cond, elif_body))
                    if self.match(TokenType.ELSE):
                        self.advance()
                        else_body = []
                        while not self.is_if_terminator():
                            else_body.append(self.parse_statement())
            self.expect(TokenType.ENDIF)
            return IfStatement(condition, then_body, elif_parts, else_body)
        else:
            statement = self.parse_statement()
            return SimpleIfStatement(condition, statement)
    def is_else_if(self) -> bool:
        if self.match(TokenType.ELSE):
            saved_pos = self.pos
            self.advance()
            if self.match(TokenType.IF):
                return True
            self.pos = saved_pos
        return False
    def is_if_terminator(self) -> bool:
        if self.match(TokenType.ELSE, TokenType.ENDIF):
            return True
        return False
    def parse_do_loop(self) -> Statement:
        self.expect(TokenType.DO)
        label = None
        if self.match(TokenType.INTEGER_LIT):
            label_token = self.expect(TokenType.INTEGER_LIT)
            label = str(label_token.value)
        if self.match(TokenType.WHILE):
            self.advance()
            self.expect(TokenType.LPAREN)
            condition = self.parse_expression()
            self.expect(TokenType.RPAREN)
            body = []
            while not self.match(TokenType.ENDDO, TokenType.EOF):
                if label and self.match(TokenType.INTEGER_LIT):
                    next_label_token = self.current()
                    if str(next_label_token.value) == label:
                        break
                if self.match(TokenType.END):
                    if self.peek().type == TokenType.DO:
                        break
                body.append(self.parse_statement())
            if label:
                label_token = self.expect(TokenType.INTEGER_LIT)
                if str(label_token.value) != label:
                    raise SyntaxError(
                        f"Expected label {label} but got {label_token.value} at {label_token.line}:{label_token.col}"
                    )
                self.expect(TokenType.CONTINUE)
                return LabeledDoWhile(label, condition, body)
            else:
                self.expect(TokenType.ENDDO)
                return DoWhile(condition, body)
        else:
            var_token = self.expect(TokenType.IDENTIFIER)
            var_name = var_token.value
            self.expect(TokenType.ASSIGN_OP)
            start = self.parse_expression()
            self.expect(TokenType.COMMA)
            end = self.parse_expression()
            step = None
            if self.match(TokenType.COMMA):
                self.advance()
                step = self.parse_expression()
            else:
                step = IntegerLiteral(1, 0, 0)
            body = []
            if label:
                while not self.match(TokenType.ENDDO, TokenType.EOF):
                    if self.match(TokenType.INTEGER_LIT):
                        next_label_token = self.current()
                        if str(next_label_token.value) == label:
                            break
                    if self.match(TokenType.END):
                        if self.peek().type == TokenType.DO:
                            break
                    body.append(self.parse_statement())
                label_token = self.expect(TokenType.INTEGER_LIT)
                if str(label_token.value) != label:
                    raise SyntaxError(
                        f"Expected label {label} but got {label_token.value} at {label_token.line}:{label_token.col}"
                    )
                self.expect(TokenType.CONTINUE)
                return LabeledDoLoop(label, var_name, start, end, step, body)
            else:
                while not self.match(TokenType.ENDDO, TokenType.EOF):
                    body.append(self.parse_statement())
                self.expect(TokenType.ENDDO)
                return DoLoop(var_name, start, end, step, body)
    def parse_print_statement(self) -> PrintStatement:
        self.expect(TokenType.PRINT)
        self.expect(TokenType.STAR)
        self.expect(TokenType.COMMA)
        items = []
        while not self.match(TokenType.ENDDO, TokenType.ENDIF, TokenType.END, TokenType.EOF,
                            TokenType.STOP, TokenType.CONTINUE, TokenType.GOTO):
            items.append(self.parse_expression())
            if not self.match(TokenType.COMMA):
                break
            self.advance()
        return PrintStatement(items)
    def parse_read_statement(self) -> ReadStatement:
        self.expect(TokenType.READ)
        if self.match(TokenType.LPAREN):
            self.advance()
            self.expect(TokenType.STAR)
            if self.match(TokenType.COMMA):
                self.advance()
                self.expect(TokenType.STAR)
            self.expect(TokenType.RPAREN)
        else:
            self.expect(TokenType.STAR)
        if self.match(TokenType.COMMA):
            self.advance()
        items = []
        while not self.match(TokenType.ENDDO, TokenType.ENDIF, TokenType.END, TokenType.EOF,
                            TokenType.STOP, TokenType.CONTINUE, TokenType.GOTO):
            item_token = self.expect(TokenType.IDENTIFIER)
            items.append(item_token.value)
            if not self.match(TokenType.COMMA):
                break
            self.advance()
        return ReadStatement("*", "*", items)
    def parse_write_statement(self) -> WriteStatement:
        self.expect(TokenType.WRITE)
        self.expect(TokenType.LPAREN)
        unit = "*"
        format_ = "*"
        self.expect(TokenType.STAR)
        if self.match(TokenType.COMMA):
            self.advance()
            self.expect(TokenType.STAR)
        self.expect(TokenType.RPAREN)
        items = []
        while not self.match(TokenType.ENDDO, TokenType.ENDIF, TokenType.END, TokenType.EOF,
                            TokenType.STOP, TokenType.CONTINUE, TokenType.GOTO):
            items.append(self.parse_expression())
            if not self.match(TokenType.COMMA):
                break
            self.advance()
        return WriteStatement(unit, format_, items)
    def parse_goto_statement(self) -> GotoStatement:
        self.expect(TokenType.GOTO)
        label_token = self.expect(TokenType.INTEGER_LIT)
        return GotoStatement(str(label_token.value))
    def parse_expression(self) -> Expression:
        return self.parse_eqv_expression()
    def parse_eqv_expression(self) -> Expression:
        left = self.parse_or_expression()
        while self.match(TokenType.EQV, TokenType.NEQV):
            op_token = self.advance()
            right = self.parse_or_expression()
            left = BinaryOp(left, op_token.value.upper(), right)
        return left
    def parse_or_expression(self) -> Expression:
        left = self.parse_and_expression()
        while self.match(TokenType.OR):
            op_token = self.advance()
            right = self.parse_and_expression()
            left = BinaryOp(left, op_token.value.upper(), right)
        return left
    def parse_and_expression(self) -> Expression:
        left = self.parse_not_expression()
        while self.match(TokenType.AND):
            op_token = self.advance()
            right = self.parse_not_expression()
            left = BinaryOp(left, op_token.value.upper(), right)
        return left
    def parse_not_expression(self) -> Expression:
        if self.match(TokenType.NOT):
            op_token = self.advance()
            expr = self.parse_not_expression()
            return UnaryOp(op_token.value.upper(), expr)
        return self.parse_relational_expression()
    def parse_relational_expression(self) -> Expression:
        left = self.parse_additive_expression()
        while self.match(TokenType.EQ, TokenType.NE, TokenType.LT, TokenType.LE,
                         TokenType.GT, TokenType.GE):
            op_token = self.advance()
            right = self.parse_additive_expression()
            left = BinaryOp(left, op_token.value.upper(), right)
        return left
    def parse_additive_expression(self) -> Expression:
        left = self.parse_concat_expression()
        while self.match(TokenType.PLUS, TokenType.MINUS):
            op_token = self.advance()
            if self.match(TokenType.PLUS, TokenType.MINUS, TokenType.STAR, TokenType.SLASH):
                next_op = self.current()
                raise SyntaxError(
                    f"Два знака арифметической операции не могут стоять рядом: "
                    f"'{op_token.value}' и '{next_op.value}' на строке {next_op.line}:{next_op.col}. "
                    f"Используйте скобки, например: X/(-Y) вместо X/-Y"
                )
            right = self.parse_concat_expression()
            left = BinaryOp(left, op_token.value, right)
        return left
    def parse_concat_expression(self) -> Expression:
        left = self.parse_multiplicative_expression()
        while self.match(TokenType.CONCAT):
            op_token = self.advance()
            right = self.parse_multiplicative_expression()
            left = BinaryOp(left, op_token.value, right)
        return left
    def parse_multiplicative_expression(self) -> Expression:
        left = self.parse_power_expression()
        while self.match(TokenType.STAR, TokenType.SLASH):
            op_token = self.advance()
            if self.match(TokenType.PLUS, TokenType.MINUS, TokenType.STAR, TokenType.SLASH):
                next_op = self.current()
                raise SyntaxError(
                    f"Два знака арифметической операции не могут стоять рядом: "
                    f"'{op_token.value}' и '{next_op.value}' на строке {next_op.line}:{next_op.col}. "
                    f"Используйте скобки, например: X/(-Y) вместо X/-Y"
                )
            right = self.parse_power_expression()
            left = BinaryOp(left, op_token.value, right)
        return left
    def parse_power_expression(self) -> Expression:
        left = self.parse_unary_expression()
        while self.match(TokenType.POWER):
            op_token = self.advance()
            right = self.parse_unary_expression()
            left = BinaryOp(left, op_token.value, right, op_token.line, op_token.col)
        return left
    def parse_unary_expression(self) -> Expression:
        if self.match(TokenType.PLUS, TokenType.MINUS):
            op_token = self.advance()
            if self.match(TokenType.PLUS, TokenType.MINUS, TokenType.STAR, TokenType.SLASH):
                next_op = self.current()
                raise SyntaxError(
                    f"Два знака арифметической операции не могут стоять рядом: "
                    f"'{op_token.value}' и '{next_op.value}' на строке {next_op.line}:{next_op.col}. "
                    f"Используйте скобки, например: X/(-Y) вместо X/-Y"
                )
            expr = self.parse_unary_expression()
            return UnaryOp(op_token.value, expr)
        return self.parse_primary_expression()
    def parse_primary_expression(self) -> Expression:
        if self.match(TokenType.LPAREN):
            self.advance()
            expr = self.parse_expression()
            self.expect(TokenType.RPAREN)
            return expr
        elif self.match(TokenType.INTEGER_LIT):
            token = self.advance()
            return IntegerLiteral(token.value, token.line, token.col)
        elif self.match(TokenType.REAL_LIT):
            token = self.advance()
            return RealLiteral(token.value, token.line, token.col)
        elif self.match(TokenType.STRING_LIT):
            token = self.advance()
            return StringLiteral(token.value, token.line, token.col)
        elif self.match(TokenType.TRUE):
            token = self.advance()
            return LogicalLiteral(True, token.line, token.col)
        elif self.match(TokenType.FALSE):
            token = self.advance()
            return LogicalLiteral(False, token.line, token.col)
        elif self.match(TokenType.SIN, TokenType.COS, TokenType.TAN,
                       TokenType.ASIN, TokenType.ACOS, TokenType.ATAN,
                       TokenType.ABS, TokenType.SQRT, TokenType.EXP,
                       TokenType.LOG, TokenType.LOG10,
                       TokenType.MIN, TokenType.MAX, TokenType.MOD,
                       TokenType.INT_FUNC, TokenType.REAL_FUNC, TokenType.FLOAT):
            func_token = self.advance()
            self.expect(TokenType.LPAREN)
            args = []
            while not self.match(TokenType.RPAREN):
                args.append(self.parse_expression())
                if not self.match(TokenType.COMMA):
                    break
                self.advance()
            self.expect(TokenType.RPAREN)
            return FunctionCall(func_token.value.upper(), args)
        elif self.match(TokenType.IDENTIFIER):
            name_token = self.advance()
            if self.match(TokenType.LPAREN):
                self.advance()
                indices = []
                while not self.match(TokenType.RPAREN):
                    indices.append(self.parse_expression())
                    if not self.match(TokenType.COMMA):
                        break
                    self.advance()
                self.expect(TokenType.RPAREN)
                return ArrayRef(name_token.value, indices, name_token.line, name_token.col)
            return Variable(name_token.value, name_token.line, name_token.col)
        raise SyntaxError(
            f"Unexpected token {self.current().type.name} "
            f"at {self.current().line}:{self.current().col}: {self.current().value}"
        )
def pretty_print_ast(node: ASTNode, indent: int = 0) -> str:
    prefix = "  " * indent
    if isinstance(node, Program):
        result = f"{prefix}PROGRAM {node.name}\n"
        for decl in node.declarations:
            result += pretty_print_ast(decl, indent + 1)
        for stmt in node.statements:
            result += pretty_print_ast(stmt, indent + 1)
        result += f"{prefix}END\n"
        return result
    elif isinstance(node, ImplicitNone):
        return f"{prefix}IMPLICIT NONE\n"
    elif isinstance(node, ImplicitStatement):
        result = f"{prefix}IMPLICIT:\n"
        for rule in node.rules:
            size_str = f"*{rule.type_size}" if rule.type_size else ""
            letters_str = ", ".join(rule.letters)
            result += f"{prefix}  {rule.type_name}{size_str}({letters_str})\n"
        return result
    elif isinstance(node, ImplicitRule):
        size_str = f"*{node.type_size}" if node.type_size else ""
        letters_str = ", ".join(node.letters)
        return f"{prefix}IMPLICIT RULE: {node.type_name}{size_str}({letters_str})\n"
    elif isinstance(node, DimensionStatement):
        result = f"{prefix}DIMENSION:\n"
        for name, dim_ranges in node.names:
            def format_dim(dim_range: Tuple[int, int]) -> str:
                k, l = dim_range
                if k == 1:
                    return str(l)                      
                return f"{k}:{l}"
            dims_str = "(" + ", ".join(format_dim(d) for d in dim_ranges) + ")"
            result += f"{prefix}  {name}{dims_str}\n"
        return result
    elif isinstance(node, ParameterStatement):
        result = f"{prefix}PARAMETER:\n"
        for name, expr in node.params:
            result += f"{prefix}  {name} = {pretty_print_ast(expr, indent + 2)}"
        return result
    elif isinstance(node, DataStatement):
        result = f"{prefix}DATA:\n"
        for vars_list, values in node.items:
            result += f"{prefix}  variables/arrays: {', '.join(str(item) for item in vars_list)}\n"
            result += f"{prefix}  values:\n"
            for val in values:
                result += f"{prefix}    {pretty_print_ast(val, indent + 3)}"
        return result
    elif isinstance(node, DataItem):
        if node.indices:
            indices_str = "(" + ", ".join(pretty_print_ast(idx, 0).strip() for idx in node.indices) + ")"
            return f"{prefix}DATA_ITEM: {node.name}{indices_str}\n"
        return f"{prefix}DATA_ITEM: {node.name}\n"
    elif isinstance(node, Declaration):
        result = f"{prefix}DECLARATION: {node.type}"
        names_parts = []
        for name, dim_ranges in node.names:
            if dim_ranges:
                def format_dim(dim_range: Tuple[int, int]) -> str:
                    k, l = dim_range
                    if k == 1:
                        return str(l)                      
                    return f"{k}:{l}"
                dims_str = "(" + ", ".join(format_dim(d) for d in dim_ranges) + ")"
                names_parts.append(f"{name}{dims_str}")
            else:
                names_parts.append(name)
        result += f" {', '.join(names_parts)}\n"
        return result
    elif isinstance(node, Assignment):
        coord_str = f" [{node.line}:{node.col}]" if node.line > 0 else ""
        result = f"{prefix}ASSIGNMENT:{coord_str}\n"
        result += f"{prefix}  target: {node.target}"
        if node.indices:
            indices_str = "[" + ", ".join(pretty_print_ast(idx, 0).strip() for idx in node.indices) + "]"
            result += indices_str
        result += "\n"
        result += f"{prefix}  value: {pretty_print_ast(node.value, indent + 1)}"
        return result
    elif isinstance(node, DoLoop):
        result = f"{prefix}DO LOOP:\n"
        result += f"{prefix}  variable: {node.var}\n"
        result += f"{prefix}  start: {pretty_print_ast(node.start, indent + 1)}"
        result += f"{prefix}  end: {pretty_print_ast(node.end, indent + 1)}"
        if node.step:
            result += f"{prefix}  step: {pretty_print_ast(node.step, indent + 1)}"
        result += f"{prefix}  body:\n"
        for stmt in node.body:
            result += pretty_print_ast(stmt, indent + 2)
        result += f"{prefix}END DO\n"
        return result
    elif isinstance(node, DoWhile):
        result = f"{prefix}DO WHILE:\n"
        result += f"{prefix}  condition: {pretty_print_ast(node.condition, indent + 1)}"
        result += f"{prefix}  body:\n"
        for stmt in node.body:
            result += pretty_print_ast(stmt, indent + 2)
        result += f"{prefix}END DO\n"
        return result
    elif isinstance(node, SimpleIfStatement):
        result = f"{prefix}SIMPLE IF STATEMENT:\n"
        result += f"{prefix}  condition: {pretty_print_ast(node.condition, indent + 1)}"
        result += f"{prefix}  statement:\n"
        result += pretty_print_ast(node.statement, indent + 2)
        return result
    elif isinstance(node, IfStatement):
        result = f"{prefix}IF STATEMENT:\n"
        result += f"{prefix}  condition: {pretty_print_ast(node.condition, indent + 1)}"
        result += f"{prefix}  THEN:\n"
        for stmt in node.then_body:
            result += pretty_print_ast(stmt, indent + 2)
        if node.elif_parts:
            for i, (cond, body) in enumerate(node.elif_parts):
                result += f"{prefix}  ELSE IF {i+1}:\n"
                result += f"{prefix}    condition: {pretty_print_ast(cond, indent + 2)}"
                for stmt in body:
                    result += pretty_print_ast(stmt, indent + 3)
        if node.else_body:
            result += f"{prefix}  ELSE:\n"
            for stmt in node.else_body:
                result += pretty_print_ast(stmt, indent + 2)
        result += f"{prefix}END IF\n"
        return result
    elif isinstance(node, PrintStatement):
        result = f"{prefix}PRINT:\n"
        for i, item in enumerate(node.items):
            result += f"{prefix}  item[{i}]: {pretty_print_ast(item, indent + 1)}"
        return result
    elif isinstance(node, ReadStatement):
        result = f"{prefix}READ:\n"
        result += f"{prefix}  unit: {node.unit}\n"
        result += f"{prefix}  format: {node.format}\n"
        result += f"{prefix}  items: {', '.join(node.items)}\n"
        return result
    elif isinstance(node, WriteStatement):
        result = f"{prefix}WRITE:\n"
        result += f"{prefix}  unit: {node.unit}\n"
        result += f"{prefix}  format: {node.format}\n"
        result += f"{prefix}  items:\n"
        for i, item in enumerate(node.items):
            result += f"{prefix}    item[{i}]: {pretty_print_ast(item, indent + 2)}"
        return result
    elif isinstance(node, CallStatement):
        result = f"{prefix}CALL {node.name}(\n"
        for i, arg in enumerate(node.args):
            result += f"{prefix}  arg[{i}]: {pretty_print_ast(arg, indent + 1)}"
        result += f"{prefix})\n"
        return result
    elif isinstance(node, ReturnStatement):
        return f"{prefix}RETURN\n"
    elif isinstance(node, StopStatement):
        return f"{prefix}STOP\n"
    elif isinstance(node, GotoStatement):
        return f"{prefix}GOTO {node.label}\n"
    elif isinstance(node, ContinueStatement):
        label_str = f" ({node.label})" if node.label else ""
        return f"{prefix}CONTINUE{label_str}\n"
    elif isinstance(node, BinaryOp):
        coord_str = f" [{node.line}:{node.col}]" if node.line > 0 else ""
        result = f"{prefix}BINARY_OP: {node.op}{coord_str}\n"
        result += f"{prefix}  left: {pretty_print_ast(node.left, indent + 1)}"
        result += f"{prefix}  right: {pretty_print_ast(node.right, indent + 1)}"
        return result
    elif isinstance(node, UnaryOp):
        result = f"{prefix}UNARY_OP: {node.op}\n"
        result += f"{prefix}  operand: {pretty_print_ast(node.operand, indent + 1)}"
        return result
    elif isinstance(node, FunctionCall):
        result = f"{prefix}FUNCTION_CALL: {node.name}(\n"
        for i, arg in enumerate(node.args):
            result += f"{prefix}  arg[{i}]: {pretty_print_ast(arg, indent + 1)}"
        result += f"{prefix})\n"
        return result
    elif isinstance(node, ArrayRef):
        result = f"{prefix}ARRAY_REF: {node.name}[\n"
        for i, idx in enumerate(node.indices):
            result += f"{prefix}  index[{i}]: {pretty_print_ast(idx, indent + 1)}"
        result += f"{prefix}]\n"
        return result
    elif isinstance(node, Variable):
        coord_str = f" [{node.line}:{node.col}]" if node.line > 0 else ""
        return f"{prefix}VARIABLE: {node.name} (value: {node.name}){coord_str}\n"
    elif isinstance(node, IntegerLiteral):
        coord_str = f" [{node.line}:{node.col}]" if node.line > 0 else ""
        return f"{prefix}INTEGER_LITERAL: {node.value} (value: {node.value}){coord_str}\n"
    elif isinstance(node, RealLiteral):
        coord_str = f" [{node.line}:{node.col}]" if node.line > 0 else ""
        return f"{prefix}REAL_LITERAL: {node.value} (value: {node.value}){coord_str}\n"
    elif isinstance(node, StringLiteral):
        coord_str = f" [{node.line}:{node.col}]" if node.line > 0 else ""
        return f"{prefix}STRING_LITERAL: {repr(node.value)} (value: {node.value}){coord_str}\n"
    elif isinstance(node, LogicalLiteral):
        val_str = ".TRUE." if node.value else ".FALSE."
        coord_str = f" [{node.line}:{node.col}]" if node.line > 0 else ""
        return f"{prefix}LOGICAL_LITERAL: {val_str} (value: {node.value}){coord_str}\n"
    else:
        return f"{prefix}{node}\n"
def main():
    test_code = """
PROGRAM FACTORIAL
    INTEGER N, I, F
    N = 5
    F = 1
    DO I = 1, N
        F = F * I
    END DO
    PRINT *, F
END
"""
    print("=== TESTING LEXER ===")
    lexer = Lexer(test_code)
    tokens = lexer.tokenize()
    print("Tokens:")
    for token in tokens:
        print(f"  {token}")
    print("\n=== TESTING PARSER ===")
    parser = Parser(tokens)
    ast = parser.parse()
    print("AST:")
    print(pretty_print_ast(ast))
if __name__ == "__main__":
    main()
