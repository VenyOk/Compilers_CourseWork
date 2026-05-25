from typing import List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from src.frontend.lexer import Token, TokenType
from src.frontend.ast import (
    ASTNode, Program, Subroutine, FunctionDef, Declaration, ImplicitNone,
    ImplicitRule, ImplicitStatement, DimensionStatement, ParameterStatement,
    Statement, DataItem, DataStatement, Assignment, DoLoop,
    DoWhile, SimpleIfStatement, IfStatement, PrintStatement, ReadStatement,
    WriteStatement, CallStatement, ReturnStatement, StopStatement, GotoStatement,
    ContinueStatement, ExternalStatement, CommonStatement, ExitStatement,
    ArithmeticIfStatement, LabeledDoLoop, LabeledDoWhile, Expression, BinaryOp,
    UnaryOp, FunctionCall, ArrayRef, Variable, IntegerLiteral, RealLiteral,
    StringLiteral, LogicalLiteral, ComplexLiteral, format_dimension_bound,
    format_dimension_spec, format_dimension_list
)
from src.config import MAX_DIMENSIONS
from src.utils.logger import debug, info, warning, error


class Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0

    def current(self) -> Token:
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return Token(type=TokenType.EOF, value=None, line=0, col=0)

    def peek(self, offset: int = 1) -> Token:
        pos = self.pos + offset
        if pos < len(self.tokens):
            return self.tokens[pos]
        return Token(type=TokenType.EOF, value=None, line=0, col=0)

    def advance(self) -> Token:
        token = self.current()
        self.pos += 1
        return token

    def expect(self, token_type: TokenType) -> Token:
        if self.current().type != token_type:
            current = self.current()
            token_names = {
                TokenType.IDENTIFIER: "",
                TokenType.INTEGER_LIT: " ",
                TokenType.REAL_LIT: " ",
                TokenType.STRING_LIT: " ",
                TokenType.LPAREN: "'('",
                TokenType.RPAREN: "')'",
                TokenType.COMMA: "','",
                TokenType.COLON: "':'",
                TokenType.ASSIGN_OP: "'='",
                TokenType.PLUS: "'+'",
                TokenType.MINUS: "'-'",
                TokenType.STAR: "'*'",
                TokenType.SLASH: "'/'",
                TokenType.END: "END",
                TokenType.THEN: "THEN",
                TokenType.ELSE: "ELSE",
                TokenType.DO: "DO",
            }
            expected_name = token_names.get(token_type, token_type.name)
            got_name = token_names.get(current.type, current.type.name)
            got_value = f" '{current.value}'" if current.value else ""
            raise SyntaxError(
                f"[ {current.line},  {current.col}]  {expected_name}, "
                f"  {got_name}{got_value}. "
                f"  . ,       ."
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
                    f"IMPLICIT         {self.current().line}:{self.current().col}"
                )
            declarations.append(self.parse_implicit_statement())
            implicit_found = True
        while self.match(TokenType.INTEGER, TokenType.REAL, TokenType.LOGICAL, TokenType.COMPLEX, TokenType.CHARACTER,
                         TokenType.DOUBLEPRECISION, TokenType.DIMENSION, TokenType.PARAMETER, TokenType.DATA,
                         TokenType.EXTERNAL, TokenType.COMMON):
            other_declarations_found = True
            if self.match(TokenType.DIMENSION):
                declarations.append(self.parse_dimension_statement())
            elif self.match(TokenType.PARAMETER):
                declarations.append(self.parse_parameter_statement())
            elif self.match(TokenType.DATA):
                declarations.append(self.parse_data_statement())
            elif self.match(TokenType.EXTERNAL):
                declarations.append(self.parse_external_statement())
            elif self.match(TokenType.COMMON):
                declarations.append(self.parse_common_statement())
            else:
                declarations.extend(self.parse_declaration())
            self.skip_comments()
            if not self.match(TokenType.INTEGER, TokenType.REAL, TokenType.LOGICAL, TokenType.COMPLEX, TokenType.CHARACTER,
                             TokenType.DOUBLEPRECISION, TokenType.DIMENSION, TokenType.PARAMETER, TokenType.DATA,
                             TokenType.EXTERNAL, TokenType.COMMON):
                break

        if (not name or name == "MAIN"):
            next_token = self.current()
            if (next_token.type == TokenType.SUBROUTINE or
                next_token.type == TokenType.FUNCTION or
                (next_token.type in (TokenType.INTEGER, TokenType.REAL, TokenType.LOGICAL, TokenType.DOUBLEPRECISION) and
                 self.peek().type == TokenType.FUNCTION)):
                subroutines = []
                functions = []
                while not self.match(TokenType.EOF):
                    self.skip_comments()
                    if self.match(TokenType.EOF):
                        break
                    if self.match(TokenType.SUBROUTINE):
                        subroutines.append(self.parse_subroutine())
                    elif self.match(TokenType.INTEGER, TokenType.REAL, TokenType.LOGICAL, TokenType.DOUBLEPRECISION):
                        saved_pos = self.pos
                        self.advance()
                        if self.match(TokenType.FUNCTION):
                            self.pos = saved_pos
                            functions.append(self.parse_function())
                        else:
                            self.pos = saved_pos
                            break
                    elif self.match(TokenType.FUNCTION):
                        functions.append(self.parse_function())
                    else:
                        break
                return Program(name="", declarations=declarations, statements=[],
                              statement_functions=[], subroutines=subroutines, functions=functions)

        statements = []
        statement_functions = []
        while not self.match(TokenType.END, TokenType.EOF):
            self.skip_comments()
            if self.match(TokenType.END, TokenType.EOF):
                break
            if self.match(TokenType.INTEGER, TokenType.REAL, TokenType.LOGICAL, TokenType.COMPLEX, TokenType.CHARACTER,
                         TokenType.DOUBLEPRECISION, TokenType.DIMENSION, TokenType.PARAMETER):
                break
            stmt = self.parse_statement()
            if stmt:
                statements.append(stmt)
        self.expect(TokenType.END)

        subroutines = []
        functions = []
        while not self.match(TokenType.EOF):
            self.skip_comments()
            if self.match(TokenType.EOF):
                break
            if self.match(TokenType.SUBROUTINE):
                subroutines.append(self.parse_subroutine())
            elif self.match(TokenType.INTEGER, TokenType.REAL, TokenType.LOGICAL, TokenType.DOUBLEPRECISION):
                saved_pos = self.pos
                self.advance()
                if self.match(TokenType.FUNCTION):
                    self.pos = saved_pos
                    functions.append(self.parse_function())
                else:
                    self.pos = saved_pos
                    break
            elif self.match(TokenType.FUNCTION):
                functions.append(self.parse_function())
            else:
                break

        return Program(name=name, declarations=declarations, statements=statements,
                      statement_functions=statement_functions, subroutines=subroutines, functions=functions)

    def parse_declaration(self) -> List[Declaration]:
        decls = []
        if self.match(TokenType.INTEGER, TokenType.REAL, TokenType.LOGICAL, TokenType.COMPLEX, TokenType.CHARACTER, TokenType.DOUBLEPRECISION):
            type_token = self.advance()
            type_name = type_token.value.upper()
            if type_name == "DOUBLEPRECISION":
                type_name = "REAL"
            type_size = None
            if self.match(TokenType.STAR):
                self.advance()
                if self.match(TokenType.INTEGER_LIT):
                    size = self.advance().value
                    type_size = size
                    type_name = f"{type_name}*{size}"
            names = []
            while True:
                if self.match(TokenType.INTEGER_LIT):
                    int_token = self.current()
                    saved_pos = self.pos
                    saved_line = self.line
                    saved_col = self.col
                    self.advance()
                    if self.match(TokenType.IDENTIFIER):
                        ident_token = self.current()
                        raise SyntaxError(
                            f"  '{int_token.value}{ident_token.value}'        {int_token.line}:{int_token.col}"
                        )
                    self.pos = saved_pos
                    self.line = saved_line
                    self.col = saved_col
                name_token = self.expect(TokenType.IDENTIFIER)
                name = name_token.value
                dim_ranges = None
                if self.match(TokenType.LPAREN):
                    self.advance()
                    dim_ranges = self.parse_dimension_specs()
                    if len(dim_ranges) > 7:
                        raise SyntaxError(
                            f" '{name}'  {len(dim_ranges)} ,   7   {name_token.line}:{name_token.col}"
                        )
                    self.expect(TokenType.RPAREN)
                names.append((name, dim_ranges))
                if not self.match(TokenType.COMMA):
                    break
                self.advance()
            decls.append(Declaration(type=type_name,
                         names=names, type_size=type_size))
        else:
            raise SyntaxError(
                f"   (INTEGER/REAL/COMPLEX/CHARACTER/LOGICAL)   {self.current().line}:{self.current().col}")
        return decls

    def parse_implicit_statement(self):
        self.expect(TokenType.IMPLICIT)
        if self.match(TokenType.NONE):
            self.advance()
            return ImplicitNone()
        rules = []
        while True:
            if not self.match(TokenType.INTEGER, TokenType.REAL, TokenType.LOGICAL, TokenType.COMPLEX, TokenType.CHARACTER, TokenType.DOUBLEPRECISION):
                raise SyntaxError(
                    f"  (INTEGER/REAL/LOGICAL/COMPLEX/CHARACTER/DOUBLE PRECISION)  IMPLICIT   {self.current().line}:{self.current().col}"
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
                        f"    '*'  IMPLICIT   {self.current().line}:{self.current().col}"
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
                            f"    IMPLICIT   {letter_token.line}:{letter_token.col},  '{letter_spec}'"
                        )
                elif self.match(TokenType.INTEGER_LIT):
                    raise SyntaxError(
                        f"   IMPLICIT   {self.current().line}:{self.current().col}"
                    )
                else:
                    raise SyntaxError(
                        f"   IMPLICIT   {self.current().line}:{self.current().col}"
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
                                f"     IMPLICIT   {end_letter_token.line}:{end_letter_token.col}"
                            )
                    else:
                        raise SyntaxError(
                            f"   '-'   IMPLICIT   {self.current().line}:{self.current().col}"
                        )
                if not self.match(TokenType.COMMA):
                    break
                self.advance()
            self.expect(TokenType.RPAREN)
            rule = ImplicitRule(type_name=type_name,
                                type_size=type_size, letters=letters)
            rules.append(rule)
            if not self.match(TokenType.COMMA):
                break
            self.advance()
        return ImplicitStatement(rules=rules)

    def parse_dimension_statement(self) -> DimensionStatement:
        self.expect(TokenType.DIMENSION)
        names = []
        while True:
            name_token = self.expect(TokenType.IDENTIFIER)
            name = name_token.value
            self.expect(TokenType.LPAREN)
            dim_ranges = self.parse_dimension_specs()
            if len(dim_ranges) > 7:
                raise SyntaxError(
                    f" '{name}'  {len(dim_ranges)} ,   7   {name_token.line}:{name_token.col}"
                )
            self.expect(TokenType.RPAREN)
            names.append((name, dim_ranges))
            if not self.match(TokenType.COMMA):
                break
            self.advance()
        return DimensionStatement(names=names)

    def parse_dimension_specs(self) -> List[Tuple[object, object]]:
        dim_ranges = []
        while True:
            lower_expr = self.parse_expression()
            if self.match(TokenType.COLON):
                self.advance()
                upper_expr = self.parse_expression()
                dim_ranges.append((lower_expr, upper_expr))
            else:
                dim_ranges.append((1, lower_expr))
            if not self.match(TokenType.COMMA):
                break
            self.advance()
        return dim_ranges

    def parse_external_statement(self) -> ExternalStatement:
        self.expect(TokenType.EXTERNAL)
        names = []
        while True:
            names.append(self.expect(TokenType.IDENTIFIER).value)
            if not self.match(TokenType.COMMA):
                break
            self.advance()
        return ExternalStatement(names=names)

    def parse_common_statement(self) -> CommonStatement:
        self.expect(TokenType.COMMON)
        blocks = []
        while True:
            block_name = ""
            if self.match(TokenType.SLASH):
                self.advance()
                if self.match(TokenType.IDENTIFIER):
                    block_name = self.advance().value
                self.expect(TokenType.SLASH)
            variables = []
            while True:
                name_token = self.expect(TokenType.IDENTIFIER)
                variables.append(Variable(name=name_token.value, line=name_token.line, col=name_token.col))
                if self.match(TokenType.COMMA) and self.peek().type == TokenType.IDENTIFIER:
                    self.advance()
                    continue
                break
            blocks.append((block_name, variables))
            if self.match(TokenType.COMMA) and self.peek().type == TokenType.SLASH:
                self.advance()
                continue
            if self.match(TokenType.SLASH):
                continue
            break
        return CommonStatement(blocks=blocks)

    def parse_parameter_statement(self) -> ParameterStatement:
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
        return ParameterStatement(params=params)

    def parse_data_statement(self) -> DataStatement:
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
                values.append(self.parse_expression_until({TokenType.COMMA, TokenType.SLASH}))
                if not self.match(TokenType.COMMA):
                    break
                self.advance()
            self.expect(TokenType.SLASH)
            items.append((vars_list, values))
            if not self.match(TokenType.COMMA):
                break
            self.advance()
        return DataStatement(items=items)

    def parse_expression_until(self, stop_tokens: Set[TokenType]) -> Expression:
        start = self.pos
        depth = 0
        while True:
            current = self.current()
            if current.type == TokenType.EOF:
                break
            if depth == 0 and current.type in stop_tokens:
                break
            if current.type == TokenType.LPAREN:
                depth += 1
            elif current.type == TokenType.RPAREN and depth > 0:
                depth -= 1
            self.pos += 1
        expr_tokens = self.tokens[start:self.pos]
        if not expr_tokens:
            raise SyntaxError(
                f"  {self.current().type.name} "
                f"  {self.current().line}:{self.current().col}: {self.current().value}"
            )
        tail = self.current()
        expr_parser = Parser(expr_tokens + [Token(type=TokenType.EOF, value=None, line=tail.line, col=tail.col)])
        expr = expr_parser.parse_expression()
        if expr_parser.current().type != TokenType.EOF:
            raise SyntaxError(
                f"  {expr_parser.current().type.name} "
                f"  {expr_parser.current().line}:{expr_parser.current().col}: {expr_parser.current().value}"
            )
        return expr

    def parse_statement(self) -> Statement:
        self.skip_comments()
        label = None
        if self.match(TokenType.INTEGER_LIT):
            saved_pos = self.pos
            label_token = self.advance()
            label = str(label_token.value)
            if not self.match(TokenType.IF, TokenType.DO, TokenType.PRINT, TokenType.READ,
                              TokenType.WRITE, TokenType.STOP, TokenType.GOTO, TokenType.CONTINUE,
                              TokenType.DATA, TokenType.CALL, TokenType.RETURN, TokenType.EXIT, TokenType.IDENTIFIER):
                self.pos = saved_pos
                label = None
        if self.match(TokenType.IF):
            stmt = self.parse_if_statement()
            if label:
                stmt.stmt_label = label
            return stmt
        elif self.match(TokenType.DO):
            stmt = self.parse_do_loop()
            if label:
                stmt.stmt_label = label
            return stmt
        elif self.match(TokenType.PRINT):
            stmt = self.parse_print_statement()
            if label:
                stmt.stmt_label = label
            return stmt
        elif self.match(TokenType.READ):
            stmt = self.parse_read_statement()
            if label:
                stmt.stmt_label = label
            return stmt
        elif self.match(TokenType.WRITE):
            stmt = self.parse_write_statement()
            if label:
                stmt.stmt_label = label
            return stmt
        elif self.match(TokenType.STOP):
            self.advance()
            stmt = StopStatement()
            if label:
                stmt.stmt_label = label
            return stmt
        elif self.match(TokenType.GOTO):
            stmt = self.parse_goto_statement()
            if label:
                stmt.stmt_label = label
            return stmt
        elif self.match(TokenType.CONTINUE):
            self.advance()
            stmt = ContinueStatement(label=label)
            if label:
                stmt.stmt_label = label
            return stmt
        elif self.match(TokenType.DATA):
            data_stmt = self.parse_data_statement()
            if label:
                data_stmt.stmt_label = label
            return data_stmt
        elif self.match(TokenType.CALL):
            stmt = self.parse_call_statement()
            if label:
                stmt.stmt_label = label
            return stmt
        elif self.match(TokenType.RETURN):
            self.advance()
            stmt = ReturnStatement()
            if label:
                stmt.stmt_label = label
            return stmt
        elif self.match(TokenType.EXIT):
            self.advance()
            stmt = ExitStatement()
            if label:
                stmt.stmt_label = label
            return stmt
        elif self.match(TokenType.IDENTIFIER):
            stmt = self.parse_assignment_or_label()
            if label:
                stmt.stmt_label = label
            return stmt
        elif self.match(TokenType.END, TokenType.ENDIF, TokenType.ENDDO, TokenType.ELSE, TokenType.ELSEIF):
            return None
        else:
            current = self.current()
            raise SyntaxError(
                f"[ {current.line},  {current.col}]   {current.type.name}"
                f"{f' ({current.value})' if current.value else ''}. "
                f"        . "
                f"  . ,  ,      ."
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
            return Assignment(target=name, value=value, indices=indices, line=name_token.line, col=name_token.col)
        else:
            current = self.current()
            raise SyntaxError(
                f"[ {current.line},  {current.col}]    '='   '{name}'. "
                f": {current.type.name}{f' ({current.value})' if current.value else ''}. "
                f",     '{name}'   '='   ."
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
            return ArithmeticIfStatement(condition=condition, label_neg=label1, label_zero=label2, label_pos=label3)
        elif self.match(TokenType.THEN):
            self.advance()
            then_body = []
            while not self.is_if_terminator():
                stmt = self.parse_statement()
                if stmt is None:
                    if self.is_if_terminator():
                        break
                    continue
                then_body.append(stmt)
            elif_parts = []
            while self.match(TokenType.ELSEIF) or self.is_else_if():
                if self.match(TokenType.ELSEIF):
                    self.advance()
                else:
                    self.advance()
                    if not self.match(TokenType.IF):
                        break
                    self.expect(TokenType.IF)
                if not self.match(TokenType.LPAREN):
                    break
                self.expect(TokenType.LPAREN)
                elif_cond = self.parse_expression()
                self.expect(TokenType.RPAREN)
                self.expect(TokenType.THEN)
                elif_body = []
                while not self.is_if_terminator():
                    stmt = self.parse_statement()
                    if stmt is None:
                        if self.is_if_terminator():
                            break
                        continue
                    elif_body.append(stmt)
                elif_parts.append((elif_cond, elif_body))
            else_body = None
            if self.match(TokenType.ELSE):
                self.advance()
                if not (self.match(TokenType.ELSEIF) or self.is_else_if()):
                    else_body = []
                    while not self.is_if_terminator():
                        stmt = self.parse_statement()
                        if stmt is None:
                            if self.is_if_terminator():
                                break
                            continue
                        else_body.append(stmt)
                else:
                    while self.match(TokenType.ELSEIF) or self.is_else_if():
                        if self.match(TokenType.ELSEIF):
                            self.advance()
                        else:
                            self.advance()
                            if not self.match(TokenType.IF):
                                break
                            self.expect(TokenType.IF)
                        if not self.match(TokenType.LPAREN):
                            break
                        self.expect(TokenType.LPAREN)
                        elif_cond = self.parse_expression()
                        self.expect(TokenType.RPAREN)
                        self.expect(TokenType.THEN)
                        elif_body = []
                        while not self.is_if_terminator():
                            stmt = self.parse_statement()
                            if stmt is None:
                                if self.is_if_terminator():
                                    break
                                continue
                            elif_body.append(stmt)
                        elif_parts.append((elif_cond, elif_body))
                    if self.match(TokenType.ELSE):
                        self.advance()
                        else_body = []
                        while not self.is_if_terminator():
                            stmt = self.parse_statement()
                            if stmt is None:
                                break
                            else_body.append(stmt)
            if self.match(TokenType.ENDIF):
                self.advance()
            elif self.match(TokenType.END):
                self.advance()
                if not self.match(TokenType.IF):
                    raise SyntaxError(
                        f" IF  END   {self.current().line}:{self.current().col}"
                    )
                self.advance()
            elif self.match(TokenType.IF) and self.pos > 0:
                prev_token = self.tokens[self.pos - 1] if self.pos > 0 else None
                if prev_token and prev_token.type == TokenType.END:
                    self.advance()
                else:
                    raise SyntaxError(
                        f" ENDIF  END IF   {self.current().line}:{self.current().col}"
                    )
            else:
                raise SyntaxError(
                    f" ENDIF  END IF   {self.current().line}:{self.current().col}"
                )
            return IfStatement(condition=condition, then_body=then_body, elif_parts=elif_parts, else_body=else_body)
        else:
            statement = self.parse_statement()
            return SimpleIfStatement(condition=condition, statement=statement)

    def is_else_if(self) -> bool:
        if self.match(TokenType.ELSE):
            saved_pos = self.pos
            self.advance()
            if self.match(TokenType.IF):
                return True
            self.pos = saved_pos
        return False

    def is_if_terminator(self) -> bool:
        if self.match(TokenType.ELSE, TokenType.ELSEIF, TokenType.ENDIF):
            return True
        if self.match(TokenType.END):
            saved_pos = self.pos
            self.advance()
            if self.match(TokenType.IF):
                return True
            self.pos = saved_pos
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
            while not self.match(TokenType.ENDDO, TokenType.END, TokenType.EOF):
                if label and self.match(TokenType.INTEGER_LIT):
                    next_label_token = self.current()
                    if str(next_label_token.value) == label:
                        break
                if self.match(TokenType.END):
                    saved_pos = self.pos
                    self.advance()
                    if self.match(TokenType.DO):
                        break
                    self.pos = saved_pos
                body.append(self.parse_statement())
            if label:
                label_token = self.expect(TokenType.INTEGER_LIT)
                if str(label_token.value) != label:
                    raise SyntaxError(
                        f"  {label},  {label_token.value}   {label_token.line}:{label_token.col}"
                    )
                self.expect(TokenType.CONTINUE)
                return LabeledDoWhile(label=label, condition=condition, body=body)
            else:
                if self.match(TokenType.ENDDO):
                    self.advance()
                elif self.match(TokenType.END):
                    self.advance()
                    self.expect(TokenType.DO)
                else:
                    raise SyntaxError(
                        f" ENDDO  END DO   {self.current().line}:{self.current().col}"
                    )
                return DoWhile(condition=condition, body=body)
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
                step = IntegerLiteral(value=1, line=0, col=0)
            body = []
            if label:
                while not self.match(TokenType.ENDDO, TokenType.END, TokenType.EOF):
                    if self.match(TokenType.INTEGER_LIT):
                        next_label_token = self.current()
                        if str(next_label_token.value) == label:
                            break
                    if self.match(TokenType.END):
                        saved_pos = self.pos
                        self.advance()
                        if self.match(TokenType.DO):
                            break
                        self.pos = saved_pos
                    body.append(self.parse_statement())
                label_token = self.expect(TokenType.INTEGER_LIT)
                if str(label_token.value) != label:
                    raise SyntaxError(
                        f"  {label},  {label_token.value}   {label_token.line}:{label_token.col}"
                    )
                self.expect(TokenType.CONTINUE)
                return LabeledDoLoop(label=label, var=var_name, start=start, end=end, step=step, body=body)
            else:
                while not self.match(TokenType.ENDDO, TokenType.END, TokenType.EOF):
                    if self.match(TokenType.END):
                        saved_pos = self.pos
                        self.advance()
                        if self.match(TokenType.DO):
                            break
                        self.pos = saved_pos
                    body.append(self.parse_statement())
                if self.match(TokenType.ENDDO):
                    self.advance()
                elif self.match(TokenType.END):
                    self.advance()
                    self.expect(TokenType.DO)
                else:
                    raise SyntaxError(
                        f" ENDDO  END DO   {self.current().line}:{self.current().col}"
                    )
                return DoLoop(var=var_name, start=start, end=end, step=step, body=body)

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
        return PrintStatement(items=items)

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
        return ReadStatement(unit="*", format="*", items=items)

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
        return WriteStatement(unit=unit, format=format_, items=items)

    def parse_goto_statement(self) -> GotoStatement:
        self.expect(TokenType.GOTO)
        label_token = self.expect(TokenType.INTEGER_LIT)
        return GotoStatement(label=str(label_token.value))

    def parse_call_statement(self) -> CallStatement:
        self.expect(TokenType.CALL)
        name_token = self.expect(TokenType.IDENTIFIER)
        name = name_token.value
        args = []
        if self.match(TokenType.LPAREN):
            self.advance()
            if not self.match(TokenType.RPAREN):
                while True:
                    args.append(self.parse_expression())
                    if not self.match(TokenType.COMMA):
                        break
                    self.advance()
            self.expect(TokenType.RPAREN)
        return CallStatement(name=name, args=args)

    def parse_subroutine(self) -> Subroutine:
        self.expect(TokenType.SUBROUTINE)
        name_token = self.expect(TokenType.IDENTIFIER)
        name = name_token.value
        params = []
        if self.match(TokenType.LPAREN):
            self.advance()
            if not self.match(TokenType.RPAREN):
                while True:
                    param_token = self.expect(TokenType.IDENTIFIER)
                    params.append(param_token.value)
                    if not self.match(TokenType.COMMA):
                        break
                    self.advance()
            self.expect(TokenType.RPAREN)

        declarations = []
        self.skip_comments()
        while self.match(TokenType.IMPLICIT):
            declarations.append(self.parse_implicit_statement())
        while self.match(TokenType.INTEGER, TokenType.REAL, TokenType.LOGICAL, TokenType.COMPLEX, TokenType.CHARACTER,
                         TokenType.DOUBLEPRECISION, TokenType.DIMENSION, TokenType.PARAMETER, TokenType.DATA,
                         TokenType.EXTERNAL, TokenType.COMMON):
            if self.match(TokenType.DIMENSION):
                declarations.append(self.parse_dimension_statement())
            elif self.match(TokenType.PARAMETER):
                declarations.append(self.parse_parameter_statement())
            elif self.match(TokenType.DATA):
                declarations.append(self.parse_data_statement())
            elif self.match(TokenType.EXTERNAL):
                declarations.append(self.parse_external_statement())
            elif self.match(TokenType.COMMON):
                declarations.append(self.parse_common_statement())
            else:
                declarations.extend(self.parse_declaration())
            self.skip_comments()

        statements = []
        while not self.match(TokenType.END, TokenType.EOF):
            self.skip_comments()
            if self.match(TokenType.END, TokenType.EOF):
                break
            stmt = self.parse_statement()
            if stmt:
                statements.append(stmt)
        self.expect(TokenType.END)

        return Subroutine(name=name, params=params, declarations=declarations, statements=statements)

    def parse_function(self) -> FunctionDef:
        return_type = None
        if self.match(TokenType.INTEGER, TokenType.REAL, TokenType.LOGICAL, TokenType.DOUBLEPRECISION):
            type_token = self.advance()
            return_type = type_token.value.upper()
            if return_type == "DOUBLEPRECISION":
                return_type = "REAL"

        self.expect(TokenType.FUNCTION)
        name_token = self.expect(TokenType.IDENTIFIER)
        name = name_token.value
        params = []
        if self.match(TokenType.LPAREN):
            self.advance()
            if not self.match(TokenType.RPAREN):
                while True:
                    param_token = self.expect(TokenType.IDENTIFIER)
                    params.append(param_token.value)
                    if not self.match(TokenType.COMMA):
                        break
                    self.advance()
            self.expect(TokenType.RPAREN)

        declarations = []
        self.skip_comments()
        while self.match(TokenType.IMPLICIT):
            declarations.append(self.parse_implicit_statement())
        while self.match(TokenType.INTEGER, TokenType.REAL, TokenType.LOGICAL, TokenType.COMPLEX, TokenType.CHARACTER,
                         TokenType.DOUBLEPRECISION, TokenType.DIMENSION, TokenType.PARAMETER, TokenType.DATA,
                         TokenType.EXTERNAL, TokenType.COMMON):
            if self.match(TokenType.DIMENSION):
                declarations.append(self.parse_dimension_statement())
            elif self.match(TokenType.PARAMETER):
                declarations.append(self.parse_parameter_statement())
            elif self.match(TokenType.DATA):
                declarations.append(self.parse_data_statement())
            elif self.match(TokenType.EXTERNAL):
                declarations.append(self.parse_external_statement())
            elif self.match(TokenType.COMMON):
                declarations.append(self.parse_common_statement())
            else:
                declarations.extend(self.parse_declaration())
            self.skip_comments()

        statements = []
        while not self.match(TokenType.END, TokenType.EOF):
            self.skip_comments()
            if self.match(TokenType.END, TokenType.EOF):
                break
            stmt = self.parse_statement()
            if stmt:
                statements.append(stmt)
        self.expect(TokenType.END)

        return FunctionDef(name=name, params=params, return_type=return_type,
                          declarations=declarations, statements=statements)

    def parse_expression(self) -> Expression:
        return self.parse_eqv_expression()

    def parse_eqv_expression(self) -> Expression:
        left = self.parse_or_expression()
        while self.match(TokenType.EQV, TokenType.NEQV):
            op_token = self.advance()
            right = self.parse_or_expression()
            left = BinaryOp(left=left, op=op_token.value.upper(),
                            right=right, line=0, col=0)
        return left

    def parse_or_expression(self) -> Expression:
        left = self.parse_and_expression()
        while self.match(TokenType.OR):
            op_token = self.advance()
            right = self.parse_and_expression()
            left = BinaryOp(left=left, op=op_token.value.upper(),
                            right=right, line=0, col=0)
        return left

    def parse_and_expression(self) -> Expression:
        left = self.parse_not_expression()
        while self.match(TokenType.AND):
            op_token = self.advance()
            right = self.parse_not_expression()
            left = BinaryOp(left=left, op=op_token.value.upper(),
                            right=right, line=0, col=0)
        return left

    def parse_not_expression(self) -> Expression:
        if self.match(TokenType.NOT):
            op_token = self.advance()
            expr = self.parse_not_expression()
            return UnaryOp(op=op_token.value.upper(), operand=expr)
        return self.parse_relational_expression()

    def parse_relational_expression(self) -> Expression:
        left = self.parse_additive_expression()
        while self.match(TokenType.EQ, TokenType.NE, TokenType.LT, TokenType.LE,
                         TokenType.GT, TokenType.GE):
            op_token = self.advance()
            right = self.parse_additive_expression()
            left = BinaryOp(left=left, op=op_token.value.upper(),
                            right=right, line=0, col=0)
        return left

    def parse_additive_expression(self) -> Expression:
        left = self.parse_concat_expression()
        while self.match(TokenType.PLUS, TokenType.MINUS):
            op_token = self.advance()
            if self.match(TokenType.PLUS, TokenType.MINUS, TokenType.STAR, TokenType.SLASH):
                next_op = self.current()
                raise SyntaxError(
                    f"       : "
                    f"'{op_token.value}'  '{next_op.value}'   {next_op.line}:{next_op.col}. "
                    f" , : X/(-Y)  X/-Y"
                )
            right = self.parse_concat_expression()
            left = BinaryOp(left=left, op=op_token.value,
                            right=right, line=0, col=0)
        return left

    def parse_concat_expression(self) -> Expression:
        left = self.parse_multiplicative_expression()
        while self.match(TokenType.CONCAT):
            op_token = self.advance()
            right = self.parse_multiplicative_expression()
            left = BinaryOp(left=left, op=op_token.value,
                            right=right, line=0, col=0)
        return left

    def parse_multiplicative_expression(self) -> Expression:
        left = self.parse_power_expression()
        while self.match(TokenType.STAR, TokenType.SLASH):
            op_token = self.advance()
            if self.match(TokenType.PLUS, TokenType.MINUS, TokenType.STAR, TokenType.SLASH):
                next_op = self.current()
                raise SyntaxError(
                    f"       : "
                    f"'{op_token.value}'  '{next_op.value}'   {next_op.line}:{next_op.col}. "
                    f" , : X/(-Y)  X/-Y"
                )
            right = self.parse_power_expression()
            left = BinaryOp(left=left, op=op_token.value,
                            right=right, line=0, col=0)
        return left

    def parse_power_expression(self) -> Expression:
        left = self.parse_unary_expression()
        while self.match(TokenType.POWER):
            op_token = self.advance()
            right = self.parse_unary_expression()
            left = BinaryOp(left=left, op=op_token.value,
                            right=right, line=op_token.line, col=op_token.col)
        return left

    def parse_unary_expression(self) -> Expression:
        if self.match(TokenType.PLUS, TokenType.MINUS):
            op_token = self.advance()
            if self.match(TokenType.PLUS, TokenType.MINUS, TokenType.STAR, TokenType.SLASH):
                next_op = self.current()
                raise SyntaxError(
                    f"       : "
                    f"'{op_token.value}'  '{next_op.value}'   {next_op.line}:{next_op.col}. "
                    f" , : X/(-Y)  X/-Y"
                )
            expr = self.parse_unary_expression()
            return UnaryOp(op=op_token.value, operand=expr)
        return self.parse_primary_expression()

    def parse_primary_expression(self) -> Expression:
        if self.match(TokenType.LPAREN):
            saved_pos = self.pos
            self.advance()
            first_neg = False
            if self.match(TokenType.MINUS):
                first_neg = True
                self.advance()
            if self.match(TokenType.REAL_LIT, TokenType.INTEGER_LIT):
                first_token = self.advance()
                if self.match(TokenType.COMMA):
                    self.advance()
                    second_neg = False
                    if self.match(TokenType.MINUS):
                        second_neg = True
                        self.advance()
                    if self.match(TokenType.REAL_LIT, TokenType.INTEGER_LIT):
                        second_token = self.advance()
                        if self.match(TokenType.RPAREN):
                            self.advance()
                            real_part = float(first_token.value) * (-1 if first_neg else 1)
                            imag_part = float(second_token.value) * (-1 if second_neg else 1)
                            return ComplexLiteral(real_part=real_part, imag_part=imag_part, line=first_token.line, col=first_token.col)
            self.pos = saved_pos
            self.advance()
            expr = self.parse_expression()
            self.expect(TokenType.RPAREN)
            return expr
        elif self.match(TokenType.INTEGER_LIT):
            token = self.advance()
            return IntegerLiteral(value=token.value, line=token.line, col=token.col)
        elif self.match(TokenType.REAL_LIT):
            token = self.advance()
            return RealLiteral(value=token.value, line=token.line, col=token.col)
        elif self.match(TokenType.STRING_LIT):
            token = self.advance()
            return StringLiteral(value=token.value, line=token.line, col=token.col)
        elif self.match(TokenType.TRUE):
            token = self.advance()
            return LogicalLiteral(value=True, line=token.line, col=token.col)
        elif self.match(TokenType.FALSE):
            token = self.advance()
            return LogicalLiteral(value=False, line=token.line, col=token.col)
        elif self.match(TokenType.SIN, TokenType.COS, TokenType.TAN,
                        TokenType.ASIN, TokenType.ACOS, TokenType.ATAN,
                        TokenType.ABS, TokenType.SQRT, TokenType.EXP,
                        TokenType.LOG, TokenType.LOG10,
                        TokenType.MIN, TokenType.MAX, TokenType.MOD, TokenType.POW,
                        TokenType.REAL_FUNC, TokenType.FLOAT):
            func_token = self.advance()
            self.expect(TokenType.LPAREN)
            args = []
            while not self.match(TokenType.RPAREN):
                args.append(self.parse_expression())
                if not self.match(TokenType.COMMA):
                    break
                self.advance()
            self.expect(TokenType.RPAREN)
            return FunctionCall(name=func_token.value.upper(), args=args)
        elif self.match(TokenType.IDENTIFIER):
            name_token = self.advance()
            if self.match(TokenType.LPAREN):
                self.advance()
                if name_token.value.upper() == "INT":
                    args = []
                    while not self.match(TokenType.RPAREN):
                        args.append(self.parse_expression())
                        if not self.match(TokenType.COMMA):
                            break
                        self.advance()
                    self.expect(TokenType.RPAREN)
                    return FunctionCall(name="INT", args=args)
                indices = []
                while not self.match(TokenType.RPAREN):
                    indices.append(self.parse_expression())
                    if not self.match(TokenType.COMMA):
                        break
                    self.advance()
                self.expect(TokenType.RPAREN)
                return ArrayRef(name=name_token.value, indices=indices, line=name_token.line, col=name_token.col)
            return Variable(name=name_token.value, line=name_token.line, col=name_token.col)
        raise SyntaxError(
            f"  {self.current().type.name} "
            f"  {self.current().line}:{self.current().col}: {self.current().value}"
        )
