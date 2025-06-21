import sys
import logging
from typing import List, Dict
from isa import Opcode, Reg, Instruction, IO_INPUT_PORT, IO_OUTPUT_PORT


class Token:
    def __init__(self, type: str, value: str, line: int = 0):
        self.type = type
        self.value = value
        self.line = line

    def __repr__(self):
        return f"Token({self.type}, {self.value})"


def tokenize(text: str) -> List[Token]:
    """Токенизер Forth кода"""
    tokens = []
    for line_num, line in enumerate(text.split('\n'), 1):
        # Удаляем комментарии (\) и чуть ниже скобочные комментарии
        if '\\' in line:
            line = line[:line.index('\\')]

        while '(' in line and ')' in line:
            start = line.find('(')
            end = line.find(')', start)
            if start < end:
                line = line[:start] + line[end + 1:]
            else:
                break

        # Обработка строковых литералов ." ... "
        while '."' in line:
            start = line.find('."')
            if start >= 0:
                quote_start = start + 2
                quote_end = line.find('"', quote_start)
                if quote_end > quote_start:
                    string_content = line[quote_start:quote_end]
                    tokens.append(Token('STRING_LITERAL', string_content, line_num))
                    line = line[:start] + ' ' + line[quote_end + 1:]
                else:
                    break
            else:
                break

        # Разбиваем на слова
        words = line.split()
        for word in words:
            if word.isdigit() or (word.startswith('-') and word[1:].isdigit()):
                tokens.append(Token('NUMBER', word, line_num))
            elif word.startswith('0x') or word.startswith('0X'):
                tokens.append(Token('HEX_NUMBER', word, line_num))
            elif word == ':':
                tokens.append(Token('COLON', word, line_num))
            elif word == ';':
                tokens.append(Token('SEMICOLON', word, line_num))
            elif word in ['if', 'then', 'else', 'begin', 'until', 'while', 'repeat',
                          'variable', 'do', 'loop', 'again']:
                tokens.append(Token(word.upper(), word, line_num))
            else:
                tokens.append(Token('WORD', word, line_num))
    return tokens


class RiscForthCompiler:
    """Компилятор Forth в RISC код """

    def __init__(self):
        self.code: List[Instruction] = []
        self.procedures: Dict[str, int] = {}
        self.variables: Dict[str, int] = {}
        self.labels: Dict[int, int] = {}
        self.label_count = 0

        # Стеки для циклов и условий
        self.loop_stack: List[int] = []
        self.begin_stack: List[int] = []
        self.if_stack: List[int] = []

        # Адреса памяти
        self.data_addr = 0x1000
        self.var_addr = 0x2000

        # Набор встроенных слов
        self.builtins = {
            # Арифметические операции
            '+': self._gen_add,
            '-': self._gen_sub,
            '*': self._gen_mul,
            '/': self._gen_div,
            'mod': self._gen_mod,

            # Битовые операции
            'and': self._gen_and,
            'or': self._gen_or,
            'xor': self._gen_xor,

            # Сравнения
            '=': self._gen_eq,
            '<': self._gen_lt,
            '>': self._gen_gt,
            '<>': self._gen_ne,
            '<=': self._gen_le,
            '>=': self._gen_ge,

            # Стековые операции
            'dup': self._gen_dup,
            'drop': self._gen_drop,
            'swap': self._gen_swap,
            'over': self._gen_over,
            'rot': self._gen_rot,

            # Память
            '@': self._gen_fetch,
            '!': self._gen_store,

            # I/O
            'emit': self._gen_emit,
            'key': self._gen_key,

            # Управление
            'halt': self._gen_halt,

            # Логические операции
            'not': self._gen_not,
            '0=': self._gen_zero_eq,
            'shl': self._gen_shl,
            'shr': self._gen_shr,
        }

        logging.info("Компилятор Forth->RISC инициализирован с полным набором операций")

    def _emit_literal(self, value: int):
        """Генерация загрузки константы"""
        if -2 ** 20 <= value <= 2 ** 20 - 1:
            self.code.append(Instruction(Opcode.LOADI, rt=Reg.T1.value, imm=value & 0x1FFFFF))
            logging.debug(f"EMIT: LOADI R{Reg.T1.value}, {value}")
        else:
            upper = (value >> 16) & 0xFFFF
            lower = value & 0xFFFF

            self.code.append(Instruction(Opcode.LUI, rt=Reg.T1.value, imm=upper))
            logging.debug(f"EMIT: LUI R{Reg.T1.value}, 0x{upper:04X}")

            if lower != 0:
                self.code.append(Instruction(Opcode.ORI, rs=Reg.T1.value, rt=Reg.T1.value, imm=lower))
                logging.debug(f"EMIT: ORI R{Reg.T1.value}, R{Reg.T1.value}, 0x{lower:04X}")

        self.code.append(Instruction(Opcode.PUSH, rs=Reg.T1.value))
        logging.debug(f"EMIT: PUSH R{Reg.T1.value}")

    def _gen_add(self):
        """+ (сложение): pop b, pop a, push(a+b)"""
        self.code.append(Instruction(Opcode.POP, rt=Reg.T2.value))  # b
        self.code.append(Instruction(Opcode.POP, rt=Reg.T1.value))  # a
        self.code.append(Instruction(Opcode.ADD, rs=Reg.T1.value, rt=Reg.T2.value, rd=Reg.T1.value))
        self.code.append(Instruction(Opcode.PUSH, rs=Reg.T1.value))

    def _gen_sub(self):
        """- (вычитание): pop b, pop a, push(a-b)"""
        self.code.append(Instruction(Opcode.POP, rt=Reg.T2.value))  # b
        self.code.append(Instruction(Opcode.POP, rt=Reg.T1.value))  # a
        self.code.append(Instruction(Opcode.SUB, rs=Reg.T1.value, rt=Reg.T2.value, rd=Reg.T1.value))
        self.code.append(Instruction(Opcode.PUSH, rs=Reg.T1.value))

    def _gen_mul(self):
        """* (умножение)"""
        self.code.append(Instruction(Opcode.POP, rt=Reg.T2.value))
        self.code.append(Instruction(Opcode.POP, rt=Reg.T1.value))
        self.code.append(Instruction(Opcode.MUL, rs=Reg.T1.value, rt=Reg.T2.value, rd=Reg.T1.value))
        self.code.append(Instruction(Opcode.PUSH, rs=Reg.T1.value))

    def _gen_div(self):
        """/ (деление)"""
        self.code.append(Instruction(Opcode.POP, rt=Reg.T2.value))
        self.code.append(Instruction(Opcode.POP, rt=Reg.T1.value))
        self.code.append(Instruction(Opcode.DIV, rs=Reg.T1.value, rt=Reg.T2.value, rd=Reg.T1.value))
        self.code.append(Instruction(Opcode.PUSH, rs=Reg.T1.value))

    def _gen_mod(self):
        """mod (остаток)"""
        self.code.append(Instruction(Opcode.POP, rt=Reg.T2.value))
        self.code.append(Instruction(Opcode.POP, rt=Reg.T1.value))
        self.code.append(Instruction(Opcode.MOD, rs=Reg.T1.value, rt=Reg.T2.value, rd=Reg.T1.value))
        self.code.append(Instruction(Opcode.PUSH, rs=Reg.T1.value))

    def _gen_and(self):
        """and (битовое И)"""
        self.code.append(Instruction(Opcode.POP, rt=Reg.T2.value))
        self.code.append(Instruction(Opcode.POP, rt=Reg.T1.value))
        self.code.append(Instruction(Opcode.AND, rs=Reg.T1.value, rt=Reg.T2.value, rd=Reg.T1.value))
        self.code.append(Instruction(Opcode.PUSH, rs=Reg.T1.value))

    def _gen_or(self):
        """or (битовое ИЛИ)"""
        self.code.append(Instruction(Opcode.POP, rt=Reg.T2.value))
        self.code.append(Instruction(Opcode.POP, rt=Reg.T1.value))
        self.code.append(Instruction(Opcode.OR, rs=Reg.T1.value, rt=Reg.T2.value, rd=Reg.T1.value))
        self.code.append(Instruction(Opcode.PUSH, rs=Reg.T1.value))

    def _gen_xor(self):
        """xor (исключающее ИЛИ)"""
        self.code.append(Instruction(Opcode.POP, rt=Reg.T2.value))
        self.code.append(Instruction(Opcode.POP, rt=Reg.T1.value))
        self.code.append(Instruction(Opcode.XOR, rs=Reg.T1.value, rt=Reg.T2.value, rd=Reg.T1.value))
        self.code.append(Instruction(Opcode.PUSH, rs=Reg.T1.value))

    def _gen_eq(self):
        """= (равенство): результат 1 если равны, 0 если не равны"""
        self.code.append(Instruction(Opcode.POP, rt=Reg.T2.value))
        self.code.append(Instruction(Opcode.POP, rt=Reg.T1.value))
        self.code.append(Instruction(Opcode.CMP, rs=Reg.T1.value, rt=Reg.T2.value, rd=Reg.T1.value))
        self.code.append(Instruction(Opcode.PUSH, rs=Reg.T1.value))

    def _gen_shl(self):
        """shl - сдвиг влево"""
        self.code.append(Instruction(Opcode.POP, rt=Reg.T2.value))  # количество бит
        self.code.append(Instruction(Opcode.POP, rt=Reg.T1.value))  # число
        self.code.append(Instruction(Opcode.SHL, rs=Reg.T1.value, rt=Reg.T2.value, rd=Reg.T1.value))
        self.code.append(Instruction(Opcode.PUSH, rs=Reg.T1.value))

    def _gen_shr(self):
        """shr - сдвиг вправо"""
        self.code.append(Instruction(Opcode.POP, rt=Reg.T2.value))  # количество бит
        self.code.append(Instruction(Opcode.POP, rt=Reg.T1.value))  # число
        self.code.append(Instruction(Opcode.SHR, rs=Reg.T1.value, rt=Reg.T2.value, rd=Reg.T1.value))
        self.code.append(Instruction(Opcode.PUSH, rs=Reg.T1.value))

    def _gen_ne(self):
        """<> (не равно): результат 1 если НЕ равны, 0 если равны"""
        self.code.append(Instruction(Opcode.POP, rt=Reg.T2.value))  # b
        self.code.append(Instruction(Opcode.POP, rt=Reg.T1.value))  # a
        self.code.append(Instruction(Opcode.CMP, rs=Reg.T1.value, rt=Reg.T2.value, rd=Reg.T1.value))
        self.code.append(Instruction(Opcode.PUSH, rs=Reg.T1.value))
        self._emit_literal(1)
        self.code.append(Instruction(Opcode.POP, rt=Reg.T2.value))
        self.code.append(Instruction(Opcode.POP, rt=Reg.T1.value))
        self.code.append(Instruction(Opcode.XOR, rs=Reg.T1.value, rt=Reg.T2.value, rd=Reg.T1.value))
        self.code.append(Instruction(Opcode.PUSH, rs=Reg.T1.value))

    def _gen_lt(self):
        """< (меньше)"""
        self.code.append(Instruction(Opcode.POP, rt=Reg.T2.value))  # b
        self.code.append(Instruction(Opcode.POP, rt=Reg.T1.value))  # a

        self.code.append(Instruction(Opcode.SUB, rs=Reg.T1.value, rt=Reg.T2.value, rd=Reg.T1.value))

        self.code.append(Instruction(Opcode.PUSH, rs=Reg.T1.value))

        self._emit_literal(31)
        self.code.append(Instruction(Opcode.POP, rt=Reg.T2.value))
        self.code.append(Instruction(Opcode.POP, rt=Reg.T1.value))

        self.code.append(Instruction(Opcode.SHR, rs=Reg.T1.value, rt=Reg.T2.value, rd=Reg.T1.value))

        self.code.append(Instruction(Opcode.PUSH, rs=Reg.T1.value))

    def _gen_gt(self):
        """> (больше) - меняем местами и вызываем <"""
        self._gen_swap()
        self._gen_lt()

    def _gen_le(self):
        """<= (меньше или равно): NOT (a > b)"""
        self._gen_gt()
        self._gen_not()

    def _gen_ge(self):
        """>= (больше или равно): NOT (a < b)"""
        self._gen_lt()
        self._gen_not()

    def _gen_dup(self):
        """dup - дублировать вершину стека ( a -- a a )"""
        self.code.append(Instruction(Opcode.POP, rt=Reg.T1.value))
        self.code.append(Instruction(Opcode.PUSH, rs=Reg.T1.value))
        self.code.append(Instruction(Opcode.PUSH, rs=Reg.T1.value))

    def _gen_drop(self):
        """drop - удалить вершину стека ( a -- )"""
        self.code.append(Instruction(Opcode.POP, rt=Reg.T1.value))

    def _gen_swap(self):
        """swap - поменять местами два верхних элемента ( a b -- b a )"""
        self.code.append(Instruction(Opcode.POP, rt=Reg.T1.value))  # b
        self.code.append(Instruction(Opcode.POP, rt=Reg.T2.value))  # a
        self.code.append(Instruction(Opcode.PUSH, rs=Reg.T1.value))  # b
        self.code.append(Instruction(Opcode.PUSH, rs=Reg.T2.value))  # a

    def _gen_over(self):
        """over - копировать второй элемент на верх ( a b -- a b a )"""
        self.code.append(Instruction(Opcode.POP, rt=Reg.T1.value))  # b
        self.code.append(Instruction(Opcode.POP, rt=Reg.T2.value))  # a
        self.code.append(Instruction(Opcode.PUSH, rs=Reg.T2.value))  # a
        self.code.append(Instruction(Opcode.PUSH, rs=Reg.T1.value))  # b
        self.code.append(Instruction(Opcode.PUSH, rs=Reg.T2.value))  # a

    def _gen_rot(self):
        """rot - поворот трех элементов ( a b c -- b c a )"""
        self.code.append(Instruction(Opcode.POP, rt=Reg.T1.value))  # c
        self.code.append(Instruction(Opcode.POP, rt=Reg.T2.value))  # b
        self.code.append(Instruction(Opcode.POP, rt=Reg.A0.value))  # a
        self.code.append(Instruction(Opcode.PUSH, rs=Reg.T2.value))  # b
        self.code.append(Instruction(Opcode.PUSH, rs=Reg.T1.value))  # c
        self.code.append(Instruction(Opcode.PUSH, rs=Reg.A0.value))  # a

    def _gen_not(self):
        """not - логическое НЕ: 0 -> 1, не-ноль -> 0"""
        self.code.append(Instruction(Opcode.POP, rt=Reg.T1.value))
        self._emit_literal(0)
        self.code.append(Instruction(Opcode.POP, rt=Reg.T2.value))
        self.code.append(Instruction(Opcode.CMP, rs=Reg.T1.value, rt=Reg.T2.value, rd=Reg.T1.value))
        self.code.append(Instruction(Opcode.PUSH, rs=Reg.T1.value))

    def _gen_zero_eq(self):
        """0= - проверка на ноль: возвращает 1 если ноль, 0 если не ноль"""
        self._gen_not()

    def _gen_fetch(self):
        """@ (load) - загрузить из памяти"""
        self.code.append(Instruction(Opcode.POP, rt=Reg.T1.value))
        self.code.append(Instruction(Opcode.LOAD, rs=Reg.T1.value, rt=Reg.T1.value, imm=0))
        self.code.append(Instruction(Opcode.PUSH, rs=Reg.T1.value))

    def _gen_store(self):
        """! (store) - сохранить в память"""
        self.code.append(Instruction(Opcode.POP, rt=Reg.T1.value))
        self.code.append(Instruction(Opcode.POP, rt=Reg.T2.value))
        self.code.append(Instruction(Opcode.STORE, rs=Reg.T2.value, rt=Reg.T1.value, imm=0))

    def _gen_emit(self):
        """emit - вывести символ"""
        self.code.append(Instruction(Opcode.POP, rt=Reg.T1.value))
        self.code.append(Instruction(Opcode.OUT, rs=Reg.T1.value, imm=IO_OUTPUT_PORT))

    def _gen_key(self):
        """key - ввести символ"""
        self.code.append(Instruction(Opcode.IN, rt=Reg.T1.value, imm=IO_INPUT_PORT))
        self.code.append(Instruction(Opcode.PUSH, rs=Reg.T1.value))

    def _gen_halt(self):
        """halt - остановка программы"""
        self.code.append(Instruction(Opcode.HALT))

    def _gen_ret(self):
        """Генерация возврата из процедуры"""
        self.code.append(Instruction(Opcode.RET))
        logging.debug("EMIT: RET")

    def _gen_call(self, addr: int):
        """Генерация вызова процедуры"""
        self.code.append(Instruction(Opcode.CALL, addr=addr))
        logging.debug(f"EMIT: CALL to {addr}")

    def _gen_string_literal(self, text: str):
        """Генерировать строковый литерал ." ... " """
        for char in text:
            char_code = ord(char) & 0xFF
            self._emit_literal(char_code)
            self._gen_emit()

    def compile(self, tokens: List[Token]) -> List[Instruction]:
        """ компилятор Forth кода с точкой входа"""
        logging.info("Начинаем компиляцию Forth кода")

        self.code.append(Instruction(Opcode.JMP, addr=0))  # Адрес будет исправлен позже

        main_code_tokens = []

        i = 0
        while i < len(tokens):
            token = tokens[i]
            if token.type == 'COLON':
                i += 1
                proc_name = tokens[i].value
                i += 1

                proc_start_addr = len(self.code)
                self.procedures[proc_name] = proc_start_addr
                logging.info(f"Найдена процедура '{proc_name}' по адресу {proc_start_addr}")

                while i < len(tokens) and tokens[i].type != 'SEMICOLON':
                    self._compile_token(tokens, i)
                    i += 1

                self._gen_ret()
                i += 1
                continue

            main_code_tokens.append(token)
            i += 1

        main_entry_point = len(self.code)
        self.code[0].addr = main_entry_point
        logging.info(f"Точка входа (main) по адресу: {main_entry_point}")

        logging.info("Компилируем основной код")
        i = 0
        while i < len(main_code_tokens):
            i = self._compile_token(main_code_tokens, i)

        if self.begin_stack or self.loop_stack or self.if_stack:
            logging.warning(f"Незакрытые стеки: BEGIN={len(self.begin_stack)}, "
                            f"LOOP={len(self.loop_stack)}, IF={len(self.if_stack)}")

        self.code.append(Instruction(Opcode.HALT))
        logging.info(f"Компиляция завершена: {len(self.code)} инструкций")
        return self.code

    def _compile_token(self, tokens: List[Token], i: int) -> int:
        """Компиляция одного токена (без логики процедур)"""
        token = tokens[i]
        logging.debug(f"Компилируем токен: {token}")

        if token.type == 'NUMBER':
            value = int(token.value)
            self._emit_literal(value)
        elif token.type == 'HEX_NUMBER':
            value = int(token.value, 16)
            self._emit_literal(value)
        elif token.type == 'STRING_LITERAL':
            self._gen_string_literal(token.value)

        elif token.type == 'BEGIN':
            begin_addr = len(self.code)
            self.begin_stack.append(begin_addr)
            logging.debug(f"BEGIN: адрес {begin_addr}")

        elif token.type == 'WHILE':
            self.code.append(Instruction(Opcode.POP, rt=Reg.T1.value))
            jump_instr_addr = len(self.code)
            self.code.append(Instruction(Opcode.JZ, rs=Reg.T1.value, imm=0))
            self.loop_stack.append(jump_instr_addr)
            logging.debug(f"WHILE: условный переход в инструкции {jump_instr_addr}")

        elif token.type == 'REPEAT':
            if not self.begin_stack or not self.loop_stack:
                logging.error("REPEAT без соответствующего BEGIN/WHILE")
                return i + 1

            begin_addr = self.begin_stack.pop()
            self.code.append(Instruction(Opcode.JMP, addr=begin_addr))

            while_jump_addr = self.loop_stack.pop()
            end_addr = len(self.code)
            self.code[while_jump_addr].imm = end_addr
            logging.debug(f"REPEAT: JMP {begin_addr}, исправили WHILE[{while_jump_addr}] -> {end_addr}")

        elif token.type == 'UNTIL':
            if not self.begin_stack:
                logging.error("UNTIL без соответствующего BEGIN")
                return i + 1

            begin_addr = self.begin_stack.pop()
            self.code.append(Instruction(Opcode.POP, rt=Reg.T1.value))  # условие
            self.code.append(Instruction(Opcode.JZ, rs=Reg.T1.value, imm=begin_addr))  # прыжок если 0
            logging.debug(f"UNTIL: условный переход к {begin_addr}")

        elif token.type == 'IF':
            self.code.append(Instruction(Opcode.POP, rt=Reg.T1.value))
            jump_instr_addr = len(self.code)
            self.code.append(Instruction(Opcode.JZ, rs=Reg.T1.value, imm=0))  # заполним позже
            self.if_stack.append(jump_instr_addr)
            logging.debug(f"IF: условный переход в инструкции {jump_instr_addr}")

        elif token.type == 'ELSE':
            if not self.if_stack:
                logging.error("ELSE без соответствующего IF")
                return i + 1

            else_jump_addr = len(self.code)
            self.code.append(Instruction(Opcode.JMP, addr=0))

            if_jump_addr = self.if_stack.pop()
            self.code[if_jump_addr].imm = len(self.code)

            self.if_stack.append(else_jump_addr)
            logging.debug(f"ELSE: исправили IF[{if_jump_addr}], новый переход {else_jump_addr}")


        elif token.type == 'THEN':
            if not self.if_stack:
                logging.error("THEN без соответствующего IF/ELSE")
                return i + 1

            jump_addr = self.if_stack.pop()
            end_addr = len(self.code)
            instruction_to_patch = self.code[jump_addr]
            if instruction_to_patch.opcode == Opcode.JMP:
                instruction_to_patch.addr = end_addr
            else:
                instruction_to_patch.imm = end_addr
            logging.debug(f"THEN: исправили переход[{jump_addr}] -> {end_addr}")

        elif token.type == 'WORD':
            if token.value in self.builtins:
                self.builtins[token.value]()
            elif token.value in self.procedures:
                addr = self.procedures[token.value]
                self._gen_call(addr)
            elif token.value in self.variables:
                addr = self.variables[token.value]
                self._emit_literal(addr)
            else:
                logging.warning(f"Неизвестное слово: {token.value} в строке {token.line}")

        return i + 1

    def save_binary(self, filename: str):
        """Сохранение в бинарный файл"""
        with open(filename, 'wb') as f:
            for instruction in self.code:
                f.write(instruction.to_binary())

        hex_filename = filename + '.hex'
        with open(hex_filename, 'w') as f:
            f.write(f"; RISC Forth Compiler Output\n")
            f.write(f"; Instructions: {len(self.code)}\n\n")

            for i, instruction in enumerate(self.code):
                f.write(instruction.to_hex(i) + '\n')  # Адрес в инструкциях, а не байтах

        logging.info(f"Сохранено: {filename} и {hex_filename}")


class RiscForthTranslator:
    """Основной класс транслятора"""

    def __init__(self):
        self.compiler = RiscForthCompiler()

    def translate_file(self, source_path: str, output_path: str):
        """Трансляция файла"""
        with open(source_path, 'r', encoding='utf-8') as f:
            source = f.read()

        print(f"Исходный код:\n{source}\n")

        tokens = tokenize(source)
        print(f"Токены: {[(t.type, t.value) for t in tokens[:20]]}\n")

        try:
            instructions = self.compiler.compile(tokens)
        except Exception as e:
            print(f"Ошибка компиляции: {e}")
            raise

        self.compiler.save_binary(output_path)

        print(f"✅ Компилировано {len(instructions)} инструкций")
        print(f"📁 Выходные файлы: {output_path}, {output_path}.hex")


def main():
    if len(sys.argv) != 3:
        print("Использование: python translator.py <source.forth> <output.bin>")
        sys.exit(1)

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    translator = RiscForthTranslator()
    try:
        translator.translate_file(sys.argv[1], sys.argv[2])
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()