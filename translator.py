import sys
import struct
from enum import Enum
from typing import List, Dict


# Новая RISC система команд - только 19 базовых команд
class Opcode(Enum):
    NOP = 0x00  # нет операции
    HALT = 0x01  # остановка
    LOAD = 0x02  # загрузить из памяти
    STORE = 0x03  # сохранить в память
    PUSH = 0x04  # положить на стек
    POP = 0x05  # снять со стека
    ADD = 0x06  # сложение
    SUB = 0x07  # вычитание
    MUL = 0x08  # умножение
    DIV = 0x09  # деление
    MOD = 0x0A  # остаток от деления
    AND = 0x0B  # битовое И
    OR = 0x0C  # битовое ИЛИ
    XOR = 0x0D  # битовое исключающее ИЛИ
    CMP = 0x0E  # сравнение (==)
    JMP = 0x0F  # безусловный переход
    JZ = 0x10  # условный переход если ноль
    IN = 0x11  # ввод
    OUT = 0x12  # вывод


class Reg(Enum):
    ZERO = 0  # всегда 0
    SP = 1  # указатель стека данных
    RSP = 2  # указатель стека возвратов
    TOS = 3  # вершина стека (Top Of Stack)
    BASE = 4  # базовый адрес
    T1 = 5  # временный регистр 1
    T2 = 6  # временный регистр 2
    PC = 7  # счетчик команд


class Instruction:
    def __init__(self, opcode: Opcode, rs=0, rt=0, rd=0, imm=0, addr=0, is_label=False):
        self.opcode = opcode
        self.rs = rs
        self.rt = rt
        self.rd = rd
        self.imm = imm
        self.addr = addr
        self.is_label = is_label

    def to_binary(self) -> bytes:
        """Упаковка в 32-битный формат"""

        def safe_uint32(value):
            return int(value) & 0xFFFFFFFF

        def safe_uint16(value):
            return int(value) & 0xFFFF

        def safe_uint26(value):
            return int(value) & 0x3FFFFFF

        # J-type для переходов
        if self.opcode in [Opcode.JMP]:
            opcode_bits = (self.opcode.value & 0x3F) << 26
            addr_bits = safe_uint26(self.addr)
            word = opcode_bits | addr_bits
        # I-type для команд с immediate
        elif self.opcode in [Opcode.LOAD, Opcode.STORE, Opcode.JZ, Opcode.IN, Opcode.OUT]:
            opcode_bits = (self.opcode.value & 0x3F) << 26
            rs_bits = (self.rs & 0x1F) << 21
            rt_bits = (self.rt & 0x1F) << 16
            imm_bits = safe_uint16(self.imm)
            word = opcode_bits | rs_bits | rt_bits | imm_bits
        # R-type для остальных
        else:
            opcode_bits = (self.opcode.value & 0x3F) << 26
            rs_bits = (self.rs & 0x1F) << 21
            rt_bits = (self.rt & 0x1F) << 16
            rd_bits = (self.rd & 0x1F) << 11
            word = opcode_bits | rs_bits | rt_bits | rd_bits

        word = safe_uint32(word)
        return struct.pack('>I', word)

    def to_hex(self, addr: int) -> str:
        """Листинг команды"""
        hex_code = self.to_binary().hex().upper()

        if self.opcode == Opcode.LOAD:
            mnemonic = f"LOAD R{self.rt}, R{self.rs}+{self.imm}"
        elif self.opcode == Opcode.STORE:
            mnemonic = f"STORE R{self.rs}, R{self.rt}+{self.imm}"
        elif self.opcode in [Opcode.ADD, Opcode.SUB, Opcode.MUL, Opcode.DIV, Opcode.MOD,
                             Opcode.AND, Opcode.OR, Opcode.XOR, Opcode.CMP]:
            mnemonic = f"{self.opcode.name} R{self.rd}, R{self.rs}, R{self.rt}"
        elif self.opcode == Opcode.PUSH:
            mnemonic = f"PUSH R{self.rs}"
        elif self.opcode == Opcode.POP:
            mnemonic = f"POP R{self.rt}"
        elif self.opcode == Opcode.JZ:
            mnemonic = f"JZ R{self.rs}, 0x{self.imm:04X}"
        elif self.opcode == Opcode.JMP:
            mnemonic = f"JMP 0x{self.addr:04X}"
        elif self.opcode == Opcode.IN:
            mnemonic = f"IN R{self.rt}, 0x{self.imm:04X}"
        elif self.opcode == Opcode.OUT:
            mnemonic = f"OUT R{self.rs}, 0x{self.imm:04X}"
        else:
            mnemonic = self.opcode.name

        return f"0x{addr:04X}: {hex_code}  {mnemonic}"


class Token:
    def __init__(self, type: str, value: str, line: int = 0):
        self.type = type
        self.value = value
        self.line = line


def tokenize(text: str) -> List[Token]:
    """Токенизер"""
    tokens = []
    for line_num, line in enumerate(text.split('\n'), 1):
        if '\\' in line:
            line = line[:line.index('\\')]

        while '(' in line and ')' in line:
            start = line.find('(')
            end = line.find(')', start)
            if start < end:
                line = line[:start] + line[end + 1:]
            else:
                break

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

        words = line.split()
        for word in words:
            if word.isdigit() or (word.startswith('-') and word[1:].isdigit()):
                tokens.append(Token('NUMBER', word, line_num))
            elif word.startswith('s"') and word.endswith('"'):
                tokens.append(Token('STRING', word[2:-1], line_num))
            elif word == ':':
                tokens.append(Token('COLON', word, line_num))
            elif word == ';':
                tokens.append(Token('SEMICOLON', word, line_num))
            elif word in ['if', 'then', 'else', 'begin', 'until', 'while', 'repeat', 'variable', 'do', 'loop', 'again']:
                tokens.append(Token(word.upper(), word, line_num))
            else:
                tokens.append(Token('WORD', word, line_num))
    return tokens


class RiscForthCompiler:
    def __init__(self):
        self.code: List[Instruction] = []
        self.data: List[int] = []
        self.strings: Dict[str, int] = {}
        self.variables: Dict[str, int] = {}
        self.words: Dict[str, int] = {}
        self.labels: Dict[int, int] = {}
        self.label_count = 0

        # Стеки для циклов
        self.loop_stack = []
        self.begin_stack = []

        self.data_addr = 0x1000
        self.var_addr = 0x2000

        # Memory-mapped I/O порты
        self.IO_INPUT_PORT = 0x8000
        self.IO_OUTPUT_PORT = 0x8001

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
            'not': self._gen_not,

            # Сравнения
            '=': self._gen_eq,
            '<': self._gen_lt,
            '>': self._gen_gt,
            '!=': self._gen_ne,
            '<=': self._gen_le,
            '>=': self._gen_ge,

            # Стековые операции
            'dup': self._gen_dup,
            'drop': self._gen_drop,
            'swap': self._gen_swap,
            'over': self._gen_over,
            'rot': self._gen_rot,

            # Стек возвратов
            '>r': self._gen_to_r,
            'r>': self._gen_from_r,
            'r@': self._gen_r_fetch,

            # Память
            '@': self._gen_fetch,  # это load
            '!': self._gen_store,  # это store

            # MMIO
            'emit': self._gen_emit,
            'key': self._gen_key,
            '.': self._gen_print_num,

            # Циклы
            'i': self._gen_loop_index,
            'j': self._gen_loop_index2,

            # Управление потоком
            'exit': self._gen_exit,
        }

    def compile(self, tokens: List[Token]) -> List[Instruction]:
        print("Собираем определения в 1 проходе.")
        i = 0
        while i < len(tokens):
            token = tokens[i]
            if token.type == 'COLON':
                i = self._collect_word_def(tokens, i)
            elif token.type == 'VARIABLE':
                i = self._collect_variable(tokens, i)
            else:
                i += 1


        # Инициализация стеков
        self._emit_literal(0x3000)  # SP = 0x3000
        self.code.append(Instruction(Opcode.POP, rt=Reg.SP.value))

        self._emit_literal(0x4000)  # RSP = 0x4000
        self.code.append(Instruction(Opcode.POP, rt=Reg.RSP.value))

        print("Генерируем код во 2 проходе.")
        i = 0
        while i < len(tokens):
            if tokens[i].type == 'COLON':
                i = self._compile_word_def(tokens, i)
            elif tokens[i].type == 'VARIABLE':
                i = self._skip_variable(tokens, i)
            else:
                i = self._compile_token(tokens, i)

        self.code.append(Instruction(Opcode.HALT))
        self._resolve_labels()
        return self.code

    def _emit_literal(self, value: int):
        """Положить литерал на стек, в реальности это будет компилироваться в LOAD + PUSH"""
        # Сохраняем значение во временную "память" (регистр T1)
        # В реальной реализации это будет загрузка из сегмента данных
        self.code.append(Instruction(Opcode.LOAD, rs=Reg.ZERO.value, rt=Reg.T1.value, imm=value & 0xFFFF))
        self.code.append(Instruction(Opcode.PUSH, rs=Reg.T1.value))

    # Генераторы для арифметических операций
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

    # Битовые операции
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

    def _gen_not(self):
        """not (битовое НЕ) - через XOR с 0xFFFF"""
        self.code.append(Instruction(Opcode.POP, rt=Reg.T1.value))
        self.code.append(Instruction(Opcode.LOAD, rs=Reg.ZERO.value, rt=Reg.T2.value, imm=0xFFFF))
        self.code.append(Instruction(Opcode.XOR, rs=Reg.T1.value, rt=Reg.T2.value, rd=Reg.T1.value))
        self.code.append(Instruction(Opcode.PUSH, rs=Reg.T1.value))

    # Сравнения
    def _gen_eq(self):
        """= (равенство)"""
        self.code.append(Instruction(Opcode.POP, rt=Reg.T2.value))
        self.code.append(Instruction(Opcode.POP, rt=Reg.T1.value))
        self.code.append(Instruction(Opcode.CMP, rs=Reg.T1.value, rt=Reg.T2.value, rd=Reg.T1.value))
        self.code.append(Instruction(Opcode.PUSH, rs=Reg.T1.value))

    def _gen_ne(self):
        """!="""
        self._gen_eq()
        self._gen_not()

    def _gen_lt(self):
        """<"""
        self.code.append(Instruction(Opcode.POP, rt=Reg.T2.value))  # b
        self.code.append(Instruction(Opcode.POP, rt=Reg.T1.value))  # a
        self.code.append(Instruction(Opcode.SUB, rs=Reg.T1.value, rt=Reg.T2.value, rd=Reg.T1.value))  # a-b
        # Если результат отрицательный, то a < b
        self.code.append(Instruction(Opcode.LOAD, rs=Reg.ZERO.value, rt=Reg.T2.value, imm=0x8000))
        self.code.append(Instruction(Opcode.AND, rs=Reg.T1.value, rt=Reg.T2.value, rd=Reg.T1.value))
        # Если результат != 0, то отрицательное число
        self.code.append(Instruction(Opcode.LOAD, rs=Reg.ZERO.value, rt=Reg.T2.value, imm=0))
        self.code.append(Instruction(Opcode.CMP, rs=Reg.T1.value, rt=Reg.T2.value, rd=Reg.T1.value))
        self._gen_not()  # инвертируем результат
        self.code.append(Instruction(Opcode.PUSH, rs=Reg.T1.value))

    def _gen_gt(self):
        """> - b < a"""
        # Меняем местами аргументы и вызываем <
        self._gen_swap()
        self._gen_lt()

    def _gen_le(self):
        """<= - NOT (a > b)"""
        self._gen_gt()
        self._gen_not()

    def _gen_ge(self):
        """>=  - NOT (a < b)"""
        self._gen_lt()
        self._gen_not()

    # Стековые операции
    def _gen_dup(self):
        """dup - дублировать вершину стека"""
        # LOAD SP, 0; PUSH T1
        self.code.append(Instruction(Opcode.LOAD, rs=Reg.SP.value, rt=Reg.T1.value, imm=0))
        self.code.append(Instruction(Opcode.PUSH, rs=Reg.T1.value))

    def _gen_drop(self):
        """drop - удалить вершину стека"""
        # POP T1
        self.code.append(Instruction(Opcode.POP, rt=Reg.T1.value))

    def _gen_swap(self):
        """swap - поменять местами два верхних элемента"""
        # POP T1; POP T2; PUSH T1; PUSH T2
        self.code.append(Instruction(Opcode.POP, rt=Reg.T1.value))
        self.code.append(Instruction(Opcode.POP, rt=Reg.T2.value))
        self.code.append(Instruction(Opcode.PUSH, rs=Reg.T1.value))
        self.code.append(Instruction(Opcode.PUSH, rs=Reg.T2.value))

    def _gen_over(self):
        """over - копировать второй элемент на вершину"""
        # LOAD SP, -1; PUSH T1
        self.code.append(Instruction(Opcode.LOAD, rs=Reg.SP.value, rt=Reg.T1.value, imm=-1 & 0xFFFF))
        self.code.append(Instruction(Opcode.PUSH, rs=Reg.T1.value))

    def _gen_rot(self):
        """rot - третий элемент на вершину: ( a b c -- b c a )"""
        # POP c, POP b, POP a, PUSH b, PUSH c, PUSH a
        self.code.append(Instruction(Opcode.POP, rt=Reg.T1.value))  # c
        self.code.append(Instruction(Opcode.POP, rt=Reg.T2.value))  # b
        self.code.append(Instruction(Opcode.POP, rt=Reg.BASE.value))  # a (используем BASE как временный)
        self.code.append(Instruction(Opcode.PUSH, rs=Reg.T2.value))  # push b
        self.code.append(Instruction(Opcode.PUSH, rs=Reg.T1.value))  # push c
        self.code.append(Instruction(Opcode.PUSH, rs=Reg.BASE.value))  # push a

    # Стек возвратов
    def _gen_to_r(self):
        """Генерация >r (переместить на стек возвратов)"""
        self.code.append(Instruction(Opcode.POP, rt=Reg.T1.value))
        self.code.append(Instruction(Opcode.STORE, rs=Reg.T1.value, rt=Reg.RSP.value, imm=0))  # сохраняем значение
        self.code.append(Instruction(Opcode.LOAD, rs=Reg.ZERO.value, rt=Reg.T1.value, imm=1))
        self.code.append(Instruction(Opcode.ADD, rs=Reg.RSP.value, rt=Reg.T1.value, rd=Reg.RSP.value))

    def _gen_from_r(self):
        """Генерация r> (снять со стека возвратов)"""
        # RSP--
        self.code.append(Instruction(Opcode.LOAD, rs=Reg.ZERO.value, rt=Reg.T1.value, imm=1))
        self.code.append(Instruction(Opcode.SUB, rs=Reg.RSP.value, rt=Reg.T1.value, rd=Reg.RSP.value))
        self.code.append(Instruction(Opcode.LOAD, rs=Reg.RSP.value, rt=Reg.T1.value, imm=0))
        self.code.append(Instruction(Opcode.PUSH, rs=Reg.T1.value))

    def _gen_r_fetch(self):
        """Генерация r@ (прочитать со стека возвратов без удаления)"""
        self.code.append(Instruction(Opcode.LOAD, rs=Reg.RSP.value, rt=Reg.T1.value, imm=-1 & 0xFFFF))
        self.code.append(Instruction(Opcode.PUSH, rs=Reg.T1.value))

    # Операции с памятью
    def _gen_fetch(self):
        """@ (load) - загрузить из памяти"""
        self.code.append(Instruction(Opcode.POP, rt=Reg.T1.value))  # адрес
        self.code.append(Instruction(Opcode.LOAD, rs=Reg.T1.value, rt=Reg.T1.value, imm=0))
        self.code.append(Instruction(Opcode.PUSH, rs=Reg.T1.value))

    def _gen_store(self):
        """! (store) - сохранить в память"""
        self.code.append(Instruction(Opcode.POP, rt=Reg.T1.value))  # адрес
        self.code.append(Instruction(Opcode.POP, rt=Reg.T2.value))  # значение
        self.code.append(Instruction(Opcode.STORE, rs=Reg.T2.value, rt=Reg.T1.value, imm=0))

    # MMIO
    def _gen_emit(self):
        """emit - вывести символ"""
        self.code.append(Instruction(Opcode.POP, rt=Reg.T1.value))
        self.code.append(Instruction(Opcode.OUT, rs=Reg.T1.value, imm=self.IO_OUTPUT_PORT))

    def _gen_key(self):
        """key - ввести символ"""
        self.code.append(Instruction(Opcode.IN, rt=Reg.T1.value, imm=self.IO_INPUT_PORT))
        self.code.append(Instruction(Opcode.PUSH, rs=Reg.T1.value))

    def _gen_print_num(self):
        """. - вывести число"""
        self.code.append(Instruction(Opcode.POP, rt=Reg.T1.value))
        self.code.append(Instruction(Opcode.OUT, rs=Reg.T1.value, imm=self.IO_OUTPUT_PORT))

    # Циклы
    def _gen_loop_index(self):
        """Генерация i (индекс цикла)"""
        # В полной реализации это будет чтение с стека возвратов
        self.code.append(Instruction(Opcode.LOAD, rs=Reg.ZERO.value, rt=Reg.T1.value, imm=0))
        self.code.append(Instruction(Opcode.PUSH, rs=Reg.T1.value))

    def _gen_loop_index2(self):
        """Генерация j (внешний индекс цикла)"""
        self.code.append(Instruction(Opcode.LOAD, rs=Reg.ZERO.value, rt=Reg.T1.value, imm=0))
        self.code.append(Instruction(Opcode.PUSH, rs=Reg.T1.value))

    # Управление потоком
    def _gen_exit(self):
        """Генерация exit (досрочный выход из слова)"""
        # Возврат из процедуры - упрощенная реализация
        self.code.append(Instruction(Opcode.POP, rt=Reg.PC.value))

    def _collect_variable(self, tokens: List[Token], i: int) -> int:
        i += 1
        name = tokens[i].value
        size = 1
        i += 1

        if i < len(tokens) and tokens[i].type == 'NUMBER':
            size = int(tokens[i].value)
            i += 1

        self.variables[name] = self.var_addr
        self.var_addr += size * 4
        print(f"Переменная '{name}' по адресу 0x{self.variables[name]:04X}, размер {size}")
        return i

    def _skip_variable(self, tokens: List[Token], i: int) -> int:
        i += 1
        i += 1
        if i < len(tokens) and tokens[i].type == 'NUMBER':
            i += 1
        return i

    def _collect_word_def(self, tokens: List[Token], i: int) -> int:
        i += 1
        name = tokens[i].value
        self.words[name] = -1
        i += 1
        while i < len(tokens) and tokens[i].type != 'SEMICOLON':
            i += 1
        return i + 1

    def _compile_word_def(self, tokens: List[Token], i: int) -> int:
        i += 1
        name = tokens[i].value
        i += 1

        word_addr = len(self.code) * 4
        self.words[name] = word_addr
        print(f"Компилирую слово '{name}' по адресу 0x{word_addr:04X}")

        while i < len(tokens) and tokens[i].type != 'SEMICOLON':
            i = self._compile_token(tokens, i)

        # RET заменяем на возврат через стек возвратов
        self.code.append(Instruction(Opcode.POP, rt=Reg.PC.value))
        return i + 1

    def _compile_token(self, tokens: List[Token], i: int) -> int:
        token = tokens[i]

        if token.type == 'NUMBER':
            self._emit_literal(int(token.value))
        elif token.type == 'STRING_LITERAL':
            self._gen_string_literal(token.value)
        elif token.type == 'WORD':
            if token.value in self.builtins:
                self.builtins[token.value]()
            elif token.value in self.words:
                addr = self.words[token.value]
                if addr == -1:
                    raise ValueError(f"Слово '{token.value}' не скомпилировано")
                # CALL заменяем на PUSH PC; JMP addr
                self.code.append(Instruction(Opcode.PUSH, rs=Reg.PC.value))
                self.code.append(Instruction(Opcode.JMP, addr=addr, is_label=True))
            elif token.value in self.variables:
                addr = self.variables[token.value]
                self._emit_literal(addr)
            else:
                raise ValueError(f"Неизвестное слово: {token.value} в строке {token.line}")
        elif token.type == 'IF':
            return self._compile_if(tokens, i)
        elif token.type == 'BEGIN':
            return self._compile_begin(tokens, i)
        elif token.type == 'UNTIL':
            return self._compile_until(tokens, i)
        elif token.type == 'WHILE':
            return self._compile_while(tokens, i)
        elif token.type == 'REPEAT':
            return self._compile_repeat(tokens, i)
        elif token.type == 'DO':
            return self._compile_do(tokens, i)
        elif token.type == 'LOOP':
            return self._compile_loop(tokens, i)
        elif token.type == 'AGAIN':
            return self._compile_again(tokens, i)

        return i + 1

    def _gen_string_literal(self, text: str):
        """Генерировать строковый литерал ." ... " """
        for char in text:
            char_code = ord(char) & 0xFF
            self._emit_literal(char_code)
            self._gen_emit()

    def _compile_if(self, tokens: List[Token], i: int) -> int:
        i += 1
        self.code.append(Instruction(Opcode.POP, rt=Reg.T1.value))

        end_label = self._new_label()
        self.code.append(Instruction(Opcode.JZ, rs=Reg.T1.value, imm=end_label, is_label=True))

        while i < len(tokens) and tokens[i].type not in ['THEN', 'ELSE']:
            i = self._compile_token(tokens, i)

        if i < len(tokens) and tokens[i].type == 'ELSE':
            else_label = end_label
            end_label = self._new_label()
            self.code.append(Instruction(Opcode.JMP, addr=else_label, is_label=True))
            self._place_label(else_label)
            i += 1

            while i < len(tokens) and tokens[i].type != 'THEN':
                i = self._compile_token(tokens, i)

        self._place_label(end_label)
        return i + 1

    def _compile_begin(self, tokens: List[Token], i: int) -> int:
        i += 1
        start_label = self._new_label()
        self._place_label(start_label)
        self.begin_stack.append(start_label)
        return i

    def _compile_until(self, tokens: List[Token], i: int) -> int:
        """Компилируем begin и until"""
        if not self.begin_stack:
            raise ValueError("UNTIL без соответствующего BEGIN")

        begin_addr = self.begin_stack.pop()

        # POP условие и проверка
        self.code.append(Instruction(Opcode.POP, rt=Reg.T1.value))
        # Если условие == 0, то переходим обратно к begin
        self.code.append(Instruction(Opcode.JZ, rs=Reg.T1.value, imm=begin_addr, is_label=True))

        return i + 1

    def _compile_while(self, tokens: List[Token], i: int) -> int:
        i += 1
        self.code.append(Instruction(Opcode.POP, rt=Reg.T1.value))

        end_label = self._new_label()
        self.code.append(Instruction(Opcode.JZ, rs=Reg.T1.value, imm=end_label, is_label=True))

        while i < len(tokens) and tokens[i].type != 'REPEAT':
            i = self._compile_token(tokens, i)

        if self.begin_stack:
            start_label = self.begin_stack.pop()
            self.code.append(Instruction(Opcode.JMP, addr=start_label, is_label=True))

        self._place_label(end_label)
        return i + 1

    def _compile_repeat(self, tokens: List[Token], i: int) -> int:
        """Компилировать begin ... while ... repeat"""
        if not self.begin_stack:
            raise ValueError("REPEAT без соответствующего BEGIN")

        begin_addr = self.begin_stack.pop()

        # Безусловный переход обратно к begin
        self.code.append(Instruction(Opcode.JMP, addr=begin_addr, is_label=True))

        return i + 1

    def _compile_do(self, tokens: List[Token], i: int) -> int:
        """Компиляция do ... loop"""
        i += 1
        loop_start = self._new_label()
        self._place_label(loop_start)
        self.loop_stack.append(loop_start)
        return i

    def _compile_loop(self, tokens: List[Token], i: int) -> int:
        """Завершение do ... loop"""
        if not self.loop_stack:
            raise ValueError("LOOP без соответствующего DO")

        loop_start = self.loop_stack.pop()
        self.code.append(Instruction(Opcode.JMP, addr=loop_start, is_label=True))
        return i + 1

    def _compile_again(self, tokens: List[Token], i: int) -> int:
        """Компиляция begin ... again (бесконечный цикл)"""
        if not self.begin_stack:
            raise ValueError("AGAIN без соответствующего BEGIN")

        begin_addr = self.begin_stack.pop()
        self.code.append(Instruction(Opcode.JMP, addr=begin_addr, is_label=True))
        return i + 1

    def _new_label(self) -> int:
        label = self.label_count
        self.label_count += 1
        return label

    def _place_label(self, label: int):
        self.labels[label] = len(self.code) * 4

    def _resolve_labels(self):
        for instr in self.code:
            if hasattr(instr, 'is_label') and instr.is_label:
                if instr.addr in self.labels:
                    resolved_addr = self.labels[instr.addr]
                    if instr.opcode == Opcode.JMP:
                        instr.addr = resolved_addr & 0x3FFFFFF
                    else:
                        instr.addr = resolved_addr & 0xFFFF

                if hasattr(instr, 'imm') and instr.imm in self.labels:
                    resolved_addr = self.labels[instr.imm]
                    instr.imm = resolved_addr & 0xFFFF


class RiscForthTranslator:
    def __init__(self):
        self.compiler = RiscForthCompiler()

    def translate_file(self, source_path: str, output_path: str):
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

        with open(output_path, 'wb') as f:
            for instr in instructions:
                f.write(instr.to_binary())

        with open(output_path + '.hex', 'w', encoding='utf-8') as f:
            f.write(f"; RISC Forth Compiler Output\n")
            f.write(f"; Source: {source_path}\n")
            f.write(f"; Instructions: {len(instructions)}\n\n")

            for i, instr in enumerate(instructions):
                f.write(instr.to_hex(i * 4) + '\n')

        print(f"Compiled {len(instructions)} instructions")
        print(f"Output: {output_path}")
        print(f"Listing: {output_path}.hex")


def main():
    if len(sys.argv) != 3:
        print("Usage: python translator.py <source.forth> <output.bin>")
        sys.exit(1)

    translator = RiscForthTranslator()
    try:
        translator.translate_file(sys.argv[1], sys.argv[2])
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()