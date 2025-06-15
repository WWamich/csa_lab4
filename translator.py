import sys
import struct
from enum import Enum
from typing import List, Dict


class Opcode(Enum):
    NOP = 0x00;
    HALT = 0x01;
    LI = 0x02;
    ADDI = 0x03
    JMP = 0x04;
    JZ = 0x05;
    JNZ = 0x06;
    CALL = 0x07;
    RET = 0x08
    PUSH = 0x10;
    POP = 0x11;
    DUP = 0x12;
    DROP = 0x13;
    SWAP = 0x14;
    OVER = 0x15;
    ROT = 0x16
    ADD = 0x20;
    SUB = 0x21;
    MUL = 0x22;
    DIV = 0x23;
    MOD = 0x24
    EQ = 0x25;
    LT = 0x26;
    GT = 0x27
    LOAD = 0x30;
    STORE = 0x31;
    IN = 0x32;
    OUT = 0x33
    RSPUSH = 0x40;
    RSPOP = 0x41;
    RSPEEK = 0x42
    EXIT = 0x43;
    LITERAL = 0x44;
    BRANCH = 0x45


class Reg(Enum):
    ZERO = 0;
    DSP = 1;
    RSP = 2;
    TOS = 3;
    T1 = 4;
    T2 = 5;
    T3 = 6;
    BASE = 7


class Instruction:
    def __init__(self, opcode: Opcode, rs=0, rt=0, rd=0, imm=0, addr=0, is_label=False):
        self.opcode = opcode
        self.rs = rs;
        self.rt = rt;
        self.rd = rd
        self.imm = imm;
        self.addr = addr
        self.is_label = is_label

    def to_binary(self) -> bytes:
        """Безопасная упаковка с проверкой границ"""

        def safe_uint32(value):
            """Безопасное приведение к 32-битному беззнаковому числу"""
            value = int(value) & 0xFFFFFFFF
            if value < 0:
                value = 0
            elif value > 0xFFFFFFFF:
                value = 0xFFFFFFFF
            return value

        def safe_uint16(value):
            """Безопасное приведение к 16-битному беззнаковому числу"""
            value = int(value) & 0xFFFF
            if value < 0:
                value = 0
            elif value > 0xFFFF:
                value = 0xFFFF
            return value

        def safe_uint26(value):
            """Безопасное приведение к 26-битному беззнаковому числу"""
            value = int(value) & 0x3FFFFFF
            if value < 0:
                value = 0
            elif value > 0x3FFFFFF:
                value = 0x3FFFFFF
            return value

        if self.opcode in [Opcode.JMP, Opcode.CALL]:
            # J-type: [opcode:6][address:26]
            opcode_bits = (self.opcode.value & 0x3F) << 26
            addr_bits = safe_uint26(self.addr)
            word = opcode_bits | addr_bits
        elif self.opcode in [Opcode.LI, Opcode.ADDI, Opcode.JZ, Opcode.JNZ, Opcode.LOAD, Opcode.STORE, Opcode.LITERAL]:
            # I-type: [opcode:6][rs:5][rt:5][immediate:16]
            opcode_bits = (self.opcode.value & 0x3F) << 26
            rs_bits = (self.rs & 0x1F) << 21
            rt_bits = (self.rt & 0x1F) << 16
            imm_bits = safe_uint16(self.imm)
            word = opcode_bits | rs_bits | rt_bits | imm_bits
        else:
            # R-type: [opcode:6][rs:5][rt:5][rd:5][shamt:5][funct:6]
            opcode_bits = (self.opcode.value & 0x3F) << 26
            rs_bits = (self.rs & 0x1F) << 21
            rt_bits = (self.rt & 0x1F) << 16
            rd_bits = (self.rd & 0x1F) << 11
            word = opcode_bits | rs_bits | rt_bits | rd_bits

        word = safe_uint32(word)

        try:
            return struct.pack('>I', word)
        except struct.error as e:
            print(
                f"ОШИБКА упаковки: word=0x{word:08X}, opcode={self.opcode}, rs={self.rs}, rt={self.rt}, rd={self.rd}, imm={self.imm}, addr={self.addr}")
            raise e

    def to_hex(self, addr: int) -> str:
        """Листинг с безопасной обработкой"""
        try:
            hex_code = self.to_binary().hex().upper()
        except:
            hex_code = "XXXXXXXX"

        if self.opcode == Opcode.LI:
            mnemonic = f"LI R{self.rt}, {self.imm}"
        elif self.opcode == Opcode.LITERAL:
            mnemonic = f"LITERAL {self.imm}"
        elif self.opcode in [Opcode.ADD, Opcode.SUB, Opcode.MUL, Opcode.DIV, Opcode.MOD, Opcode.EQ, Opcode.LT,
                             Opcode.GT]:
            mnemonic = f"{self.opcode.name} R{self.rd}, R{self.rs}, R{self.rt}"
        elif self.opcode in [Opcode.PUSH, Opcode.OUT, Opcode.RSPUSH]:
            mnemonic = f"{self.opcode.name} R{self.rs}"
        elif self.opcode in [Opcode.POP, Opcode.IN, Opcode.RSPOP]:
            mnemonic = f"{self.opcode.name} R{self.rt}"
        elif self.opcode == Opcode.JZ:
            mnemonic = f"JZ R{self.rs}, 0x{self.imm:04X}"
        elif self.opcode == Opcode.JNZ:
            mnemonic = f"JNZ R{self.rs}, 0x{self.imm:04X}"
        elif self.opcode in [Opcode.JMP, Opcode.CALL]:
            mnemonic = f"{self.opcode.name} 0x{self.addr:04X}"
        else:
            mnemonic = self.opcode.name
        return f"0x{addr:04X}: {hex_code}  {mnemonic}"


class Token:
    def __init__(self, type: str, value: str, line: int = 0):
        self.type = type;
        self.value = value;
        self.line = line


def tokenize(text: str) -> List[Token]:
    """Улучшенный токенизер с лучшей обработкой строк"""
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


class FixedForthCompiler:
    def __init__(self):
        self.code: List[Instruction] = []
        self.data: List[int] = []
        self.strings: Dict[str, int] = {}
        self.variables: Dict[str, int] = {}
        self.words: Dict[str, int] = {}
        self.labels: Dict[int, int] = {}
        self.label_count = 0

        # Стеки для циклов
        self.loop_stack = []  # для do/loop
        self.begin_stack = []  # для begin/until/while/repeat

        self.data_addr = 0x1000
        self.var_addr = 0x2000

        self.builtins = {
            # Арифметика
            '+': self._gen_binary_op(Opcode.ADD),
            '-': self._gen_binary_op(Opcode.SUB),
            '*': self._gen_binary_op(Opcode.MUL),
            '/': self._gen_binary_op(Opcode.DIV),
            'mod': self._gen_binary_op(Opcode.MOD),

            # Сравнения
            '=': self._gen_binary_op(Opcode.EQ),
            '<': self._gen_binary_op(Opcode.LT),
            '>': self._gen_binary_op(Opcode.GT),

            # Стековые операции
            'dup': lambda: self.code.append(Instruction(Opcode.DUP)),
            'drop': lambda: self.code.append(Instruction(Opcode.DROP)),
            'swap': lambda: self.code.append(Instruction(Opcode.SWAP)),
            'over': lambda: self.code.append(Instruction(Opcode.OVER)),
            'rot': lambda: self.code.append(Instruction(Opcode.ROT)),

            # Стек возвратов
            '>r': self._gen_to_r,
            'r>': self._gen_from_r,
            'r@': self._gen_r_fetch,

            # Память
            '@': self._gen_fetch,
            '!': self._gen_store,

            # I/O
            'emit': self._gen_emit,
            'key': self._gen_key,
            '.': self._gen_print_num,

            # Управление потоком
            'exit': self._gen_exit,

            # Счетчики циклов (заглушки - возвращают 0)
            'i': self._gen_loop_index,
            'j': self._gen_loop_index2,
        }

    def compile(self, tokens: List[Token]) -> List[Instruction]:
        print("Проход 1: сбор определений...")
        i = 0
        while i < len(tokens):
            token = tokens[i]
            if token.type == 'COLON':
                i = self._collect_word_def(tokens, i)
            elif token.type == 'VARIABLE':
                i = self._collect_variable(tokens, i)
            else:
                i += 1

        print(f"Найдено слов: {list(self.words.keys())}")
        print(f"Найдено переменных: {list(self.variables.keys())}")

        # Инициализация стеков
        self.code.append(Instruction(Opcode.LI, rt=Reg.DSP.value, imm=0x3000))
        self.code.append(Instruction(Opcode.LI, rt=Reg.RSP.value, imm=0x4000))

        print("Проход 2: генерация кода...")
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

    def _collect_variable(self, tokens: List[Token], i: int) -> int:
        """Ходим по файлу и собираем переменные."""
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
        """Пропускаем переменную на 2 проходе"""
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

        self.code.append(Instruction(Opcode.RET))
        return i + 1

    def _compile_token(self, tokens: List[Token], i: int) -> int:
        token = tokens[i]

        if token.type == 'NUMBER':
            self._gen_number(int(token.value))
        elif token.type == 'STRING':
            self._gen_string(token.value)
        elif token.type == 'STRING_LITERAL':
            self._gen_string_literal(token.value)
        elif token.type == 'WORD':
            if token.value in self.builtins:
                self.builtins[token.value]()
            elif token.value in self.words:
                addr = self.words[token.value]
                if addr == -1:
                    raise ValueError(f"Слово '{token.value}' не скомпилировано")
                self.code.append(Instruction(Opcode.CALL, addr=addr, is_label=True))
            elif token.value in self.variables:
                addr = self.variables[token.value]
                self.code.append(Instruction(Opcode.LI, rt=Reg.T1.value, imm=addr & 0xFFFF))
                self.code.append(Instruction(Opcode.PUSH, rs=Reg.T1.value))
            else:
                raise ValueError(f"Неизвестное слово: {token.value} в строке {token.line}")
        elif token.type == 'IF':
            return self._compile_if(tokens, i)
        elif token.type == 'BEGIN':
            return self._compile_begin(tokens, i)
        elif token.type == 'DO':
            return self._compile_do(tokens, i)
        elif token.type == 'LOOP':
            return self._compile_loop(tokens, i)
        elif token.type == 'WHILE':
            return self._compile_while(tokens, i)
        elif token.type == 'AGAIN':
            return self._compile_again(tokens, i)

        return i + 1

    def _compile_do(self, tokens: List[Token], i: int) -> int:
        """Компилировать do ... loop"""
        i += 1
        loop_start = len(self.code) * 4
        self.loop_stack.append(loop_start)

        return i

    def _compile_loop(self, tokens: List[Token], i: int) -> int:
        """Завершить do ... loop"""
        if not self.loop_stack:
            raise ValueError("LOOP без соответствующего DO")

        loop_start = self.loop_stack.pop()

        self.code.append(Instruction(Opcode.JMP, addr=loop_start, is_label=True))

        return i + 1

    def _compile_again(self, tokens: List[Token], i: int) -> int:
        """Компилировать begin ... again (бесконечный цикл)"""
        if not self.begin_stack:
            raise ValueError("AGAIN без соответствующего BEGIN")

        begin_addr = self.begin_stack.pop()
        self.code.append(Instruction(Opcode.JMP, addr=begin_addr, is_label=True))
        return i + 1

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

    def _gen_number(self, value: int):
        """Генерируем числа"""
        safe_value = value & 0xFFFF
        if value < 0:
            safe_value = 0
        elif value > 0xFFFF:
            safe_value = 0xFFFF

        self.code.append(Instruction(Opcode.LI, rt=Reg.T1.value, imm=safe_value))
        self.code.append(Instruction(Opcode.PUSH, rs=Reg.T1.value))

    def _gen_string_literal(self, text: str):
        """Генерировать строковый литерал ." ... " """
        for char in text:
            char_code = ord(char) & 0xFF
            self.code.append(Instruction(Opcode.LI, rt=Reg.T1.value, imm=char_code))
            self.code.append(Instruction(Opcode.OUT, rs=Reg.T1.value))

    def _gen_string(self, text: str):
        if text not in self.strings:
            addr = self.data_addr
            self.strings[text] = addr
            self.data.extend([len(text)] + [ord(c) for c in text])
            self.data_addr += (len(text) + 1) * 4

        addr = self.strings[text]
        safe_addr = addr & 0xFFFF
        self.code.append(Instruction(Opcode.LI, rt=Reg.T1.value, imm=safe_addr))
        self.code.append(Instruction(Opcode.PUSH, rs=Reg.T1.value))

    def _gen_binary_op(self, opcode: Opcode):
        def gen():
            self.code.append(Instruction(Opcode.POP, rt=Reg.T2.value))
            self.code.append(Instruction(Opcode.POP, rt=Reg.T1.value))
            self.code.append(Instruction(opcode, rs=Reg.T1.value, rt=Reg.T2.value, rd=Reg.T1.value))
            self.code.append(Instruction(Opcode.PUSH, rs=Reg.T1.value))

        return gen

    def _gen_to_r(self):
        """Генерация >r (переместить на стек возвратов)"""
        self.code.append(Instruction(Opcode.POP, rt=Reg.T1.value))
        self.code.append(Instruction(Opcode.RSPUSH, rs=Reg.T1.value))

    def _gen_from_r(self):
        """Генерация r> (снять со стека возвратов)"""
        self.code.append(Instruction(Opcode.RSPOP, rt=Reg.T1.value))
        self.code.append(Instruction(Opcode.PUSH, rs=Reg.T1.value))

    def _gen_r_fetch(self):
        """Генерация r@ (прочитать со стека возвратов)"""
        self.code.append(Instruction(Opcode.RSPEEK, rt=Reg.T1.value))
        self.code.append(Instruction(Opcode.PUSH, rs=Reg.T1.value))

    def _gen_fetch(self):
        self.code.append(Instruction(Opcode.POP, rt=Reg.T1.value))
        self.code.append(Instruction(Opcode.LOAD, rs=Reg.T1.value, rt=Reg.T1.value))
        self.code.append(Instruction(Opcode.PUSH, rs=Reg.T1.value))

    def _gen_store(self):
        self.code.append(Instruction(Opcode.POP, rt=Reg.T1.value))  # addr
        self.code.append(Instruction(Opcode.POP, rt=Reg.T2.value))  # value
        self.code.append(Instruction(Opcode.STORE, rs=Reg.T2.value, rt=Reg.T1.value))

    def _gen_emit(self):
        self.code.append(Instruction(Opcode.POP, rt=Reg.T1.value))
        self.code.append(Instruction(Opcode.OUT, rs=Reg.T1.value))

    def _gen_key(self):
        self.code.append(Instruction(Opcode.IN, rt=Reg.T1.value))
        self.code.append(Instruction(Opcode.PUSH, rs=Reg.T1.value))

    def _gen_print_num(self):
        self.code.append(Instruction(Opcode.POP, rt=Reg.T1.value))
        self.code.append(Instruction(Opcode.OUT, rs=Reg.T1.value))

    def _gen_exit(self):
        """Генерировать exit (досрочный выход из слова)"""
        self.code.append(Instruction(Opcode.RET))

    def _gen_loop_index(self):
        """Генерировать i (индекс цикла) - упрощенная версия"""

        self.code.append(Instruction(Opcode.LI, rt=Reg.T1.value, imm=0))
        self.code.append(Instruction(Opcode.PUSH, rs=Reg.T1.value))

    def _gen_loop_index2(self):
        """Генерировать j (внешний индекс цикла)"""
        self.code.append(Instruction(Opcode.LI, rt=Reg.T1.value, imm=0))
        self.code.append(Instruction(Opcode.PUSH, rs=Reg.T1.value))

    def _new_label(self) -> int:
        label = self.label_count
        self.label_count += 1
        return label

    def _place_label(self, label: int):
        self.labels[label] = len(self.code) * 4

    def _resolve_labels(self):
        """Безопасное разрешение меток"""
        for instr in self.code:
            if hasattr(instr, 'is_label') and instr.is_label:
                if instr.addr in self.labels:
                    resolved_addr = self.labels[instr.addr]
                    if instr.opcode in [Opcode.JMP, Opcode.CALL]:
                        instr.addr = resolved_addr & 0x3FFFFFF
                    else:
                        instr.addr = resolved_addr & 0xFFFF

                if hasattr(instr, 'imm') and instr.imm in self.labels:
                    resolved_addr = self.labels[instr.imm]
                    instr.imm = resolved_addr & 0xFFFF


class FixedForthTranslator:
    def __init__(self):
        self.compiler = FixedForthCompiler()

    def translate_file(self, source_path: str, output_path: str):
        with open(source_path, 'r', encoding='utf-8') as f:
            source = f.read()

        print(f"Исходный код:\n{source}\n")

        tokens = tokenize(source)
        print(f"Токены: {[(t.type, t.value) for t in tokens[:20]]}\n")  # показать первые 20

        try:
            instructions = self.compiler.compile(tokens)
        except Exception as e:
            print(f"Ошибка компиляции: {e}")
            raise

        try:
            with open(output_path, 'wb') as f:
                for instr in instructions:
                    try:
                        f.write(instr.to_binary())
                    except Exception as e:
                        print(f"Ошибка записи инструкции: {instr}")
                        print(f"Детали: {e}")
                        raise
        except Exception as e:
            print(f"Ошибка записи файла: {e}")
            raise

        try:
            with open(output_path + '.hex', 'w', encoding='utf-8') as f:
                f.write(f"; Fixed Forth Compiler Output\n")
                f.write(f"; Source: {source_path}\n")
                f.write(f"; Instructions: {len(instructions)}\n\n")

                for i, instr in enumerate(instructions):
                    try:
                        f.write(instr.to_hex(i * 4) + '\n')
                    except Exception as e:
                        f.write(f"0x{i * 4:04X}: ERROR - {instr.opcode}\n")
        except Exception as e:
            print(f"Ошибка создания листинга: {e}")

        print(f"Compiled {len(instructions)} instructions")
        print(f"Output: {output_path}")
        print(f"Listing: {output_path}.hex")


def main():
    if len(sys.argv) != 3:
        print("Usage: python translator_fixed.py <source.forth> <output.bin>")
        sys.exit(1)

    translator = FixedForthTranslator()
    try:
        translator.translate_file(sys.argv[1], sys.argv[2])
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()