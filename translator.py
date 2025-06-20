import sys
import logging
import struct
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
                    tokens.append(Token('STRING_LITERAL', line[quote_start:quote_end], line_num))
                    line = line[:start] + ' ' + line[quote_end + 1:]
                else:
                    break
            else:
                break
        for word in line.split():
            if word.isdigit() or (word.startswith('-') and word[1:].isdigit()):
                tokens.append(Token('NUMBER', word, line_num))
            elif word.startswith('0x'):
                tokens.append(Token('HEX_NUMBER', word, line_num))
            elif word == ':':
                tokens.append(Token('COLON', word, line_num))
            elif word == ';':
                tokens.append(Token('SEMICOLON', word, line_num))
            elif word == 'variable':
                tokens.append(Token('VARIABLE', word, line_num))
            elif word in ['if', 'then', 'else', 'begin', 'until', 'while', 'repeat', 'do', 'loop', 'again']:
                tokens.append(Token(word.upper(), word, line_num))
            else:
                tokens.append(Token('WORD', word, line_num))
    return tokens

class RiscForthCompiler:
    def __init__(self):
        self.code: List[Instruction] = []
        self.data_segment: List[int] = []
        self.procedures: Dict[str, int] = {}
        self.variables: Dict[str, int] = {}
        self.loop_stack: List[int] = []
        self.begin_stack: List[int] = []
        self.if_stack: List[Instruction] = []
        self.data_addr = 0x1000
        self.var_addr = 0x2000
        self.builtins = {
            '+': self._gen_add, '-': self._gen_sub, '*': self._gen_mul, '/': self._gen_div, 'mod': self._gen_mod,
            'and': self._gen_and, 'or': self._gen_or, 'xor': self._gen_xor,
            '=': self._gen_eq, '<': self._gen_lt, '>': self._gen_gt, '<>': self._gen_ne, '<=': self._gen_le, '>=': self._gen_ge,
            'dup': self._gen_dup, 'drop': self._gen_drop, 'swap': self._gen_swap, 'over': self._gen_over, 'rot': self._gen_rot,
            '@': self._gen_fetch, '!': self._gen_store,
            'emit': self._gen_emit, 'key': self._gen_key,
            'halt': self._gen_halt, 'not': self._gen_not, '0=': self._gen_zero_eq,
            'shl': self._gen_shl, 'shr': self._gen_shr,
        }
        logging.info("Компилятор Forth->RISC инициализирован")

    def _emit_literal(self, value: int):
        self.code.append(Instruction(Opcode.LOADI, rt=Reg.T1.value, imm=value))
        self.code.append(Instruction(Opcode.PUSH, rs=Reg.T1.value))

    def _gen_op(self, op: Opcode):
        self.code.append(Instruction(Opcode.POP, rt=Reg.T2.value))
        self.code.append(Instruction(Opcode.POP, rt=Reg.T1.value))
        self.code.append(Instruction(op, rd=Reg.T1.value, rs=Reg.T1.value, rt=Reg.T2.value))
        self.code.append(Instruction(Opcode.PUSH, rs=Reg.T1.value))

    def _gen_add(self): self._gen_op(Opcode.ADD)
    def _gen_sub(self): self._gen_op(Opcode.SUB)
    def _gen_mul(self): self._gen_op(Opcode.MUL)
    def _gen_div(self): self._gen_op(Opcode.DIV)
    def _gen_mod(self): self._gen_op(Opcode.MOD)
    def _gen_and(self): self._gen_op(Opcode.AND)
    def _gen_or(self): self._gen_op(Opcode.OR)
    def _gen_xor(self): self._gen_op(Opcode.XOR)
    def _gen_shl(self): self._gen_op(Opcode.SHL)
    def _gen_shr(self): self._gen_op(Opcode.SHR)

    def _gen_eq(self): self._gen_op(Opcode.CMP)
    def _gen_ne(self): self._gen_eq(); self._gen_not()
    def _gen_lt(self): self._gen_op(Opcode.SUB); self._emit_literal(31); self._gen_op(Opcode.SHR)
    def _gen_gt(self): self._gen_swap(); self._gen_lt()
    def _gen_le(self): self._gen_gt(); self._gen_not()
    def _gen_ge(self): self._gen_lt(); self._gen_not()

    def _gen_dup(self):
        self.code.extend([Instruction(Opcode.POP, rt=Reg.T1.value), Instruction(Opcode.PUSH, rs=Reg.T1.value), Instruction(Opcode.PUSH, rs=Reg.T1.value)])
    def _gen_drop(self):
        self.code.append(Instruction(Opcode.POP, rt=Reg.T1.value))
    def _gen_swap(self):
        self.code.extend([Instruction(Opcode.POP, rt=Reg.T1.value), Instruction(Opcode.POP, rt=Reg.T2.value), Instruction(Opcode.PUSH, rs=Reg.T1.value), Instruction(Opcode.PUSH, rs=Reg.T2.value)])
    def _gen_over(self):
        self.code.extend([Instruction(Opcode.POP, rt=Reg.T1.value), Instruction(Opcode.POP, rt=Reg.T2.value), Instruction(Opcode.PUSH, rs=Reg.T2.value), Instruction(Opcode.PUSH, rs=Reg.T1.value), Instruction(Opcode.PUSH, rs=Reg.T2.value)])
    def _gen_rot(self):
        self.code.extend([Instruction(Opcode.POP, rt=Reg.T1.value), Instruction(Opcode.POP, rt=Reg.T2.value), Instruction(Opcode.POP, rt=Reg.A0.value), Instruction(Opcode.PUSH, rs=Reg.T2.value), Instruction(Opcode.PUSH, rs=Reg.T1.value), Instruction(Opcode.PUSH, rs=Reg.A0.value)])

    def _gen_not(self):
        self._emit_literal(0)
        self._gen_eq()
    def _gen_zero_eq(self):
        self._gen_not()

    def _gen_fetch(self):
        self.code.extend([Instruction(Opcode.POP, rt=Reg.T1.value), Instruction(Opcode.LOAD, rt=Reg.T1.value, rs=Reg.T1.value, imm=0), Instruction(Opcode.PUSH, rs=Reg.T1.value)])
    def _gen_store(self):
        self.code.extend([Instruction(Opcode.POP, rt=Reg.T1.value), Instruction(Opcode.POP, rt=Reg.T2.value), Instruction(Opcode.STORE, rs=Reg.T2.value, rt=Reg.T1.value, imm=0)])

    def _gen_emit(self):
        self.code.extend([Instruction(Opcode.POP, rt=Reg.T1.value), Instruction(Opcode.OUT, rs=Reg.T1.value, imm=IO_OUTPUT_PORT)])
    def _gen_key(self):
        self.code.extend([Instruction(Opcode.IN, rt=Reg.T1.value, imm=IO_INPUT_PORT), Instruction(Opcode.PUSH, rs=Reg.T1.value)])
    def _gen_halt(self):
        self.code.append(Instruction(Opcode.HALT))

    def _gen_string_literal(self, text: str):
        string_addr = self.data_addr
        self.data_segment.append(len(text))
        for char in text:
            self.data_segment.append(ord(char))
        self.data_addr += len(text) + 1
        self._emit_literal(string_addr)

    def compile(self, tokens: List[Token]) -> List[Instruction]:
        logging.info("Начинаем компиляцию Forth кода")
        i = 0
        while i < len(tokens):
            token = tokens[i]
            logging.debug(f"Компилируем токен: {token}")

            if token.type == 'COLON':
                i += 1
                proc_name = tokens[i].value
                self.procedures[proc_name] = len(self.code)
                logging.info(f"Определена процедура '{proc_name}' по адресу {len(self.code)}")
            elif token.type == 'SEMICOLON':
                self.code.append(Instruction(Opcode.RET))
            elif token.type == 'VARIABLE':
                i += 1
                var_name = tokens[i].value
                self.variables[var_name] = self.var_addr
                self.var_addr += 4  # Резервируем 4 байта (одно машинное слово)
                logging.info(f"Определена переменная '{var_name}' по адресу {self.variables[var_name]}")
            elif token.type == 'STRING_LITERAL':
                self._gen_string_literal(token.value)
            elif token.type == 'NUMBER':
                self._emit_literal(int(token.value))
            elif token.type == 'HEX_NUMBER':
                self._emit_literal(int(token.value, 16))
            elif token.type == 'IF':
                self.code.append(Instruction(Opcode.POP, rt=Reg.T1.value))
                jump_instr = Instruction(Opcode.JZ, rs=Reg.T1.value, imm=0)
                self.code.append(jump_instr)
                self.if_stack.append(jump_instr)
            elif token.type == 'ELSE':
                jump_instr = Instruction(Opcode.JMP, addr=0)
                self.code.append(jump_instr)
                if_instr = self.if_stack.pop()
                if_instr.imm = len(self.code)
                self.if_stack.append(jump_instr)
            elif token.type == 'THEN':
                jump_instr = self.if_stack.pop()
                if jump_instr.opcode == Opcode.JMP:
                    jump_instr.addr = len(self.code)
                else:
                    jump_instr.imm = len(self.code)
            elif token.type == 'WORD':
                if token.value in self.builtins:
                    self.builtins[token.value]()
                elif token.value in self.procedures:
                    self.code.append(Instruction(Opcode.CALL, addr=self.procedures[token.value]))
                elif token.value in self.variables:
                    self._emit_literal(self.variables[token.value])
                else:
                    logging.warning(f"Неизвестное слово: {token.value} в строке {token.line}")
            i += 1

        self.code.append(Instruction(Opcode.HALT))
        logging.info(f"Компиляция завершена: {len(self.code)} инструкций")
        return self.code

    def save_binary(self, filename: str):
        with open(filename, 'wb') as f:
            for instruction in self.code:
                f.write(instruction.to_binary())
            # Дописываем сегмент данных в конец
            for val in self.data_segment:
                f.write(struct.pack('>I', val))

        with open(filename + '.hex', 'w') as f:
            f.write(f"; Instructions: {len(self.code)}\n")
            for i, instr in enumerate(self.code):
                f.write(instr.to_hex(i * 4) + '\n')
            f.write("\n; Data Segment\n")
            data_start_addr = len(self.code) * 4
            for i, val in enumerate(self.data_segment):
                f.write(f"0x{data_start_addr + i*4:04X}: {val:08X} ({chr(val) if 32 <= val <= 126 else '.'})\n")
        logging.info(f"Сохранено: {filename} и {filename}.hex")

class RiscForthTranslator:
    def __init__(self):
        self.compiler = RiscForthCompiler()

    def translate_file(self, source_path: str, output_path: str):
        with open(source_path, 'r', encoding='utf-8') as f:
            source = f.read()
        tokens = tokenize(source)
        instructions = self.compiler.compile(tokens)
        self.compiler.save_binary(output_path)
        print(f"✅ Компилировано {len(instructions)} инструкций. Выход: {output_path}, {output_path}.hex")

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