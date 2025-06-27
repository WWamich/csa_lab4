from __future__ import annotations
import logging
import struct
import sys
from typing import Any, Iterator
from isa import IO_INPUT_PORT, IO_OUTPUT_PORT, Instruction, Opcode, Reg


class Translator:
    """
    Класс транслятора. Преобразует исходный код на Forth в список
    машинных инструкций и секцию данных.
    """

    def __init__(self):
        self.code: list[Instruction] = []
        self.symbols: dict[str, Any] = self._get_built_in_symbols()
        self.string_relocations: dict[int, str] = {}
        self.control_flow_stack = []
        self.data_variables: dict[str, dict[str, int]] = {}  # name -> {'offset_words': word_offset, 'size_words':
        # word_size }
        self.current_data_section_offset_words: int = 0
        self.variable_relocations: dict[int, str] = {}

    def _get_built_in_symbols(self) -> dict[str, Any]:
        """
        Определяет встроенные слова.
        Простые слова отображаются на лямбда-функции, которые генерируют код напрямую.
        Сложные слова помечены как "subroutine" и будут заменены на CALLы к подпрограммам.
        """
        return {
            # Арифметика (простые)
            "+": lambda: self.emit([self.emit_binary_op_instr(Opcode.ADD)]),
            "-": lambda: self.emit([self.emit_binary_op_instr(Opcode.SUB)]),
            "*": lambda: self.emit([self.emit_binary_op_instr(Opcode.MUL)]),
            "/": lambda: self.emit([self.emit_binary_op_instr(Opcode.DIV)]),
            "MOD": lambda: self.emit([self.emit_binary_op_instr(Opcode.MOD)]),
            # ОПЕРАЦИИ СРАВНЕНИЯ И ЛОГИКИ:
            "AND": lambda: self.emit([self.emit_binary_op_instr(Opcode.AND)]),
            "OR": lambda: self.emit([self.emit_binary_op_instr(Opcode.OR)]),
            "XOR": lambda: self.emit([self.emit_binary_op_instr(Opcode.XOR)]),
            # Комплексные слова, реализуемые как подпрограммы
            "DROP": "subroutine",
            "=": "subroutine",
            "DUP": "subroutine",
            "SWAP": "subroutine",
            "/MOD": "subroutine",
            "2+": "subroutine",
            "1-": "subroutine",
            "2DROP": "subroutine",
            "OVER": "subroutine",
            "@": "subroutine",
            "!": "subroutine",
            "NOT": "subroutine",
            "EMIT": "subroutine",
            "1+": "subroutine",
            "KEY": "subroutine",
            "CR": "subroutine",
            "TYPE": "subroutine",
            "ROT": "subroutine",
            "2DUP": "subroutine",
            "0=": "subroutine",
            "<>": "subroutine",
            "<": "subroutine",
            ">": "subroutine",
            "<=": "subroutine",
            ">=": "subroutine",
            "0<>": "subroutine",
            # Управляющие структуры (обрабатываются во время компиляции)
            "IF": self.emit_if,
            "ELSE": self.emit_else,
            "THEN": self.emit_then,
            ":": self.start_word_definition,
            ";": self.end_word_definition,
            # Циклы
            "BEGIN": self.emit_begin,
            "WHILE": self.emit_while,
            "REPEAT": self.emit_repeat,
            "UNTIL": self.emit_until,
            # Системные слова
            "HALT": lambda: self.emit([Instruction(Opcode.HALT)]),
            "EXIT": lambda: self.emit([Instruction(Opcode.RET)]),
            "NOP": lambda: self.emit([Instruction(Opcode.NOP)]),
        }

    def emit(self, instructions: list[Instruction | list[Instruction]]):
        """Добавляет инструкции в код, рекурсивно раскрывая списки."""
        for instr in instructions:
            if isinstance(instr, list):
                self.emit(instr)
            else:
                self.code.append(instr)

    def emit_load_immediate(self, value: int, target_reg: int) -> list[Instruction]:
        """
        Генерирует инструкции для загрузки 32-битного непосредственного значения
        в регистр. Использует LUI и ORI для полной поддержки 32-битного диапазона.
        """
        if -32768 <= value <= 32767:
            return [Instruction(Opcode.ADDI, rs=Reg.ZERO.value, rt=target_reg, imm=value)]

        upper = (value >> 16) & 0xFFFF
        lower = value & 0xFFFF

        instructions = [Instruction(Opcode.LUI, rt=target_reg, imm=upper)]
        if lower != 0:
            instructions.append(Instruction(Opcode.ORI, rs=target_reg, rt=target_reg, imm=lower))
        return instructions

    def emit_push_instr(self, value: int, target_reg: int = Reg.T0.value) -> list[Instruction]:
        """Генерация инструкций для PUSH числа на стек с поддержкой 32-битных чисел."""
        load_instructions = self.emit_load_immediate(value, target_reg)
        push_instruction = Instruction(Opcode.PUSH, rs=target_reg)
        return load_instructions + [push_instruction]

    def emit_binary_op_instr(self, opcode: Opcode) -> list[Instruction]:
        """Генерация инструкций для бинарных операций (+, -, * и т.д.)."""
        return [
            Instruction(Opcode.POP, rt=Reg.T1.value),
            Instruction(Opcode.POP, rt=Reg.T0.value),
            Instruction(opcode, rd=Reg.T0.value, rs=Reg.T0.value, rt=Reg.T1.value),
            Instruction(Opcode.PUSH, rs=Reg.T0.value),
        ]

    def emit_if(self):
        # ( flag -- )
        self.emit([Instruction(Opcode.POP, rt=Reg.T0.value)])
        jz_instr = Instruction(Opcode.JZ, rt=Reg.T0.value, imm=0)
        self.emit([jz_instr])
        self.control_flow_stack.append({'type': 'IF', 'addr': len(self.code) - 1})

    def emit_else(self):
        if_info = self.control_flow_stack.pop()
        if if_info['type'] != 'IF':
            raise SyntaxError("ELSE without IF")

        jmp_instr = Instruction(Opcode.JMP, addr=0)
        self.emit([jmp_instr])

        else_addr = len(self.code)
        self.code[if_info['addr']].imm = else_addr - (if_info['addr'] + 1)

        self.control_flow_stack.append({'type': 'ELSE', 'addr': len(self.code) - 1})

    def emit_then(self):
        last_info = self.control_flow_stack.pop()
        if last_info['type'] not in ['IF', 'ELSE']:
            raise SyntaxError("THEN without matching IF/ELSE")

        exit_addr = len(self.code)
        jump_instr = self.code[last_info['addr']]

        if jump_instr.opcode == Opcode.JMP:
            jump_instr.addr = exit_addr
        else:
            jump_instr.imm = exit_addr - (last_info['addr'] + 1)

    def emit_begin(self):
        self.control_flow_stack.append({'type': 'BEGIN', 'addr': len(self.code)})

    def emit_while(self):
        self.emit([Instruction(Opcode.POP, rt=Reg.T0.value)])
        jz_instr = Instruction(Opcode.JZ, rt=Reg.T0.value, imm=0)
        self.emit([jz_instr])
        self.control_flow_stack.append({'type': 'WHILE', 'addr': len(self.code) - 1})

    def emit_repeat(self):
        while_info = self.control_flow_stack.pop()
        begin_info = self.control_flow_stack.pop()
        self.emit([Instruction(Opcode.JMP, addr=begin_info['addr'])])
        exit_addr = len(self.code)
        self.code[while_info['addr']].imm = exit_addr - (while_info['addr'] + 1)

    def emit_until(self):
        begin_info = self.control_flow_stack.pop()
        self.emit([
            Instruction(Opcode.POP, rt=Reg.T0.value),
            Instruction(Opcode.JZ, rt=Reg.T0.value, imm=begin_info['addr'] - (len(self.code) + 1))
        ])

    def _inject_built_in_subroutines(self):
        """Генерирует код для всех подпрограмм и сохраняет их адреса."""
        simple_subroutines = [
            ("EMIT", self._generate_emit_sub),
            ("KEY", self._generate_key_sub),
            ("DUP", self._generate_dup_sub),
            ("SWAP", self._generate_swap_sub),
            ("ROT", self._generate_rot_sub),
            ("DROP", self._generate_drop_sub),
            ("=", self._generate_equal_sub),
            ("1+", self._generate_plus_one_sub),
            ("OVER", self._generate_over_sub),
            ("2DUP", self._generate_2dup_sub),
            ("@", self._generate_fetch_sub),
            ("/MOD", self._generate_slash_mod_sub),
            ("2+", self._generate_plus_two_sub),
            ("1-", self._generate_minus_one_sub),
            ("2DROP", self._generate_two_drop_sub),
            ("!", self._generate_store_sub),
            ("0=", self._generate_zero_equal_sub),
            ("<>", self._generate_not_equal_sub),
            ("<", self._generate_less_than_sub),
            (">", self._generate_greater_than_sub),
            ("<=", self._generate_less_equal_sub),
            (">=", self._generate_greater_equal_sub),
            ("0<>", self._generate_zero_not_equal_sub),
        ]

        for name, generator in simple_subroutines:
            if self.symbols.get(name) == "subroutine":
                self.symbols[name] = len(self.code)
                generator()
        if self.symbols.get("NOT") == "subroutine":
            self.symbols["NOT"] = self.symbols["0="]
        dependent_subroutines = [
            ("CR", self._generate_cr_sub),
            ("TYPE", self._generate_type_sub),
        ]

        for name, generator in dependent_subroutines:
            if self.symbols.get(name.upper()) == "subroutine":
                self.symbols[name] = len(self.code)
                generator()

    def _generate_slash_mod_sub(self):
        self.emit([
            Instruction(Opcode.POP, rt=Reg.T1.value),
            Instruction(Opcode.POP, rt=Reg.T0.value),
            Instruction(Opcode.MOD, rd=Reg.T2.value, rs=Reg.T0.value, rt=Reg.T1.value),
            Instruction(Opcode.DIV, rd=Reg.A1.value, rs=Reg.T0.value, rt=Reg.T1.value),
            Instruction(Opcode.PUSH, rs=Reg.A1.value),
            Instruction(Opcode.PUSH, rs=Reg.T2.value),
            Instruction(Opcode.RET),
        ])

    def _generate_plus_two_sub(self):
        self.emit([
            Instruction(Opcode.POP, rt=Reg.T0.value),
            Instruction(Opcode.ADDI, rt=Reg.T0.value, rs=Reg.T0.value, imm=2),
            Instruction(Opcode.PUSH, rs=Reg.T0.value),
            Instruction(Opcode.RET)
        ])

    def _generate_minus_one_sub(self):
        self.emit([
            Instruction(Opcode.POP, rt=Reg.T0.value),
            Instruction(Opcode.ADDI, rt=Reg.T0.value, rs=Reg.T0.value, imm=-1),
            Instruction(Opcode.PUSH, rs=Reg.T0.value),
            Instruction(Opcode.RET)
        ])

    def _generate_two_drop_sub(self):
        self.emit([
            Instruction(Opcode.POP, rt=Reg.T0.value),
            Instruction(Opcode.POP, rt=Reg.T1.value),
            Instruction(Opcode.RET)
        ])

    def _generate_equal_sub(self):
        self.emit([
            Instruction(Opcode.POP, rt=Reg.T1.value),
            Instruction(Opcode.POP, rt=Reg.T0.value),
            Instruction(Opcode.CMP, rd=Reg.T0.value, rs=Reg.T0.value, rt=Reg.T1.value),
            Instruction(Opcode.PUSH, rs=Reg.T0.value),
            Instruction(Opcode.RET)
        ])

    def _generate_plus_one_sub(self):
        self.emit([
            Instruction(Opcode.POP, rt=Reg.T0.value),
            Instruction(Opcode.ADDI, rt=Reg.T0.value, rs=Reg.T0.value, imm=1),
            Instruction(Opcode.PUSH, rs=Reg.T0.value),
            Instruction(Opcode.RET)
        ])

    def _generate_drop_sub(self):
        self.emit([
            Instruction(Opcode.POP, rt=Reg.T0.value),
            Instruction(Opcode.RET)
        ])

    def _generate_rot_sub(self):
        """Генерирует подпрограмму для ROT ( a b c -- b c a )."""
        self.emit([
            Instruction(Opcode.POP, rt=Reg.T2.value),  # T2 <- c (верхний элемент)
            Instruction(Opcode.POP, rt=Reg.T1.value),  # T1 <- b
            Instruction(Opcode.POP, rt=Reg.T0.value),  # T0 <- a
            Instruction(Opcode.PUSH, rs=Reg.T1.value),
            Instruction(Opcode.PUSH, rs=Reg.T2.value),
            Instruction(Opcode.PUSH, rs=Reg.T0.value),
            Instruction(Opcode.RET),
        ])

    def _generate_type_sub(self):
        """
        Генерирует подпрограмму TYPE ( addr -- )
        addr - адрес p-строки (указывает на байт/слово с длиной)
        """
        emit_sub_addr = self.symbols.get("EMIT")
        if not isinstance(emit_sub_addr, int):
            raise RuntimeError("EMIT subroutine not found when generating TYPE subroutine")

        pre_loop_code = [
            Instruction(Opcode.POP, rt=Reg.T0.value),  # T0 = base_addr
            Instruction(Opcode.LOAD, rt=Reg.T1.value, rs=Reg.T0.value, imm=0),  # T1 = length
            Instruction(Opcode.ADDI, rs=Reg.T0.value, rt=Reg.T0.value, imm=1),  # T0 = &char[0]
        ]
        self.emit(pre_loop_code)
        type_loop_start_addr = len(self.code)
        jz_instruction = Instruction(Opcode.JZ, rt=Reg.T1.value, imm=0)
        self.emit([jz_instruction])
        type_jz_to_end_loop_idx = len(self.code) - 1
        loop_body_code = [
            Instruction(Opcode.PUSH, rs=Reg.T0.value),
            Instruction(Opcode.PUSH, rs=Reg.T1.value),
            Instruction(Opcode.LOAD, rt=Reg.T2.value, rs=Reg.T0.value, imm=0),
            Instruction(Opcode.PUSH, rs=Reg.T2.value),
            Instruction(Opcode.CALL, addr=emit_sub_addr),
            Instruction(Opcode.POP, rt=Reg.T1.value),
            Instruction(Opcode.POP, rt=Reg.T0.value),
            Instruction(Opcode.ADDI, rs=Reg.T0.value, rt=Reg.T0.value, imm=1),
            Instruction(Opcode.ADDI, rs=Reg.T1.value, rt=Reg.T1.value, imm=-1),
            Instruction(Opcode.JMP, addr=type_loop_start_addr),
        ]
        self.emit(loop_body_code)
        after_loop_addr = len(self.code)
        self.code[type_jz_to_end_loop_idx].imm = after_loop_addr - (type_jz_to_end_loop_idx + 1)
        self.emit([Instruction(Opcode.RET)])

    def _generate_not_equal_sub(self):
        """Генерирует подпрограмму для <> (не равно)."""
        self.emit([
            Instruction(Opcode.POP, rt=Reg.T1.value),
            Instruction(Opcode.POP, rt=Reg.T0.value),
            Instruction(Opcode.CMP, rd=Reg.T2.value, rs=Reg.T0.value, rt=Reg.T1.value),
            self.emit_load_immediate(1, Reg.T1.value),
            Instruction(Opcode.XOR, rd=Reg.T0.value, rs=Reg.T2.value, rt=Reg.T1.value),
            Instruction(Opcode.PUSH, rs=Reg.T0.value),
            Instruction(Opcode.RET),
        ])

    def _generate_2dup_sub(self):
        """2DUP ( a b -- a b a b ) дублирует два верхних элемента."""
        self.emit([
            Instruction(Opcode.POP, rt=Reg.T1.value),
            Instruction(Opcode.POP, rt=Reg.T0.value),
            Instruction(Opcode.PUSH, rs=Reg.T0.value),
            Instruction(Opcode.PUSH, rs=Reg.T1.value),
            Instruction(Opcode.PUSH, rs=Reg.T0.value),
            Instruction(Opcode.PUSH, rs=Reg.T1.value),
            Instruction(Opcode.RET),
        ])

    def _generate_less_than_sub(self):
        """
        Генерирует подпрограмму для < (a < b).
        """
        self.emit([
            Instruction(Opcode.POP, rt=Reg.T1.value),  # T1 <- b
            Instruction(Opcode.POP, rt=Reg.T0.value),  # T0 <- a
            Instruction(Opcode.SUB, rd=Reg.T2.value, rs=Reg.T0.value, rt=Reg.T1.value),
            Instruction(Opcode.ADDI, rt=Reg.T0.value, rs=Reg.ZERO.value, imm=31),
            Instruction(Opcode.SHR, rd=Reg.T2.value, rs=Reg.T2.value, rt=Reg.T0.value),
            Instruction(Opcode.PUSH, rs=Reg.T2.value),
            Instruction(Opcode.RET),
        ])

    def _generate_greater_than_sub(self):
        """Генерирует подпрограмму для > (больше). a b -- (a > b)"""
        less_than_addr = self.symbols.get("<")
        if not isinstance(less_than_addr, int):
            raise RuntimeError("< subroutine not found for >")

        self.emit([
            Instruction(Opcode.POP, rt=Reg.T1.value),
            Instruction(Opcode.POP, rt=Reg.T0.value),
            Instruction(Opcode.PUSH, rs=Reg.T1.value),
            Instruction(Opcode.PUSH, rs=Reg.T0.value),
            Instruction(Opcode.CALL, addr=less_than_addr),
            Instruction(Opcode.RET),
        ])

    def _generate_less_equal_sub(self):
        """Генерирует подпрограмму для <=. Логика: NOT (a > b)."""
        greater_than_addr = self.symbols.get(">")
        if not isinstance(greater_than_addr, int):
            raise RuntimeError("> subroutine not found for <=")

        zero_equal_addr = self.symbols.get("0=")
        if not isinstance(zero_equal_addr, int):
            raise RuntimeError("0= subroutine not found for <=")

        self.emit([
            Instruction(Opcode.CALL, addr=greater_than_addr),
            Instruction(Opcode.CALL, addr=zero_equal_addr),
            Instruction(Opcode.RET),
        ])

        self.emit([
            Instruction(Opcode.CALL, addr=greater_than_addr),
            Instruction(Opcode.POP, rt=Reg.T0.value),
            self.emit_load_immediate(1, Reg.T1.value),
            Instruction(Opcode.XOR, rd=Reg.T0.value, rs=Reg.T0.value, rt=Reg.T1.value),
            Instruction(Opcode.PUSH, rs=Reg.T0.value),
            Instruction(Opcode.RET),
        ])

    def _generate_greater_equal_sub(self):
        """Генерирует подпрограмму для >=. Логика: NOT (a < b)."""
        less_than_addr = self.symbols.get("<")
        if not isinstance(less_than_addr, int):
            raise RuntimeError("< subroutine not found for >=")

        zero_equal_addr = self.symbols.get("0=")
        if not isinstance(zero_equal_addr, int):
            raise RuntimeError("0= subroutine not found for >=")

        self.emit([
            Instruction(Opcode.CALL, addr=less_than_addr),
            Instruction(Opcode.CALL, addr=zero_equal_addr),
            Instruction(Opcode.RET),
        ])

    def _generate_zero_not_equal_sub(self):
        """Генерирует подпрограмму для 0<> (не равно нулю)."""
        self.emit([
            Instruction(Opcode.POP, rt=Reg.T0.value),
            self.emit_load_immediate(0, Reg.T1.value),
            Instruction(Opcode.CMP, rd=Reg.T2.value, rs=Reg.T0.value, rt=Reg.T1.value),
            self.emit_load_immediate(1, Reg.T1.value),
            Instruction(Opcode.XOR, rd=Reg.T0.value, rs=Reg.T2.value, rt=Reg.T1.value),
            Instruction(Opcode.PUSH, rs=Reg.T0.value),
            Instruction(Opcode.RET),
        ])

    def _generate_emit_sub(self):
        load_addr_instr = self.emit_load_immediate(IO_OUTPUT_PORT, Reg.T1.value)
        self.emit([
            Instruction(Opcode.POP, rt=Reg.T0.value),
            load_addr_instr,
            Instruction(Opcode.STORE, rt=Reg.T0.value, rs=Reg.T1.value, imm=0),
            Instruction(Opcode.RET),
        ])

    def _generate_zero_equal_sub(self):
        """Генерирует подпрограмму для 0= (проверка на ноль)."""
        self.emit([
            Instruction(Opcode.POP, rt=Reg.T0.value),
            self.emit_load_immediate(0, Reg.T1.value),
            Instruction(Opcode.CMP, rd=Reg.T0.value, rs=Reg.T0.value, rt=Reg.T1.value),
            Instruction(Opcode.PUSH, rs=Reg.T0.value),
            Instruction(Opcode.RET),
        ])

    def _generate_key_sub(self):
        load_addr_instr = self.emit_load_immediate(IO_INPUT_PORT, Reg.T1.value)
        self.emit([
            load_addr_instr,
            Instruction(Opcode.LOAD, rt=Reg.T0.value, rs=Reg.T1.value, imm=0),
            Instruction(Opcode.PUSH, rs=Reg.T0.value),
            Instruction(Opcode.RET),
        ])

    def _generate_dup_sub(self):
        self.emit([
            Instruction(Opcode.POP, rt=Reg.T0.value),
            Instruction(Opcode.PUSH, rs=Reg.T0.value),
            Instruction(Opcode.PUSH, rs=Reg.T0.value),
            Instruction(Opcode.RET),
        ])

    def _generate_swap_sub(self):
        self.emit([
            Instruction(Opcode.POP, rt=Reg.T0.value),
            Instruction(Opcode.POP, rt=Reg.T1.value),
            Instruction(Opcode.PUSH, rs=Reg.T0.value),
            Instruction(Opcode.PUSH, rs=Reg.T1.value),
            Instruction(Opcode.RET),
        ])

    def _generate_over_sub(self):
        self.emit([
            Instruction(Opcode.POP, rt=Reg.T0.value),
            Instruction(Opcode.POP, rt=Reg.T1.value),
            Instruction(Opcode.PUSH, rs=Reg.T1.value),
            Instruction(Opcode.PUSH, rs=Reg.T0.value),
            Instruction(Opcode.PUSH, rs=Reg.T1.value),
            Instruction(Opcode.RET),
        ])

    def _generate_fetch_sub(self):  # @
        self.emit([
            Instruction(Opcode.POP, rt=Reg.T0.value),
            Instruction(Opcode.LOAD, rt=Reg.T0.value, rs=Reg.T0.value, imm=0),
            Instruction(Opcode.PUSH, rs=Reg.T0.value),
            Instruction(Opcode.RET),
        ])

    def _generate_store_sub(self):  # !
        self.emit([
            Instruction(Opcode.POP, rt=Reg.T0.value),
            Instruction(Opcode.POP, rt=Reg.T1.value),
            Instruction(Opcode.STORE, rt=Reg.T1.value, rs=Reg.T0.value, imm=0),
            Instruction(Opcode.RET),
        ])

    def _generate_cr_sub(self):
        emit_addr = self.symbols.get("EMIT")
        if not isinstance(emit_addr, int):
            raise RuntimeError("EMIT subroutine not found when generating CR")

        self.emit([
            self.emit_push_instr(10),
            Instruction(Opcode.CALL, addr=emit_addr),
            Instruction(Opcode.RET),
        ])

    def start_word_definition(self, word: str):
        self.symbols[word.upper()] = len(self.code)

    def end_word_definition(self):
        self.emit([Instruction(Opcode.RET)])

    def _parse_string(self, token_stream: Iterator[str]) -> str:
        buffer = []
        for token in token_stream:
            if token.endswith('"'):
                if len(token) > 1:
                    buffer.append(token[:-1])
                return " ".join(buffer)
            buffer.append(token)
        raise SyntaxError("Unterminated string literal")

    def _compile_string_literal(self, content: str):
        """
        Компилирует строковый литерал S" ... "
        """
        string_addr_load_instr_index = len(self.code)
        self.string_relocations[string_addr_load_instr_index] = content
        self.emit([
            Instruction(Opcode.LUI, rt=Reg.T0.value, imm=0),  # Placeholder для старших бит адреса
            Instruction(Opcode.ORI, rs=Reg.T0.value, rt=Reg.T0.value, imm=0),  # Placeholder для младших бит
            Instruction(Opcode.PUSH, rs=Reg.T0.value)  # Помещаем адрес строки (из T0) на стек
        ])

    def _compile_char_literal(self, token_stream: Iterator[str]):
        char_token = next(token_stream)
        if not char_token:
            raise SyntaxError("[CHAR] must be followed by a character token.")
        char_code = ord(char_token[0])
        self.emit(self.emit_push_instr(char_code))

    def _patch_string_addresses(self, base_byte_address_for_strings: int) -> list[int]:
        data_section_words_for_strings: list[int] = []
        string_content_to_offset_map: dict[str, int] = {}
        unique_strings = sorted(list(set(self.string_relocations.values())))
        current_string_block_offset_words = 0
        for s_content in unique_strings:
            string_content_to_offset_map[s_content] = current_string_block_offset_words
            encoded_bytes = s_content.encode('utf-8')
            data_section_words_for_strings.append(len(encoded_bytes))
            current_string_block_offset_words += 1
            for char_byte in encoded_bytes:
                data_section_words_for_strings.append(int(char_byte))
                current_string_block_offset_words += 1
        string_data_start_word_addr = base_byte_address_for_strings // 4

        for instr_idx, string_content in self.string_relocations.items():
            string_relative_offset_words = string_content_to_offset_map[string_content]

            final_string_address_word = string_data_start_word_addr + string_relative_offset_words
            upper_bits = (final_string_address_word >> 16) & 0xFFFF
            lower_bits = final_string_address_word & 0xFFFF

            self.code[instr_idx].imm = upper_bits
            self.code[instr_idx + 1].imm = lower_bits
            logging.info(
                f"Patched string '{string_content}': LUI@idx={instr_idx} imm=0x{upper_bits:04X}, "
                f"ORI@idx={instr_idx + 1} imm=0x{lower_bits:04X} -> final_word_addr=0x{final_string_address_word:08X}"
            )
        return data_section_words_for_strings

    def _tokenize(self, source: str) -> list[str]:
        """Улучшенный токенизатор с поддержкой комментариев и сохранением регистра строк."""
        lines = source.split('\n')
        tokens = []

        for line in lines:
            comment_pos = line.find('\\')
            if comment_pos != -1:
                line = line[:comment_pos]

            line = line.replace('s"', ' s" ').replace('S"', ' s" ')

            line_tokens = line.strip().split()
            processed_tokens = []
            in_string = False

            for token in line_tokens:
                if token.lower() == 's"':
                    processed_tokens.append('S"')
                    in_string = True
                elif in_string and token.endswith('"'):
                    processed_tokens.append(token[:-1])
                    processed_tokens.append('"')
                    in_string = False
                elif in_string:
                    processed_tokens.append(token)
                else:
                    processed_tokens.append(token.upper())

            tokens.extend(processed_tokens)

        return [token for token in tokens if token]

    def _process_token_in_colon_definition(self, token: str, token_stream: Iterator[str]):
        if token == 'S"':
            content = self._parse_string(token_stream)
            self._compile_string_literal(content)  # Строки внутри слов тоже компилируются
            return

        try:
            num = int(token)
            self.emit(self.emit_push_instr(num))
            return
        except ValueError:
            pass  # Не число, продолжаем

        upper_token = token.upper()
        if upper_token in self.symbols:
            action_or_info = self.symbols[upper_token]

            if isinstance(action_or_info, int):
                self.emit([Instruction(Opcode.CALL, addr=action_or_info)])
            elif callable(action_or_info):
                action_or_info()
            elif isinstance(action_or_info, tuple) and action_or_info[0] == 'data_variable':
                var_name_from_symbols = action_or_info[1]
                variable_addr_load_instr_index = len(self.code)
                self.variable_relocations[variable_addr_load_instr_index] = var_name_from_symbols
                self.emit([
                    Instruction(Opcode.LUI, rt=Reg.T0.value, imm=0),
                    Instruction(Opcode.ORI, rs=Reg.T0.value, rt=Reg.T0.value, imm=0),
                    Instruction(Opcode.PUSH, rs=Reg.T0.value)
                ])
            elif isinstance(action_or_info, str) and action_or_info == "subroutine":
                raise NotImplementedError(
                    f"Built-in subroutine '{upper_token}' was not correctly pre-compiled with an address.")
            else:
                raise ValueError(f"Unknown type of symbol '{upper_token}' in colon definition: {action_or_info}")
        elif upper_token == "[CHAR]":
            self._compile_char_literal(token_stream)
            return
        else:
            raise ValueError(f"Unknown word in colon definition: {token} (uppercase: {upper_token})")

    def _process_executable_token(self, token: str, token_stream: Iterator[str]):
        if token == 'S"':
            content = self._parse_string(token_stream)
            self._compile_string_literal(content)
            return

        try:
            num = int(token)
            self.emit(self.emit_push_instr(num))
            return
        except ValueError:
            pass

        upper_token = token.upper()
        if upper_token in self.symbols:
            action_or_info = self.symbols[upper_token]
            if isinstance(action_or_info, int):
                self.emit([Instruction(Opcode.CALL, addr=action_or_info)])
            elif callable(action_or_info):
                action_or_info()
            elif isinstance(action_or_info, tuple) and action_or_info[0] == 'data_variable':
                var_name_from_symbols = action_or_info[1]
                variable_addr_load_instr_index = len(self.code)
                self.variable_relocations[variable_addr_load_instr_index] = var_name_from_symbols
                self.emit([
                    Instruction(Opcode.LUI, rt=Reg.T0.value, imm=0),
                    Instruction(Opcode.ORI, rs=Reg.T0.value, rt=Reg.T0.value, imm=0),
                    Instruction(Opcode.PUSH, rs=Reg.T0.value)
                ])
            elif isinstance(action_or_info, str) and action_or_info == "subroutine":
                subroutine_addr = self.symbols.get(upper_token + "_ADDR")
                if isinstance(subroutine_addr, int):
                    self.emit([Instruction(Opcode.CALL, addr=subroutine_addr)])
                else:
                    raise NotImplementedError(
                        f"Built-in subroutine '{upper_token}' was not correctly resolved to an address.")
            else:
                raise ValueError(f"Unknown type of symbol '{upper_token}': {action_or_info}")
        elif upper_token == "[CHAR]":
            self._compile_char_literal(token_stream)
            return
        else:
            raise ValueError(f"Unknown word: {token} (uppercase: {upper_token})")

    def _process_token(self, token: str, token_stream: Iterator[str]):
        """Обрабатывает один токен, будь то число, слово, переменная и т.д."""
        upper_token = token.upper()
        if upper_token == '>R':
            self.emit([
                Instruction(Opcode.POP, rt=Reg.T0.value),
                Instruction(Opcode.ADDI, rt=Reg.SP.value, rs=Reg.SP.value, imm=-1),
                Instruction(Opcode.STORE, rt=Reg.T0.value, rs=Reg.SP.value, imm=0),
            ])
            return
        if upper_token == 'R>':
            self.emit([
                Instruction(Opcode.LOAD, rt=Reg.T0.value, rs=Reg.SP.value, imm=0),
                Instruction(Opcode.ADDI, rt=Reg.SP.value, rs=Reg.SP.value, imm=1),
                Instruction(Opcode.PUSH, rs=Reg.T0.value),
            ])
            return
        if upper_token == 'R@':
            self.emit([
                Instruction(Opcode.LOAD, rt=Reg.T0.value, rs=Reg.SP.value, imm=0),
                Instruction(Opcode.PUSH, rs=Reg.T0.value),
            ])
            return
        if token == 'S"':
            content = self._parse_string(token_stream)
            self._compile_string_literal(content)
            return

        if token.upper() == "[CHAR]":
            self._compile_char_literal(token_stream)
            return

        try:
            num = int(token)
            self.emit(self.emit_push_instr(num))
            return
        except (ValueError, TypeError):
            pass

        if upper_token in self.symbols:
            action_or_info = self.symbols[upper_token]

            if isinstance(action_or_info, int):
                self.emit([Instruction(Opcode.CALL, addr=action_or_info)])
            elif callable(action_or_info):
                action_or_info()
            elif isinstance(action_or_info, tuple) and action_or_info[0] == 'data_variable':
                var_name_from_symbols = action_or_info[1]
                variable_addr_load_instr_index = len(self.code)
                self.variable_relocations[variable_addr_load_instr_index] = var_name_from_symbols
                self.emit([
                    Instruction(Opcode.LUI, rt=Reg.T0.value, imm=0),
                    Instruction(Opcode.ORI, rs=Reg.T0.value, rt=Reg.T0.value, imm=0),
                    Instruction(Opcode.PUSH, rs=Reg.T0.value)
                ])
            else:
                raise ValueError(f"Unknown symbol type for '{upper_token}': {action_or_info}")
        else:
            raise ValueError(f"Unknown word: {token} (uppercase: {upper_token})")

    def translate(self, source: str) -> tuple[list[Instruction], list[int]]:
        self.emit([Instruction(Opcode.JMP, addr=0)])
        jmp_to_main_instruction_index = 0
        self._inject_built_in_subroutines()
        token_stream = iter(self._tokenize(source))

        main_code_started = False

        while True:
            try:
                token = next(token_stream)
            except StopIteration:
                break

            upper_token = token.upper()
            if upper_token == ':':
                if main_code_started: raise SyntaxError("Definition ':' not allowed after main code.")
                word_name = next(token_stream)
                self.start_word_definition(word_name)
                while True:
                    body_token = next(token_stream)
                    if body_token.upper() == ';':
                        self.end_word_definition()
                        break
                    self._process_token(body_token, token_stream)
                continue

            elif upper_token == "CREATE":
                if main_code_started: raise SyntaxError("Definition 'CREATE' not allowed after main code.")
                var_name = next(token_stream).upper()
                allot_token = next(token_stream).upper()
                if allot_token != "ALLOT":
                    raise SyntaxError(f"Expected ALLOT after CREATE {var_name}, but got {allot_token}")
                size_token = next(token_stream)
                try:
                    size_in_words = int(size_token)
                    self.data_variables[var_name] = {
                        'offset_words': self.current_data_section_offset_words,
                        'size_words': size_in_words
                    }
                    self.symbols[var_name] = ('data_variable', var_name)
                    self.current_data_section_offset_words += size_in_words
                except ValueError:
                    raise SyntaxError(f"Invalid size '{size_token}' for ALLOT")
                continue

            elif upper_token == "VARIABLE":
                if main_code_started: raise SyntaxError("Definition 'VARIABLE' not allowed after main code.")
                var_name = next(token_stream).upper()
                self.data_variables[var_name] = {
                    'offset_words': self.current_data_section_offset_words,
                    'size_words': 1
                }
                self.symbols[var_name] = ('data_variable', var_name)
                self.current_data_section_offset_words += 1
                continue

            if not main_code_started:
                main_code_started = True
                main_code_start_addr = len(self.code)
                self.code[jmp_to_main_instruction_index].addr = main_code_start_addr

            self._process_token(token, token_stream)

        if not main_code_started:
            main_code_start_addr = len(self.code)
            self.code[jmp_to_main_instruction_index].addr = main_code_start_addr

        self.emit([Instruction(Opcode.HALT)])

        code_size_in_bytes = len(self.code) * 4
        self._patch_variable_addresses(code_size_in_bytes)
        base_byte_address_for_strings = code_size_in_bytes + (self.current_data_section_offset_words * 4)
        string_data_section_words = self._patch_string_addresses(base_byte_address_for_strings)

        variable_data_words = [0] * self.current_data_section_offset_words
        data_section_words = variable_data_words + string_data_section_words

        return self.code, data_section_words

    def _patch_variable_addresses(self, code_section_size_bytes: int):
        data_variables_start_word_addr = code_section_size_bytes // 4

        for instr_idx, var_name in self.variable_relocations.items():
            if var_name not in self.data_variables:
                raise RuntimeError(
                    f"Internal error: Variable '{var_name}' in relocations but not in data_variables.")

            var_info = self.data_variables[var_name]
            variable_absolute_word_addr = data_variables_start_word_addr + var_info['offset_words']

            upper_bits = (variable_absolute_word_addr >> 16) & 0xFFFF
            lower_bits = variable_absolute_word_addr & 0xFFFF
            self.code[instr_idx].imm = upper_bits
            self.code[instr_idx + 1].imm = lower_bits

            logging.info(
                f"Patched variable '{var_name}': LUI@idx={instr_idx} imm=0x{upper_bits:04X}, "
                f"ORI@idx={instr_idx + 1} imm=0x{lower_bits:04X} -> final_word_addr=0x{variable_absolute_word_addr:08X}"
            )


def main(source_file: str, target_file: str):
    """Главная функция."""
    with open(source_file, 'r', encoding='utf-8') as f:
        source_code = f.read()

    translator = Translator()
    instructions, data_words = translator.translate(source_code)

    with open(target_file, 'wb') as f:
        for instr in instructions:
            f.write(instr.to_binary())
        for word_val in data_words:
            f.write(struct.pack('>I', word_val))  # упаковка каждого слова

    data_section_size_bytes = len(data_words) * 4  # Размер секции данных в байтах
    with open(target_file + ".txt", 'w', encoding='utf-8') as f:
        f.write(f"; Source: {source_file}\n")
        f.write(f"; Code section (size: {len(instructions) * 4} bytes)\n")
        for i, instr in enumerate(instructions):
            f.write(instr.to_hex(i) + '\n')
        f.write(f"\n; Data section (size: {data_section_size_bytes} bytes)\n")
        f.write("; Data words (decimal):\n")
        f.write(str(data_words) + "\n")
        f.write("; Data words (hex):\n")
        f.write("[" + ", ".join([f"0x{dw:08X}" for dw in data_words]) + "]\n")

    print(f"Successfully translated {source_file} to {target_file}")
    print(
        f"Total instructions: {len(instructions)}, Data size: {data_section_size_bytes} bytes")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) != 3:
        print("Usage: python translator.py <source_file.f> <target_file.bin>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
