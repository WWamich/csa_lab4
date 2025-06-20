#!/usr/bin/env python3
"""
Модель RISC Forth процессора с микрокодным управлением и кэшем
Архитектура: risc | neum | mc | tick | cache
"""

import logging
import sys
from typing import List, Tuple

from isa import Opcode, Reg, IO_INPUT_PORT, IO_OUTPUT_PORT, MEMORY_SIZE
from microcode import (
    MicroOp, MicroInstruction, SimpleCache,
    get_microcode
)


def decode_instruction(instruction: int) -> Tuple[Opcode, int, int, int, int, int]:
    """Декодирование с поддержкой всех форматов инструкций"""
    opcode_val = (instruction >> 26) & 0x3F
    rs = (instruction >> 21) & 0x1F
    rt = (instruction >> 16) & 0x1F
    rd = (instruction >> 11) & 0x1F
    imm16 = instruction & 0xFFFF
    addr = instruction & 0x3FFFFFF

    try:
        opcode = Opcode(opcode_val)
    except ValueError:
        logging.warning(f"Неизвестный опкод: 0x{opcode_val:02X} в инструкции 0x{instruction:08X}")
        opcode = Opcode.NOP

    # Для J-типа (JMP, CALL) важен только addr
    if opcode in [Opcode.JMP, Opcode.CALL]:
        logging.debug(f"   DECODE J-Type: OP={opcode.name}, ADDR=0x{addr:07X}")
        return opcode, 0, 0, 0, 0, addr

    # Для I-типа важны rs, rt, imm16
    if opcode in [Opcode.LOAD, Opcode.STORE, Opcode.JZ, Opcode.IN, Opcode.OUT, Opcode.ORI, Opcode.LUI, Opcode.LOADI]:
        # У LOADI и LUI rt в другом месте, но для упрощения оставим так
        imm = imm16
        if imm & 0x8000: # знаковое расширение для 16-битных чисел
            imm |= 0xFFFF0000
        logging.debug(f"   DECODE I-Type: OP={opcode.name}, RS={rs}, RT={rt}, IMM={imm}")
        return opcode, rs, rt, 0, imm, 0

    # Для R-типа важны rs, rt, rd
    logging.debug(f"   DECODE R-Type: OP={opcode.name}, RS={rs}, RT={rt}, RD={rd}")
    return opcode, rs, rt, rd, 0, 0


class DataPath:
    """Тракт данных процессора"""

    def __init__(self, input_buffer: List[str]):
        self.registers = [0] * 8
        self.memory = [0] * MEMORY_SIZE
        self.stack_memory = [0] * 1024

        self.cache = SimpleCache(size=16)

        self.input_buffer = input_buffer.copy()
        self.output_buffer = []

        self.temp_addr = 0
        self.alu_result = 0
        self.zero_flag = False
        self.memory_accesses = 0
        logging.info("DataPath инициализирован с чистой памятью")

    def read_register(self, reg_id: int) -> int:
        if reg_id == Reg.ZERO.value:
            return 0
        if 0 <= reg_id < 8:
            return self.registers[reg_id] & 0xFFFFFFFF
        return 0

    def write_register(self, reg_id: int, value: int):
        if reg_id != Reg.ZERO.value and 0 <= reg_id < 8:
            self.registers[reg_id] = value & 0xFFFFFFFF
            self.zero_flag = (value == 0)
            logging.debug(f"WRITE REG: R{reg_id} = 0x{value:08X} ({value})")

    def memory_read(self, addr: int) -> Tuple[bool, int, int]:
        self.memory_accesses += 1
        hit, result = self.cache.access(addr)
        if hit:
            logging.debug(f"CACHE HIT: mem[0x{addr:04X}] = {result}")
            return True, result, 1
        else:
            data = self.memory[addr & 0xFFFF]
            self.cache.write(addr, data)
            logging.debug(f"CACHE MISS: mem[0x{addr:04X}] = {data} (10 cycles)")
            return False, data, 10

    def memory_write(self, addr: int, data: int) -> Tuple[bool, int]:
        self.memory_accesses += 1
        addr = addr & 0xFFFF
        self.memory[addr] = data & 0xFFFFFFFF
        self.cache.write(addr, data)
        logging.debug(f"MEM WRITE: mem[0x{addr:04X}] = {data}")
        return True, 1

    def stack_push(self, value: int):
        sp = self.read_register(Reg.SP.value)
        if sp >= len(self.stack_memory):
            raise OverflowError(f"Stack overflow: SP={sp}, stack_size={len(self.stack_memory)}")
        self.stack_memory[sp] = value & 0xFFFFFFFF
        self.write_register(Reg.SP.value, sp + 1)
        logging.debug(f"STACK PUSH: stack[{sp}] = {value}, SP = {sp + 1}")

    def stack_pop(self) -> int:
        sp = self.read_register(Reg.SP.value) - 1
        if sp < 0:
            raise OverflowError(f"Stack underflow: SP={sp}")
        self.write_register(Reg.SP.value, sp)
        value = self.stack_memory[sp]
        logging.debug(f"STACK POP: stack[{sp}] = {value}, SP = {sp}")
        return value

    def alu_operation(self, op: MicroOp, src1: int, src2: int) -> int:
        result = 0
        if op == MicroOp.ALU_ADD: result = (src1 + src2) & 0xFFFFFFFF
        elif op == MicroOp.ALU_SUB: result = (src1 - src2) & 0xFFFFFFFF
        elif op == MicroOp.ALU_MUL: result = (src1 * src2) & 0xFFFFFFFF
        elif op == MicroOp.ALU_DIV: result = (src1 // src2) if src2 != 0 else 0
        elif op == MicroOp.ALU_MOD: result = (src1 % src2) if src2 != 0 else 0
        elif op == MicroOp.ALU_AND: result = src1 & src2
        elif op == MicroOp.ALU_OR: result = src1 | src2
        elif op == MicroOp.ALU_XOR: result = src1 ^ src2
        elif op == MicroOp.ALU_CMP: result = 1 if src1 == src2 else 0
        elif op == MicroOp.ALU_SHL:
            shift_amount = src2 & 0x1F
            result = (src1 << shift_amount) & 0xFFFFFFFF
            logging.debug(f"SHL: {src1} << {shift_amount} = {result}")
        elif op == MicroOp.ALU_SHR:
            shift_amount = src2 & 0x1F
            if src1 & 0x80000000:
                signed_src1 = src1 - 0x100000000
            else:
                signed_src1 = src1
            signed_result = signed_src1 >> shift_amount
            result = signed_result & 0xFFFFFFFF
            logging.debug(f"ARITHMETIC SHR: {src1} >> {shift_amount} = {result}")

        self.alu_result = result
        self.zero_flag = (result == 0)
        logging.debug(f"ALU: {op.name} {src1} {src2} = {result}")
        return result

    def io_read(self, port: int) -> int:
        if port == IO_INPUT_PORT and self.input_buffer:
            char = self.input_buffer.pop(0)
            return ord(char) if char else 0
        return 0

    def io_write(self, port: int, value: int):
        if port == IO_OUTPUT_PORT:
            char = chr(value & 0xFF)
            self.output_buffer.append(char)
            logging.debug(f"I/O OUT: port 0x{port:04X} = {value} ('{char}')")

    def get_stats(self) -> dict:
        cache_stats = self.cache.get_stats()
        return {
            "memory_accesses": self.memory_accesses,
            "cache_hits": cache_stats["hits"],
            "cache_misses": cache_stats["misses"],
            "cache_hit_rate": cache_stats["hit_rate"],
            "output": "".join(self.output_buffer)
        }


class ControlUnit:
    """Микрокодный блок управления"""

    def __init__(self, program: List[int], data_path: DataPath):
        self.program = program
        self.data_path = data_path
        self.pc = 0
        self.tick_count = 0
        self.current_instruction = None
        self.current_microcode = []
        self.micro_step = 0
        self.opcode, self.rs, self.rt, self.rd, self.imm, self.addr = (None, 0, 0, 0, 0, 0)
        self.halted = False
        self.waiting_for_cache = 0

    def fetch_instruction(self):
        if self.pc >= len(self.program):
            self.halted = True
            return
        instruction = self.program[self.pc]
        self.current_instruction = instruction
        self.opcode, self.rs, self.rt, self.rd, self.imm, self.addr = decode_instruction(instruction)
        self.current_microcode = get_microcode(self.opcode)
        self.micro_step = 0
        logging.debug(f"FETCH: PC=0x{self.pc:04X} INSTR=0x{instruction:08X} OP={self.opcode.name}")

    def execute_micro_instruction(self, micro_instr: MicroInstruction) -> bool:
        logging.debug(f"MICRO: {micro_instr}")
        op = micro_instr.op

        if op == MicroOp.NOP: return True
        if op == MicroOp.HALT: self.halted = True; return False
        if op == MicroOp.PC_INC: self.pc += 1; return True
        if op == MicroOp.PC_LOAD: self.pc = self.addr; return True
        if op == MicroOp.PC_LOAD_TEMP: self.pc = self.data_path.temp_addr; return True

        if op == MicroOp.LOAD_IMM:
            self.data_path.write_register(self.rt, self.imm)
            return True

        if op in [MicroOp.ALU_ADD, MicroOp.ALU_SUB, MicroOp.ALU_MUL, MicroOp.ALU_DIV, MicroOp.ALU_MOD,
                  MicroOp.ALU_AND, MicroOp.ALU_OR, MicroOp.ALU_XOR, MicroOp.ALU_CMP, MicroOp.ALU_SHL, MicroOp.ALU_SHR]:
            src1_val = self.data_path.read_register(self.rs)
            src2_val = self.data_path.read_register(self.rt) if micro_instr.imm == "rt" else self.imm
            result = self.data_path.alu_operation(op, src1_val, src2_val)
            if micro_instr.dst == "rd": self.data_path.write_register(self.rd, result)
            elif micro_instr.dst == "rt": self.data_path.write_register(self.rt, result)
            elif micro_instr.dst == "TEMP_ADDR": self.data_path.temp_addr = result
            return True

        if op == MicroOp.ALU_PC_ADD: # Для CALL
            self.data_path.temp_addr = self.pc + 1
            return True

        if op == MicroOp.CACHE_WAIT:
            if self.waiting_for_cache > 0: self.waiting_for_cache -= 1; return False
            return True

        if op == MicroOp.MEM_READ:
            addr = self.data_path.temp_addr
            hit, data, cycles = self.data_path.memory_read(addr)
            self.data_path.write_register(self.rt, data)
            if not hit: self.waiting_for_cache = cycles - 1
            return True

        if op == MicroOp.MEM_WRITE:
            addr = self.data_path.temp_addr
            data = self.data_path.read_register(self.rs)
            hit, cycles = self.data_path.memory_write(addr, data)
            return True

        if op == MicroOp.STACK_PUSH:
            value = self.data_path.read_register(self.rs) if micro_instr.src == "rs" else self.data_path.temp_addr
            self.data_path.stack_push(value)
            return True

        if op == MicroOp.STACK_POP:
            value = self.data_path.stack_pop()
            if micro_instr.dst == "rt": self.data_path.write_register(self.rt, value)
            elif micro_instr.dst == "TEMP_ADDR": self.data_path.temp_addr = value
            return True

        if op == MicroOp.JUMP_COND:
            condition_met = (self.data_path.read_register(self.rs) == 0)
            if condition_met: self.pc = self.imm
            else: self.pc += 1
            return True

        if op == MicroOp.IO_READ:
            value = self.data_path.io_read(self.imm)
            self.data_path.write_register(self.rt, value)
            return True

        if op == MicroOp.IO_WRITE:
            value = self.data_path.read_register(self.rs)
            self.data_path.io_write(self.imm, value)
            return True

        logging.warning(f"Неизвестная микрооперация: {op}")
        return True

    def step(self) -> bool:
        if self.halted: return False
        if self.waiting_for_cache > 0:
            self.waiting_for_cache -= 1
            self.tick_count += 1
            return True
        if not self.current_microcode or self.micro_step >= len(self.current_microcode):
            self.fetch_instruction()
            if self.halted: return False
        micro_instr = self.current_microcode[self.micro_step]
        if self.execute_micro_instruction(micro_instr):
            self.micro_step += 1
        self.tick_count += 1
        return True

def simulate(program: List[int], input_data: str, max_ticks: int = 10000) -> dict:
    """Основная функция симуляции"""
    input_buffer = list(input_data)
    data_path = DataPath(input_buffer)
    control_unit = ControlUnit(program, data_path)
    logging.info(f"Программа загружена: {len(program)} инструкций, Входные данные: '{input_data}'")
    tick_count = 0
    while tick_count < max_ticks and control_unit.step():
        tick_count += 1
    stats = data_path.get_stats()
    return {
        'ticks': tick_count,
        'halted': control_unit.halted,
        'pc': control_unit.pc,
        'output': stats['output'],
        'registers': [data_path.read_register(i) for i in range(8)],
        'cache_stats': { 'accesses': stats['memory_accesses'], 'hits': stats['cache_hits'], 'misses': stats['cache_misses'], 'hit_rate': stats['cache_hit_rate'] }
    }

def load_program(filename: str) -> List[int]:
    program = []
    try:
        with open(filename, 'rb') as f:
            while (data := f.read(4)):
                program.append(int.from_bytes(data, byteorder='big'))
    except FileNotFoundError:
        print(f"❌ Файл не найден: {filename}"); sys.exit(1)
    return program

def main():
    if len(sys.argv) != 3:
        print("Использование: python machine.py <program.bin> <input.txt>"); sys.exit(1)

    logging.basicConfig(level=logging.INFO, format='%(levelname)-8s %(message)s')
    program = load_program(sys.argv[1])
    try:
        with open(sys.argv[2], 'r') as f: input_data = f.read().strip()
    except FileNotFoundError: input_data = ""

    result = simulate(program, input_data)
    print("\n" + "=" * 50 + "\nРЕЗУЛЬТАТЫ СИМУЛЯЦИИ\n" + "=" * 50)
    print(f"Тактов выполнено: {result['ticks']}")
    print(f"Программа завершена: {'Да' if result['halted'] else 'Нет'}")
    print(f"Финальный PC: 0x{result['pc']:04X}")
    cs = result['cache_stats']
    print(f"\nКЭШ СТАТИСТИКА:\n  Обращений: {cs['accesses']}, Попаданий: {cs['hits']}, Промахов: {cs['misses']}, Рейтинг: {cs['hit_rate'] * 100:.2f}%")
    print(f"\nВЫВОД ПРОГРАММЫ:\n'{result['output']}'")
    print(f"\nРЕГИСТРЫ:")
    for i, val in enumerate(result['registers']):
        print(f"  R{i}: 0x{val:08X} ({val})")

if __name__ == "__main__":
    main()