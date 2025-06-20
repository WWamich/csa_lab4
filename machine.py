import logging
import sys
from typing import List, Tuple

from isa import Opcode, Reg, IO_INPUT_PORT, IO_OUTPUT_PORT, MEMORY_SIZE
from microcode import (
    MicroOp, MicroInstruction, SimpleCache,
    get_microcode
)


def decode_instruction(instruction: int) -> Tuple[Opcode, int, int, int, int, int]:
    """
    Декодирование машинной инструкции с корректной обработкой immediate.
    """
    opcode_val = (instruction >> 26) & 0x3F
    rs = (instruction >> 21) & 0x1F
    rt = (instruction >> 16) & 0x1F
    rd = (instruction >> 11) & 0x1F
    addr = instruction & 0x3FFFFFF
    imm16 = instruction & 0xFFFF

    try:
        opcode = Opcode(opcode_val)
    except ValueError:
        opcode = Opcode.NOP

    if opcode in [Opcode.JMP, Opcode.CALL]:
        return opcode, 0, 0, 0, 0, addr

    if opcode == Opcode.LOADI:
        rt_l = (instruction >> 21) & 0x1F
        imm21 = instruction & 0x1FFFFF
        if imm21 & 0x100000: imm21 -= 0x200000
        return opcode, 0, rt_l, 0, imm21, 0

    if opcode in [Opcode.IN, Opcode.OUT, Opcode.LOAD, Opcode.STORE]:
        imm_final = imm16
    else:
        imm_final = imm16
        if imm_final & 0x8000: imm_final -= 0x10000

    return opcode, rs, rt, rd, imm_final, 0


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

    def read_register(self, reg_id: int) -> int:
        if reg_id == Reg.ZERO.value: return 0
        return self.registers[reg_id] & 0xFFFFFFFF

    def write_register(self, reg_id: int, value: int):
        if reg_id != Reg.ZERO.value:
            self.registers[reg_id] = value & 0xFFFFFFFF
            self.zero_flag = (value == 0)

    def alu_operation(self, op: MicroOp, src1: int, src2: int) -> int:
        result = 0
        if op == MicroOp.ALU_ADD:
            result = (src1 + src2) & 0xFFFFFFFF
        elif op == MicroOp.ALU_SUB:
            result = (src1 - src2) & 0xFFFFFFFF
        elif op == MicroOp.ALU_MUL:
            result = (src1 * src2) & 0xFFFFFFFF
        elif op == MicroOp.ALU_DIV:
            result = (src1 // src2) if src2 != 0 else 0
        elif op == MicroOp.ALU_MOD:
            result = (src1 % src2) if src2 != 0 else 0
        elif op == MicroOp.ALU_AND:
            result = src1 & src2
        elif op == MicroOp.ALU_OR:
            result = src1 | src2
        elif op == MicroOp.ALU_XOR:
            result = src1 ^ src2
        elif op == MicroOp.ALU_CMP:
            result = 1 if src1 == src2 else 0
        elif op == MicroOp.ALU_SHL:
            result = (src1 << (src2 & 0x1F)) & 0xFFFFFFFF
        elif op == MicroOp.ALU_SHR:
            shift = src2 & 0x1F
            if src1 & 0x80000000:
                result = (src1 - (1 << 32)) >> shift
            else:
                result = src1 >> shift
            result &= 0xFFFFFFFF
        self.alu_result = result
        self.zero_flag = (result == 0)
        return result

    def stack_push(self, value: int):
        sp = self.read_register(Reg.SP.value)
        if sp >= len(self.stack_memory): raise OverflowError("Stack overflow")
        self.stack_memory[sp] = value & 0xFFFFFFFF
        self.write_register(Reg.SP.value, sp + 1)

    def stack_pop(self) -> int:
        sp = self.read_register(Reg.SP.value) - 1
        if sp < 0: raise OverflowError("Stack underflow")
        self.write_register(Reg.SP.value, sp)
        return self.stack_memory[sp]

    def io_write(self, port: int, value: int):
        if port == IO_OUTPUT_PORT:
            self.output_buffer.append(chr(value & 0xFF))

    def get_stats(self) -> dict:
        cache_stats = self.cache.get_stats()
        return {"memory_accesses": self.memory_accesses, "output": "".join(self.output_buffer), **cache_stats}


class ControlUnit:
    def __init__(self, program: List[int], data_path: DataPath):
        self.program = program
        self.data_path = data_path
        self.pc = 0
        self.tick_count = 0
        self.current_microcode = []
        self.micro_step = 0
        self.opcode, self.rs, self.rt, self.rd, self.imm, self.addr = (None, 0, 0, 0, 0, 0)
        self.halted = False

    def fetch_instruction(self):
        if self.pc >= len(self.program):
            self.halted = True
            return
        instruction = self.program[self.pc]
        self.opcode, self.rs, self.rt, self.rd, self.imm, self.addr = decode_instruction(instruction)
        self.current_microcode = get_microcode(self.opcode)
        self.micro_step = 0

    def execute_micro_instruction(self, micro_instr: MicroInstruction):
        op, src, dst, imm = micro_instr.op, micro_instr.src, micro_instr.dst, micro_instr.imm

        if op == MicroOp.HALT: self.halted = True; return
        if op == MicroOp.PC_INC: self.pc += 1; return
        if op == MicroOp.PC_LOAD: self.pc = self.addr; return
        if op == MicroOp.PC_LOAD_TEMP: self.pc = self.data_path.temp_addr; return
        if op == MicroOp.LOAD_IMM: self.data_path.write_register(self.rt, self.imm); return
        if op == MicroOp.IO_WRITE: self.data_path.io_write(self.imm, self.data_path.read_register(self.rs)); return

        if op == MicroOp.STACK_PUSH:
            value = self.data_path.temp_addr if src == "TEMP_ADDR" else self.data_path.read_register(
                self.rs)
            self.data_path.stack_push(value)
            return

        if op == MicroOp.STACK_POP:
            value = self.data_path.stack_pop()
            if dst == "TEMP_ADDR":
                self.data_path.temp_addr = value
            else:
                self.data_path.write_register(self.rt, value)
            return

        if op == MicroOp.JUMP_COND:
            if self.data_path.read_register(self.rs) == 0:
                self.pc = self.imm
            else:
                self.pc += 1
            return

        if op.value.startswith("alu"):
            src1 = self.pc if src == "PC" else self.data_path.read_register(self.rs)
            src2 = self.data_path.read_register(self.rt) if imm == "rt" else self.imm
            result = self.data_path.alu_operation(op, src1, src2)
            if dst == "rd":
                self.data_path.write_register(self.rd, result)
            elif dst == "rt":
                self.data_path.write_register(self.rt, result)
            elif dst == "TEMP_ADDR":
                self.data_path.temp_addr = result

    def step(self):
        if self.halted: return
        if not self.current_microcode or self.micro_step >= len(self.current_microcode):
            self.fetch_instruction()
            if self.halted: return

        micro_instr = self.current_microcode[self.micro_step]
        self.execute_micro_instruction(micro_instr)
        self.tick_count += 1
        self.micro_step += 1


def simulate(program: List[int], input_data: str, max_ticks: int) -> dict:
    input_buffer = list(input_data)
    data_path = DataPath(input_buffer)
    control_unit = ControlUnit(program, data_path)
    logging.info(f"Программа загружена: {len(program)} инструкций, Входные данные: '{input_data}'")
    while control_unit.tick_count < max_ticks and not control_unit.halted:
        control_unit.step()
    if not control_unit.halted:
        logging.warning(f"Симуляция остановлена по достижению лимита в {max_ticks} тактов.")
    stats = data_path.get_stats()
    return {'ticks': control_unit.tick_count, 'halted': control_unit.halted, 'pc': control_unit.pc,
            'registers': data_path.registers, 'stats': stats}


def main():
    if len(sys.argv) != 3:
        print("Использование: python machine.py <program.bin> <input.txt>");
        sys.exit(1)

    logging.basicConfig(level=logging.INFO, format='%(levelname)-8s: %(message)s')

    try:
        with open(sys.argv[1], 'rb') as f:
            program_bytes = f.read()
            program = [int.from_bytes(program_bytes[i:i + 4], 'big') for i in range(0, len(program_bytes), 4)]
        with open(sys.argv[2], 'r') as f:
            input_data = f.read().strip()
    except FileNotFoundError as e:
        print(f"❌ Файл не найден: {e.filename}");
        sys.exit(1)

    result = simulate(program, input_data, max_ticks=10000)
    print("\n" + "=" * 50 + "\nРЕЗУЛЬТАТЫ СИМУЛЯЦИИ\n" + "=" * 50)
    print(f"Тактов выполнено: {result['ticks']}")
    print(f"Программа завершена: {'Да' if result['halted'] else 'Нет'}")
    print(f"Финальный PC: 0x{result['pc']:04X}")
    stats = result['stats']
    print(f"\nКЭШ СТАТИСТИКА:")
    print(f"  Обращений к памяти: {stats['memory_accesses']}")
    print(f"  Попаданий в кэш: {stats['hits']}")
    print(f"  Промахов кэша: {stats['misses']}")
    print(f"  Коэффициент попаданий: {stats['hit_rate'] * 100:.2f}%")
    print(f"\nВЫВОД ПРОГРАММЫ:\n'{stats['output']}'")
    print("\nРЕГИСТРЫ:")
    for i, val in enumerate(result['registers']):
        print(f"  R{i}: 0x{val:08X} ({val})")


if __name__ == "__main__":
    main()