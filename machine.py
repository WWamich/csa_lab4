from __future__ import annotations
import logging
import struct
import sys
from collections import deque
from isa import (IO_INPUT_PORT, IO_OUTPUT_PORT, MEMORY_SIZE, Instruction,
                 Opcode, Reg)
from microcode import MicroOp, get_microcode_rom

CACHE_HIT_LATENCY = 1
CACHE_MISS_LATENCY = 10


def is_io_address(addr: int) -> bool:
    """Проверяет, является ли адрес адресом порта ввода-вывода."""
    return addr == IO_INPUT_PORT or addr == IO_OUTPUT_PORT


class CacheLine:
    """Моделирует одну строку кеша."""

    def __init__(self):
        self.valid = False
        self.tag = -1
        self.data = 0

    def __repr__(self):
        return f"CacheLine(valid={self.valid}, tag={self.tag}, data={hex(self.data)})"


class Cache:
    """
    Модель кеш-памяти. Реализован как direct-mapped кеш.
    Взаимодействует с основной памятью.
    """

    def __init__(self, size_in_lines, main_memory):
        assert size_in_lines > 0 and (size_in_lines & (size_in_lines - 1) == 0), "Cache size must be a power of 2"
        self.size = size_in_lines
        self.lines = [CacheLine() for _ in range(size_in_lines)]
        self.main_memory = main_memory

    def _get_line_and_tag(self, addr):
        # Используем log2(size) младших бит для индекса
        index_bits = (self.size - 1).bit_length() - 1
        line_index = addr & ((1 << index_bits) - 1)
        tag = addr >> index_bits
        return line_index, tag

    def read(self, addr):
        """Чтение данных из кеша. Возвращает (данные, задержка)."""
        if is_io_address(addr):
            logging.info(f"CACHE: Bypassing for I/O read at 0x{addr:04X}")
            return self.main_memory.read(addr), CACHE_HIT_LATENCY
        line_index, tag = self._get_line_and_tag(addr)
        line = self.lines[line_index]
        if line.valid and line.tag == tag:
            logging.info(f"CACHE: HIT on read at addr 0x{addr:04X}")
            return line.data, CACHE_HIT_LATENCY
        else:
            logging.warning(f"CACHE: MISS on read at addr 0x{addr:04X}. Accessing main memory.")
            data = self.main_memory.read(addr)
            line.valid = True
            line.tag = tag
            line.data = data
            return data, CACHE_MISS_LATENCY

    def write(self, addr, data):
        """
        Запись данных в кеш (write-through, no-write-allocate).
        Возвращает задержку.
        """
        if is_io_address(addr):
            logging.info(f"CACHE: Bypassing for I/O write at 0x{addr:04X}")
            self.main_memory.write(addr, data)
            return CACHE_HIT_LATENCY

        line_index, tag = self._get_line_and_tag(addr)
        line = self.lines[line_index]
        is_hit = line.valid and line.tag == tag
        if is_hit:
            logging.info(f"CACHE: HIT on write at addr 0x{addr:04X}. Updating cache and memory.")
            latency = CACHE_HIT_LATENCY
        else:
            logging.warning(f"CACHE: MISS on write at addr 0x{addr:04X}. Writing to main memory.")
            latency = CACHE_MISS_LATENCY

        self.main_memory.write(addr, data)

        if is_hit:
            line.data = data

        return latency


class MainMemory:
    """Модель основной памяти с memory-mapped I/O."""

    def __init__(self, size, input_buffer):
        self.size = size
        self.memory = [0] * size
        self.input_buffer = deque(input_buffer)
        self.output_buffer = []

    def write(self, addr, value):
        assert 0 <= addr < self.size, f"Invalid memory address: {addr}"
        if addr == IO_OUTPUT_PORT:
            char = chr(value & 0xFF)
            logging.info(f"IO: Write to OUTPUT port: '{char}'")
            self.output_buffer.append(char)
        else:
            self.memory[addr] = value

    def read(self, addr):
        assert 0 <= addr < self.size, f"Invalid memory address: {addr}"
        if addr == IO_INPUT_PORT:
            if not self.input_buffer:
                logging.warning("IO: Input buffer is empty. Returning 0.")
                return 0
            char = self.input_buffer.popleft()
            logging.info(f"IO: Read from INPUT port: '{char}'")
            return ord(char)
        else:
            return self.memory[addr]

    def load_program(self, code: list[Instruction], data: bytes):
        for i, instr in enumerate(code):
            instr_bytes = instr.to_binary()
            word = struct.unpack('>I', instr_bytes)[0]
            self.memory[i] = word

        code_size_bytes = len(code) * 4
        for i, byte in enumerate(data):
            if code_size_bytes + i < self.size:
                word_addr = (code_size_bytes + i) // 4
                byte_offset = (code_size_bytes + i) % 4

                current_word = self.memory[word_addr]
                mask = ~(0xFF << ((3 - byte_offset) * 8))
                new_byte = byte << ((3 - byte_offset) * 8)

                self.memory[word_addr] = (current_word & mask) | new_byte


class DataPath:
    """
    Тракт данных. Содержит все регистры, АЛУ и интерфейс к памяти.
    Управляется сигналами от ControlUnit.
    """

    def __init__(self, memory_size, cache_size, input_buffer):
        self.main_memory = MainMemory(memory_size, input_buffer)
        self.cache = Cache(cache_size, self.main_memory)

        # Регистры
        self.gpr = [0] * 8
        self.gpr[Reg.SP.value] = memory_size - 1000
        self.data_sp = memory_size - 5000
        self.pc = 0
        self.mar = 0
        self.mdr = 0
        self.ir_reg = 0
        self.alu_a = 0
        self.alu_b = 0
        self.alu_out = 0
        self.zero_flag = False

    @property
    def sp(self):
        return self.gpr[Reg.SP.value]

    @sp.setter
    def sp(self, value):
        self.gpr[Reg.SP.value] = value

    def decode_ir(self):
        """Декодирует инструкцию из `ir_reg` в структурированный вид."""
        opcode_val = (self.ir_reg >> 26) & 0x3F
        rs = (self.ir_reg >> 21) & 0x1F
        rt = (self.ir_reg >> 16) & 0x1F
        rd = (self.ir_reg >> 11) & 0x1F
        imm = self.ir_reg & 0xFFFF
        addr = self.ir_reg & 0x03FFFFFF

        try:
            opcode = Opcode(opcode_val)
        except ValueError:
            logging.error(f"Unknown opcode value: {hex(opcode_val)}. Treating as NOP.")
            opcode = Opcode.NOP

        sign_extended_opcodes = {Opcode.ADDI, Opcode.LOAD, Opcode.STORE, Opcode.JZ, Opcode.JNZ}
        if opcode in sign_extended_opcodes:
            if (imm & 0x8000) == 0x8000:
                imm -= (1 << 16)
        return {'opcode': opcode, 'rs': rs, 'rt': rt, 'rd': rd, 'imm': imm, 'addr': addr}

    def alu_op(self, op: MicroOp):
        """Выполняет операцию в АЛУ."""
        if op == MicroOp.ALU_ADD:
            self.alu_out = self.alu_a + self.alu_b
        elif op == MicroOp.ALU_SUB:
            self.alu_out = self.alu_a - self.alu_b
        elif op == MicroOp.ALU_MUL:
            self.alu_out = self.alu_a * self.alu_b
        elif op == MicroOp.ALU_DIV:
            self.alu_out = self.alu_a // self.alu_b if self.alu_b != 0 else 0
        elif op == MicroOp.ALU_MOD:
            self.alu_out = self.alu_a % self.alu_b if self.alu_b != 0 else 0
        elif op == MicroOp.ALU_OR:
            self.alu_out = self.alu_a | self.alu_b
        elif op == MicroOp.ALU_AND:
            self.alu_out = self.alu_a & self.alu_b
        elif op == MicroOp.ALU_XOR:
            self.alu_out = self.alu_a ^ self.alu_b
        elif op == MicroOp.ALU_CMP:
            self.alu_out = 1 if self.alu_a == self.alu_b else 0
        elif op == MicroOp.ALU_SHL:
            self.alu_out = self.alu_a << self.alu_b
        elif op == MicroOp.ALU_SHR:
            self.alu_out = self.alu_a >> self.alu_b
        elif op == MicroOp.ALU_LUI:
            self.alu_out = self.alu_b << 16

        else:
            raise ValueError(f"Unknown ALU micro-op: {op}")

        self.zero_flag = self.alu_out == 0
        self.gpr[0] = 0


class ControlUnit:
    """
    Микрокомандный блок управления. "Проигрывает" микрокод для каждой инструкции.
    """

    def __init__(self, datapath: DataPath):
        self.datapath = datapath
        self.microcode_rom = get_microcode_rom()
        self.micro_pc = 0
        self.tick_counter = 0
        self.stall_cycles = 0
        self.halted = False
        self.current_decoded_ir = {'opcode': Opcode.NOP, 'rs': 0, 'rt': 0, 'rd': 0, 'imm': 0, 'addr': 0}

    def tick(self):
        """Выполняет один такт симуляции."""
        self.tick_counter += 1

        if self.stall_cycles > 0:
            logging.info(f"STALL: {self.stall_cycles - 1} cycles remaining.")
            self.stall_cycles -= 1
            return

        if self.halted:
            return

        decoded_ir = self.current_decoded_ir
        opcode = decoded_ir['opcode']
        micro_program = self.microcode_rom.get(opcode)

        if not micro_program:
            raise ValueError(f"No microprogram for opcode: {opcode}")

        if len(micro_program) == 0:
            logging.error(f"Empty microprogram for opcode: {opcode}")
            self.halted = True
            return

        if self.micro_pc >= len(micro_program):
            logging.error(f"MicroPC out of bounds for {opcode}: {self.micro_pc}")
            self.micro_pc = 0
            return

        micro_op = micro_program[self.micro_pc]
        self.execute_micro_op(micro_op, decoded_ir)
        if self.halted:
            return  # Не вычисляем следующий MicroPC если процессор остановлен
        next_micro_pc = self.micro_pc + 1

        if micro_op == MicroOp.FINISH_INSTRUCTION:
            next_micro_pc = 0
        if next_micro_pc >= len(micro_program) and next_micro_pc != 0:
            logging.error(f"MicroPC for {opcode} will be out of bounds ({next_micro_pc}). Resetting.")
            self.micro_pc = 0
        else:
            self.micro_pc = next_micro_pc

    def execute_micro_op(self, op: MicroOp, ir: dict):
        """Исполнение одной микро-операции."""
        dp = self.datapath
        logging.debug(f"TICK {self.tick_counter}: Executing micro-op: {op.name}")
        if op == MicroOp.HALT_PROCESSOR:
            self.halted = True
            logging.info("HALT instruction executed. Stopping simulation.")
        elif op == MicroOp.LATCH_PC_INC:
            dp.pc += 1
        elif op == MicroOp.LATCH_PC_ADDR:
            logging.info(f"JMP/CALL: Setting PC to 0x{ir['addr']:04X}")
            dp.pc = ir['addr']
        elif op == MicroOp.LATCH_PC_ALU:
            dp.pc = dp.alu_out
        elif op == MicroOp.LATCH_MAR_PC:
            dp.mar = dp.pc
        elif op == MicroOp.LATCH_MAR_ALU:
            if ir['opcode'] == Opcode.RET:
                dp.mar = dp.sp
                logging.debug(f"RET: Reading from call_sp=0x{dp.sp:04X}")
            elif ir['opcode'] == Opcode.POP:
                dp.mar = dp.alu_a
                logging.debug(f"POP: Reading from data_sp=0x{dp.alu_a:04X}")
            else:
                dp.mar = dp.alu_out
        elif op == MicroOp.LATCH_IR:
            dp.ir_reg = dp.mdr
            self.current_decoded_ir = dp.decode_ir()
        elif op == MicroOp.LATCH_MDR_RT:
            dp.mdr = dp.gpr[ir['rt']]
            if ir['opcode'] == Opcode.STORE and dp.mar == 0xFF01:
                logging.warning(
                    f"EMIT: R{ir['rt']} contains value {dp.gpr[ir['rt']]} ('{chr(dp.gpr[ir['rt']] & 0xFF)}')")
        elif op == MicroOp.LATCH_MDR_A:
            dp.mdr = dp.alu_a
        elif op == MicroOp.LATCH_A_RS:
            dp.alu_a = dp.gpr[ir['rs']]
        elif op == MicroOp.LATCH_A_RT:
            dp.alu_a = dp.gpr[ir['rt']]
            dp.zero_flag = (dp.alu_a == 0)
        elif op == MicroOp.LATCH_A_SP:
            if ir['opcode'] in [Opcode.PUSH, Opcode.POP]:
                dp.alu_a = dp.data_sp
            else:
                dp.alu_a = dp.sp
        elif op == MicroOp.LATCH_A_MDR:
            dp.alu_a = dp.mdr
            if ir['opcode'] == Opcode.RET:
                logging.error(f"RET: Read return address 0x{dp.mdr:04X} from stack")
        elif op == MicroOp.LATCH_A_PC:
            if ir['opcode'] == Opcode.CALL:
                dp.alu_a = dp.pc
                logging.info(f"CALL: Saving return address 0x{dp.pc:04X}")
            else:
                dp.alu_a = dp.pc
        elif op == MicroOp.BRANCH_IF_ZERO:
            if dp.zero_flag:
                dp.alu_a = dp.pc
                dp.alu_b = ir['imm']
                dp.alu_op(MicroOp.ALU_ADD)
                dp.pc = dp.alu_out
        elif op == MicroOp.BRANCH_IF_NOT_ZERO:
            if not dp.zero_flag:
                dp.alu_a = dp.pc
                dp.alu_b = ir['imm']
                dp.alu_op(MicroOp.ALU_ADD)
                dp.pc = dp.alu_out
        elif op == MicroOp.LATCH_B_RT:
            dp.alu_b = dp.gpr[ir['rt']]
        elif op == MicroOp.LATCH_B_IMM:
            dp.alu_b = ir['imm']
        elif op == MicroOp.LATCH_B_CONST_1:
            dp.alu_b = 1
        elif op == MicroOp.LATCH_RD_ALU:
            dp.gpr[ir['rd']] = dp.alu_out
        elif op == MicroOp.LATCH_RT_ALU:
            dp.gpr[ir['rt']] = dp.alu_out
        elif op == MicroOp.LATCH_RT_MDR:
            dp.gpr[ir['rt']] = dp.mdr
        elif op == MicroOp.LATCH_SP_ALU:
            if ir['opcode'] in [Opcode.PUSH, Opcode.POP]:
                if not (0 <= dp.alu_out < dp.cache.main_memory.size):
                    logging.error(f"Data stack overflow: {dp.alu_out}")
                    raise RuntimeError("Data stack overflow")
                dp.data_sp = dp.alu_out
            else:
                if not (0 <= dp.alu_out < dp.cache.main_memory.size):
                    logging.error(f"Call stack overflow: {dp.alu_out}")
                    raise RuntimeError("Call stack overflow")
                dp.sp = dp.alu_out
        elif op.name.startswith("ALU_"):
            dp.alu_op(op)
        elif op == MicroOp.CACHE_READ:
            data, latency = dp.cache.read(dp.mar)
            dp.mdr = data
            if latency > 1:
                self.stall_cycles = latency - 1
        elif op == MicroOp.CACHE_WRITE:
            if dp.mar == IO_OUTPUT_PORT:
                logging.warning(f"EMIT: Writing value {dp.mdr} ('{chr(dp.mdr & 0xFF)}') to output")
            latency = dp.cache.write(dp.mar, dp.mdr)
            if latency > 1:
                self.stall_cycles = latency - 1
        elif op == MicroOp.FINISH_INSTRUCTION:
            pass
        else:
            raise ValueError(f"Unknown micro-op during execution: {op}")
        dp.gpr[0] = 0


def simulation(binary_code: bytes, input_str: str, limit: int, cache_size: int):
    """Основной цикл симуляции."""

    words = []
    for i in range(0, len(binary_code), 4):
        words.append(struct.unpack('>I', binary_code[i:i + 4])[0])

    last_halt_idx = -1
    for i, word in enumerate(words):
        opcode_val = (word >> 26) & 0x3F
        if opcode_val == Opcode.HALT.value:
            last_halt_idx = i

    if last_halt_idx == -1:
        raise ValueError("HALT instruction not found in the binary file.")

    code_words = words[:last_halt_idx + 1]
    data_words = words[last_halt_idx + 1:]
    datapath = DataPath(MEMORY_SIZE, cache_size, list(input_str))
    for i, word in enumerate(code_words):
        datapath.main_memory.memory[i] = word
    code_size_words = len(code_words)
    for i, word in enumerate(data_words):
        if code_size_words + i < datapath.main_memory.size:
            datapath.main_memory.memory[code_size_words + i] = word
        else:
            logging.warning("Data section overflows memory. Truncating.")
            break

    print(f"Code size: {len(code_words)} words. Data size: {len(data_words)} words.")

    control_unit = ControlUnit(datapath)

    logging.info("Starting simulation...")
    while not control_unit.halted and control_unit.tick_counter < limit:
        try:
            decoded = control_unit.current_decoded_ir
            mnemonic = Instruction(
                decoded['opcode'], decoded['rs'], decoded['rt'],
                decoded['rd'], decoded['imm'], decoded['addr']
            ).get_mnemonic()

            log_msg = (
                f"TICK: {control_unit.tick_counter:4} | "
                f"PC: 0x{datapath.pc:04X} | "
                f"IR: 0x{control_unit.datapath.ir_reg:08X} ({mnemonic}) | "
                f"SP: {datapath.sp} | DSP: {datapath.data_sp} | "
                f"MicroPC: {control_unit.micro_pc} | "
                f"Zero: {datapath.zero_flag}"
            )
            logging.info(log_msg)

            control_unit.tick()

        except (ValueError, IndexError) as e:
            logging.error(f"Error during simulation: {e}")
            break
        except Exception:
            logging.exception("An unexpected error occurred")
            break

    if control_unit.halted:
        logging.info("Simulation halted by HALT instruction.")
    elif control_unit.tick_counter >= limit:
        logging.warning("Simulation limit reached.")

    output = "".join(datapath.main_memory.output_buffer)
    logging.info(f"Simulation finished. Total ticks: {control_unit.tick_counter}. Output: '{output}'")

    return output, control_unit.tick_counter


def main(code_file: str, input_file: str):
    """Главная функция для запуска симулятора из командной строки."""
    logging.basicConfig(level=logging.INFO, format='%(levelname)-8s %(message)s')
    try:
        with open(code_file, 'rb') as f:
            binary_code = f.read()
    except FileNotFoundError:
        logging.critical(f"Error: Code file not found at '{code_file}'")
        sys.exit(1)

    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            input_data = f.read()
    except FileNotFoundError:
        logging.warning(f"Input file not found at '{input_file}', using empty input.")
        input_data = ""

    output, ticks = simulation(
        binary_code=binary_code,
        input_str=input_data,
        limit=500000,
        cache_size=32
    )

    print("-" * 40)
    print(f"Simulation output: '{output}'")
    print(f"Total ticks: {ticks}")
    print("-" * 40)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python machine.py <binary_code_file> <input_file>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
