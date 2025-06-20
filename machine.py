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


def decode_instruction(instruction: int) -> Tuple[Opcode, int, int, int, int]:
    """Декодирование с поддержкой новых форматов"""
    opcode_val = (instruction >> 26) & 0x3F
    rs = (instruction >> 21) & 0x1F
    rt = (instruction >> 16) & 0x1F
    rd = (instruction >> 11) & 0x1F
    imm16 = instruction & 0xFFFF
    imm21 = instruction & 0x1FFFFF
    addr = instruction & 0x3FFFFFF

    try:
        opcode = Opcode(opcode_val)
    except ValueError:
        logging.warning(f"Неизвестный опкод: 0x{opcode_val:02X} в инструкции 0x{instruction:08X}")
        opcode = Opcode.NOP

    if opcode == Opcode.LOADI:
        rt = (instruction >> 21) & 0x1F
        imm21 = instruction & 0x1FFFFF
        if imm21 & 0x100000:
            imm21 |= 0xFFE00000
        print(f"   LOADI: RT={rt}, IMM21={imm21}")
        return opcode, 0, rt, 0, imm21

    # Для LUI - только RT и IMM16
    elif opcode == Opcode.LUI:
        print(f"   LUI: RT={rt}, IMM16={imm16}")
        return opcode, 0, rt, 0, imm16

    # Для ORI - RS, RT, IMM16
    elif opcode == Opcode.ORI:
        print(f"   ORI: RS={rs}, RT={rt}, IMM16={imm16}")
        return opcode, rs, rt, 0, imm16

    print(f"   STANDARD: RS={rs}, RT={rt}, RD={rd}, IMM={imm16}")
    return opcode, rs, rt, rd, imm16


class DataPath:
    """Тракт данных процессора"""

    def __init__(self, input_buffer: List[str]):
        # Регистровый файл
        self.registers = [0] * 8
        self.memory = [0] * MEMORY_SIZE  # ЧИСТАЯ память
        self.stack_memory = [0] * 1024  #  стековая память

        # Кэш
        self.cache = SimpleCache(size=16)

        # I/O
        self.input_buffer = input_buffer.copy()
        self.output_buffer = []

        # Временные регистры для микрокода
        self.temp_addr = 0
        self.alu_result = 0

        # Флаги
        self.zero_flag = False

        # Статистика
        self.memory_accesses = 0

        logging.info("DataPath инициализирован с чистой памятью")

    def read_register(self, reg_id: int) -> int:
        """Читать значение из регистра"""
        if reg_id == Reg.ZERO.value:
            return 0
        if 0 <= reg_id < 8:
            return self.registers[reg_id] & 0xFFFFFFFF
        return 0

    def write_register(self, reg_id: int, value: int):
        """Записать значение в регистр"""
        if reg_id != Reg.ZERO.value and 0 <= reg_id < 8:  # R0 нельзя изменить
            self.registers[reg_id] = value & 0xFFFFFFFF
            self.zero_flag = (value == 0)
            logging.debug(f"WRITE REG: R{reg_id} = 0x{value:08X} ({value})")

    def memory_read(self, addr: int) -> Tuple[bool, int, int]:
        """Чтение из памяти через кэш"""
        self.memory_accesses += 1
        hit, result = self.cache.access(addr)

        if hit:
            logging.debug(f"CACHE HIT: mem[0x{addr:04X}] = {result}")
            return True, result, 1
        else:
            # Промах - читаем из основной памяти
            data = self.memory[addr & 0xFFFF]
            self.cache.write(addr, data)  # обновляем кэш
            logging.debug(f"CACHE MISS: mem[0x{addr:04X}] = {data} (10 cycles)")
            return False, data, 10  # 10 тактов на промах

    def memory_write(self, addr: int, data: int) -> Tuple[bool, int]:
        """Запись в память через кэш"""
        self.memory_accesses += 1
        addr = addr & 0xFFFF
        self.memory[addr] = data & 0xFFFFFFFF
        self.cache.write(addr, data)
        logging.debug(f"MEM WRITE: mem[0x{addr:04X}] = {data}")
        return True, 1  # запись всегда попадание

    def stack_push(self, value: int):
        """ИСПРАВЛЕННАЯ функция: положить значение на стек данных"""
        sp = self.read_register(Reg.SP.value)

        # ПРОВЕРКА ГРАНИЦ СТЕКА
        if sp >= len(self.stack_memory):
            raise OverflowError(f"Stack overflow: SP={sp}, stack_size={len(self.stack_memory)}")

        self.stack_memory[sp] = value & 0xFFFFFFFF  # ← УБРАЛИ & 0x3FF!
        self.write_register(Reg.SP.value, sp + 1)
        logging.debug(f"STACK PUSH: stack[{sp}] = {value}, SP = {sp + 1}")

    def stack_pop(self) -> int:
        """ИСПРАВЛЕННАЯ функция: снять значение со стека данных"""
        sp = self.read_register(Reg.SP.value) - 1

        # ПРОВЕРКА ГРАНИЦ СТЕКА
        if sp < 0:
            raise OverflowError(f"Stack underflow: SP={sp}")

        self.write_register(Reg.SP.value, sp)
        value = self.stack_memory[sp]  # ← УБРАЛИ & 0x3FF!
        logging.debug(f"STACK POP: stack[{sp}] = {value}, SP = {sp}")
        return value

    def alu_operation(self, op: MicroOp, src1: int, src2: int) -> int:
        """Выполнить операцию в АЛУ"""
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
            shift_amount = src2 & 0x1F
            result = (src1 << shift_amount) & 0xFFFFFFFF
            logging.debug(f"SHL: {src1} << {shift_amount} = {result}")
        elif op == MicroOp.ALU_SHR:
            shift_amount = src2 & 0x1F
            result = src1 >> shift_amount
            logging.debug(f"SHR: {src1} >> {shift_amount} = {result}")

        self.alu_result = result
        self.zero_flag = (result == 0)
        logging.debug(f"ALU: {op.name} {src1} {src2} = {result}")
        return result

    def io_read(self, port: int) -> int:
        """Чтение с порта ввода"""
        if port == IO_INPUT_PORT and self.input_buffer:
            char = self.input_buffer.pop(0)
            return ord(char) if char else 0
        return 0

    def io_write(self, port: int, value: int):
        """Запись в порт вывода"""
        if port == IO_OUTPUT_PORT:
            char = chr(value & 0xFF)
            self.output_buffer.append(char)
            logging.debug(f"I/O OUT: port 0x{port:04X} = {value} ('{char}')")

    def get_stats(self) -> dict:
        """Получить статистику работы DataPath"""
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

        # Счетчики
        self.pc = 0  # счетчик команд
        self.tick_count = 0  # счетчик тактов

        # Состояние микропрограммы
        self.current_instruction = None
        self.current_microcode = []
        self.micro_step = 0

        # Декодированная инструкция
        self.opcode = None
        self.rs = 0
        self.rt = 0
        self.rd = 0
        self.imm = 0

        # Флаги состояния
        self.halted = False
        self.waiting_for_cache = 0

    def fetch_instruction(self):
        """Выборка инструкции из памяти"""
        if self.pc >= len(self.program):
            self.halted = True
            return

        instruction = self.program[self.pc]
        self.current_instruction = instruction

        # Декодирование инструкции
        self.opcode, self.rs, self.rt, self.rd, self.imm = decode_instruction(instruction)

        # Получаем микрокод для инструкции
        self.current_microcode = get_microcode(self.opcode)
        self.micro_step = 0

        logging.debug(f"FETCH: PC=0x{self.pc:04X} INSTR=0x{instruction:08X} "
                      f"OP={self.opcode.name} RS=R{self.rs} RT=R{self.rt} IMM={self.imm}")

    def execute_micro_instruction(self, micro_instr: MicroInstruction) -> bool:
        """Выполнить одну микроинструкцию"""
        logging.debug(f"MICRO: {micro_instr}")

        if micro_instr.op == MicroOp.NOP:
            return True

        elif micro_instr.op == MicroOp.HALT:
            self.halted = True
            return False

        elif micro_instr.op == MicroOp.PC_INC:
            self.pc += 1
            return True

        elif micro_instr.op == MicroOp.PC_LOAD:
            self.pc = self.imm
            return True

        elif micro_instr.op == MicroOp.LOAD_IMM:
            value = self.imm
            reg_num = self.rt
            self.data_path.write_register(reg_num, value)
            logging.debug(f"LOAD_IMM: R{reg_num} = {value}")
            return True

        elif micro_instr.op == MicroOp.SHIFT_LEFT:
            shift_amount = micro_instr.src
            value = self.imm << shift_amount
            reg_num = self.rt if micro_instr.dst == "rt" else int(micro_instr.dst)
            self.data_path.write_register(reg_num, value)
            logging.debug(f"SHIFT_LEFT: R{reg_num} = 0x{value:08X} ({self.imm} << {shift_amount})")
            return True

        elif micro_instr.op == MicroOp.ALU_ADD:
            src1 = self.data_path.read_register(self.rs)
            if micro_instr.imm == "rt":
                src2 = self.data_path.read_register(self.rt)
            elif micro_instr.imm == "imm":
                src2 = self.imm
            elif isinstance(micro_instr.imm, int):
                src2 = micro_instr.imm
            else:
                src2 = self.imm

            result = self.data_path.alu_operation(MicroOp.ALU_ADD, src1, src2)

            if micro_instr.dst == "TEMP_ADDR":
                self.data_path.temp_addr = result
            elif micro_instr.dst == "rd":
                self.data_path.write_register(self.rd, result)
            elif micro_instr.dst == "rt":
                self.data_path.write_register(self.rt, result)
            elif micro_instr.dst == "SP":
                self.data_path.write_register(Reg.SP.value, result)
            return True

        elif micro_instr.op == MicroOp.ALU_SUB:
            src1 = self.data_path.read_register(self.rs if micro_instr.src == "rs" else Reg.SP.value)
            src2 = self.data_path.read_register(self.rt) if micro_instr.imm == "rt" else (
                micro_instr.imm if isinstance(micro_instr.imm, int) else self.imm)
            result = self.data_path.alu_operation(MicroOp.ALU_SUB, src1, src2)

            if micro_instr.dst == "rd":
                self.data_path.write_register(self.rd, result)
            elif micro_instr.dst == "SP":
                self.data_path.write_register(Reg.SP.value, result)
            return True

        elif micro_instr.op in [MicroOp.ALU_MUL, MicroOp.ALU_DIV, MicroOp.ALU_MOD,
                                MicroOp.ALU_AND, MicroOp.ALU_OR, MicroOp.ALU_XOR, MicroOp.ALU_CMP, MicroOp.ALU_SHL, MicroOp.ALU_SHR]:
            src1 = self.data_path.read_register(self.rs)
            if micro_instr.imm == "rt":
                src2 = self.data_path.read_register(self.rt)
            elif micro_instr.imm == "imm":
                src2 = self.imm
            else:
                src2 = self.imm
            result = self.data_path.alu_operation(micro_instr.op, src1, src2)

            if micro_instr.dst == "rd":
                self.data_path.write_register(self.rd, result)
            elif micro_instr.dst == "rt":
                self.data_path.write_register(self.rt, result)
            return True

        elif micro_instr.op == MicroOp.CACHE_CHECK:
            # Проверка кэша - пока просто помечаем что будем читать
            return True

        elif micro_instr.op == MicroOp.CACHE_WAIT:
            # Ожидание кэша - здесь будет логика ожидания промаха
            if self.waiting_for_cache > 0:
                self.waiting_for_cache -= 1
                return False
            return True

        elif micro_instr.op == MicroOp.MEM_READ:
            addr = self.data_path.temp_addr
            hit, data, cycles = self.data_path.memory_read(addr)
            self.data_path.write_register(self.rt, data)
            if not hit:
                self.waiting_for_cache = cycles - 1
            return True

        elif micro_instr.op == MicroOp.MEM_WRITE:
            addr = self.data_path.temp_addr
            data = self.data_path.read_register(self.rs)
            hit, cycles = self.data_path.memory_write(addr, data)
            return True

        elif micro_instr.op == MicroOp.STACK_PUSH:
            value = self.data_path.read_register(self.rs)
            self.data_path.stack_push(value)
            return True

        elif micro_instr.op == MicroOp.STACK_POP:
            value = self.data_path.stack_pop()
            self.data_path.write_register(self.rt, value)
            return True


        elif micro_instr.op == MicroOp.JUMP_COND:

            condition_met = False

            if micro_instr.condition == "zero":

                condition_met = (self.data_path.read_register(self.rs) == 0)

            if condition_met:

                self.pc = self.imm

                logging.debug(f"JUMP: PC -> {self.imm}")

            else:

                self.pc += 1

                logging.debug(f"NO JUMP: PC -> {self.pc}")

            return True

        elif micro_instr.op == MicroOp.IO_READ:
            value = self.data_path.io_read(self.imm)
            self.data_path.write_register(self.rt, value)
            return True

        elif micro_instr.op == MicroOp.IO_WRITE:
            value = self.data_path.read_register(self.rs)
            self.data_path.io_write(self.imm, value)
            return True
        elif micro_instr.op == MicroOp.ALU_SHL:
            src1 = self.data_path.read_register(self.rs)
            if micro_instr.imm == "rt":
                src2 = self.data_path.read_register(self.rt)
            else:
                src2 = self.imm
            result = self.data_path.alu_operation(micro_instr.op, src1, src2)
            if micro_instr.dst == "rd":
                self.data_path.write_register(self.rd, result)

        elif micro_instr.op == MicroOp.ALU_SHR:
            src1 = self.data_path.read_register(self.rs)
            if micro_instr.imm == "rt":
                src2 = self.data_path.read_register(self.rt)
            else:
                src2 = self.imm
            result = self.data_path.alu_operation(micro_instr.op, src1, src2)
            if micro_instr.dst == "rd":
                self.data_path.write_register(self.rd, result)


        else:
            logging.warning(f"Неизвестная микрооперация: {micro_instr.op}")
            return True

    def step(self) -> bool:
        """Выполнить один такт процессора"""
        if self.halted:
            return False

        # Проверяем, есть ли ожидание кэша
        if self.waiting_for_cache > 0:
            self.waiting_for_cache -= 1
            self.tick_count += 1
            return True

        # Если нет текущего микрокода, загружаем инструкцию
        if not self.current_microcode or self.micro_step >= len(self.current_microcode):
            self.fetch_instruction()
            if self.halted:
                return False

        # Выполняем текущую микроинструкцию
        micro_instr = self.current_microcode[self.micro_step]
        completed = self.execute_micro_instruction(micro_instr)

        if completed:
            self.micro_step += 1

        self.tick_count += 1
        return True


def simulate(program: List[int], input_data: str, max_ticks: int = 10000) -> tuple:
    """Основная функция симуляции"""
    input_buffer = list(input_data)
    data_path = DataPath(input_buffer)
    control_unit = ControlUnit(program, data_path)

    logging.info(f"Программа загружена: {len(program)} инструкций")
    logging.info(f"Входные данные: '{input_data}'")

    print("ОТЛАДКА ПРОГРАММЫ:")
    for i in range(min(10, len(program))):
        instruction = program[i]
        print(f"🔍 DECODE 0x{instruction:08X}:")
        print(f"   Raw opcode: 0x{(instruction >> 26) & 0x3F:02X}")

        try:
            opcode = Opcode((instruction >> 26) & 0x3F)
            print(f"   ✅ Mapped to: {opcode.name}")

            decoded = decode_instruction(instruction)
            opcode_name = decoded[0].name
            rs, rt, rd, imm = decoded[1], decoded[2], decoded[3], decoded[4]
            print(f"   {i}: 0x{instruction:08X} -> {opcode_name} RS=R{rs} RT=R{rt} IMM={imm}")

        except ValueError:
            print(f"   ❌ Unknown opcode")

    tick_count = 0
    while tick_count < max_ticks and control_unit.step():
        tick_count += 1
        if tick_count % 10 == 0:  # Логирование каждые 10 тактов
            stats = data_path.get_stats()
            output_preview = stats['output'][:20] + ('...' if len(stats['output']) > 20 else '')
            logging.info(f"TICK {tick_count:5}: PC=0x{control_unit.pc:04X} "
                         f"OP={control_unit.opcode.name if control_unit.opcode else 'NONE':<6} "
                         f"μ{control_unit.micro_step} "
                         f"R5=0x{data_path.read_register(5):04X} "
                         f"OUT='{output_preview}'")

    stats = data_path.get_stats()

    return {
        'ticks': tick_count,
        'halted': control_unit.halted,
        'pc': control_unit.pc,
        'output': stats['output'],
        'registers': [data_path.read_register(i) for i in range(8)],
        'cache_stats': {
            'accesses': stats['memory_accesses'],
            'hits': stats['cache_hits'],
            'misses': stats['cache_misses'],
            'hit_rate': stats['cache_hit_rate']
        }
    }


def load_program(filename: str) -> List[int]:
    """Загрузка программы из бинарного файла"""
    program = []
    try:
        with open(filename, 'rb') as f:
            while True:
                data = f.read(4)
                if len(data) < 4:
                    break
                instruction = int.from_bytes(data, byteorder='big')
                program.append(instruction)
    except FileNotFoundError:
        print(f"❌ Файл не найден: {filename}")
        sys.exit(1)
    return program


def main():
    """Главная функция"""
    if len(sys.argv) != 3:
        print("Использование: python machine.py <program.bin> <input.txt>")
        sys.exit(1)

    program_file = sys.argv[1]
    input_file = sys.argv[2]

    # Настройка логирования
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(levelname)-8s %(name)s:%(funcName)-13s %(message)s'
    )

    # Загрузка программы
    program = load_program(program_file)

    # Загрузка входных данных
    try:
        with open(input_file, 'r') as f:
            input_data = f.read().strip()
    except FileNotFoundError:
        input_data = ""

    # Запуск симуляции
    result = simulate(program, input_data)

    # Вывод результатов
    print("\n" + "=" * 50)
    print("РЕЗУЛЬТАТЫ СИМУЛЯЦИИ")
    print("=" * 50)
    print(f"Тактов выполнено: {result['ticks']}")
    print(f"Программа завершена: {'Да' if result['halted'] else 'Нет'}")
    print(f"Финальный PC: 0x{result['pc']:04X}")

    cache_stats = result['cache_stats']
    print(f"\nКЭШ СТАТИСТИКА:")
    print(f"  Обращений к памяти: {cache_stats['accesses']}")
    print(f"  Попаданий в кэш: {cache_stats['hits']}")
    print(f"  Промахов кэша: {cache_stats['misses']}")
    print(f"  Коэффициент попаданий: {cache_stats['hit_rate'] * 100:.2f}%")

    print(f"\nВЫВОД ПРОГРАММЫ:")
    print(f"'{result['output']}'")

    print(f"\nРЕГИСТРЫ:")
    for i, val in enumerate(result['registers']):
        print(f"  R{i}: 0x{val:08X} ({val})")


if __name__ == "__main__":
    main()