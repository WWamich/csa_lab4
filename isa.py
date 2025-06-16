"""
ISA определения для RISC Forth процессора
Общие определения для транслятора и модели процессора
"""

from enum import Enum


class Opcode(Enum):
    """Коды операций RISC процессора"""
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
    """Регистры процессора"""
    ZERO = 0  # всегда 0
    SP = 1  # указатель стека данных
    RSP = 2  # указатель стека возвратов
    TOS = 3  # вершина стека (Top Of Stack)
    BASE = 4  # базовый адрес
    T1 = 5  # временный регистр 1
    T2 = 6  # временный регистр 2
    PC = 7  # счетчик команд


# MMIO
IO_INPUT_PORT = 0x8000
IO_OUTPUT_PORT = 0x8001

# Адреса памяти
DATA_START = 0x1000
VARIABLES_START = 0x2000
STACK_START = 0x3000
RETURN_STACK_START = 0x4000

# Размеры
MEMORY_SIZE = 0x10000  # 64K
CACHE_SIZE = 16  # 16 строк кэша
WORD_SIZE = 4  # 32-битные слова


def format_instruction(opcode: Opcode, rs=0, rt=0, rd=0, imm=0, addr=0) -> str:
    """Форматирование инструкции"""
    if opcode == Opcode.LOAD:
        return f"LOAD R{rt}, R{rs}+{imm}"
    elif opcode == Opcode.STORE:
        return f"STORE R{rs}, R{rt}+{imm}"
    elif opcode in [Opcode.ADD, Opcode.SUB, Opcode.MUL, Opcode.DIV, Opcode.MOD,
                    Opcode.AND, Opcode.OR, Opcode.XOR, Opcode.CMP]:
        return f"{opcode.name} R{rd}, R{rs}, R{rt}"
    elif opcode == Opcode.PUSH:
        return f"PUSH R{rs}"
    elif opcode == Opcode.POP:
        return f"POP R{rt}"
    elif opcode == Opcode.JZ:
        return f"JZ R{rs}, 0x{imm:04X}"
    elif opcode == Opcode.JMP:
        return f"JMP 0x{addr:04X}"
    elif opcode == Opcode.IN:
        return f"IN R{rt}, 0x{imm:04X}"
    elif opcode == Opcode.OUT:
        return f"OUT R{rs}, 0x{imm:04X}"
    else:
        return opcode.name


if __name__ == "__main__":
    print("🏗️  RISC Forth ISA")
    print("=" * 40)
    print(f"📋 Опкоды: {len(Opcode)} команд")
    for op in Opcode:
        print(f"  0x{op.value:02X}: {op.name}")

    print(f"\n🔧 Регистры: {len(Reg)} регистров")
    for reg in Reg:
        print(f"  R{reg.value}: {reg.name}")

    print(f"\n💾 Карта памяти:")
    print(f"  Данные:        0x{DATA_START:04X}")
    print(f"  Переменные:    0x{VARIABLES_START:04X}")
    print(f"  Стек данных:   0x{STACK_START:04X}")
    print(f"  Стек возврата: 0x{RETURN_STACK_START:04X}")

    print(f"\n🔌 I/O порты:")
    print(f"  Ввод:  0x{IO_INPUT_PORT:04X}")
    print(f"  Вывод: 0x{IO_OUTPUT_PORT:04X}")