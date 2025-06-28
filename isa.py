import struct
from enum import Enum


class Opcode(Enum):
    """
    Коды операций для нашего RISC-процессора.
    Архитектура фон-Неймановская, стек-ориентированная (для Forth), с кешем.
    """

    # --- Системные инструкции ---
    NOP = 0x00  # Ничего не делать
    HALT = 0x01  # Остановка процессора

    # --- Арифметические и логические R-type (Регистр-Регистр) ---
    ADD = 0x10  # Сложение: rd <- rs + rt
    SUB = 0x11  # Вычитание: rd <- rs - rt
    MUL = 0x12  # Умножение: rd <- rs * rt
    DIV = 0x13  # Деление: rd <- rs / rt
    MOD = 0x14  # Остаток от деления: rd <- rs % rt
    AND = 0x15  # Битовое И: rd <- rs & rt
    OR = 0x16  # Битовое ИЛИ: rd <- rs | rt
    XOR = 0x17  # Битовое XOR: rd <- rs ^ rt
    CMP = 0x18  # Сравнение: rd <- 1 if rs == rt else 0
    SHL = 0x19  # Сдвиг влево: rd <- rs << rt
    SHR = 0x1A  # Сдвиг вправо: rd <- rs >> rt

    # --- Инструкции I-type (с непосредственным значением) ---
    ADDI = 0x20  # Сложение с константой: rt <- rs + imm16
    LOAD = 0x21  # Загрузка из памяти: rt <- mem[rs + imm16]
    STORE = 0x22  # Запись в память: mem[rs + imm16] <- rt
    JZ = 0x23  # Условный переход если rt == 0: pc <- pc + imm16
    JNZ = 0x24  # Условный переход если rt != 0: pc <- pc + imm16
    LUI = 0x25  # Загрузка старших бит: rt <- imm16 << 16
    ORI = 0x26  # ИЛИ с константой: rt <- rs | imm16

    # --- Инструкции для работы со стеком (реализуются через ADDI/LOAD/STORE) ---
    # Например, PUSH rs -> STORE rs, [SP-1]; ADDI SP, SP, -1
    # POP rt -> ADDI SP, SP, 1; LOAD rt, [SP]
    # Для простоты трансляции, введем отдельные опкоды.
    PUSH = 0x30
    POP = 0x31

    # --- Инструкции J-type (безусловные переходы) ---
    JMP = 0x32  # Безусловный переход: pc <- addr
    CALL = 0x33  # Вызов процедуры: stack.push(pc+1); pc <- addr
    RET = 0x34  # Возврат из процедуры: pc <- stack.pop()


class Reg(Enum):
    """Регистры процессора. 8 регистров общего назначения."""

    ZERO = 0  # Всегда ноль
    SP = 1  # Stack Pointer (указатель на вершину стека)
    RA = 2  # Return Address (адрес возврата для CALL)
    T0 = 3  # Temporary 0
    T1 = 4  # Temporary 1
    T2 = 5  # Temporary 2
    T3 = 6  # Temporary 3
    A1 = 7  # Argument 1 / Return Value 1


# Константы для memory-mapped I/O
MEMORY_SIZE = 65536  # 64K слов (256KB)
IO_INPUT_PORT = 0xFF00  # Адрес порта ввода
IO_OUTPUT_PORT = 0xFF01  # Адрес порта вывода


class Instruction:
    """
    Представление одной 32-битной машинной инструкции.
    Определяет методы для кодирования в бинарный формат.
    """

    def __init__(
        self,
        opcode: Opcode,
        rs: int = 0,
        rt: int = 0,
        rd: int = 0,
        imm: int = 0,
        addr: int = 0,
    ) -> None:
        self.opcode = opcode
        self.rs = rs
        self.rt = rt
        self.rd = rd
        self.imm = imm  # 16-bit immediate
        self.addr = addr  # 26-bit address for J-type

    def to_binary(self) -> bytes:
        """Кодирование инструкции в 32-битное слово (big-endian)."""
        opcode_val = self.opcode.value & 0x3F
        word = 0

        # R-type: [opcode:6][rs:5][rt:5][rd:5][unused:11]
        if self.opcode in [
            Opcode.ADD,
            Opcode.SUB,
            Opcode.MUL,
            Opcode.DIV,
            Opcode.MOD,
            Opcode.AND,
            Opcode.OR,
            Opcode.XOR,
            Opcode.CMP,
            Opcode.SHL,
            Opcode.SHR,
        ]:
            word = (
                (opcode_val << 26) | (self.rs << 21) | (self.rt << 16) | (self.rd << 11)
            )

        # I-type: [opcode:6][rs:5][rt:5][imm:16]
        elif self.opcode in [
            Opcode.ADDI,
            Opcode.LOAD,
            Opcode.STORE,
            Opcode.JZ,
            Opcode.JNZ,
            Opcode.LUI,
            Opcode.ORI,
        ]:
            if self.opcode == Opcode.LUI:
                self.rs = 0
            word = (
                (opcode_val << 26)
                | (self.rs << 21)
                | (self.rt << 16)
                | (self.imm & 0xFFFF)
            )

        # J-type: [opcode:6][addr:26]
        elif self.opcode in [Opcode.JMP, Opcode.CALL]:
            word = (opcode_val << 26) | (self.addr & 0x03FFFFFF)

        elif self.opcode in [Opcode.PUSH]:
            word = (opcode_val << 26) | (self.rs << 21)
        elif self.opcode in [Opcode.POP]:
            word = (opcode_val << 26) | (self.rt << 16)
        else:  # HALT, NOP, RET
            word = opcode_val << 26

        return struct.pack(">I", word)

    def to_hex(self, addr: int) -> str:
        """Генерация строкового представления для листинга (адрес, код, мнемоника)."""
        hex_code = self.to_binary().hex().upper()
        mnemonic = self.get_mnemonic()
        return f"0x{addr:04X}: {hex_code}  {mnemonic}"

    def get_mnemonic(self) -> str:
        """Получить мнемонику инструкции."""
        if self.opcode in [
            Opcode.ADD,
            Opcode.SUB,
            Opcode.MUL,
            Opcode.DIV,
            Opcode.MOD,
            Opcode.AND,
            Opcode.OR,
            Opcode.XOR,
            Opcode.CMP,
            Opcode.SHL,
            Opcode.SHR,
        ]:
            return f"{self.opcode.name:<5} R{self.rd}, R{self.rs}, R{self.rt}"

        elif self.opcode in [Opcode.ADDI, Opcode.ORI]:
            return f"{self.opcode.name:<5} R{self.rt}, R{self.rs}, {self.imm}"
        elif self.opcode in [Opcode.LOAD, Opcode.STORE]:
            return f"{self.opcode.name:<5} R{self.rt}, {self.imm}(R{self.rs})"
        elif self.opcode in [Opcode.JZ, Opcode.JNZ]:
            return f"{self.opcode.name:<5} R{self.rt}, {self.imm}"

        elif self.opcode == Opcode.LUI:
            return f"LUI   R{self.rt}, 0x{self.imm:04X}"

        elif self.opcode in [Opcode.JMP, Opcode.CALL]:
            return f"{self.opcode.name:<5} 0x{self.addr:07X}"

        elif self.opcode == Opcode.PUSH:
            return f"PUSH  R{self.rs}"
        elif self.opcode == Opcode.POP:
            return f"POP   R{self.rt}"

        else:
            return self.opcode.name

    def __repr__(self):
        return f"Instruction({self.get_mnemonic()})"
