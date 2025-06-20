from enum import Enum
import struct


class Opcode(Enum):
    """Коды операций RISC процессора"""
    NOP = 0x00
    HALT = 0x01
    LOAD = 0x02
    STORE = 0x03
    PUSH = 0x04
    POP = 0x05
    ADD = 0x06
    SUB = 0x07
    MUL = 0x08
    DIV = 0x09
    MOD = 0x0A
    AND = 0x0B
    OR = 0x0C
    XOR = 0x0D
    CMP = 0x0E
    JMP = 0x0F
    JZ = 0x10
    IN = 0x11
    OUT = 0x12
    LOADI = 0x13
    LUI = 0x14
    ORI = 0x15
    SHL = 0x16
    SHR = 0x17
    CALL = 0x18
    RET = 0x19



class Reg(Enum):
    """Регистры процессора"""
    ZERO = 0  # 0
    SP = 1  # stack pointer
    T1 = 2  # temporary 1
    T2 = 3  # temporary 2
    A0 = 4  # argument 0
    A1 = 5  # argument 1
    V0 = 6  # return value 0
    V1 = 7  # return value 1
    RSP = 1  # return stack pointer (alias для SP)
    BASE = 4  # base pointer (alias для A0)
    PC = 6  # program counter (alias для V0)


# Константы памяти и I/O
MEMORY_SIZE = 65536  # 64K памяти
IO_INPUT_PORT = 0x8000
IO_OUTPUT_PORT = 0x8001


def format_instruction(opcode: Opcode, rs=0, rt=0, rd=0, imm=0, addr=0) -> str:
    """Форматировать инструкцию для отладки"""
    if opcode == Opcode.LOAD:
        return f"LOAD R{rt}, R{rs}+{imm}"
    elif opcode == Opcode.STORE:
        return f"STORE R{rs}, R{rt}+{imm}"
    elif opcode in [Opcode.ADD, Opcode.SUB, Opcode.MUL, Opcode.DIV, Opcode.MOD,
                    Opcode.AND, Opcode.OR, Opcode.XOR, Opcode.CMP, Opcode.SHL, Opcode.SHR]:
        return f"{opcode.name} R{rd}, R{rs}, R{rt}"
    elif opcode == Opcode.PUSH:
        return f"PUSH R{rs}"
    elif opcode == Opcode.POP:
        return f"POP R{rt}"
    elif opcode == Opcode.JZ:
        return f"JZ R{rs}, 0x{imm:04X}"
    elif opcode == Opcode.JMP:
        return f"JMP 0x{addr:04X}"
    elif opcode == Opcode.CALL:
        return f"CALL 0x{addr:04X}"
    elif opcode == Opcode.RET:
        return f"RET"
    elif opcode == Opcode.IN:
        return f"IN R{rt}, 0x{imm:04X}"
    elif opcode == Opcode.OUT:
        return f"OUT R{rs}, 0x{imm:04X}"
    elif opcode == Opcode.LOADI:
        return f"LOADI R{rt}, {imm}"
    elif opcode == Opcode.LUI:
        return f"LUI R{rt}, 0x{imm:04X}"
    elif opcode == Opcode.ORI:
        return f"ORI R{rt}, R{rs}, 0x{imm:04X}"
    else:
        return opcode.name


class Instruction:
    """Представление машинной инструкции"""

    def __init__(self, opcode: Opcode, rs=0, rt=0, rd=0, imm=0, addr=0, is_label=False):
        self.opcode = opcode
        self.rs = rs
        self.rt = rt
        self.rd = rd
        self.imm = imm
        self.addr = addr
        self.is_label = is_label

    def to_binary(self) -> bytes:
        """ИСПРАВЛЕННАЯ упаковка в 32-битный формат"""

        def safe_uint32(value):
            return int(value) & 0xFFFFFFFF

        def safe_uint16(value):
            return int(value) & 0xFFFF

        def safe_uint21(value):
            return int(value) & 0x1FFFFF

        def safe_uint26(value):
            return int(value) & 0x3FFFFFF
        # J-type для переходов
        if self.opcode in [Opcode.JMP, Opcode.CALL]: # <-- ДОБАВЛЕН CALL
            opcode_bits = (self.opcode.value & 0x3F) << 26
            addr_bits = safe_uint26(self.addr)
            word = opcode_bits | addr_bits
        # L-type для LOADI (21-битный immediate)
        elif self.opcode in [Opcode.LOADI]:
            opcode_bits = (self.opcode.value & 0x3F) << 26  # биты 26-31
            rt_bits = (self.rt & 0x1F) << 21  # биты 21-25 ← ИСПРАВЛЕНО!
            imm_bits = safe_uint21(self.imm)  # биты 0-20
            word = opcode_bits | rt_bits | imm_bits
        # U-type для LUI (16-битный immediate в старшие биты)
        elif self.opcode in [Opcode.LUI]:
            opcode_bits = (self.opcode.value & 0x3F) << 26
            rt_bits = (self.rt & 0x1F) << 21
            imm_bits = safe_uint16(self.imm)
            word = opcode_bits | rt_bits | imm_bits
        # I-type для команд с immediate
        elif self.opcode in [Opcode.LOAD, Opcode.STORE, Opcode.JZ, Opcode.IN, Opcode.OUT, Opcode.ORI]:
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
                             Opcode.AND, Opcode.OR, Opcode.XOR, Opcode.CMP, Opcode.SHL, Opcode.SHR]:
            mnemonic = f"{self.opcode.name} R{self.rd}, R{self.rs}, R{self.rt}"
        elif self.opcode == Opcode.PUSH:
            mnemonic = f"PUSH R{self.rs}"
        elif self.opcode == Opcode.POP:
            mnemonic = f"POP R{self.rt}"
        elif self.opcode == Opcode.JZ:
            mnemonic = f"JZ R{self.rs}, 0x{self.imm:04X}"
        elif self.opcode == Opcode.JMP:
            mnemonic = f"JMP 0x{self.addr:04X}"
        elif self.opcode == Opcode.CALL: # <-- НОВОЕ
            mnemonic = f"CALL 0x{self.addr:04X}"
        elif self.opcode == Opcode.RET: # <-- НОВОЕ
            mnemonic = "RET"
        elif self.opcode == Opcode.IN:
            mnemonic = f"IN R{self.rt}, 0x{self.imm:04X}"
        elif self.opcode == Opcode.OUT:
            mnemonic = f"OUT R{self.rs}, 0x{self.imm:04X}"
        elif self.opcode == Opcode.LOADI:
            mnemonic = f"LOADI R{self.rt}, {self.imm}"
        elif self.opcode == Opcode.LUI:
            mnemonic = f"LUI R{self.rt}, 0x{self.imm:04X}"
        elif self.opcode == Opcode.ORI:
            mnemonic = f"ORI R{self.rt}, R{self.rs}, 0x{self.imm:04X}"
        else:
            mnemonic = self.opcode.name

        return f"0x{addr:04X}: {hex_code}  {mnemonic}"

    def __repr__(self):
        return f"Instruction({self.opcode.name}, rs={self.rs}, rt={self.rt}, rd={self.rd}, imm={self.imm}, addr={self.addr})"