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
    ZERO = 0
    SP = 1
    T1 = 2
    T2 = 3
    A0 = 4
    A1 = 5
    V0 = 6
    V1 = 7

MEMORY_SIZE = 65536
IO_INPUT_PORT = 0x8000
IO_OUTPUT_PORT = 0x8001


class Instruction:
    """Представление машинной инструкции"""
    def __init__(self, opcode: Opcode, rs=0, rt=0, rd=0, imm=0, addr=0, is_label=False, label_name=""):
        self.opcode = opcode
        self.rs = rs
        self.rt = rt
        self.rd = rd
        self.imm = imm
        self.addr = addr
        self.is_label = is_label
        self.label_name = label_name

    def to_binary(self) -> bytes:
        """ИСПРАВЛЕННАЯ упаковка в 32-битный формат с учетом всех типов"""
        opcode_bits = (self.opcode.value & 0x3F) << 26
        word = 0

        # J-type (JMP, CALL) - 26 бит на адрес
        if self.opcode in [Opcode.JMP, Opcode.CALL]:
            addr_bits = int(self.addr) & 0x3FFFFFF
            word = opcode_bits | addr_bits
        # I-type (LOAD, STORE, JZ, IN, OUT, ORI) - 16 бит на imm
        elif self.opcode in [Opcode.LOAD, Opcode.STORE, Opcode.JZ, Opcode.IN, Opcode.OUT, Opcode.ORI]:
            rs_bits = (self.rs & 0x1F) << 21
            rt_bits = (self.rt & 0x1F) << 16
            imm_bits = int(self.imm) & 0xFFFF
            word = opcode_bits | rs_bits | rt_bits | imm_bits
        # L-type (LOADI) - 21 бит на imm
        elif self.opcode == Opcode.LOADI:
            rt_bits = (self.rt & 0x1F) << 21
            imm_bits = int(self.imm) & 0x1FFFFF
            word = opcode_bits | rt_bits | imm_bits
        # U-type (LUI)
        elif self.opcode == Opcode.LUI:
            rt_bits = (self.rt & 0x1F) << 16
            imm_bits = int(self.imm) & 0xFFFF
            word = opcode_bits | rt_bits | imm_bits
        # R-type (остальные)
        else:
            rs_bits = (self.rs & 0x1F) << 21
            rt_bits = (self.rt & 0x1F) << 16
            rd_bits = (self.rd & 0x1F) << 11
            word = opcode_bits | rs_bits | rt_bits | rd_bits

        return struct.pack('>I', word)

    def to_hex(self, addr: int) -> str:
        hex_code = self.to_binary().hex().upper()
        mnemonic = self.opcode.name
        if self.opcode in [Opcode.ADD, Opcode.SUB, Opcode.MUL, Opcode.DIV, Opcode.MOD, Opcode.AND, Opcode.OR, Opcode.XOR, Opcode.CMP, Opcode.SHL, Opcode.SHR, Opcode.RET]:
             mnemonic = f"{self.opcode.name} R{self.rd}, R{self.rs}, R{self.rt}"
        elif self.opcode in [Opcode.LOAD, Opcode.STORE]:
            mnemonic = f"{self.opcode.name} R{self.rt}, {self.imm}(R{self.rs})"
        elif self.opcode in [Opcode.JMP, Opcode.CALL]:
            mnemonic = f"{self.opcode.name} 0x{self.addr:04X}"
        elif self.opcode == Opcode.JZ:
            mnemonic = f"JZ R{self.rs}, 0x{self.imm:04X}"
        elif self.opcode in [Opcode.PUSH, Opcode.POP]:
            mnemonic = f"{self.opcode.name} R{self.rs or self.rt}"
        elif self.opcode == Opcode.LOADI:
            mnemonic = f"LOADI R{self.rt}, {self.imm}"
        elif self.opcode == Opcode.LUI:
            mnemonic = f"LUI R{self.rt}, 0x{self.imm:04X}"
        elif self.opcode == Opcode.ORI:
            mnemonic = f"ORI R{self.rt}, R{self.rs}, {self.imm}"
        return f"0x{addr:04X}: {hex_code}  {mnemonic}"

    def __repr__(self):
        return f"Instruction({self.opcode.name}, rs={self.rs}, rt={self.rt}, rd={self.rd}, imm={self.imm}, addr={self.addr})"