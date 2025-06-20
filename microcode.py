from enum import Enum
from typing import List, Dict

class MicroOp(Enum):
    """Микрооперации"""
    NOP = "nop"
    HALT = "halt"
    PC_INC = "pc_inc"
    PC_LOAD = "pc_load"
    PC_LOAD_TEMP = "pc_load_temp"
    ALU_ADD = "alu_add"
    ALU_SUB = "alu_sub"
    ALU_MUL = "alu_mul"
    ALU_DIV = "alu_div"
    ALU_MOD = "alu_mod"
    ALU_AND = "alu_and"
    ALU_OR = "alu_or"
    ALU_XOR = "alu_xor"
    ALU_CMP = "alu_cmp"
    ALU_SHL = "alu_shl"
    ALU_SHR = "alu_shr"
    ALU_PC_ADD = "alu_pc_add"
    CACHE_CHECK = "cache_check"
    CACHE_WAIT = "cache_wait"
    MEM_READ = "mem_read"
    MEM_WRITE = "mem_write"
    STACK_PUSH = "stack_push"
    STACK_POP = "stack_pop"
    JUMP_COND = "jump_cond"
    IO_READ = "io_read"
    IO_WRITE = "io_write"
    LOAD_IMM = "load_imm"
    SHIFT_LEFT = "shift_left"

class MicroInstruction:
    """Микроинструкция"""
    def __init__(self, op: MicroOp, src=None, dst=None, imm=None, condition=None):
        self.op = op
        self.src = src
        self.dst = dst
        self.imm = imm
        self.condition = condition

    def __repr__(self):
        parts = [f"μ({self.op.value}"]
        if self.src is not None: parts.append(f"src={self.src}")
        if self.dst is not None: parts.append(f"dst={self.dst}")
        if self.imm is not None: parts.append(f"imm={self.imm}")
        if self.condition is not None: parts.append(f"cond={self.condition}")
        return ", ".join(parts) + ")"

class SimpleCache:
    """Простой кэш"""
    def __init__(self, size: int = 16):
        self.size = size
        self.data: Dict[int, int] = {}
        self.access_order: List[int] = []
        self.hits = 0
        self.misses = 0

    def access(self, addr: int) -> tuple[bool, int]:
        if addr in self.data:
            self.hits += 1
            self.access_order.remove(addr)
            self.access_order.append(addr)
            return True, self.data[addr]
        else:
            self.misses += 1
            return False, 0

    def write(self, addr: int, data: int):
        if len(self.data) >= self.size and addr not in self.data:
            oldest = self.access_order.pop(0)
            del self.data[oldest]
        self.data[addr] = data
        if addr in self.access_order:
            self.access_order.remove(addr)
        self.access_order.append(addr)

    def get_stats(self) -> dict:
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0.0
        return {"hits": self.hits, "misses": self.misses, "hit_rate": hit_rate}

MICROCODE: Dict[str, List[MicroInstruction]] = {
    "NOP": [ MicroInstruction(MicroOp.PC_INC) ],
    "HALT": [ MicroInstruction(MicroOp.HALT) ],
    "LOADI": [ MicroInstruction(MicroOp.LOAD_IMM, dst="rt", imm="imm"), MicroInstruction(MicroOp.PC_INC) ],
    "LUI": [ MicroInstruction(MicroOp.SHIFT_LEFT, dst="rt", src=16, imm="imm"), MicroInstruction(MicroOp.PC_INC) ],
    "ORI": [ MicroInstruction(MicroOp.ALU_OR, src="rs", dst="rt", imm="imm"), MicroInstruction(MicroOp.PC_INC) ],
    "PUSH": [ MicroInstruction(MicroOp.STACK_PUSH, src="rs"), MicroInstruction(MicroOp.PC_INC) ],
    "POP": [ MicroInstruction(MicroOp.STACK_POP, dst="rt"), MicroInstruction(MicroOp.PC_INC) ],
    "ADD": [ MicroInstruction(MicroOp.ALU_ADD, src="rs", dst="rd", imm="rt"), MicroInstruction(MicroOp.PC_INC) ],
    "SUB": [ MicroInstruction(MicroOp.ALU_SUB, src="rs", dst="rd", imm="rt"), MicroInstruction(MicroOp.PC_INC) ],
    "MUL": [ MicroInstruction(MicroOp.ALU_MUL, src="rs", dst="rd", imm="rt"), MicroInstruction(MicroOp.PC_INC) ],
    "DIV": [ MicroInstruction(MicroOp.ALU_DIV, src="rs", dst="rd", imm="rt"), MicroInstruction(MicroOp.PC_INC) ],
    "MOD": [ MicroInstruction(MicroOp.ALU_MOD, src="rs", dst="rd", imm="rt"), MicroInstruction(MicroOp.PC_INC) ],
    "AND": [ MicroInstruction(MicroOp.ALU_AND, src="rs", dst="rd", imm="rt"), MicroInstruction(MicroOp.PC_INC) ],
    "OR": [ MicroInstruction(MicroOp.ALU_OR, src="rs", dst="rd", imm="rt"), MicroInstruction(MicroOp.PC_INC) ],
    "XOR": [ MicroInstruction(MicroOp.ALU_XOR, src="rs", dst="rd", imm="rt"), MicroInstruction(MicroOp.PC_INC) ],
    "CMP": [ MicroInstruction(MicroOp.ALU_CMP, src="rs", dst="rd", imm="rt"), MicroInstruction(MicroOp.PC_INC) ],
    "SHL": [ MicroInstruction(MicroOp.ALU_SHL, src="rs", dst="rd", imm="rt"), MicroInstruction(MicroOp.PC_INC) ],
    "SHR": [ MicroInstruction(MicroOp.ALU_SHR, src="rs", dst="rd", imm="rt"), MicroInstruction(MicroOp.PC_INC) ],
    "LOAD": [ MicroInstruction(MicroOp.ALU_ADD, src="rs", dst="TEMP_ADDR", imm="imm"), MicroInstruction(MicroOp.MEM_READ), MicroInstruction(MicroOp.PC_INC) ],
    "STORE": [ MicroInstruction(MicroOp.ALU_ADD, src="rt", dst="TEMP_ADDR", imm="imm"), MicroInstruction(MicroOp.MEM_WRITE), MicroInstruction(MicroOp.PC_INC) ],
    "JMP": [ MicroInstruction(MicroOp.PC_LOAD) ],
    "JZ": [ MicroInstruction(MicroOp.JUMP_COND, condition="zero") ],
    "IN": [ MicroInstruction(MicroOp.IO_READ, dst="rt", imm="imm"), MicroInstruction(MicroOp.PC_INC) ],
    "OUT": [ MicroInstruction(MicroOp.IO_WRITE, src="rs", imm="imm"), MicroInstruction(MicroOp.PC_INC) ],
    "CALL": [
        MicroInstruction(MicroOp.ALU_PC_ADD),
        MicroInstruction(MicroOp.STACK_PUSH, src="TEMP_ADDR"),
        MicroInstruction(MicroOp.PC_LOAD),
    ],
    "RET": [
        MicroInstruction(MicroOp.STACK_POP, dst="TEMP_ADDR"),
        MicroInstruction(MicroOp.PC_LOAD_TEMP),
    ]
}

def get_microcode(opcode) -> List[MicroInstruction]:
    opcode_name = opcode.name if hasattr(opcode, 'name') else str(opcode)
    return MICROCODE.get(opcode_name, [MicroInstruction(MicroOp.NOP)])