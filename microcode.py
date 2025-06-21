from enum import Enum
from typing import List, Dict


class MicroOp(Enum):
    """Атомарные микрооперации, выполняемые за 1 такт."""

    # Общие
    NOP = "nop"
    HALT = "halt"

    # Управление PC
    PC_INC = "pc_inc"
    PC_LOAD_FROM_ADDR = "pc_load_from_addr"  # Загрузка PC из регистра адреса

    # Управление микро-PC (uPC)
    JUMP_COND = "jump_cond"  # Условный микро-переход

    # Регистры и АЛУ
    ALU_OP = "alu_op"          # Выполнить операцию АЛУ (add, sub, etc.)
    LATCH_REG = "latch_reg"    # Защёлкнуть значение в регистр
    LOAD_IMM = "load_imm"      # Загрузка непосредственного значения

    # Память и кэш
    CACHE_READ = "cache_read"      # Чтение из кэша
    CACHE_WRITE = "cache_write"    # Запись в кэш
    MEM_READ_START = "mem_read_start" # Начать чтение из основной памяти
    MEM_WRITE_START = "mem_write_start" # Начать запись в основную память
    AWAIT_MEMORY = "await_memory"  # Ожидание ответа от памяти (stall)

    # Ввод/Вывод
    IO_READ = "io_read"
    IO_WRITE = "io_write"


class MicroInstruction:
    """
    Микроинструкция. Представляет одну атомарную операцию.
    - op: тип микрооперации (MicroOp)
    - dst: целевой регистр/ресурс
    - src1, src2: исходные регистры/ресурсы
    - imm: непосредственное значение
    - alu_op: операция для АЛУ ('add', 'sub', 'or', 'and', ...)
    - condition: условие для JUMP_COND ('CACHE_HIT', 'ZERO', ...)
    - target_uPC: адрес для микро-перехода
    """
    def __init__(self, op: MicroOp, dst=None, src1=None, src2=None, imm=None, alu_op=None, condition=None, target_uPC=None):
        self.op = op
        self.dst = dst
        self.src1 = src1
        self.src2 = src2
        self.imm = imm
        self.alu_op = alu_op
        self.condition = condition
        self.target_uPC = target_uPC

    def __repr__(self):
        parts = [f"μ({self.op.value}"]
        if self.dst is not None: parts.append(f"dst={self.dst}")
        if self.src1 is not None: parts.append(f"src1={self.src1}")
        if self.src2 is not None: parts.append(f"src2={self.src2}")
        if self.imm is not None: parts.append(f"imm={self.imm}")
        if self.alu_op is not None: parts.append(f"alu_op={self.alu_op}")
        if self.condition is not None: parts.append(f"if({self.condition})->{self.target_uPC}")
        return ", ".join(parts) + ")"

SP = "SP"         # Stack Pointer
RT = "RT_REG"     # Рег-р временного хранения (для результата LOAD или операнда STORE)
RS = "RS_REG"     # Регистр-источник (базовый адрес)
RD = "RD_REG"     # Регистр-приемник (для арифметических операций)
ADDR = "ADDR_REG" # Регистр для хранения вычисленного адреса памяти
DATA = "DATA_REG" # Регистр для данных, читаемых/записываемых в память


# Каждая машинная инструкция реализуется последовательностью микроинструкций.
# target_uPC - это индекс в списке микроинструкций для текущей команды.


MICROCODE: Dict[str, List[MicroInstruction]] = {
    "NOP": [
        MicroInstruction(MicroOp.PC_INC),
    ],
    "HALT": [
        MicroInstruction(MicroOp.HALT),
    ],

    # --- Арифметические и логические операции ---
    # Пример для ADD: rt <- rs + rd
    "ADD": [
        MicroInstruction(MicroOp.ALU_OP, alu_op='pop', dst=RS),         # 1. Pop в rs
        MicroInstruction(MicroOp.ALU_OP, alu_op='pop', dst=RD),         # 2. Pop в rd
        MicroInstruction(MicroOp.ALU_OP, alu_op='add', src1=RS, src2=RD, dst=RT), # 3. rt = rs + rd
        MicroInstruction(MicroOp.ALU_OP, alu_op='push', src1=RT),       # 4. Push rt
        MicroInstruction(MicroOp.PC_INC),                               # 5. PC++
    ],
    # SUB, MUL, DIV и т.д. реализуются аналогично
    "SUB": [
        MicroInstruction(MicroOp.ALU_OP, alu_op='pop', dst=RS),
        MicroInstruction(MicroOp.ALU_OP, alu_op='pop', dst=RD),
        MicroInstruction(MicroOp.ALU_OP, alu_op='sub', src1=RS, src2=RD, dst=RT),
        MicroInstruction(MicroOp.ALU_OP, alu_op='push', src1=RT),
        MicroInstruction(MicroOp.PC_INC),
    ],

    # --- Работа с памятью ---
    # LOAD rt, imm(rs) -> rt = memory[rs + imm]
    # Forth-семантика: pop(addr), push(value)
    "LOAD": [
        MicroInstruction(MicroOp.ALU_OP, alu_op='pop', dst=ADDR),         # ADDR <- pop()
        MicroInstruction(MicroOp.CACHE_READ, src1=ADDR, dst=DATA),       # Попытка чтения из кэша
        MicroInstruction(MicroOp.JUMP_COND, condition="CACHE_HIT", target_uPC=5), # Если попали, идем на шаг 5
        MicroInstruction(MicroOp.MEM_READ_START, src1=ADDR),             # Начать чтение из памяти
        MicroInstruction(MicroOp.AWAIT_MEMORY),                          # Ждать 10 тактов
        MicroInstruction(MicroOp.LATCH_REG, src1="MEM_BUS", dst=DATA),   # DATA <- данные с шины памяти
        MicroInstruction(MicroOp.CACHE_WRITE, src1=ADDR, src2=DATA),     # Записать в кэш
        MicroInstruction(MicroOp.ALU_OP, alu_op='push', src1=DATA),      # push(DATA)
        MicroInstruction(MicroOp.PC_INC),                                # PC++
    ],

    # STORE rt, imm(rs) -> memory[rs + imm] = rt
    # Forth-семантика: pop(addr), pop(value), store
    "STORE": [
        MicroInstruction(MicroOp.ALU_OP, alu_op='pop', dst=ADDR),         # ADDR <- pop()
        MicroInstruction(MicroOp.ALU_OP, alu_op='pop', dst=DATA),         # DATA <- pop()
        MicroInstruction(MicroOp.CACHE_WRITE, src1=ADDR, src2=DATA),     # апись в кэш (write-through)
        MicroInstruction(MicroOp.MEM_WRITE_START, src1=ADDR, src2=DATA), # Начать запись в память (не блокирует)
        MicroInstruction(MicroOp.PC_INC),                                # PC++
    ],

    # PUSH/POP - это частные случаи STORE/LOAD, где адрес - указатель стека SP
    # PUSH rs -> memory[--SP] = rs
    "PUSH": [
        MicroInstruction(MicroOp.LATCH_REG, src1="INSTR_REG_RS", dst=DATA), # 1. DATA <- rs
        MicroInstruction(MicroOp.ALU_OP, alu_op='dec', src1=SP, dst=SP),
        MicroInstruction(MicroOp.CACHE_WRITE, src1=SP, src2=DATA),
        MicroInstruction(MicroOp.MEM_WRITE_START, src1=SP, src2=DATA),
        MicroInstruction(MicroOp.PC_INC),                                  # 5. PC++
    ],
    # POP rt -> rt = memory[SP++]
    "POP": [
        MicroInstruction(MicroOp.CACHE_READ, src1=SP, dst=DATA),
        MicroInstruction(MicroOp.JUMP_COND, condition="CACHE_HIT", target_uPC=4),
        MicroInstruction(MicroOp.MEM_READ_START, src1=SP),
        MicroInstruction(MicroOp.AWAIT_MEMORY),
        MicroInstruction(MicroOp.LATCH_REG, src1="MEM_BUS", dst=DATA),
        MicroInstruction(MicroOp.CACHE_WRITE, src1=SP, src2=DATA),
        MicroInstruction(MicroOp.LATCH_REG, dst="INSTR_REG_RT", src1=DATA), # rt <- DATA
        MicroInstruction(MicroOp.ALU_OP, alu_op='inc', src1=SP, dst=SP),    # SP++
        MicroInstruction(MicroOp.PC_INC),
    ],


    # --- Переходы ---
    # JMP target
    "JMP": [
        MicroInstruction(MicroOp.PC_LOAD_FROM_ADDR, src1="INSTR_TARGET"), # PC <- адрес из инструкции
    ],

    # JZ target -> if (zero_flag) PC = target
    "JZ": [
        MicroInstruction(MicroOp.ALU_OP, alu_op='pop', dst=RT),
        MicroInstruction(MicroOp.JUMP_COND, condition="NOT_ZERO", target_uPC=4),
        MicroInstruction(MicroOp.PC_LOAD_FROM_ADDR, src1="INSTR_TARGET"),
        MicroInstruction(MicroOp.JUMP_COND, target_uPC=5),
        MicroInstruction(MicroOp.PC_INC),
        MicroInstruction(MicroOp.NOP),
    ],

    # --- Ввод/Вывод ---
    "IN": [
        MicroInstruction(MicroOp.IO_READ, dst=RT),
        MicroInstruction(MicroOp.ALU_OP, alu_op='push', src1=RT),
        MicroInstruction(MicroOp.PC_INC),
    ],
    "OUT": [
        MicroInstruction(MicroOp.ALU_OP, alu_op='pop', dst=RT),
        MicroInstruction(MicroOp.IO_WRITE, src1=RT),
        MicroInstruction(MicroOp.PC_INC),
    ],
}

def get_microcode(opcode_name: str) -> List[MicroInstruction]:
    """Получить микрокод для инструкции по ее имени."""
    return MICROCODE.get(opcode_name, [MicroInstruction(MicroOp.NOP), MicroInstruction(MicroOp.PC_INC)])
