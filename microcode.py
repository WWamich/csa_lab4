from __future__ import annotations

from enum import Enum, auto

from isa import Opcode


class MicroOp(Enum):
    """
    Микро-операции (микроинструкции), составляющие основу нашего микрокода.
    Каждая микро-операция выполняется за один такт (`tick`).
    Они представляют собой управляющие сигналы, которые активируют различные части DataPath.

    Соглашения по именованию:
    - LATCH_*: Защелкнуть значение в регистр.
    - ALU_*: Выполнить операцию в АЛУ.
    - MEM_*: Операция, связанная с памятью/кешем.
    - JUMP_*: Изменение счетчика микрокоманд (внутренние переходы в микрокоде).
    """

    # --- Управление потоком команд (PC) и выборка инструкций (Fetch) ---
    LATCH_PC_INC = auto()  # PC <- PC + 1
    LATCH_PC_ADDR = auto()  # PC <- IR.addr (для JMP, CALL)
    LATCH_PC_ALU = auto()  # PC <- ALU_OUT (для условных переходов)

    LATCH_MAR_PC = auto()  # MAR <- PC (Memory Address Register <- Program Counter)
    LATCH_IR = auto()  # IR <- MDR (Instruction Register <- Memory Data Register)

    # --- Декодирование и работа с регистрами ---
    # `DECODE` не является явной микрокомандой, а действием CU после LATCH_IR.
    LATCH_A_RS = auto()  # Внутренний регистр АЛУ 'A' <- GPR[rs]
    LATCH_A_RT = auto()  # Внутренний регистр АЛU 'A' <- GPR[rt]
    LATCH_A_MDR = auto()  # A <- MDR
    LATCH_A_SP = auto()  # Внутренний регистр АЛУ 'A' <- SP
    LATCH_A_PC = (
        auto()
    )  # Внутренний регистр АЛУ 'A' <- PC (для сохранения адреса возврата)

    LATCH_B_RT = auto()  # Внутренний регистр АЛУ 'B' <- GPR[rt]
    LATCH_B_IMM = auto()  # Внутренний регистр АЛУ 'B' <- IR.imm
    LATCH_B_CONST_1 = (
        auto()
    )  # Внутренний регистр АЛУ 'B' <- 1 (для инкремента/декремента SP)

    LATCH_RD_ALU = auto()  # GPR[rd] <- ALU_OUT
    LATCH_RT_ALU = auto()  # GPR[rt] <- ALU_OUT
    LATCH_RT_MDR = auto()  # GPR[rt] <- MDR (для LOAD)
    LATCH_SP_ALU = auto()  # SP <- ALU_OUT (для изменения указателя стека)

    # --- Операции АЛУ ---
    ALU_ADD = auto()  # ALU_OUT <- A + B
    ALU_SUB = auto()  # ALU_OUT <- A - B
    ALU_MUL = auto()  # ALU_OUT <- A * B
    ALU_DIV = auto()  # ALU_OUT <- A / B
    ALU_MOD = auto()  # ALU_OUT <- A % B
    ALU_OR = auto()  # ALU_OUT <- A | B
    ALU_AND = auto()  # ALU_OUT <- A & B
    ALU_XOR = auto()  # ALU_OUT <- A ^ B
    ALU_CMP = auto()  # ALU_OUT <- 1 if A == B else 0
    ALU_SHL = auto()  # ALU_OUT <- A << B
    ALU_SHR = auto()  # ALU_OUT <- A >> B
    ALU_LUI = auto()  # ALU_OUT <- B << 16 (B содержит imm)

    # --- Операции с памятью и кешем ---
    LATCH_MAR_ALU = auto()  # MAR <- ALU_OUT (для вычисленного адреса LOAD/STORE)
    LATCH_MDR_RT = auto()  # MDR <- GPR[rt] (для STORE)
    LATCH_MDR_A = auto()  # MDR <- A (для PUSH, A содержит PC или значение из регистра)

    CACHE_READ = auto()  # Сигнал на чтение из кеша/памяти. Может вызвать STALL.
    CACHE_WRITE = auto()  # Сигнал на запись в кеш/память. Может вызвать STALL.

    # --- Управление потоком микрокода ---
    BRANCH_IF_ZERO = auto()  # Переход, если ALU_OUT == 0 (используется в JZ)
    BRANCH_IF_NOT_ZERO = auto()  # Переход, если ALU_OUT != 0 (используется в JNZ)
    FINISH_INSTRUCTION = auto()  # Сигнал для CU, что выполнение инструкции завершено, можно начинать новую выборку.
    HALT_PROCESSOR = auto()


def get_microcode_rom() -> dict[Opcode, list[MicroOp]]:
    """
    Фабричная функция, создающая и возвращающая "ROM" с микрокодом.
    Ключ словаря - Opcode машинной инструкции, значение - список микро-операций.
    """
    # Общая последовательность для выборки любой инструкции (Fetch Cycle)
    fetch_cycle = [
        MicroOp.LATCH_MAR_PC,
        MicroOp.CACHE_READ,
        MicroOp.LATCH_PC_INC,
        MicroOp.LATCH_IR,
    ]

    microcode = {
        # --- Системные инструкции ---
        Opcode.NOP: [MicroOp.FINISH_INSTRUCTION],
        Opcode.HALT: [MicroOp.HALT_PROCESSOR],
        # --- R-type инструкции ---
        Opcode.ADD: [
            MicroOp.LATCH_A_RS,
            MicroOp.LATCH_B_RT,
            MicroOp.ALU_ADD,
            MicroOp.LATCH_RD_ALU,
            MicroOp.FINISH_INSTRUCTION,
        ],
        Opcode.SUB: [
            MicroOp.LATCH_A_RS,
            MicroOp.LATCH_B_RT,
            MicroOp.ALU_SUB,
            MicroOp.LATCH_RD_ALU,
            MicroOp.FINISH_INSTRUCTION,
        ],
        Opcode.MUL: [
            MicroOp.LATCH_A_RS,
            MicroOp.LATCH_B_RT,
            MicroOp.ALU_MUL,
            MicroOp.LATCH_RD_ALU,
            MicroOp.FINISH_INSTRUCTION,
        ],
        Opcode.DIV: [
            MicroOp.LATCH_A_RS,
            MicroOp.LATCH_B_RT,
            MicroOp.ALU_DIV,
            MicroOp.LATCH_RD_ALU,
            MicroOp.FINISH_INSTRUCTION,
        ],
        Opcode.MOD: [
            MicroOp.LATCH_A_RS,
            MicroOp.LATCH_B_RT,
            MicroOp.ALU_MOD,
            MicroOp.LATCH_RD_ALU,
            MicroOp.FINISH_INSTRUCTION,
        ],
        Opcode.OR: [
            MicroOp.LATCH_A_RS,
            MicroOp.LATCH_B_RT,
            MicroOp.ALU_OR,
            MicroOp.LATCH_RD_ALU,
            MicroOp.FINISH_INSTRUCTION,
        ],
        Opcode.AND: [
            MicroOp.LATCH_A_RS,
            MicroOp.LATCH_B_RT,
            MicroOp.ALU_AND,
            MicroOp.LATCH_RD_ALU,
            MicroOp.FINISH_INSTRUCTION,
        ],
        Opcode.XOR: [
            MicroOp.LATCH_A_RS,
            MicroOp.LATCH_B_RT,
            MicroOp.ALU_XOR,
            MicroOp.LATCH_RD_ALU,
            MicroOp.FINISH_INSTRUCTION,
        ],
        Opcode.CMP: [
            MicroOp.LATCH_A_RS,
            MicroOp.LATCH_B_RT,
            MicroOp.ALU_CMP,
            MicroOp.LATCH_RD_ALU,
            MicroOp.FINISH_INSTRUCTION,
        ],
        Opcode.SHL: [
            MicroOp.LATCH_A_RS,
            MicroOp.LATCH_B_RT,
            MicroOp.ALU_SHL,
            MicroOp.LATCH_RD_ALU,
            MicroOp.FINISH_INSTRUCTION,
        ],
        Opcode.SHR: [
            MicroOp.LATCH_A_RS,
            MicroOp.LATCH_B_RT,
            MicroOp.ALU_SHR,
            MicroOp.LATCH_RD_ALU,
            MicroOp.FINISH_INSTRUCTION,
        ],
        # --- I-type инструкции ---
        Opcode.ADDI: [
            MicroOp.LATCH_A_RS,
            MicroOp.LATCH_B_IMM,
            MicroOp.ALU_ADD,
            MicroOp.LATCH_RT_ALU,
            MicroOp.FINISH_INSTRUCTION,
        ],
        Opcode.ORI: [
            MicroOp.LATCH_A_RS,
            MicroOp.LATCH_B_IMM,
            MicroOp.ALU_OR,
            MicroOp.LATCH_RT_ALU,
            MicroOp.FINISH_INSTRUCTION,
        ],
        Opcode.LUI: [
            MicroOp.LATCH_B_IMM,
            MicroOp.ALU_LUI,
            MicroOp.LATCH_RT_ALU,
            MicroOp.FINISH_INSTRUCTION,
        ],
        Opcode.LOAD: [
            MicroOp.LATCH_A_RS,
            MicroOp.LATCH_B_IMM,
            MicroOp.ALU_ADD,
            MicroOp.LATCH_MAR_ALU,
            MicroOp.CACHE_READ,
            MicroOp.LATCH_RT_MDR,
            MicroOp.FINISH_INSTRUCTION,
        ],
        Opcode.STORE: [
            MicroOp.LATCH_A_RS,
            MicroOp.LATCH_B_IMM,
            MicroOp.ALU_ADD,
            MicroOp.LATCH_MAR_ALU,
            MicroOp.LATCH_MDR_RT,
            MicroOp.CACHE_WRITE,
            MicroOp.FINISH_INSTRUCTION,
        ],
        Opcode.JZ: [
            MicroOp.LATCH_A_RT,
            MicroOp.BRANCH_IF_ZERO,
            MicroOp.FINISH_INSTRUCTION,
        ],
        Opcode.JNZ: [
            MicroOp.LATCH_A_RT,
            MicroOp.BRANCH_IF_NOT_ZERO,
            MicroOp.FINISH_INSTRUCTION,
        ],
        # --- Стек ---
        # PUSH rs: SP <- SP - 1; mem[SP] <- GPR[rs]
        Opcode.PUSH: [
            MicroOp.LATCH_A_SP,
            MicroOp.LATCH_B_CONST_1,
            MicroOp.ALU_SUB,
            MicroOp.LATCH_SP_ALU,
            MicroOp.LATCH_MAR_ALU,
            MicroOp.LATCH_A_RS,
            MicroOp.LATCH_MDR_A,
            MicroOp.CACHE_WRITE,
            MicroOp.FINISH_INSTRUCTION,
        ],
        # POP rt: rt <- mem[SP]; SP <- SP + 1
        Opcode.POP: [
            MicroOp.LATCH_A_SP,
            MicroOp.LATCH_MAR_ALU,
            MicroOp.CACHE_READ,
            MicroOp.LATCH_A_SP,
            MicroOp.LATCH_B_CONST_1,
            MicroOp.ALU_ADD,
            MicroOp.LATCH_SP_ALU,
            MicroOp.LATCH_RT_MDR,
            MicroOp.FINISH_INSTRUCTION,
        ],
        # --- J-type и процедуры ---
        Opcode.JMP: [MicroOp.LATCH_PC_ADDR, MicroOp.FINISH_INSTRUCTION],
        Opcode.CALL: [
            MicroOp.LATCH_A_SP,
            MicroOp.LATCH_B_CONST_1,
            MicroOp.ALU_SUB,
            MicroOp.LATCH_SP_ALU,
            MicroOp.LATCH_MAR_ALU,
            MicroOp.LATCH_A_PC,
            MicroOp.LATCH_MDR_A,
            MicroOp.CACHE_WRITE,
            MicroOp.LATCH_PC_ADDR,
            MicroOp.FINISH_INSTRUCTION,
        ],
        Opcode.RET: [
            MicroOp.LATCH_A_SP,
            MicroOp.LATCH_MAR_ALU,
            MicroOp.CACHE_READ,
            MicroOp.LATCH_A_SP,
            MicroOp.LATCH_B_CONST_1,
            MicroOp.ALU_ADD,
            MicroOp.LATCH_SP_ALU,
            MicroOp.LATCH_A_MDR,
            MicroOp.LATCH_B_IMM,
            MicroOp.ALU_ADD,
            MicroOp.LATCH_PC_ALU,
            MicroOp.FINISH_INSTRUCTION,
        ],
    }
    full_microcode = {}
    for opcode, micro_ops in microcode.items():
        full_microcode[opcode] = fetch_cycle + micro_ops

    return full_microcode
