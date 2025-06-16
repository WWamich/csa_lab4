from enum import Enum
from typing import List
from isa import Opcode

class MicroOp(Enum):
    """Микроинструкции для микрокодного блока управления"""
    # Управление регистрами
    REG_LOAD = "reg_load"  # загрузить в регистр
    REG_STORE = "reg_store"  # сохранить из регистра
    REG_COPY = "reg_copy"  # копировать регистр в регистр

    # Операции АЛУ
    ALU_ADD = "alu_add"  # сложение
    ALU_SUB = "alu_sub"  # вычитание
    ALU_MUL = "alu_mul"  # умножение
    ALU_DIV = "alu_div"  # деление
    ALU_MOD = "alu_mod"  # остаток
    ALU_AND = "alu_and"  # битовое И
    ALU_OR = "alu_or"  # битовое ИЛИ
    ALU_XOR = "alu_xor"  # битовое исключающее ИЛИ
    ALU_CMP = "alu_cmp"  # сравнение

    # Управление памятью/кэшем
    MEM_READ = "mem_read"  # чтение из памяти
    MEM_WRITE = "mem_write"  # запись в память
    CACHE_CHECK = "cache_check"  # проверка кэша
    CACHE_WAIT = "cache_wait"  # ожидание кэша

    # Управление стеком
    STACK_PUSH = "stack_push"  # положить на стек
    STACK_POP = "stack_pop"  # снять со стека
    STACK_PEEK = "stack_peek"  # прочитать со стека без удаления

    # Управление потоком
    PC_INC = "pc_inc"  # PC++
    PC_LOAD = "pc_load"  # загрузить PC
    JUMP = "jump"  # переход
    JUMP_COND = "jump_cond"  # условный переход

    # I/O операции
    IO_READ = "io_read"  # чтение с порта
    IO_WRITE = "io_write"  # запись в порт

    # Управляющие сигналы
    NOP = "nop"  # нет операции
    HALT = "halt"  # остановка
    FETCH = "fetch"  # выборка инструкции


class MicroInstruction:
    """Одна микроинструкция"""

    def __init__(self, op: MicroOp, src=None, dst=None, imm=None, condition=None):
        self.op = op
        self.src = src  # источник (регистр/адрес)
        self.dst = dst  # назначение (регистр/адрес)
        self.imm = imm  # immediate значение
        self.condition = condition  # условие для условных переходов

    def __repr__(self):
        parts = [self.op.value]
        if self.src:
            parts.append(f"src={self.src}")
        if self.dst:
            parts.append(f"dst={self.dst}")
        if self.imm is not None:
            parts.append(f"imm={self.imm}")
        if self.condition:
            parts.append(f"cond={self.condition}")
        return f"μ({', '.join(parts)})"


class SimpleCache:
    """Простой кэш с прямым отображением"""

    def __init__(self, size=16):
        self.size = size
        self.valid = [False] * size
        self.tags = [0] * size
        self.data = [0] * size

        # Статистика
        self.hits = 0
        self.misses = 0

    def get_index(self, addr: int) -> int:
        """Получить индекс в кэше по адресу"""
        return (addr // 4) % self.size  # word-aligned доступ

    def get_tag(self, addr: int) -> int:
        """Получить тег по адресу"""
        return addr // (self.size * 4)

    def access(self, addr: int) -> tuple[bool, int]:
        """
        Обратиться к кэшу
        Возвращает: (hit, data_or_cycles)
        hit=True: попадание, data_or_cycles = данные
        hit=False: промах, data_or_cycles = количество тактов ожидания
        """
        index = self.get_index(addr)
        tag = self.get_tag(addr)

        # Проверяем попадание
        if self.valid[index] and self.tags[index] == tag:
            self.hits += 1
            return True, self.data[index]
        else:
            # Промах - загружаем из "памяти"
            self.misses += 1
            self.valid[index] = True
            self.tags[index] = tag
            self.data[index] = addr & 0xFFFF  # данные из "памяти"
            return False, 10  # 10 тактов ожидания

    def write(self, addr: int, value: int):
        """Записать в кэш"""
        index = self.get_index(addr)
        tag = self.get_tag(addr)

        self.valid[index] = True
        self.tags[index] = tag
        self.data[index] = value

    def get_stats(self) -> dict:
        """Получить статистику кэша"""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "total_accesses": total
        }


# Импортируем из translator.py чтобы избежать дублирования

# Микрокод для каждой RISC команды
MICROCODE = {
    # NOP - ничего не делать
    Opcode.NOP: [
        MicroInstruction(MicroOp.NOP),
        MicroInstruction(MicroOp.PC_INC)
    ],

    # HALT - остановка
    Opcode.HALT: [
        MicroInstruction(MicroOp.HALT)
    ],

    # LOAD rs, rt, imm - загрузить из памяти
    Opcode.LOAD: [
        # μ1: вычислить адрес = rs + imm
        MicroInstruction(MicroOp.ALU_ADD, src="rs", dst="TEMP_ADDR", imm="imm"),
        # μ2: проверить кэш
        MicroInstruction(MicroOp.CACHE_CHECK, src="TEMP_ADDR"),
        # μ3: если промах - ждать 10 тактов
        MicroInstruction(MicroOp.CACHE_WAIT),
        # μ4: загрузить данные в rt
        MicroInstruction(MicroOp.MEM_READ, src="TEMP_ADDR", dst="rt"),
        # μ5: PC++
        MicroInstruction(MicroOp.PC_INC)
    ],

    # STORE rs, rt, imm - сохранить в память
    Opcode.STORE: [
        # μ1: вычислить адрес = rt + imm
        MicroInstruction(MicroOp.ALU_ADD, src="rt", dst="TEMP_ADDR", imm="imm"),
        # μ2: проверить кэш
        MicroInstruction(MicroOp.CACHE_CHECK, src="TEMP_ADDR"),
        # μ3: если промах - ждать 10 тактов
        MicroInstruction(MicroOp.CACHE_WAIT),
        # μ4: сохранить rs в память
        MicroInstruction(MicroOp.MEM_WRITE, src="rs", dst="TEMP_ADDR"),
        # μ5: PC++
        MicroInstruction(MicroOp.PC_INC)
    ],

    # PUSH rs - положить на стек
    Opcode.PUSH: [
        # μ1: memory[SP] = rs
        MicroInstruction(MicroOp.STACK_PUSH, src="rs"),
        # μ2: SP++
        MicroInstruction(MicroOp.ALU_ADD, src="SP", dst="SP", imm=1),
        # μ3: PC++
        MicroInstruction(MicroOp.PC_INC)
    ],

    # POP rt - снять со стека
    Opcode.POP: [
        # μ1: SP--
        MicroInstruction(MicroOp.ALU_SUB, src="SP", dst="SP", imm=1),
        # μ2: rt = memory[SP]
        MicroInstruction(MicroOp.STACK_POP, dst="rt"),
        # μ3: PC++
        MicroInstruction(MicroOp.PC_INC)
    ],

    # ADD rs, rt, rd - сложение
    Opcode.ADD: [
        # μ1: rd = rs + rt
        MicroInstruction(MicroOp.ALU_ADD, src="rs", dst="rd", imm="rt"),
        # μ2: PC++
        MicroInstruction(MicroOp.PC_INC)
    ],

    # SUB rs, rt, rd - вычитание
    Opcode.SUB: [
        # μ1: rd = rs - rt
        MicroInstruction(MicroOp.ALU_SUB, src="rs", dst="rd", imm="rt"),
        # μ2: PC++
        MicroInstruction(MicroOp.PC_INC)
    ],

    # MUL rs, rt, rd - умножение
    Opcode.MUL: [
        # μ1: rd = rs * rt
        MicroInstruction(MicroOp.ALU_MUL, src="rs", dst="rd", imm="rt"),
        # μ2: PC++
        MicroInstruction(MicroOp.PC_INC)
    ],

    # DIV rs, rt, rd - деление
    Opcode.DIV: [
        # μ1: rd = rs / rt
        MicroInstruction(MicroOp.ALU_DIV, src="rs", dst="rd", imm="rt"),
        # μ2: PC++
        MicroInstruction(MicroOp.PC_INC)
    ],

    # MOD rs, rt, rd - остаток от деления
    Opcode.MOD: [
        # μ1: rd = rs % rt
        MicroInstruction(MicroOp.ALU_MOD, src="rs", dst="rd", imm="rt"),
        # μ2: PC++
        MicroInstruction(MicroOp.PC_INC)
    ],

    # AND rs, rt, rd - битовое И
    Opcode.AND: [
        # μ1: rd = rs & rt
        MicroInstruction(MicroOp.ALU_AND, src="rs", dst="rd", imm="rt"),
        # μ2: PC++
        MicroInstruction(MicroOp.PC_INC)
    ],

    # OR rs, rt, rd - битовое ИЛИ
    Opcode.OR: [
        # μ1: rd = rs | rt
        MicroInstruction(MicroOp.ALU_OR, src="rs", dst="rd", imm="rt"),
        # μ2: PC++
        MicroInstruction(MicroOp.PC_INC)
    ],

    # XOR rs, rt, rd - битовое исключающее ИЛИ
    Opcode.XOR: [
        # μ1: rd = rs ^ rt
        MicroInstruction(MicroOp.ALU_XOR, src="rs", dst="rd", imm="rt"),
        # μ2: PC++
        MicroInstruction(MicroOp.PC_INC)
    ],

    # CMP rs, rt, rd - сравнение (==)
    Opcode.CMP: [
        # μ1: rd = (rs == rt) ? 1 : 0
        MicroInstruction(MicroOp.ALU_CMP, src="rs", dst="rd", imm="rt"),
        # μ2: PC++
        MicroInstruction(MicroOp.PC_INC)
    ],

    # JMP addr - безусловный переход
    Opcode.JMP: [
        # μ1: PC = addr
        MicroInstruction(MicroOp.PC_LOAD, imm="addr")
    ],

    # JZ rs, imm - условный переход
    Opcode.JZ: [
        # μ1: если rs == 0, то PC = imm, иначе PC++
        MicroInstruction(MicroOp.JUMP_COND, src="rs", imm="imm", condition="zero"),
        # μ2: PC++ (если переход не выполнен)
        MicroInstruction(MicroOp.PC_INC)
    ],

    # IN rt, imm - ввод с порта
    Opcode.IN: [
        # μ1: rt = input[port]
        MicroInstruction(MicroOp.IO_READ, dst="rt", imm="imm"),
        # μ2: PC++
        MicroInstruction(MicroOp.PC_INC)
    ],

    # OUT rs, imm - вывод в порт
    Opcode.OUT: [
        # μ1: output[port] = rs
        MicroInstruction(MicroOp.IO_WRITE, src="rs", imm="imm"),
        # μ2: PC++
        MicroInstruction(MicroOp.PC_INC)
    ]
}


def get_microcode(opcode: Opcode) -> List[MicroInstruction]:
    """Получить микрокод для инструкции"""
    return MICROCODE.get(opcode, [MicroInstruction(MicroOp.NOP)])


def print_microcode_summary():
    """Вывести сводку по микрокоду"""
    print("📋 МИКРОКОД RISC ПРОЦЕССОРА")
    print("=" * 50)

    total_micro_ops = 0
    for opcode, micro_instructions in MICROCODE.items():
        print(f"{opcode.name:6} : {len(micro_instructions)} μ-ops")
        for i, micro_op in enumerate(micro_instructions, 1):
            print(f"      μ{i}: {micro_op}")
        total_micro_ops += len(micro_instructions)
        print()

    print(f"📊 Всего RISC команд: {len(MICROCODE)}")
    print(f"🔧 Всего микроопераций: {total_micro_ops}")
    print(f"📈 Среднее μ-ops на команду: {total_micro_ops / len(MICROCODE):.1f}")


if __name__ == "__main__":
    # Тестирование микрокода
    print_microcode_summary()

    # Пример использования кэша - лучший тест
    print("\n🧪 ТЕСТ КЭША")
    print("=" * 30)
    cache = SimpleCache(size=4)

    # Серия обращений к памяти - теперь с попаданиями
    addresses = [0x1000, 0x1004, 0x1008, 0x100C, 0x1000, 0x1004, 0x1010, 0x1000]

    for addr in addresses:
        hit, result = cache.access(addr)
        status = "HIT " if hit else "MISS"
        cycles = 1 if hit else result
        print(f"Обращение к 0x{addr:04X}: {status} ({cycles} тактов)")

    stats = cache.get_stats()
    print(f"\n📊 Статистика кэша:")
    print(f"Попадания: {stats['hits']}")
    print(f"Промахи: {stats['misses']}")
    print(f"Коэффициент попаданий: {stats['hit_rate']:.2%}")