
VARIABLE A_LOW
VARIABLE A_HIGH
VARIABLE B_LOW
VARIABLE B_HIGH
VARIABLE CARRY_VAR

\ Переменные для <U
VARIABLE UNSIGNED_MASK
VARIABLE VAL_A
VARIABLE VAL_B


: CR 10 EMIT ;
: DIGIT>CHAR 48 + ;
: U.SIMPLE
    DUP 10 < IF DIGIT>CHAR EMIT
    ELSE 10 /MOD SWAP U.SIMPLE DIGIT>CHAR EMIT
    THEN ;
: PRINT-NUMBER U.SIMPLE 32 EMIT ;

\ <U ( a b -- flag ) - Беззнаковое сравнение "меньше"
: <U
    VAL_B !
    VAL_A !
    VAL_A @ UNSIGNED_MASK @ XOR
    VAL_B @ UNSIGNED_MASK @ XOR
    < \ Выполняем обычное знаковое сравнение над измененными числами
;

\ D. ( high low -- ) - печать 64-битного числа
: D.
    SWAP
    S" High: " TYPE PRINT-NUMBER
    S" Low: " TYPE PRINT-NUMBER
    CR ;

\ D+ ( -- ) - складывает числа из переменных A и B,
\ результат оставляет на стеке ( low high )
: D+
    0 CARRY_VAR !
    A_LOW @ B_LOW @ +
    DUP A_LOW @
    <U IF                 \ ИСПОЛЬЗУЕМ БЕЗЗНАКОВОЕ СРАВНЕНИЕ
        1 CARRY_VAR !
    THEN
    A_HIGH @ B_HIGH @ + CARRY_VAR @ + \ -> sum_high
    A_LOW @ B_LOW @ + SWAP
;

\ Инициализация маски для <U
2147483648 UNSIGNED_MASK !  \ 0x80000000 - знаковый бит

S" --- Test 1: Addition with carry ---" TYPE CR
1 A_HIGH !
4294967295 A_LOW !
0 B_HIGH !
2 B_LOW !
D+ SWAP D. CR

S" --- Test 2: Simple addition ---" TYPE CR
0 A_HIGH !
4096 A_LOW !
0 B_HIGH !
8192 B_LOW !
D+ SWAP D. CR

HALT