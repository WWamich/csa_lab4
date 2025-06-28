\ Тест, демонстрирующий влияние кэш-памяти на производительность
\ Оптимизирован для cache_size = 32 слова. Размер массива = 64 слова.

\ ======================================================================
\ --- Утилиты для вывода ---
\ ======================================================================
: CR 10 EMIT ;
: DIGIT>CHAR 48 + ;
: U.SIMPLE
    DUP 10 < IF
        DIGIT>CHAR EMIT
    ELSE
        10 /MOD SWAP U.SIMPLE DIGIT>CHAR EMIT
    THEN ;


VARIABLE I_VAR
VARIABLE SUM_VAR
VARIABLE ARRAY-SIZE-VAR

\ Создаем массив в 2 раза больше размера кеша
CREATE BIG-ARRAY ALLOT 64


: FILL-ARRAY
    0 I_VAR !
    BEGIN
        I_VAR @ ARRAY-SIZE-VAR @ <
    WHILE
        I_VAR @ 1+
        BIG-ARRAY I_VAR @ + !
        I_VAR @ 1+ I_VAR !
    REPEAT
;

: SUM-ARRAY
    0 SUM_VAR !
    0 I_VAR !
    BEGIN
        I_VAR @ ARRAY-SIZE-VAR @ <
    WHILE
        SUM_VAR @ BIG-ARRAY I_VAR @ + @ +
        SUM_VAR !
        I_VAR @ 1+ I_VAR !
    REPEAT
;

\ Устанавливаем размер массива в переменную
64 ARRAY-SIZE-VAR !

FILL-ARRAY

SUM-ARRAY
S" Sum (Cold Cache): " TYPE SUM_VAR @ U.SIMPLE CR
SUM-ARRAY
S" Sum (Warm Cache): " TYPE SUM_VAR @ U.SIMPLE CR

HALT