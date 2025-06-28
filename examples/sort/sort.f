CREATE NUMBERS ALLOT 51  \ Массив: NUMBERS[0]=N, NUMBERS[1..N] = значения
VARIABLE I_VAR           \ Счетчик для внешнего цикла
VARIABLE J_VAR           \ Счетчик для внутреннего цикла
VARIABLE V_J             \ Временная переменная для array[j]
VARIABLE V_J1            \ Временная переменная для array[j+1]

: CR 10 EMIT ;
: DIGIT>CHAR 48 + ;
: U.SIMPLE DUP 10 < IF DIGIT>CHAR EMIT ELSE 10 /MOD SWAP U.SIMPLE DIGIT>CHAR EMIT THEN ;
: PRINT-NUMBER U.SIMPLE 32 EMIT ;

VARIABLE PARSE_ACCUM

: PARSE-INT \ читает число до пробела/перевода строки, оставляет его на стеке
    0 PARSE_ACCUM !
    BEGIN
        KEY DUP 48 <
    WHILE
        DROP
        KEY DUP 48 <
    REPEAT

    BEGIN
        DUP 48 >= OVER 57 <= AND
    WHILE
        48 -
        PARSE_ACCUM @ 10 * +
        PARSE_ACCUM !
        KEY
    REPEAT
    DROP
    PARSE_ACCUM @
;

: READ_ARRAY
    PARSE-INT NUMBERS !
    0 I_VAR !
    BEGIN
        I_VAR @ NUMBERS @ <
    WHILE
        NUMBERS I_VAR @ 1+ +
        PARSE-INT SWAP !
        I_VAR @ 1+ I_VAR !
    REPEAT
;

: PRINT_ARRAY
    0 I_VAR !
    BEGIN
        I_VAR @ NUMBERS @ <
    WHILE
        NUMBERS I_VAR @ 1+ + @ PRINT-NUMBER
        I_VAR @ 1+ I_VAR !
    REPEAT
    CR
;

: BUBBLE_SORT
    0 I_VAR !
    BEGIN                 \ Внешний цикл for(i=0; i < N-1; i++)
        I_VAR @ NUMBERS @ 1- <
    WHILE
        0 J_VAR !         \ Внутренний цикл for(j=0; j < N-i-1; j++)
        BEGIN
            J_VAR @ NUMBERS @ I_VAR @ - 1- <
        WHILE
            NUMBERS J_VAR @ 1+ + @  V_J !
            NUMBERS J_VAR @ 2+ + @  V_J1 !

            V_J @ V_J1 @ > IF
                V_J1 @  NUMBERS J_VAR @ 1+ + !
                V_J @   NUMBERS J_VAR @ 2+ + !
            THEN
            J_VAR @ 1+ J_VAR !
        REPEAT
        I_VAR @ 1+ I_VAR !
    REPEAT
;

READ_ARRAY
BUBBLE_SORT
PRINT_ARRAY
HALT