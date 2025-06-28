\ Задача №6 проекта "Эйлер"
\

\ ======================================================================
\ --- Утилиты для вывода  ---
\ ======================================================================
: CR 10 EMIT ;
: DIGIT>CHAR 48 + ;
: U.SIMPLE
    DUP 10 < IF
        DIGIT>CHAR EMIT
    ELSE
        10 /MOD SWAP U.SIMPLE DIGIT>CHAR EMIT
    THEN
;


VARIABLE I_VAR           \ Счетчик цикла
VARIABLE SUM             \ Для хранения (1 + 2 + ... + 100)
VARIABLE SUM-OF-SQUARES  \ Для хранения (1*1 + 2*2 + ... + 100*100)


: SOLVE-EULER-6
    0 SUM !
    0 SUM-OF-SQUARES !
    1 I_VAR !

    BEGIN
        I_VAR @ 101 <
    WHILE
        SUM @ I_VAR @ + SUM !

        I_VAR @ DUP *          \ ( i*i )
        SUM-OF-SQUARES @ +     \ ( sum_sq + i*i )
        SUM-OF-SQUARES !       \ Сохраняем

        I_VAR @ 1+ I_VAR !
    REPEAT

    SUM @ DUP *              \ ( square_of_sum )
    SUM-OF-SQUARES @         \ ( square_of_sum  sum_of_squares )
    -                        \ ( difference )

    U.SIMPLE
    CR
;

SOLVE-EULER-6
HALT