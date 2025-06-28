s" What is your name?" TYPE CR

s" Hello, " TYPE

BEGIN
    KEY DUP      \ ( char char ) -- читаем символ и дублируем
    10 <>        \ ( char flag ) -- сравниваем с ENTER, flag=true если НЕ ENTER
WHILE
    DUP EMIT     \ ( char ) -- печатаем символ
REPEAT

2DROP

s" !" TYPE
HALT