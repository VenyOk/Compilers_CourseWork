@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

set OUTPUT_FILE=result.txt

if exist !OUTPUT_FILE! (
    del !OUTPUT_FILE!
)

powershell -Command "[System.IO.File]::WriteAllText('%OUTPUT_FILE%', '', [System.Text.Encoding]::UTF8)" >nul 2>&1

echo ========================================
echo   ПОЛНОЕ ТЕСТИРОВАНИЕ КОМПИЛЯТОРА FORTRAN
echo ========================================
echo.
echo Вывод сохраняется в файл: !OUTPUT_FILE!
echo.

set PASSED=0
set FAILED=0
set TOTAL=0

if exist build (
    echo Очистка папки build...
    rmdir /s /q build
)
mkdir build
mkdir build\ssa
mkdir build\llvm
mkdir build\exes

for %%f in (inputs\*.f) do (
    call :run_test "%%f" "%%~nf"
)
goto :end_tests

:run_test
set /a TOTAL+=1
set "TEST_FILE=%~1"
set "TEST_NAME=%~2"

(
echo ========================================
echo   ТЕСТ !TOTAL!: !TEST_NAME!
echo ========================================
echo.

echo [1/4] Лексический анализ и парсинг...
python -m src.main -f "!TEST_FILE!" -s "build\ssa\!TEST_NAME!.ssa" -l "build\llvm\!TEST_NAME!.ll"
) >> "!OUTPUT_FILE!" 2>&1
set COMPILE_RESULT=!errorlevel!

if !COMPILE_RESULT! neq 0 (
    (
    echo [ОШИБКА] Не удалось скомпилировать !TEST_FILE!
    echo.
    ) >> "!OUTPUT_FILE!" 2>&1
    set /a FAILED+=1
    goto :eof
)

(
echo.
echo [2/4] Генерация SSA формы...
) >> "!OUTPUT_FILE!" 2>&1
if not exist "build\ssa\!TEST_NAME!.ssa" (
    (
    echo [ОШИБКА] SSA файл не был создан
    echo.
    ) >> "!OUTPUT_FILE!" 2>&1
    set /a FAILED+=1
    goto :eof
)
(
echo [OK] SSA форма создана: build\ssa\!TEST_NAME!.ssa
echo.

echo [3/4] Генерация LLVM IR...
) >> "!OUTPUT_FILE!" 2>&1
if not exist "build\llvm\!TEST_NAME!.ll" (
    (
    echo [ОШИБКА] LLVM IR файл не был создан
    echo.
    ) >> "!OUTPUT_FILE!" 2>&1
    set /a FAILED+=1
    goto :eof
)
(
echo [OK] LLVM IR создан: build\llvm\!TEST_NAME!.ll
echo.

echo [4/4] Компиляция LLVM IR в исполняемый файл...
clang "build\llvm\!TEST_NAME!.ll" -o "build\exes\!TEST_NAME!.exe"
) >> "!OUTPUT_FILE!" 2>&1
set CLANG_RESULT=!errorlevel!

if !CLANG_RESULT! neq 0 (
    (
    echo [ОШИБКА] Clang не смог скомпилировать build\llvm\!TEST_NAME!.ll
    echo.
    ) >> "!OUTPUT_FILE!" 2>&1
    set /a FAILED+=1
    goto :eof
)

(
echo [OK] Исполняемый файл создан: build\exes\!TEST_NAME!.exe
echo.

echo Запуск программы...
echo ----------------------------------------
echo ВЫВОД ПРОГРАММЫ:
echo ----------------------------------------
"build\exes\!TEST_NAME!.exe"
echo ----------------------------------------
) >> "!OUTPUT_FILE!" 2>&1
set EXE_RESULT=!errorlevel!

(
if !EXE_RESULT! neq 0 (
    echo [ОШИБКА] Программа завершилась с кодом ошибки: !EXE_RESULT!
) else (
    echo [OK] Программа выполнена успешно
)
echo.
echo [OK] Тест !TEST_NAME! завершен
echo.
) >> "!OUTPUT_FILE!" 2>&1

if !EXE_RESULT! neq 0 (
    set /a FAILED+=1
) else (
    set /a PASSED+=1
)
goto :eof

:end_tests

(
echo ========================================
echo   ИТОГОВЫЕ РЕЗУЛЬТАТЫ
echo ========================================
echo Всего тестов: !TOTAL!
echo Успешно:      !PASSED!
echo Ошибок:       !FAILED!
echo ========================================
echo.

if !FAILED! gtr 0 (
    echo Некоторые тесты не прошли!
) else (
    echo Все тесты пройдены успешно!
)
) >> "!OUTPUT_FILE!" 2>&1

echo.
echo Результаты сохранены в: !OUTPUT_FILE!

if !FAILED! gtr 0 (
    exit /b 1
) else (
    exit /b 0
)

