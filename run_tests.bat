@echo off
chcp 65001 >nul 2>&1
cd /d "%~dp0"
py run_all_tests.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo TESTS FAILED
) else (
    echo.
    echo ALL TESTS PASSED
)
pause
