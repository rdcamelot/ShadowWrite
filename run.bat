@echo off
setlocal
cd /d "%~dp0"
title ShadowWrite Local Server

echo [ShadowWrite] Starting local server...

py -3 --version >nul 2>nul
if not errorlevel 1 goto run_with_py

python --version >nul 2>nul
if not errorlevel 1 goto run_with_python

echo.
echo [ShadowWrite] Python 3 was not found.
echo Install Python 3.10 or newer, then run this file again.
pause
exit /b 1

:run_with_py
py -3 shadowwrite_server.py %*
set "EXIT_CODE=%ERRORLEVEL%"
goto finished

:run_with_python
python shadowwrite_server.py %*
set "EXIT_CODE=%ERRORLEVEL%"

:finished
if not "%EXIT_CODE%"=="0" (
  echo.
  echo [ShadowWrite] Server exited with code %EXIT_CODE%.
  pause
)
endlocal & exit /b %EXIT_CODE%
