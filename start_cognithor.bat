@echo off
setlocal enabledelayedexpansion
title Cognithor Control Center
color 0F

:: ============================================================
::  COGNITHOR ONE-CLICK LAUNCHER
::  Prueft Abhaengigkeiten, bootstrappt beim ersten Start,
::  startet dann die Web-UI.
:: ============================================================

:: UTF-8 fuer Python-Output aktivieren
chcp 65001 >nul 2>&1

echo.
echo    ____  ___   ____ _   _ ___ _____ _   _  ___  ____
echo   / ___|/ _ \ / ___| \ | |_ _|_   _| | | |/ _ \|  _ \
echo  | |  _| | | | |  _|  \| || |  | | | |_| | | | | |_) |
echo  | |_| | |_| | |_| | |\  || |  | | |  _  | |_| |  _ ^<
echo   \____|\___/ \____|_| \_|___| |_| |_| |_|\___/|_| \_\
echo.

set "REPO_ROOT=%~dp0"
:: Trailing backslash entfernen
if "%REPO_ROOT:~-1%"=="\" set "REPO_ROOT=%REPO_ROOT:~0,-1%"

:: ============================================================
::  1. Python im PATH? (versucht python, dann py Launcher)
:: ============================================================
set "PYTHON_CMD="
where python >nul 2>&1
if not errorlevel 1 (
    :: Teste ob es echtes Python ist (nicht Microsoft Store Stub)
    python -c "import sys" >nul 2>&1
    if not errorlevel 1 (
        set "PYTHON_CMD=python"
    )
)

if "%PYTHON_CMD%"=="" (
    where py >nul 2>&1
    if not errorlevel 1 (
        py -c "import sys" >nul 2>&1
        if not errorlevel 1 (
            set "PYTHON_CMD=py"
        )
    )
)

if "%PYTHON_CMD%"=="" (
    echo   [FEHLER] Python wurde nicht gefunden!
    echo.
    echo   Bitte installiere Python 3.12+:
    echo   https://www.python.org/downloads/
    echo.
    echo   WICHTIG: Bei der Installation "Add Python to PATH" ankreuzen!
    echo.
    pause
    exit /b 1
)

:: ============================================================
::  2. Python ^>= 3.12?
:: ============================================================
%PYTHON_CMD% -c "import sys; exit(0 if sys.version_info >= (3,12) else 1)" 2>nul
if errorlevel 1 (
    echo   [FEHLER] Python 3.12 oder neuer wird benoetigt!
    echo.
    for /f "tokens=*" %%v in ('%PYTHON_CMD% --version 2^>^&1') do echo   Installiert: %%v
    echo.
    echo   Bitte upgrade Python:
    echo   https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)

:: ============================================================
::  3. Node.js im PATH?
:: ============================================================
where node >nul 2>&1
if errorlevel 1 (
    echo   [FEHLER] Node.js wurde nicht gefunden!
    echo.
    echo   Bitte installiere Node.js 18+:
    echo   https://nodejs.org/
    echo.
    pause
    exit /b 1
)

:: ============================================================
::  4. Bootstrap ausfuehren
:: ============================================================
echo   Bootstrap wird gestartet...
echo.

%PYTHON_CMD% "%REPO_ROOT%\scripts\bootstrap_windows.py" --repo-root "%REPO_ROOT%"
if errorlevel 1 (
    echo.
    echo   [FEHLER] Bootstrap fehlgeschlagen!
    echo   Bitte pruefe die Ausgabe oben fuer Details.
    echo.
    pause
    exit /b 1
)

:: ============================================================
::  5. Web-UI starten
:: ============================================================
echo.
echo   Web-UI wird gestartet...
echo.
cd /d "%REPO_ROOT%\ui"
npm run dev
pause
