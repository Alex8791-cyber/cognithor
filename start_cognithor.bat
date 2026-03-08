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
echo   / ___^|/ _ \ / ___^| \ ^| ^|_ _^|_   _^| ^| ^| ^|/ _ \^|  _ \
echo  ^| ^|  _^| ^| ^| ^| ^|  _^|  \^| ^|^| ^|  ^| ^| ^| ^|_^| ^| ^| ^| ^| ^|_^) ^|
echo  ^| ^|_^| ^| ^|_^| ^| ^|_^| ^| ^|\  ^|^| ^|  ^| ^| ^|  _  ^| ^|_^| ^|  _ ^<
echo   \____^|\___/ \____^|_^| \_^|___^| ^|_^| ^|_^| ^|_^|\___/^|_^| \_\
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
    REM Teste ob es echtes Python ist, nicht Microsoft Store Stub
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
    echo   Python wurde nicht gefunden.
    echo.
    :: Pruefen ob winget verfuegbar ist
    where winget >nul 2>&1
    if errorlevel 1 goto :no_winget_python
    echo   Python 3.12 kann automatisch installiert werden.
    echo.
    CHOICE /C JN /M "  Python 3.12 jetzt installieren? (J=Ja, N=Nein)"
    if errorlevel 2 goto :no_winget_python
    echo.
    echo   Installiere Python 3.12 via winget...
    winget install --id Python.Python.3.12 -e --accept-source-agreements --accept-package-agreements
    if errorlevel 1 (
        echo.
        echo   [WARNUNG] winget-Installation fehlgeschlagen.
        goto :no_winget_python
    )
    echo.
    echo   Python installiert. Aktualisiere PATH...
    :: PATH aus Registry refreshen (delayed expansion noetig innerhalb Block)
    for /f "tokens=2*" %%A in ('reg query "HKCU\Environment" /v Path 2^>nul') do set "USER_PATH=%%B"
    for /f "tokens=2*" %%A in ('reg query "HKLM\SYSTEM\CurrentControlSet\Control\Session Manager\Environment" /v Path 2^>nul') do set "SYS_PATH=%%B"
    if defined USER_PATH set "PATH=!USER_PATH!;!PATH!"
    if defined SYS_PATH set "PATH=!SYS_PATH!;!PATH!"
    :: Python erneut suchen
    where python >nul 2>&1
    if not errorlevel 1 (
        python -c "import sys" >nul 2>&1
        if not errorlevel 1 set "PYTHON_CMD=python"
    )
    if "!PYTHON_CMD!"=="" (
        where py >nul 2>&1
        if not errorlevel 1 (
            py -c "import sys" >nul 2>&1
            if not errorlevel 1 set "PYTHON_CMD=py"
        )
    )
    if "!PYTHON_CMD!"=="" (
        echo.
        echo   [INFO] Python wurde installiert, ist aber noch nicht im PATH.
        echo   Bitte dieses Fenster schliessen und neu oeffnen.
        echo.
        pause
        exit /b 1
    )
    goto :python_found
)

:no_winget_python
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
:python_found

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
set "HAS_NODE=0"
where node >nul 2>&1
if not errorlevel 1 set "HAS_NODE=1"

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
::  5. UI-Modus waehlen: Vite Dev -> Pre-built -> CLI
:: ============================================================

:: Modus A: Node.js vorhanden -> Vite Dev Server
if "%HAS_NODE%"=="0" goto :check_prebuilt
if not exist "%REPO_ROOT%\ui\node_modules" (
    echo.
    echo   [INFO] node_modules nicht gefunden. Versuche npm install...
    cd /d "%REPO_ROOT%\ui"
    call npm install >nul 2>&1
    if errorlevel 1 goto :check_prebuilt
    cd /d "%REPO_ROOT%"
)

:: ============================================================
::  6a. Web-UI starten (Vite Dev Server)
:: ============================================================
echo.
echo   Web-UI wird gestartet (Vite Dev Server)...
echo   Oeffne http://127.0.0.1:5173 im Browser.
echo.
cd /d "%REPO_ROOT%\ui"
call npm run dev
echo.
echo   Web-UI wurde beendet.
pause
exit /b 0

:: ============================================================
::  6b. Pre-built UI (kein Node.js, aber ui/dist/ vorhanden)
:: ============================================================
:check_prebuilt
if not exist "%REPO_ROOT%\ui\dist\index.html" goto :cli_fallback
echo.
echo   Node.js nicht gefunden -- starte Pre-built UI.
echo   Backend + UI auf http://localhost:8741
echo.
cd /d "%REPO_ROOT%"
start "" http://localhost:8741
%PYTHON_CMD% -m jarvis --no-cli
echo.
echo   Cognithor wurde beendet.
pause
exit /b 0

:: ============================================================
::  6c. CLI-Fallback (kein Node.js, kein Pre-built UI)
:: ============================================================
:cli_fallback
echo.
echo   Node.js nicht gefunden und kein Pre-built UI vorhanden.
echo   Starte im CLI-Modus.
echo   Fuer die Web-UI installiere Node.js 18+: https://nodejs.org/
echo.
echo   ============================================================
cd /d "%REPO_ROOT%"
%PYTHON_CMD% -m jarvis
echo.
echo   Cognithor wurde beendet.
pause
