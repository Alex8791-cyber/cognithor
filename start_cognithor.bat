@echo off
setlocal enabledelayedexpansion
title Cognithor Control Center
color 0F

:: ============================================================
::  COGNITHOR ONE-CLICK LAUNCHER
::  Prueft Abhaengigkeiten, bootstrappt beim ersten Start,
::  startet dann die Web-UI.
::
::  UI Priority: Flutter pre-built > Flutter SDK > CLI
::  React UI is deprecated since v0.42.0.
:: ============================================================

call :main
echo.
echo   ============================================================
echo   Press any key to close this window...
echo   ============================================================
pause >nul
exit /b 0

:: ============================================================
:main
:: ============================================================

:: UTF-8 fuer Python-Output aktivieren
chcp 65001 >nul 2>&1

echo.
echo    ____  ___   ____ _   _ ___ _____ _   _  ___  ____
echo   / ___^|/ _ \ / ___^| \ ^| ^|_ _^|_   _^| ^| ^| ^|/ _ \^|  _ \
echo  ^| ^|   ^| ^| ^| ^| ^|  _^|  \^| ^|^| ^|  ^| ^| ^| ^|_^| ^| ^| ^| ^| ^|_^) ^|
echo  ^| ^|___^| ^|_^| ^| ^|_^| ^| ^|\  ^|^| ^|  ^| ^| ^|  _  ^| ^|_^| ^|  _ ^<
echo   \____^|\___/ \____^|_^| \_^|___^| ^|_^| ^|_^| ^|_^|\___/^|_^| \_\
echo.
echo   v0.50.0 - Flutter UI
echo.

set "REPO_ROOT=%~dp0"
:: Trailing backslash entfernen
if "%REPO_ROOT:~-1%"=="\" set "REPO_ROOT=%REPO_ROOT:~0,-1%"

:: ============================================================
::  1. Python im PATH?
:: ============================================================
call :find_python
if "!PYTHON_CMD!"=="" (
    echo   [ERROR] Python not found!
    echo.
    echo   Please install Python 3.12+:
    echo   https://www.python.org/downloads/
    echo.
    echo   IMPORTANT: Check "Add Python to PATH" during installation!
    goto :eof
)

:: ============================================================
::  2. Python >= 3.12?
:: ============================================================
%PYTHON_CMD% -c "import sys; sys.exit(0 if sys.version_info >= (3,12) else 1)" 2>nul
if errorlevel 1 (
    echo   [ERROR] Python 3.12 or newer is required!
    echo.
    for /f "tokens=*" %%v in ('%PYTHON_CMD% --version 2^>^&1') do echo   Installed: %%v
    echo.
    echo   Please upgrade Python:
    echo   https://www.python.org/downloads/
    goto :eof
)

:: ============================================================
::  3. Flutter erkennen
:: ============================================================
call :detect_flutter

:: ============================================================
::  4. Bootstrap ausfuehren
:: ============================================================
echo   Starting bootstrap...
echo.

%PYTHON_CMD% "%REPO_ROOT%\scripts\bootstrap_windows.py" --repo-root "%REPO_ROOT%"
if errorlevel 1 (
    echo.
    echo   [ERROR] Bootstrap failed!
    echo   Please check the output above for details.
    goto :eof
)

:: ============================================================
::  4b. Verify identity module (now part of [all])
:: ============================================================
%PYTHON_CMD% -c "import jarvis.identity" >nul 2>&1
if errorlevel 1 (
    echo   [INFO] Identity module missing. Installing (may take a few minutes^)...
    echo.
    cd /d "%REPO_ROOT%"
    %PYTHON_CMD% -m pip install -e ".[identity]"
    echo.
    %PYTHON_CMD% -c "import jarvis.identity" >nul 2>&1
    if not errorlevel 1 (
        echo   [OK] Identity module installed.
    ) else (
        echo   [WARNING] Identity install failed. Try: pip install -e ".[identity]"
    )
) else (
    echo   [OK] Identity module available.
)

:: ── Desktop automation (Computer Use) ──
%PYTHON_CMD% -c "import pyautogui, mss" >nul 2>&1
if errorlevel 1 (
    echo   [INFO] Installing Desktop automation deps (Computer Use^)...
    %PYTHON_CMD% -m pip install --quiet pyautogui mss pyperclip Pillow >nul 2>&1
    echo   [OK] Desktop automation ready.
) else (
    echo   [OK] Desktop automation available.
)

:: ============================================================
::  5. UI-Modus waehlen (Flutter-first)
:: ============================================================

:: --- Modus 1: Pre-built Flutter Web (bundled in repo) ---
if exist "%REPO_ROOT%\flutter_app\build\web\index.html" (
    echo.
    echo   Starting with Flutter UI...
    echo   Backend + UI at http://localhost:8741
    echo.
    cd /d "%REPO_ROOT%"
    start "" http://localhost:8741
    %PYTHON_CMD% -m jarvis --no-cli --api-host 0.0.0.0
    echo.
    echo   Cognithor stopped.
    goto :eof
)

:: --- Modus 2: Flutter SDK vorhanden -> Build + Start ---
if "!HAS_FLUTTER!"=="1" (
    if exist "%REPO_ROOT%\flutter_app\pubspec.yaml" (
        echo.
        echo   Flutter SDK detected. Building Flutter Web UI...

        :: Dependencies holen
        if not exist "%REPO_ROOT%\flutter_app\.dart_tool" (
            echo   [INFO] Fetching Flutter dependencies...
            cd /d "%REPO_ROOT%\flutter_app"
            cmd /c flutter pub get >nul 2>&1
            cd /d "%REPO_ROOT%"
        )

        :: Build
        echo   [INFO] Running flutter build web --release --no-tree-shake-icons...
        cd /d "%REPO_ROOT%\flutter_app"
        cmd /c flutter build web --release --no-tree-shake-icons
        cd /d "%REPO_ROOT%"

        if exist "%REPO_ROOT%\flutter_app\build\web\index.html" (
            echo   [OK] Flutter Web UI built successfully.
            echo.
            echo   Starting with Flutter UI...
            echo   Backend + UI at http://localhost:8741
            echo.
            start "" http://localhost:8741
            %PYTHON_CMD% -m jarvis --no-cli --api-host 0.0.0.0
            echo.
            echo   Cognithor stopped.
            goto :eof
        ) else (
            echo   [WARNING] Flutter build failed.
        )
    )
)

:: --- Modus 3: CLI-Fallback ---
echo.
echo   ============================================================
echo   No pre-built Flutter UI found.
echo.
if "!HAS_FLUTTER!"=="0" (
    echo   To get the Flutter UI:
    echo     1. Install Flutter: https://docs.flutter.dev/get-started/install
    echo     2. cd flutter_app
    echo     3. flutter build web --release --no-tree-shake-icons
    echo     4. Re-run start_cognithor.bat
) else (
    echo   Flutter SDK found but build failed. Try manually:
    echo     cd flutter_app
    echo     flutter build web --release --no-tree-shake-icons
)
echo.
echo   Starting in CLI mode...
echo   ============================================================
echo.
cd /d "%REPO_ROOT%"
%PYTHON_CMD% -m jarvis
echo.
echo   Cognithor stopped.
goto :eof

:: ============================================================
::  SUBROUTINEN
:: ============================================================

:find_python
set "PYTHON_CMD="
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
goto :eof

:detect_flutter
set "HAS_FLUTTER=0"
where flutter >nul 2>&1
if not errorlevel 1 (
    cmd /c flutter --version >nul 2>&1
    if not errorlevel 1 set "HAS_FLUTTER=1"
)
if "!HAS_FLUTTER!"=="0" (
    if exist "C:\flutter\bin\flutter.bat" (
        set "PATH=C:\flutter\bin;!PATH!"
        cmd /c C:\flutter\bin\flutter.bat --version >nul 2>&1
        if not errorlevel 1 set "HAS_FLUTTER=1"
    )
)
goto :eof
