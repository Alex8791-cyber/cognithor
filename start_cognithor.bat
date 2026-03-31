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

:: Start companion services (Tailscale + AltServer)
call :start_services

call :main

:: Stop companion services on exit
call :stop_services

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
echo   v0.69.0 - ATL + CAPTCHA + AACS
echo.

set "REPO_ROOT=%~dp0"
:: Trailing backslash entfernen
if "%REPO_ROOT:~-1%"=="\" set "REPO_ROOT=%REPO_ROOT:~0,-1%"

:: ============================================================
::  0. llama.cpp Server (falls konfiguriert + nicht schon laufend)
:: ============================================================
call :start_llama_server

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

:: ── ARC-AGI-3 Benchmark Agent (optional) ──
%PYTHON_CMD% -c "import arc_agi" >nul 2>&1
if errorlevel 1 (
    echo   [INFO] ARC-AGI-3 SDK not installed. Install with: pip install -e ".[arc]"
) else (
    echo   [OK] ARC-AGI-3 SDK available.
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
%PYTHON_CMD% -m jarvis --api-host 0.0.0.0
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

:start_services
:: Start Tailscale (VPN for mobile access)
tasklist /FI "IMAGENAME eq tailscale-ipn.exe" 2>nul | find /i "tailscale-ipn.exe" >nul
if errorlevel 1 (
    if exist "C:\Program Files\Tailscale\tailscale-ipn.exe" (
        echo   [INFO] Starting Tailscale...
        start "" "C:\Program Files\Tailscale\tailscale-ipn.exe"
        echo   [OK] Tailscale started.
    ) else (
        echo   [SKIP] Tailscale not installed.
    )
) else (
    echo   [OK] Tailscale already running.
)

:: Start SearXNG (local meta search engine for Evolution Engine)
where docker >nul 2>&1
if errorlevel 1 (
    echo   [SKIP] Docker not installed - SearXNG unavailable.
    goto :searxng_done
)
:: Ensure Docker daemon is running
docker info >nul 2>&1
if errorlevel 1 (
    echo   [INFO] Starting Docker Desktop...
    start "" "C:\Program Files\Docker\Docker\Docker Desktop.exe" 2>nul
    :: Wait up to 30 seconds for daemon
    for /L %%i in (1,1,15) do (
        timeout /t 2 /nobreak >nul
        docker info >nul 2>&1
        if not errorlevel 1 goto :docker_ready
    )
    echo   [SKIP] Docker Desktop did not start in time.
    goto :searxng_done
)
:docker_ready
docker ps -q -f "name=cognithor-searxng" 2>nul | find /v "" >nul
if errorlevel 1 (
    docker ps -aq -f "name=cognithor-searxng" 2>nul | find /v "" >nul
    if not errorlevel 1 (
        echo   [INFO] Starting SearXNG container...
        docker start cognithor-searxng >nul 2>&1
        echo   [OK] SearXNG started.
    ) else (
        echo   [INFO] Creating SearXNG container...
        docker run -d --name cognithor-searxng -p 8888:8080 --restart unless-stopped -v "%REPO_ROOT%\docker\searxng\settings.yml:/etc/searxng/settings.yml:ro" -v "%REPO_ROOT%\docker\searxng\limiter.toml:/etc/searxng/limiter.toml:ro" -e SEARXNG_SECRET=cognithor-local searxng/searxng >nul 2>&1
        if not errorlevel 1 (
            echo   [OK] SearXNG created and started on port 8888.
        ) else (
            echo   [SKIP] SearXNG container creation failed.
        )
    )
) else (
    echo   [OK] SearXNG already running.
)
:searxng_done

:: Check GDPR encryption dependencies (SQLCipher + keyring)
python -c "import pysqlcipher3" >nul 2>&1
if errorlevel 1 (
    python -c "import sqlcipher3" >nul 2>&1
    if errorlevel 1 (
        echo   [INFO] Installing sqlcipher3 for GDPR encryption...
        pip install sqlcipher3 >nul 2>&1
        if errorlevel 1 (
            echo   [INFO] sqlcipher3 failed, trying pysqlcipher3...
            pip install pysqlcipher3 >nul 2>&1
            if errorlevel 1 (
                echo   [WARN] SQLCipher installation failed. Database encryption unavailable.
                echo          Try manually: pip install sqlcipher3
            ) else (
                echo   [OK] pysqlcipher3 installed.
            )
        ) else (
            echo   [OK] sqlcipher3 installed -- database encryption active.
        )
    ) else (
        echo   [OK] sqlcipher3 available.
    )
) else (
    echo   [OK] pysqlcipher3 available.
)
python -c "import keyring; keyring.get_password('test','test')" >nul 2>&1
if errorlevel 1 (
    echo   [INFO] Installing keyring for OS-level key protection...
    pip install keyring >nul 2>&1
    if errorlevel 1 (
        echo   [WARN] keyring installation failed. DB key stored in file instead of OS keyring.
    ) else (
        echo   [OK] keyring installed -- DB encryption key protected by Windows Credential Locker.
    )
) else (
    echo   [OK] keyring available -- DB key protected by OS credential store.
)

:: Start AltServer (iOS sideloading)
tasklist /FI "IMAGENAME eq AltServer.exe" 2>nul | find /i "AltServer.exe" >nul
if errorlevel 1 (
    if exist "C:\Program Files (x86)\AltServer\AltServer.exe" (
        echo   [INFO] Starting AltServer...
        start "" "C:\Program Files (x86)\AltServer\AltServer.exe"
        echo   [OK] AltServer started.
    ) else (
        echo   [SKIP] AltServer not installed.
    )
) else (
    echo   [OK] AltServer already running.
)
goto :eof

:start_llama_server
:: Check if llama_cpp is configured as backend
findstr /C:"llm_backend_type: llama_cpp" "%USERPROFILE%\.jarvis\config.yaml" >nul 2>&1
if errorlevel 1 (
    echo   [SKIP] llama.cpp not configured as backend.
    goto :eof
)

:: Check if llama-server is already running
tasklist /FI "IMAGENAME eq llama-server.exe" 2>nul | find /i "llama-server.exe" >nul
if not errorlevel 1 (
    echo   [OK] llama-server already running.
    goto :eof
)

:: Check if llama-server is installed
set "LLAMA_SERVER=C:\Users\ArtiCall\llama-cpp-cuda\llama-server.exe"
if not exist "%LLAMA_SERVER%" (
    echo   [ERROR] llama-server CUDA build not found at %LLAMA_SERVER%
    goto :eof
)

:: Official GGUF from Hugging Face (bartowski Q4_K_M)
set "LLAMA_MODEL=C:\Users\ArtiCall\models\Qwen3.5-27B-Q4_K_M.gguf"
if not exist "%LLAMA_MODEL%" (
    echo   [ERROR] qwen3.5:27b GGUF not found at %LLAMA_MODEL%
    echo          Run: ollama pull qwen3.5:27b
    goto :eof
)

echo   [INFO] Starting llama-server (qwen3.5:27b, 252K context, KV Q8_0^)...
start "llama-server" /MIN "%LLAMA_SERVER%" -m "%LLAMA_MODEL%" -c 258048 -ctk q8_0 -ctv q8_0 -ngl 99 -fa on --port 8080 --host 0.0.0.0 -t 8

:: Wait for server to be ready (up to 60 seconds)
echo   [INFO] Waiting for llama-server to load model...
for /L %%i in (1,1,30) do (
    timeout /t 2 /nobreak >nul
    curl -s http://localhost:8080/health >nul 2>&1
    if not errorlevel 1 (
        echo   [OK] llama-server ready on port 8080.
        goto :eof
    )
)
echo   [WARN] llama-server did not respond in 60s. It may still be loading.
goto :eof

:stop_services
echo.
echo   [INFO] Stopping companion services...

:: Stop llama-server
tasklist /FI "IMAGENAME eq llama-server.exe" 2>nul | find /i "llama-server.exe" >nul
if not errorlevel 1 (
    taskkill /IM llama-server.exe /F >nul 2>&1
    echo   [OK] llama-server stopped.
)

:: Stop AltServer
tasklist /FI "IMAGENAME eq AltServer.exe" 2>nul | find /i "AltServer.exe" >nul
if not errorlevel 1 (
    taskkill /IM AltServer.exe /F >nul 2>&1
    echo   [OK] AltServer stopped.
)

:: Stop SearXNG container (keep for fast restart)
docker ps -q -f "name=cognithor-searxng" 2>nul | find /v "" >nul
if not errorlevel 1 (
    docker stop cognithor-searxng >nul 2>&1
    echo   [OK] SearXNG stopped.
)

:: Stop Tailscale GUI (not the system service)
tasklist /FI "IMAGENAME eq tailscale-ipn.exe" 2>nul | find /i "tailscale-ipn.exe" >nul
if not errorlevel 1 (
    taskkill /IM tailscale-ipn.exe /F >nul 2>&1
    echo   [OK] Tailscale GUI stopped.
)
goto :eof
