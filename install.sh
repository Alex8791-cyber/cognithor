#!/usr/bin/env bash
# ============================================================================
# Jarvis · Agent OS – Installations-Script
# ============================================================================
#
# Nutzung:
#   chmod +x install.sh
#   ./install.sh              Vollinstallation (interaktiv)
#   ./install.sh --minimal    Nur Core (kein Web, kein Telegram)
#   ./install.sh --full       Alles inkl. Voice
#   ./install.sh --use-uv     uv statt pip verwenden (10x schneller)
#   ./install.sh --systemd    Nur Systemd-Services installieren
#   ./install.sh --uninstall  Deinstallation
#
# Voraussetzungen:
#   - Python 3.12+
#   - pip (oder uv mit --use-uv)
#   - Ollama (installiert und gestartet)
#
# ============================================================================
set -euo pipefail

# --- Farben ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# --- Konfiguration ---
JARVIS_HOME="${JARVIS_HOME:-$HOME/.jarvis}"
VENV_DIR="${JARVIS_HOME}/venv"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MIN_PYTHON="3.12"
OLLAMA_URL="${JARVIS_OLLAMA_BASE_URL:-http://localhost:11434}"
USE_UV=false
PKG_INSTALLER=""  # "uv" or "pip", set in detect_installer

# --- Error Tracker ---
INSTALL_FAILED=false

# ============================================================================
# Hilfsfunktionen
# ============================================================================

info()    { echo -e "${BLUE}[i]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}  $*"; }
warn()    { echo -e "${YELLOW}[!]${NC}  $*"; }
error()   { echo -e "${RED}[X]${NC}  $*" >&2; }
fatal()   {
    error "$*"
    INSTALL_FAILED=true
    show_error_submission
    exit 1
}
header()  { echo -e "\n${BOLD}${CYAN}=== $* ===${NC}\n"; }

check_command() {
    command -v "$1" &>/dev/null
}

version_ge() {
    # Prueft ob Version $1 >= $2
    printf '%s\n%s\n' "$2" "$1" | sort -V | head -n1 | grep -qF "$2"
}

# ============================================================================
# Error Submission Helper (Fix #9)
# ============================================================================

show_error_submission() {
    echo ""
    echo -e "${RED}${BOLD}[X] Installation fehlgeschlagen.${NC}"
    echo ""
    echo "  Bitte oeffne ein Issue auf GitHub:"
    echo "  https://github.com/Alex8791-cyber/cognithor/issues/new"
    echo ""
    echo "  Fuege die obige Ausgabe als Log bei."
    echo ""
}

# ============================================================================
# Banner (Fix #7: dynamic version from pyproject.toml)
# ============================================================================

show_banner() {
    local version="unknown"
    if [[ -f "$REPO_DIR/pyproject.toml" ]]; then
        version=$(grep '^version' "$REPO_DIR/pyproject.toml" | head -1 | cut -d'"' -f2)
    fi

    echo -e "${BOLD}${CYAN}"
    cat << 'BANNER'

       ██╗ █████╗ ██████╗ ██╗   ██╗██╗███████╗
       ██║██╔══██╗██╔══██╗██║   ██║██║██╔════╝
       ██║███████║██████╔╝██║   ██║██║███████╗
  ██   ██║██╔══██║██╔══██╗╚██╗ ██╔╝██║╚════██║
  ╚█████╔╝██║  ██║██║  ██║ ╚████╔╝ ██║███████║
   ╚════╝ ╚═╝  ╚═╝╚═╝  ╚═╝  ╚═══╝  ╚═╝╚══════╝
BANNER
    echo -e "           Agent OS · v${version} · Installer"
    echo ""
    echo -e "${NC}"
}

# ============================================================================
# Installer-Erkennung (uv / pip)
# ============================================================================

detect_installer() {
    if [[ "$USE_UV" == true ]]; then
        if check_command uv; then
            local uv_ver
            uv_ver=$(uv --version 2>/dev/null | head -1)
            PKG_INSTALLER="uv"
            success "uv gefunden ($uv_ver)"
            return 0
        else
            warn "uv nicht gefunden, installiere automatisch..."
            local _uv_install_script
            _uv_install_script=$(mktemp)
            if curl -LsSf --max-time 30 https://astral.sh/uv/install.sh -o "$_uv_install_script"; then
                warn "Downloaded uv installer to $_uv_install_script — executing..."
                sh "$_uv_install_script" 2>/dev/null
                rm -f "$_uv_install_script"
                # Pfad aktualisieren
                export PATH="$HOME/.local/bin:$PATH"
                if check_command uv; then
                    local uv_ver
                    uv_ver=$(uv --version 2>/dev/null | head -1)
                    PKG_INSTALLER="uv"
                    success "uv installiert ($uv_ver)"
                    return 0
                fi
            else
                rm -f "$_uv_install_script"
            fi
            warn "uv konnte nicht installiert werden -- Fallback auf pip"
        fi
    fi

    # Auto-Detect: uv bevorzugen wenn vorhanden
    if check_command uv; then
        local uv_ver
        uv_ver=$(uv --version 2>/dev/null | head -1)
        PKG_INSTALLER="uv"
        success "uv erkannt ($uv_ver) -- wird bevorzugt verwendet"
        return 0
    fi

    # Fallback: pip (Fix #2: abort with helpful message if missing)
    if python3 -m pip --version &>/dev/null; then
        PKG_INSTALLER="pip"
        success "pip wird verwendet"
        return 0
    fi

    echo ""
    error "pip nicht gefunden!"
    echo ""
    echo "  Installiere pip mit:"
    echo ""
    echo "    sudo apt install python3-pip"
    echo ""
    echo "  Danach erneut ausfuehren:"
    echo ""
    echo "    ./install.sh"
    echo ""
    INSTALL_FAILED=true
    show_error_submission
    exit 1
}

# ============================================================================
# Helper: Distro-spezifischer Python-Installationshinweis
# ============================================================================

_python_install_hint() {
    local distro_id=""
    if [[ -f /etc/os-release ]]; then
        distro_id=$(. /etc/os-release && echo "${ID:-}")
    fi

    echo ""
    info "Python ${MIN_PYTHON}+ installieren:"
    echo ""
    case "$distro_id" in
        ubuntu)
            echo "    sudo add-apt-repository ppa:deadsnakes/ppa"
            echo "    sudo apt update"
            echo "    sudo apt install python3.12 python3.12-venv python3.12-dev"
            ;;
        debian)
            echo "    # Option 1: Backports aktivieren"
            echo "    # Option 2: pyenv verwenden:"
            echo "    curl https://pyenv.run | bash"
            echo "    pyenv install 3.12"
            ;;
        fedora|rhel|centos|rocky|alma)
            echo "    sudo dnf install python3.12 python3.12-devel"
            ;;
        arch|manjaro)
            echo "    sudo pacman -S python"
            ;;
        opensuse*|sles)
            echo "    sudo zypper install python312 python312-devel"
            ;;
        *)
            echo "    https://www.python.org/downloads/"
            ;;
    esac
    echo ""
}

# ============================================================================
# Schritt 1: Systemvoraussetzungen pruefen
# ============================================================================

check_prerequisites() {
    header "Systemvoraussetzungen pruefen"
    local errors=0

    # Python
    if check_command python3; then
        local py_version
        py_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        if version_ge "$py_version" "$MIN_PYTHON"; then
            success "Python $py_version gefunden"
        else
            error "Python $py_version zu alt (mindestens $MIN_PYTHON benoetigt)"
            _python_install_hint
            errors=$((errors + 1))
        fi
    else
        error "Python3 nicht gefunden"
        _python_install_hint
        errors=$((errors + 1))
    fi

    # pip (Fix #2: abort immediately with exact fix command)
    if [[ "$USE_UV" != true ]]; then
        if python3 -m pip --version &>/dev/null; then
            success "pip verfuegbar"
        else
            error "pip nicht gefunden"
            echo ""
            echo "  Behebe mit:"
            echo "    sudo apt install python3-pip"
            echo ""
            fatal "pip ist eine Pflicht-Abhaengigkeit. Bitte installieren und erneut starten."
        fi
    fi

    # venv
    if python3 -m venv --help &>/dev/null; then
        success "venv-Modul verfuegbar"
    else
        error "venv nicht gefunden"
        echo ""
        echo "  Behebe mit:"
        echo "    sudo apt install python3.12-venv"
        echo ""
        fatal "python3-venv ist eine Pflicht-Abhaengigkeit. Bitte installieren und erneut starten."
    fi

    # git (optional)
    if check_command git; then
        success "git verfuegbar"
    else
        warn "git nicht gefunden (optional, fuer Updates empfohlen)"
    fi

    # Ollama
    if check_command ollama; then
        local ollama_version
        ollama_version=$(ollama --version 2>/dev/null | head -1 || echo "unbekannt")
        success "Ollama installiert ($ollama_version)"
    else
        warn "Ollama nicht gefunden -- LLM-Funktionen sind ohne Ollama eingeschraenkt"
        read -rp "  Ollama jetzt installieren? [j/N] " _ollama_answer
        if [[ "$_ollama_answer" =~ ^[jJyY]$ ]]; then
            info "Installiere Ollama..."
            local _ollama_install_script
            _ollama_install_script=$(mktemp)
            curl -fsSL --max-time 60 https://ollama.com/install.sh -o "$_ollama_install_script"
            if [ $? -eq 0 ] && sh "$_ollama_install_script"; then
                rm -f "$_ollama_install_script"
                success "Ollama installiert"
                # Ollama-Server starten
                nohup ollama serve &>/dev/null &
                info "Warte auf Ollama-Server..."
                local _ollama_delay=1
                local _ollama_total=0
                while [[ $_ollama_total -lt 15 ]]; do
                    if curl -sf --max-time 2 "${OLLAMA_URL}/api/version" &>/dev/null; then
                        success "Ollama-Server gestartet"
                        break
                    fi
                    sleep "$_ollama_delay"
                    _ollama_total=$((_ollama_total + _ollama_delay))
                    # Exponential backoff: 1, 2, 4 (capped)
                    _ollama_delay=$((_ollama_delay * 2))
                    [[ $_ollama_delay -gt 4 ]] && _ollama_delay=4
                done
                if [[ $_ollama_total -ge 15 ]]; then
                    warn "Ollama-Server nicht erreichbar nach 15s -- manuell starten: ollama serve"
                fi
            else
                rm -f "$_ollama_install_script"
                error "Ollama-Installation fehlgeschlagen"
                info "Manuell installieren: curl -fsSL https://ollama.com/install.sh | sh"
            fi
        else
            info "Installiere spaeter: curl -fsSL https://ollama.com/install.sh | sh"
        fi
    fi

    # Ollama-Server erreichbar?
    if curl -sf --max-time 3 "${OLLAMA_URL}/api/version" &>/dev/null; then
        success "Ollama-Server erreichbar (${OLLAMA_URL})"
    else
        warn "Ollama-Server nicht erreichbar auf ${OLLAMA_URL}"
        info "Starte mit: ollama serve"
    fi

    if [[ $errors -gt 0 ]]; then
        fatal "$errors kritische Voraussetzung(en) fehlen"
    fi
}

# ============================================================================
# Schritt 2: Ollama-Modelle (Fix #4: optional, never blocking)
# ============================================================================

detect_hardware_tier() {
    # Ermittelt Hardware-Tier basierend auf VRAM und RAM
    local vram_mb=0
    local ram_mb=0
    local gpu_count=0
    local vram_str="keine GPU"

    # VRAM via nvidia-smi (supports multi-GPU: sums all GPUs)
    if check_command nvidia-smi; then
        local _nvsmi_output
        _nvsmi_output=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null) || true
        if [[ -n "$_nvsmi_output" ]]; then
            # Sum VRAM across all GPUs (one value per line)
            local _total_vram=0
            while IFS= read -r _line; do
                _line=$(echo "$_line" | tr -d ' \r')
                if [[ "$_line" =~ ^[0-9]+$ ]]; then
                    _total_vram=$(( _total_vram + _line ))
                    gpu_count=$(( gpu_count + 1 ))
                fi
            done <<< "$_nvsmi_output"
            vram_mb=$_total_vram
        fi
        if [[ "$vram_mb" -gt 0 ]]; then
            local vram_gb=$(( vram_mb / 1024 ))
            if [[ "$gpu_count" -gt 1 ]]; then
                vram_str="${vram_gb} GB VRAM (${gpu_count} GPUs)"
            else
                vram_str="${vram_gb} GB VRAM"
            fi
        fi
    # AMD GPU via rocm-smi (Radeon)
    elif check_command rocm-smi; then
        local _rocm_output
        _rocm_output=$(rocm-smi --showmeminfo vram --json 2>/dev/null) || true
        if [[ -n "$_rocm_output" ]]; then
            local _total_vram=0
            # Parse JSON output: extract "Total Memory (B)" values
            while IFS= read -r _vram_bytes; do
                _vram_bytes=$(echo "$_vram_bytes" | tr -d ' ",' | grep -oP '[0-9]+')
                if [[ -n "$_vram_bytes" && "$_vram_bytes" =~ ^[0-9]+$ ]]; then
                    _total_vram=$(( _total_vram + _vram_bytes / 1048576 ))
                    gpu_count=$(( gpu_count + 1 ))
                fi
            done < <(echo "$_rocm_output" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    for card in data.values():
        if isinstance(card, dict):
            total = card.get('Total Memory (B)', card.get('VRAM Total Memory (B)', 0))
            if total: print(total)
except: pass
" 2>/dev/null)
            vram_mb=$_total_vram
        fi
        if [[ "$vram_mb" -gt 0 ]]; then
            local vram_gb=$(( vram_mb / 1024 ))
            if [[ "$gpu_count" -gt 1 ]]; then
                vram_str="${vram_gb} GB VRAM AMD (${gpu_count} GPUs)"
            else
                vram_str="${vram_gb} GB VRAM AMD"
            fi
        fi
    fi

    # RAM via /proc/meminfo
    if [[ -f /proc/meminfo ]]; then
        local ram_kb
        ram_kb=$(grep MemTotal /proc/meminfo | awk '{print $2}')
        ram_mb=$(( ram_kb / 1024 ))
    fi

    local ram_gb=$(( ram_mb / 1024 ))
    local cpu_cores
    cpu_cores=$(nproc 2>/dev/null || echo "4")

    # Tier bestimmen
    local tier="minimal"
    if [[ $ram_gb -ge 64 && $cpu_cores -ge 16 && $vram_mb -ge 49152 ]]; then
        tier="enterprise"
    elif [[ $vram_mb -ge 16384 && $ram_gb -ge 32 ]]; then
        tier="power"
    elif [[ $vram_mb -ge 8192 && $ram_gb -ge 16 ]]; then
        tier="standard"
    fi

    # Anzeige
    echo ""
    info "Dein System: ${vram_str}, ${ram_gb} GB RAM, ${cpu_cores} Kerne"
    info "Hardware-Tier: ${BOLD}$(echo "$tier" | tr '[:lower:]' '[:upper:]')${NC}"

    case "$tier" in
        minimal)
            info "Modelle: qwen3:8b, nomic-embed-text"
            info "Tipp: Fuer bessere Qualitaet mindestens 8 GB VRAM empfohlen"
            ;;
        standard)
            info "Modelle: qwen3:8b, qwen3:32b, nomic-embed-text"
            info "Tipp: 'cognithor --lite' fuer nur 6 GB VRAM"
            ;;
        power|enterprise)
            info "Modelle: qwen3:8b, qwen3:32b, qwen3-coder:30b, nomic-embed-text"
            info "Tipp: 'cognithor --lite' fuer nur 6 GB VRAM"
            ;;
    esac
    echo ""
}

ensure_ollama_models() {
    header "Ollama-Modelle pruefen"

    detect_hardware_tier

    if ! curl -sf "${OLLAMA_URL}/api/version" &>/dev/null; then
        warn "Ollama nicht erreichbar -- Modelle manuell herunterladen:"
        info "  ollama pull qwen3:8b"
        info "  ollama pull nomic-embed-text"
        info "  ollama pull qwen3:32b     (optional, wenn >=24GB VRAM)"
        return 0
    fi

    # Installierte Modelle abfragen
    local installed
    installed=$(curl -sf "${OLLAMA_URL}/api/tags" | python3 -c "
import sys, json
data = json.load(sys.stdin)
for m in data.get('models', []):
    print(m['name'])
" 2>/dev/null || echo "")

    # Pflicht-Modelle (Fix #4: nur pruefen, nicht automatisch downloaden)
    local required_models=("qwen3:8b" "nomic-embed-text")
    local optional_models=("qwen3:32b")
    local missing_required=()

    for model in "${required_models[@]}"; do
        local base_name="${model%%:*}"
        if echo "$installed" | grep -q "$base_name"; then
            success "$model bereits installiert"
        else
            warn "$model FEHLT (Pflicht-Modell)"
            missing_required+=("$model")
        fi
    done

    for model in "${optional_models[@]}"; do
        local base_name="${model%%:*}"
        if echo "$installed" | grep -q "$base_name"; then
            success "$model bereits installiert"
        else
            info "$model nicht installiert (optional, fuer bessere Qualitaet)"
        fi
    done

    # Zusammenfassung fehlender Modelle
    if [[ ${#missing_required[@]} -gt 0 ]]; then
        echo ""
        warn "Fehlende Pflicht-Modelle! Bitte manuell herunterladen:"
        echo ""
        for model in "${missing_required[@]}"; do
            echo "    ollama pull $model"
        done
        echo ""
        info "Die Installation wird fortgesetzt -- Modelle koennen spaeter geladen werden."
        echo ""
    fi
}

# ============================================================================
# Schritt 3: Virtual Environment + Installation
# ============================================================================

setup_venv() {
    header "Python Virtual Environment"

    # Fix #3: If venv dir exists but activate is missing, it's corrupted
    if [[ -d "$VENV_DIR" ]]; then
        if [[ -f "$VENV_DIR/bin/activate" ]]; then
            info "Bestehendes venv gefunden: $VENV_DIR"
            # shellcheck disable=SC1091
            source "$VENV_DIR/bin/activate"
            success "venv aktiviert"
            return 0
        else
            warn "Korruptes venv erkannt (bin/activate fehlt) -- wird neu erstellt"
            rm -rf "$VENV_DIR"
        fi
    fi

    # Create fresh venv
    if [[ "$PKG_INSTALLER" == "uv" ]]; then
        info "Erstelle venv mit uv in $VENV_DIR ..."
        uv venv "$VENV_DIR" --python python3
    else
        info "Erstelle venv in $VENV_DIR ..."
        python3 -m venv "$VENV_DIR"
    fi

    # Verify activate exists before sourcing
    if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
        fatal "venv-Erstellung fehlgeschlagen: $VENV_DIR/bin/activate nicht gefunden"
    fi

    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
    if [[ "$PKG_INSTALLER" == "pip" ]]; then
        pip install --upgrade pip setuptools wheel --quiet
    fi
    success "venv erstellt und aktiviert"
}

install_jarvis() {
    header "Jarvis installieren"

    local install_extras="$1"
    local spec="${REPO_DIR}"
    if [[ -n "$install_extras" ]]; then
        spec="${REPO_DIR}[${install_extras}]"
    fi

    if [[ "$PKG_INSTALLER" == "uv" ]]; then
        info "Installiere jarvis[$install_extras] mit uv aus $REPO_DIR ..."
        uv pip install -e "$spec" --quiet 2>&1 | tail -5
    else
        # Fix #5: progress feedback for pip
        echo ""
        info "Installiere jarvis[$install_extras] mit pip aus $REPO_DIR ..."
        info "Installiere Pakete... (kann 2-5 Minuten dauern)"
        echo ""
        pip install -e "$spec" --progress-bar on 2>&1 | tail -20
    fi

    # Verifiziere Installation
    if python3 -c "import jarvis; print(f'Jarvis v{jarvis.__version__}')" 2>/dev/null; then
        success "Jarvis erfolgreich installiert"
    else
        fatal "Installation fehlgeschlagen -- pip install hat Fehler verursacht"
    fi

    # Jarvis-CLI pruefen
    if "$VENV_DIR/bin/jarvis" --version &>/dev/null; then
        local ver
        ver=$("$VENV_DIR/bin/jarvis" --version 2>&1)
        success "CLI verfuegbar: $ver"
    else
        warn "CLI 'jarvis' nicht im PATH -- nutze: $VENV_DIR/bin/jarvis"
    fi

    # Web-UI Build (Node.js optional)
    local ui_dir="$REPO_DIR/ui"
    local ui_dist="$ui_dir/dist/index.html"
    if check_command node && check_command npm; then
        if [[ ! -d "$ui_dir/node_modules" ]]; then
            info "Installiere UI-Abhaengigkeiten (npm install)..."
            (cd "$ui_dir" && npm install --silent 2>&1 | tail -5) || true
        fi
        if [[ ! -f "$ui_dist" ]]; then
            info "Erstelle UI-Build (npm run build)..."
            if (cd "$ui_dir" && npm run build --silent 2>&1 | tail -5); then
                success "UI-Build erstellt (ui/dist/)"
            else
                warn "npm run build fehlgeschlagen -- CLI-Modus verfuegbar"
            fi
        else
            success "UI-Build vorhanden (ui/dist/)"
        fi
    elif [[ -f "$ui_dist" ]]; then
        success "Pre-built UI vorhanden (Node.js nicht noetig)"
    else
        echo ""
        warn "Node.js nicht gefunden -- Web-UI kann nicht erstellt werden"
        info "CLI-Modus ist vollstaendig verfuegbar (kein Node.js noetig)"
        echo ""
        info "Fuer Web-UI spaeter Node.js installieren:"
        local _distro_id=""
        if [[ -f /etc/os-release ]]; then
            _distro_id=$(. /etc/os-release && echo "${ID:-}")
        fi
        case "$_distro_id" in
            ubuntu|debian)
                echo "    curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -"
                echo "    sudo apt install -y nodejs"
                ;;
            fedora|rhel|centos|rocky|alma)
                echo "    sudo dnf install nodejs"
                ;;
            arch|manjaro)
                echo "    sudo pacman -S nodejs npm"
                ;;
            *)
                echo "    https://nodejs.org/en/download/"
                ;;
        esac
        echo "    cd $ui_dir && npm install && npm run build"
        echo ""
    fi
}

# ============================================================================
# Schritt 4: Verzeichnisstruktur + Config (Fix #6: verbose + timeout + perms)
# ============================================================================

create_directory_safe() {
    # Creates a single directory with error handling and verbose output
    local dir="$1"
    if [[ -d "$dir" ]]; then
        info "  [vorhanden] $dir"
        return 0
    fi
    local err_file
    err_file=$(mktemp "${TMPDIR:-/tmp}/jarvis_mkdir_XXXXXX" 2>/dev/null || echo "/tmp/jarvis_mkdir_err")
    if mkdir -p "$dir" 2>"$err_file"; then
        success "  [erstellt]  $dir"
        rm -f "$err_file" 2>/dev/null
    else
        local err
        err=$(cat "$err_file" 2>/dev/null || echo "unbekannter Fehler")
        rm -f "$err_file" 2>/dev/null
        error "Verzeichnis konnte nicht erstellt werden: $dir"
        error "Fehler: $err"
        echo ""
        echo "  Behebe mit:"
        echo "    sudo mkdir -p $dir"
        echo "    sudo chown \$(whoami) $dir"
        echo ""
        fatal "Verzeichnis-Erstellung fehlgeschlagen. Berechtigungen pruefen."
    fi
}

setup_directories() {
    header "Verzeichnisstruktur erstellen"

    # Core directories that Jarvis needs
    local dirs=(
        "$JARVIS_HOME"
        "$JARVIS_HOME/memory"
        "$JARVIS_HOME/memory/episodes"
        "$JARVIS_HOME/memory/procedures"
        "$JARVIS_HOME/memory/knowledge"
        "$JARVIS_HOME/memory/sessions"
        "$JARVIS_HOME/index"
        "$JARVIS_HOME/logs"
        "$JARVIS_HOME/cache"
        "$JARVIS_HOME/workspace"
        "$JARVIS_HOME/workspace/tmp"
    )

    for dir in "${dirs[@]}"; do
        create_directory_safe "$dir"
    done

    # Additionally run jarvis --init-only for any extra setup (with timeout)
    info "Fuehre jarvis --init-only aus..."
    if timeout 30 "$VENV_DIR/bin/jarvis" --init-only 2>/dev/null; then
        success "jarvis --init-only erfolgreich"
    else
        local _exit_code=$?
        if [[ $_exit_code -eq 124 ]]; then
            warn "jarvis --init-only Timeout nach 30s -- uebersprungen"
        else
            warn "jarvis --init-only fehlgeschlagen (Exit-Code: $_exit_code) -- uebersprungen"
        fi
    fi
    success "Verzeichnisstruktur in $JARVIS_HOME vollstaendig"

    # Config-Datei
    local config_file="$JARVIS_HOME/config.yaml"
    if [[ -f "$config_file" ]]; then
        info "config.yaml bereits vorhanden -- wird nicht ueberschrieben"
    else
        if [[ -f "$REPO_DIR/config.yaml.example" ]]; then
            cp "$REPO_DIR/config.yaml.example" "$config_file"
            success "config.yaml erstellt aus Vorlage"
        else
            warn "config.yaml.example nicht gefunden -- uebersprungen"
        fi
    fi

    # Locale-basierte Spracherkennung
    if [[ -f "$config_file" ]] && ! grep -q "^language:" "$config_file" 2>/dev/null; then
        local _lang_code="${LANG%%_*}"
        _lang_code="${_lang_code:-de}"
        local _detected_lang="en"
        if [[ "$_lang_code" == "de" ]]; then
            _detected_lang="de"
        fi
        # Sprache an den Anfang der config.yaml schreiben
        local _tmp_cfg
        _tmp_cfg=$(mktemp "${TMPDIR:-/tmp}/jarvis_cfg_XXXXXX")
        echo "language: \"${_detected_lang}\"" > "$_tmp_cfg"
        cat "$config_file" >> "$_tmp_cfg"
        mv "$_tmp_cfg" "$config_file"
        success "Sprache erkannt: ${_detected_lang} (Locale: ${_lang_code})"
    fi

    # .env (optional)
    local env_file="$JARVIS_HOME/.env"
    if [[ ! -f "$env_file" ]]; then
        if [[ -f "$REPO_DIR/.env.example" ]]; then
            cp "$REPO_DIR/.env.example" "$env_file"
            chmod 600 "$env_file"
            success ".env erstellt (Berechtigungen: 600)"
        fi
    fi
}

# ============================================================================
# Schritt 5: Systemd-Services
# ============================================================================

install_systemd() {
    header "Systemd-Services"

    local service_dir="$HOME/.config/systemd/user"
    mkdir -p "$service_dir"

    # Jarvis Core Service
    local service_file="$service_dir/jarvis.service"
    cat > "$service_file" << UNIT
[Unit]
Description=Jarvis Agent OS
Documentation=https://github.com/team-soellner/jarvis
After=network-online.target ollama.service
Wants=network-online.target

[Service]
Type=simple
ExecStart=${VENV_DIR}/bin/jarvis
WorkingDirectory=${JARVIS_HOME}
EnvironmentFile=-${JARVIS_HOME}/.env
Restart=on-failure
RestartSec=10
WatchdogSec=300

# Sicherheit
NoNewPrivileges=true
ProtectHome=read-only
ReadWritePaths=${JARVIS_HOME} /tmp/jarvis

# Logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier=jarvis

[Install]
WantedBy=default.target
UNIT
    success "jarvis.service erstellt"

    # WebUI Service (optional)
    local webui_file="$service_dir/jarvis-webui.service"
    cat > "$webui_file" << UNIT
[Unit]
Description=Jarvis Web-UI (FastAPI)
Documentation=https://github.com/team-soellner/jarvis
After=jarvis.service
BindsTo=jarvis.service

[Service]
Type=simple
ExecStart=${VENV_DIR}/bin/python -m uvicorn jarvis.channels.webui:create_app --host 127.0.0.1 --port 8080 --factory
WorkingDirectory=${JARVIS_HOME}
EnvironmentFile=-${JARVIS_HOME}/.env
Restart=on-failure
RestartSec=5

# Sicherheit
NoNewPrivileges=true
ProtectHome=read-only
ReadWritePaths=${JARVIS_HOME} /tmp/jarvis

# Logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier=jarvis-webui

[Install]
WantedBy=default.target
UNIT
    success "jarvis-webui.service erstellt"

    # Daemon Reload
    systemctl --user daemon-reload 2>/dev/null || true
    success "systemd daemon-reload"

    info "Services verwalten:"
    info "  systemctl --user start jarvis        # Starten"
    info "  systemctl --user stop jarvis         # Stoppen"
    info "  systemctl --user enable jarvis       # Autostart"
    info "  journalctl --user -u jarvis -f       # Logs"
    info "  systemctl --user start jarvis-webui  # Web-UI starten"
}

# ============================================================================
# Schritt 6: Logrotate
# ============================================================================

setup_logrotate() {
    header "Log-Rotation"

    local logrotate_dir="$JARVIS_HOME/logrotate.d"
    mkdir -p "$logrotate_dir"

    cat > "$logrotate_dir/jarvis" << CONF
${JARVIS_HOME}/logs/*.log {
    daily
    missingok
    rotate 14
    compress
    delaycompress
    notifempty
    dateext
    dateformat -%Y-%m-%d
    create 0640
}

${JARVIS_HOME}/logs/*.jsonl {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    dateext
    dateformat -%Y-%m-%d
    create 0640
}
CONF
    success "logrotate-Konfiguration erstellt"
    info "Fuer System-Logrotate: sudo ln -s $logrotate_dir/jarvis /etc/logrotate.d/jarvis"
}

# ============================================================================
# Schritt 7: Smoke-Test
# ============================================================================

run_smoke_test() {
    header "Smoke-Test"

    if [[ -f "$REPO_DIR/scripts/smoke_test.py" ]]; then
        "$VENV_DIR/bin/python" "$REPO_DIR/scripts/smoke_test.py" \
            --jarvis-home "$JARVIS_HOME" \
            --ollama-url "$OLLAMA_URL" \
            --venv "$VENV_DIR"
    else
        warn "smoke_test.py nicht gefunden -- uebersprungen"
    fi

    # LLM-Rauchtest: Kurze Anfrage an Ollama
    if curl -sf --max-time 3 "${OLLAMA_URL}/api/version" &>/dev/null; then
        info "LLM-Rauchtest..."
        local _llm_response
        _llm_response=$(curl -sf --max-time 30 "${OLLAMA_URL}/api/chat" \
            -d '{"model":"qwen3:8b","messages":[{"role":"user","content":"Sage kurz Hallo."}],"stream":false}' \
            2>/dev/null)
        if [[ -n "$_llm_response" ]]; then
            local _llm_answer
            _llm_answer=$(echo "$_llm_response" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(data.get('message', {}).get('content', '').strip()[:80])
except: pass
" 2>/dev/null)
            if [[ -n "$_llm_answer" ]]; then
                success "LLM antwortet: $_llm_answer"
            else
                warn "LLM antwortete leer -- Modell evtl. noch nicht bereit"
            fi
        else
            warn "LLM-Rauchtest: Keine Antwort (Timeout oder Modell nicht geladen)"
        fi
    else
        info "LLM-Rauchtest uebersprungen (Ollama nicht erreichbar)"
    fi
}

# ============================================================================
# Schritt 8: Shell-Integration
# ============================================================================

setup_shell_integration() {
    header "Shell-Integration"

    local shell_rc=""
    if [[ -f "$HOME/.bashrc" ]]; then
        shell_rc="$HOME/.bashrc"
    elif [[ -f "$HOME/.zshrc" ]]; then
        shell_rc="$HOME/.zshrc"
    fi

    local alias_line="alias jarvis='${VENV_DIR}/bin/jarvis'"
    local activate_line="# Jarvis Agent OS"

    if [[ -n "$shell_rc" ]]; then
        if grep -qF "Jarvis Agent OS" "$shell_rc" 2>/dev/null; then
            info "Shell-Integration bereits vorhanden in $shell_rc"
        else
            {
                echo ""
                echo "$activate_line"
                echo "$alias_line"
            } >> "$shell_rc"
            success "Alias 'jarvis' zu $shell_rc hinzugefuegt"
        fi
    else
        info "Kein .bashrc/.zshrc gefunden -- fuege manuell hinzu:"
        info "  $alias_line"
    fi
}

# ============================================================================
# Schritt 8b: .desktop-Dateien (Linux Desktop-Integration)
# ============================================================================

create_desktop_entry() {
    header "Desktop-Integration"

    local apps_dir="$HOME/.local/share/applications"
    mkdir -p "$apps_dir"

    # CLI-Launcher
    cat > "$apps_dir/cognithor.desktop" << DESKTOP
[Desktop Entry]
Version=1.0
Type=Application
Name=Cognithor (CLI)
Comment=Cognithor Agent OS - Terminal
Exec=${VENV_DIR}/bin/jarvis
Icon=utilities-terminal
Terminal=true
Categories=Utility;Development;
StartupNotify=false
DESKTOP
    success "cognithor.desktop erstellt"

    # Web-UI Launcher
    cat > "$apps_dir/cognithor-webui.desktop" << DESKTOP
[Desktop Entry]
Version=1.0
Type=Application
Name=Cognithor Web-UI
Comment=Cognithor Agent OS - Web-Interface
Exec=bash -c '${VENV_DIR}/bin/python -m jarvis --no-cli & sleep 3 && xdg-open http://localhost:8741; wait'
Icon=applications-internet
Terminal=false
Categories=Utility;Development;
StartupNotify=true
DESKTOP
    success "cognithor-webui.desktop erstellt"

    # Desktop-Datenbank aktualisieren
    if check_command update-desktop-database; then
        update-desktop-database "$apps_dir" 2>/dev/null || true
        success "Desktop-Datenbank aktualisiert"
    fi
}

# ============================================================================
# Deinstallation
# ============================================================================

uninstall() {
    header "Jarvis deinstallieren"

    warn "Dies entfernt die Jarvis-Installation (NICHT die Daten in ~/.jarvis)"

    read -rp "Fortfahren? [y/N] " confirm
    if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
        info "Abgebrochen."
        exit 0
    fi

    # Services stoppen
    systemctl --user stop jarvis jarvis-webui 2>/dev/null || true
    systemctl --user disable jarvis jarvis-webui 2>/dev/null || true

    # Service-Dateien entfernen
    rm -f "$HOME/.config/systemd/user/jarvis.service"
    rm -f "$HOME/.config/systemd/user/jarvis-webui.service"
    systemctl --user daemon-reload 2>/dev/null || true

    # venv entfernen
    if [[ -d "$VENV_DIR" ]]; then
        rm -rf "$VENV_DIR"
        success "Virtual Environment entfernt"
    fi

    # Shell-Alias entfernen (portable: works on GNU sed and BSD/macOS sed)
    for rc in "$HOME/.bashrc" "$HOME/.zshrc"; do
        if [[ -f "$rc" ]] && grep -q "Jarvis Agent OS\|jarvis.*venv.*bin.*jarvis" "$rc" 2>/dev/null; then
            grep -v "Jarvis Agent OS" "$rc" | grep -v "jarvis.*venv.*bin.*jarvis" > "${rc}.jarvis_tmp" \
                && mv "${rc}.jarvis_tmp" "$rc" \
                || rm -f "${rc}.jarvis_tmp"
        fi
    done

    success "Jarvis deinstalliert"
    info "Daten in $JARVIS_HOME wurden NICHT geloescht"
    info "Zum vollstaendigen Entfernen: rm -rf $JARVIS_HOME"
}

# ============================================================================
# Zusammenfassung
# ============================================================================

show_summary() {
    header "Installation abgeschlossen"

    echo -e "${GREEN}${BOLD}"
    cat << 'DONE'
  [OK] Jarvis Agent OS erfolgreich installiert!
DONE
    echo -e "${NC}"

    echo "  Starten:"
    echo "    jarvis                              # CLI-Chat"
    echo "    jarvis --config ~/meine-config.yaml # Eigene Config"
    echo ""
    echo "  Systemd:"
    echo "    systemctl --user start jarvis       # Als Service"
    echo "    systemctl --user enable jarvis      # Autostart"
    echo ""
    echo "  Verzeichnisse:"
    echo "    $JARVIS_HOME/                       # Home"
    echo "    $JARVIS_HOME/config.yaml            # Konfiguration"
    echo "    $JARVIS_HOME/memory/                # Alle Erinnerungen"
    echo "    $JARVIS_HOME/logs/                  # Logs + Audit"
    echo ""
    echo "  Naechste Schritte:"
    echo "    1. config.yaml pruefen und anpassen"
    echo "    2. jarvis starten und testen"
    echo "    3. Optional: Telegram/WebUI aktivieren"
    echo ""
}

# ============================================================================
# Main
# ============================================================================

main() {
    show_banner

    # Parse Argumente
    local mode="interactive"
    for arg in "$@"; do
        case "$arg" in
            --use-uv) USE_UV=true ;;
            --minimal|--full|--systemd|--uninstall|--help|-h) mode="$arg" ;;
        esac
    done

    case "$mode" in
        --uninstall)
            uninstall
            exit 0
            ;;
        --systemd)
            install_systemd
            exit 0
            ;;
        --minimal)
            check_prerequisites
            detect_installer
            setup_venv
            install_jarvis ""
            setup_directories
            run_smoke_test
            show_summary
            ;;
        --full)
            check_prerequisites
            detect_installer
            ensure_ollama_models
            setup_venv
            install_jarvis "full"
            setup_directories
            install_systemd
            setup_logrotate
            setup_shell_integration
            create_desktop_entry
            run_smoke_test
            show_summary
            ;;
        --help|-h)
            echo "Nutzung: $0 [--minimal|--full|--use-uv|--systemd|--uninstall|--help]"
            echo ""
            echo "  (ohne Argumente)  Interaktive Installation"
            echo "  --minimal         Nur Core-Pakete"
            echo "  --full            Alles inkl. Voice + Systemd"
            echo "  --use-uv          uv statt pip verwenden (10x schneller)"
            echo "  --systemd         Nur Systemd-Services installieren"
            echo "  --uninstall       Deinstallation"
            exit 0
            ;;
        *)
            # Interaktive Installation (Default)
            check_prerequisites
            detect_installer
            ensure_ollama_models
            setup_venv
            install_jarvis "all,dev"
            setup_directories
            install_systemd
            setup_logrotate
            setup_shell_integration
            create_desktop_entry
            run_smoke_test
            show_summary
            ;;
    esac
}

main "$@"
