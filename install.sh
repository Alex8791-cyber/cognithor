#!/usr/bin/env bash
# ============================================================================
# Jarvis · Agent OS – Installations-Script
# ============================================================================
#
# Nutzung:
#   ./install.sh              Vollinstallation (interaktiv)
#   ./install.sh --minimal    Nur Core (kein Web, kein Telegram)
#   ./install.sh --full       Alles inkl. Voice
#   ./install.sh --systemd    Nur Systemd-Services installieren
#   ./install.sh --uninstall  Deinstallation
#
# Voraussetzungen:
#   - Python 3.12+
#   - pip
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

# ============================================================================
# Hilfsfunktionen
# ============================================================================

info()    { echo -e "${BLUE}ℹ${NC}  $*"; }
success() { echo -e "${GREEN}✓${NC}  $*"; }
warn()    { echo -e "${YELLOW}⚠${NC}  $*"; }
error()   { echo -e "${RED}✗${NC}  $*" >&2; }
fatal()   { error "$*"; exit 1; }
header()  { echo -e "\n${BOLD}${CYAN}═══ $* ═══${NC}\n"; }

check_command() {
    command -v "$1" &>/dev/null
}

version_ge() {
    # Prüft ob Version $1 >= $2
    printf '%s\n%s\n' "$2" "$1" | sort -V | head -n1 | grep -qF "$2"
}

# ============================================================================
# Banner
# ============================================================================

show_banner() {
    echo -e "${BOLD}${CYAN}"
    cat << 'BANNER'

       ██╗ █████╗ ██████╗ ██╗   ██╗██╗███████╗
       ██║██╔══██╗██╔══██╗██║   ██║██║██╔════╝
       ██║███████║██████╔╝██║   ██║██║███████╗
  ██   ██║██╔══██║██╔══██╗╚██╗ ██╔╝██║╚════██║
  ╚█████╔╝██║  ██║██║  ██║ ╚████╔╝ ██║███████║
   ╚════╝ ╚═╝  ╚═╝╚═╝  ╚═╝  ╚═══╝  ╚═╝╚══════╝
           Agent OS · v0.1.0 · Installer

BANNER
    echo -e "${NC}"
}

# ============================================================================
# Schritt 1: Systemvoraussetzungen prüfen
# ============================================================================

check_prerequisites() {
    header "Systemvoraussetzungen prüfen"
    local errors=0

    # Python
    if check_command python3; then
        local py_version
        py_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        if version_ge "$py_version" "$MIN_PYTHON"; then
            success "Python $py_version gefunden"
        else
            error "Python $py_version zu alt (mindestens $MIN_PYTHON benötigt)"
            ((errors++))
        fi
    else
        error "Python3 nicht gefunden"
        info  "Installiere mit: sudo apt install python3.12 python3.12-venv"
        ((errors++))
    fi

    # pip
    if python3 -m pip --version &>/dev/null; then
        success "pip verfügbar"
    else
        error "pip nicht gefunden"
        info  "Installiere mit: sudo apt install python3-pip"
        ((errors++))
    fi

    # venv
    if python3 -m venv --help &>/dev/null; then
        success "venv-Modul verfügbar"
    else
        error "venv nicht gefunden"
        info  "Installiere mit: sudo apt install python3.12-venv"
        ((errors++))
    fi

    # git (optional)
    if check_command git; then
        success "git verfügbar"
    else
        warn "git nicht gefunden (optional, für Updates empfohlen)"
    fi

    # Ollama
    if check_command ollama; then
        local ollama_version
        ollama_version=$(ollama --version 2>/dev/null | head -1 || echo "unbekannt")
        success "Ollama installiert ($ollama_version)"
    else
        warn "Ollama nicht gefunden – LLM-Funktionen sind ohne Ollama eingeschränkt"
        info "Installiere: curl -fsSL https://ollama.com/install.sh | sh"
    fi

    # Ollama-Server erreichbar?
    if curl -sf "${OLLAMA_URL}/api/version" &>/dev/null; then
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
# Schritt 2: Ollama-Modelle
# ============================================================================

ensure_ollama_models() {
    header "Ollama-Modelle prüfen"

    if ! curl -sf "${OLLAMA_URL}/api/version" &>/dev/null; then
        warn "Ollama nicht erreichbar – Modelle manuell herunterladen:"
        info "  ollama pull qwen3:8b"
        info "  ollama pull nomic-embed-text"
        info "  ollama pull qwen3:32b     (optional, wenn ≥24GB VRAM)"
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

    # Pflicht-Modelle
    local required_models=("qwen3:8b" "nomic-embed-text")
    local optional_models=("qwen3:32b")

    for model in "${required_models[@]}"; do
        # Prüfe ob Modell-Name (ohne Tag-Variante) installiert ist
        local base_name="${model%%:*}"
        if echo "$installed" | grep -q "$base_name"; then
            success "$model bereits installiert"
        else
            info "Lade $model herunter (Pflicht-Modell)..."
            ollama pull "$model" || warn "Download von $model fehlgeschlagen – manuell nachholen"
        fi
    done

    for model in "${optional_models[@]}"; do
        local base_name="${model%%:*}"
        if echo "$installed" | grep -q "$base_name"; then
            success "$model bereits installiert"
        else
            info "$model nicht installiert (optional, für bessere Qualität)"
            info "  Installiere später mit: ollama pull $model"
        fi
    done
}

# ============================================================================
# Schritt 3: Virtual Environment + Installation
# ============================================================================

setup_venv() {
    header "Python Virtual Environment"

    if [[ -d "$VENV_DIR" ]]; then
        info "Bestehendes venv gefunden: $VENV_DIR"
        # shellcheck disable=SC1091
        source "$VENV_DIR/bin/activate"
        success "venv aktiviert"
    else
        info "Erstelle venv in $VENV_DIR ..."
        python3 -m venv "$VENV_DIR"
        # shellcheck disable=SC1091
        source "$VENV_DIR/bin/activate"
        pip install --upgrade pip setuptools wheel --quiet
        success "venv erstellt und aktiviert"
    fi
}

install_jarvis() {
    header "Jarvis installieren"

    local install_extras="$1"

    info "Installiere jarvis[$install_extras] aus $REPO_DIR ..."
    pip install -e "${REPO_DIR}[${install_extras}]" --quiet 2>&1 | tail -5

    # Verifiziere Installation
    if python3 -c "import jarvis; print(f'Jarvis v{jarvis.__version__}')" 2>/dev/null; then
        success "Jarvis erfolgreich installiert"
    else
        fatal "Installation fehlgeschlagen"
    fi

    # Jarvis-CLI prüfen
    if "$VENV_DIR/bin/jarvis" --version &>/dev/null; then
        local ver
        ver=$("$VENV_DIR/bin/jarvis" --version 2>&1)
        success "CLI verfügbar: $ver"
    else
        warn "CLI 'jarvis' nicht im PATH – nutze: $VENV_DIR/bin/jarvis"
    fi
}

# ============================================================================
# Schritt 4: Verzeichnisstruktur + Config
# ============================================================================

setup_directories() {
    header "Verzeichnisstruktur erstellen"

    # Jarvis --init-only erstellt alles
    "$VENV_DIR/bin/jarvis" --init-only 2>/dev/null || true
    success "Verzeichnisstruktur in $JARVIS_HOME erstellt"

    # Config-Datei
    local config_file="$JARVIS_HOME/config.yaml"
    if [[ -f "$config_file" ]]; then
        info "config.yaml bereits vorhanden – wird nicht überschrieben"
    else
        cp "$REPO_DIR/config.yaml.example" "$config_file"
        success "config.yaml erstellt aus Vorlage"
    fi

    # .env (optional)
    local env_file="$JARVIS_HOME/.env"
    if [[ ! -f "$env_file" ]]; then
        cp "$REPO_DIR/.env.example" "$env_file"
        chmod 600 "$env_file"
        success ".env erstellt (Berechtigungen: 600)"
    fi

    # Verzeichnis-Übersicht
    info "Struktur:"
    find "$JARVIS_HOME" -maxdepth 2 -type d | sort | head -20 | while read -r dir; do
        echo "    ${dir#$HOME/}"
    done
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
    info "Für System-Logrotate: sudo ln -s $logrotate_dir/jarvis /etc/logrotate.d/jarvis"
}

# ============================================================================
# Schritt 7: Smoke-Test
# ============================================================================

run_smoke_test() {
    header "Smoke-Test"

    "$VENV_DIR/bin/python" "$REPO_DIR/scripts/smoke_test.py" \
        --jarvis-home "$JARVIS_HOME" \
        --ollama-url "$OLLAMA_URL" \
        --venv "$VENV_DIR"
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
            success "Alias 'jarvis' zu $shell_rc hinzugefügt"
        fi
    else
        info "Kein .bashrc/.zshrc gefunden – füge manuell hinzu:"
        info "  $alias_line"
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

    # Shell-Alias entfernen
    for rc in "$HOME/.bashrc" "$HOME/.zshrc"; do
        if [[ -f "$rc" ]]; then
            sed -i '/Jarvis Agent OS/d' "$rc"
            sed -i "/jarvis.*venv.*bin.*jarvis/d" "$rc"
        fi
    done

    success "Jarvis deinstalliert"
    info "Daten in $JARVIS_HOME wurden NICHT gelöscht"
    info "Zum vollständigen Entfernen: rm -rf $JARVIS_HOME"
}

# ============================================================================
# Zusammenfassung
# ============================================================================

show_summary() {
    header "Installation abgeschlossen"

    echo -e "${GREEN}${BOLD}"
    cat << 'DONE'
  ✓ Jarvis Agent OS erfolgreich installiert!
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
    echo "  Nächste Schritte:"
    echo "    1. config.yaml prüfen und anpassen"
    echo "    2. jarvis starten und testen"
    echo "    3. Optional: Telegram/WebUI aktivieren"
    echo ""
}

# ============================================================================
# Main
# ============================================================================

main() {
    show_banner

    local mode="${1:-interactive}"

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
            setup_venv
            install_jarvis ""
            setup_directories
            run_smoke_test
            show_summary
            ;;
        --full)
            check_prerequisites
            ensure_ollama_models
            setup_venv
            install_jarvis "full"
            setup_directories
            install_systemd
            setup_logrotate
            setup_shell_integration
            run_smoke_test
            show_summary
            ;;
        --help|-h)
            echo "Nutzung: $0 [--minimal|--full|--systemd|--uninstall|--help]"
            echo ""
            echo "  (ohne Argumente)  Interaktive Installation"
            echo "  --minimal         Nur Core-Pakete"
            echo "  --full            Alles inkl. Voice + Systemd"
            echo "  --systemd         Nur Systemd-Services installieren"
            echo "  --uninstall       Deinstallation"
            exit 0
            ;;
        *)
            # Interaktive Installation (Default)
            check_prerequisites
            ensure_ollama_models
            setup_venv
            install_jarvis "all,dev"
            setup_directories
            install_systemd
            setup_logrotate
            setup_shell_integration
            run_smoke_test
            show_summary
            ;;
    esac
}

main "$@"
