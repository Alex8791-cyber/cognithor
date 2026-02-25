#!/usr/bin/env bash
# ============================================================================
# Jarvis · Backup-Script
# ============================================================================
# Erstellt ein komprimiertes Backup von ~/.jarvis/
#
# Nutzung:
#   ./scripts/backup.sh                     # Backup nach ~/.jarvis/backups/
#   ./scripts/backup.sh /mnt/nas/backup     # Backup in eigenes Verzeichnis
#   ./scripts/backup.sh --restore latest    # Letztes Backup wiederherstellen
#   ./scripts/backup.sh --list              # Backups auflisten
#
# Gesichert wird:
#   ✓ config.yaml           Konfiguration
#   ✓ .env                  Umgebungsvariablen
#   ✓ memory/CORE.md        Identität + Regeln
#   ✓ memory/episodes/      Episodisches Gedächtnis
#   ✓ memory/knowledge/     Wissensdateien
#   ✓ memory/procedures/    Gelernte Abläufe
#   ✓ memory/index/         SQLite-Index + Embeddings
#   ✓ logs/audit.jsonl      Audit-Trail (Hash-Chain)
#   ✓ credentials.enc       Verschlüsselte Credentials
#
# NICHT gesichert (wird automatisch erstellt):
#   ✗ venv/                 Virtual Environment
#   ✗ logs/*.log            Normale Logs
#   ✗ workspace/            Temporäre Dateien
# ============================================================================
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

JARVIS_HOME="${JARVIS_HOME:-$HOME/.jarvis}"
DEFAULT_BACKUP_DIR="$JARVIS_HOME/backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

info()    { echo -e "${CYAN}ℹ${NC}  $*"; }
success() { echo -e "${GREEN}✓${NC}  $*"; }
warn()    { echo -e "${YELLOW}⚠${NC}  $*"; }
error()   { echo -e "${RED}✗${NC}  $*" >&2; }

# ============================================================================
# Backup erstellen
# ============================================================================

create_backup() {
    local backup_dir="${1:-$DEFAULT_BACKUP_DIR}"
    mkdir -p "$backup_dir"

    local backup_name="jarvis-backup-${TIMESTAMP}.tar.gz"
    local backup_path="$backup_dir/$backup_name"

    echo -e "\n${BOLD}${CYAN}Jarvis · Backup${NC}\n"
    info "Quelle:  $JARVIS_HOME"
    info "Ziel:    $backup_path"
    echo ""

    # Prüfe ob JARVIS_HOME existiert
    if [[ ! -d "$JARVIS_HOME" ]]; then
        error "$JARVIS_HOME existiert nicht"; exit 1
    fi

    # Dateien zum Sichern sammeln (relativ zu JARVIS_HOME)
    local include_files=()
    local total_size=0

    for item in \
        "config.yaml" \
        ".env" \
        "memory/CORE.md" \
        "memory/episodes" \
        "memory/knowledge" \
        "memory/procedures" \
        "memory/index" \
        "memory/semantic" \
        "logs/audit.jsonl" \
        "credentials.enc" \
        "policies"
    do
        local full_path="$JARVIS_HOME/$item"
        if [[ -e "$full_path" ]]; then
            include_files+=("$item")
            local size
            if [[ -d "$full_path" ]]; then
                size=$(du -sb "$full_path" 2>/dev/null | cut -f1)
            else
                size=$(stat -c%s "$full_path" 2>/dev/null || echo 0)
            fi
            total_size=$((total_size + size))
            success "$item ($(numfmt --to=iec "$size" 2>/dev/null || echo "${size}B"))"
        else
            warn "$item nicht vorhanden – übersprungen"
        fi
    done

    if [[ ${#include_files[@]} -eq 0 ]]; then
        error "Keine Dateien zum Sichern gefunden"; exit 1
    fi

    echo ""
    info "Erstelle Backup ($(numfmt --to=iec "$total_size" 2>/dev/null || echo "${total_size}B"))..."

    # Tar erstellen
    tar -czf "$backup_path" \
        -C "$JARVIS_HOME" \
        "${include_files[@]}" \
        2>/dev/null

    local backup_size
    backup_size=$(stat -c%s "$backup_path" 2>/dev/null || echo "?")

    echo ""
    success "Backup erstellt: $backup_path"
    success "Größe: $(numfmt --to=iec "$backup_size" 2>/dev/null || echo "${backup_size}B") (komprimiert)"
    success "Dateien: ${#include_files[@]}"

    # Alte Backups aufräumen (behalte die letzten 10)
    local count
    count=$(find "$backup_dir" -name "jarvis-backup-*.tar.gz" | wc -l)
    if [[ $count -gt 10 ]]; then
        info "Entferne alte Backups (behalte 10 neueste)..."
        find "$backup_dir" -name "jarvis-backup-*.tar.gz" -type f | \
            sort | head -n $((count - 10)) | \
            xargs rm -f
        success "$((count - 10)) alte Backup(s) entfernt"
    fi
}

# ============================================================================
# Backup wiederherstellen
# ============================================================================

restore_backup() {
    local backup_file="$1"
    local backup_dir="${2:-$DEFAULT_BACKUP_DIR}"

    echo -e "\n${BOLD}${CYAN}Jarvis · Restore${NC}\n"

    # "latest" → Neuestes Backup finden
    if [[ "$backup_file" == "latest" ]]; then
        backup_file=$(find "$backup_dir" -name "jarvis-backup-*.tar.gz" -type f | sort -r | head -1)
        if [[ -z "$backup_file" ]]; then
            error "Kein Backup gefunden in $backup_dir"; exit 1
        fi
        info "Neuestes Backup: $(basename "$backup_file")"
    fi

    if [[ ! -f "$backup_file" ]]; then
        error "Backup nicht gefunden: $backup_file"; exit 1
    fi

    # Sicherheitsabfrage
    warn "Dies überschreibt bestehende Daten in $JARVIS_HOME!"
    read -rp "Fortfahren? [y/N] " confirm
    if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
        info "Abgebrochen."; exit 0
    fi

    # Sicherungskopie des aktuellen Zustands
    if [[ -d "$JARVIS_HOME/memory" ]]; then
        local pre_restore="$JARVIS_HOME/backups/pre-restore-${TIMESTAMP}.tar.gz"
        mkdir -p "$(dirname "$pre_restore")"
        info "Erstelle Sicherungskopie vor Restore..."
        tar -czf "$pre_restore" -C "$JARVIS_HOME" memory/ config.yaml 2>/dev/null || true
        success "Sicherungskopie: $pre_restore"
    fi

    # Restore
    info "Stelle wieder her aus: $(basename "$backup_file")"
    tar -xzf "$backup_file" -C "$JARVIS_HOME"

    success "Restore abgeschlossen"
    info "Starte Jarvis neu, um Änderungen zu laden"
}

# ============================================================================
# Backups auflisten
# ============================================================================

list_backups() {
    local backup_dir="${1:-$DEFAULT_BACKUP_DIR}"

    echo -e "\n${BOLD}${CYAN}Jarvis · Backups${NC}\n"

    if [[ ! -d "$backup_dir" ]]; then
        info "Kein Backup-Verzeichnis: $backup_dir"; return
    fi

    local count=0
    while IFS= read -r file; do
        local size
        size=$(stat -c%s "$file" 2>/dev/null || echo "?")
        local date
        date=$(stat -c%y "$file" 2>/dev/null | cut -d. -f1)
        printf "  %s  %s  %s\n" \
            "$(numfmt --to=iec "$size" 2>/dev/null || echo "${size}B")" \
            "$date" \
            "$(basename "$file")"
        ((count++))
    done < <(find "$backup_dir" -name "jarvis-backup-*.tar.gz" -type f | sort -r)

    echo ""
    if [[ $count -eq 0 ]]; then
        info "Keine Backups vorhanden"
    else
        info "$count Backup(s) gefunden"
    fi
}

# ============================================================================
# Main
# ============================================================================

case "${1:-}" in
    --restore)
        restore_backup "${2:-latest}" "${3:-$DEFAULT_BACKUP_DIR}"
        ;;
    --list)
        list_backups "${2:-$DEFAULT_BACKUP_DIR}"
        ;;
    --help|-h)
        echo "Nutzung: $0 [BACKUP_DIR]"
        echo "         $0 --restore [BACKUP_FILE|latest]"
        echo "         $0 --list [BACKUP_DIR]"
        ;;
    *)
        create_backup "${1:-$DEFAULT_BACKUP_DIR}"
        ;;
esac
