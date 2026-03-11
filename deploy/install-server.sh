#!/usr/bin/env bash
# ============================================================================
# Cognithor · Bare-Metal Server Install Script
# ============================================================================
# Supported: Ubuntu 22.04 / 24.04, Debian 12
#
# Usage:
#   sudo bash deploy/install-server.sh --domain jarvis.example.com --email admin@example.com
#   sudo bash deploy/install-server.sh --domain test.local --self-signed
#   sudo bash deploy/install-server.sh --uninstall
#
# Options:
#   --domain DOMAIN    Server domain (required for install)
#   --email EMAIL      Email for Let's Encrypt (optional)
#   --no-ollama        Skip Ollama installation
#   --no-nginx         Skip Nginx installation
#   --self-signed      Use self-signed certificate instead of Let's Encrypt
#   --uninstall        Remove Cognithor installation (keeps data)
# ============================================================================

set -euo pipefail

# ── Constants ──────────────────────────────────────────────────
INSTALL_DIR="/opt/cognithor"
DATA_DIR="/var/lib/cognithor"
VENV_DIR="${INSTALL_DIR}/venv"
SERVICE_USER="cognithor"
SERVICE_GROUP="cognithor"
PYTHON_MIN="3.12"

# ── Colors ─────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }
log_step()  { echo -e "\n${BLUE}──── $* ────${NC}"; }

# ── Parse Arguments ───────────────────────────────────────────
DOMAIN=""
EMAIL=""
INSTALL_OLLAMA=true
INSTALL_NGINX=true
SELF_SIGNED=false
UNINSTALL=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --domain)      DOMAIN="$2"; shift 2 ;;
        --email)       EMAIL="$2"; shift 2 ;;
        --no-ollama)   INSTALL_OLLAMA=false; shift ;;
        --no-nginx)    INSTALL_NGINX=false; shift ;;
        --self-signed) SELF_SIGNED=true; shift ;;
        --uninstall)   UNINSTALL=true; shift ;;
        -h|--help)
            echo "Usage: sudo bash $0 --domain DOMAIN [--email EMAIL] [--no-ollama] [--no-nginx] [--self-signed]"
            echo "       sudo bash $0 --uninstall"
            exit 0
            ;;
        *) log_error "Unknown option: $1"; exit 1 ;;
    esac
done

# ── Root Check ─────────────────────────────────────────────────
if [[ $EUID -ne 0 ]]; then
    log_error "This script must be run as root (sudo)"
    exit 1
fi

# ── OS Check ───────────────────────────────────────────────────
check_os() {
    if [[ ! -f /etc/os-release ]]; then
        log_error "Cannot determine OS. Only Ubuntu 22.04/24.04 and Debian 12 are supported."
        exit 1
    fi
    source /etc/os-release
    case "${ID}-${VERSION_ID}" in
        ubuntu-22.04|ubuntu-24.04|debian-12)
            log_info "Detected: ${PRETTY_NAME}"
            ;;
        *)
            log_warn "Untested OS: ${PRETTY_NAME}. Proceeding anyway..."
            ;;
    esac
}

# ── Uninstall ──────────────────────────────────────────────────
do_uninstall() {
    log_step "Uninstalling Cognithor"

    # Stop and disable services
    systemctl stop cognithor-webui.service 2>/dev/null || true
    systemctl stop cognithor.service 2>/dev/null || true
    systemctl disable cognithor-webui.service 2>/dev/null || true
    systemctl disable cognithor.service 2>/dev/null || true

    # Remove service files
    rm -f /etc/systemd/system/cognithor.service
    rm -f /etc/systemd/system/cognithor-webui.service
    systemctl daemon-reload

    # Remove installation directory (not data)
    if [[ -d "${INSTALL_DIR}" ]]; then
        rm -rf "${INSTALL_DIR}"
        log_info "Removed ${INSTALL_DIR}"
    fi

    # Keep data directory
    if [[ -d "${DATA_DIR}" ]]; then
        log_warn "Data directory preserved: ${DATA_DIR}"
        log_warn "To remove data: rm -rf ${DATA_DIR}"
    fi

    # Remove user (optional)
    if id "${SERVICE_USER}" &>/dev/null; then
        log_warn "System user '${SERVICE_USER}' preserved."
        log_warn "To remove: userdel ${SERVICE_USER}"
    fi

    log_info "Uninstall complete."
    exit 0
}

if $UNINSTALL; then
    do_uninstall
fi

# ── Validate Arguments ────────────────────────────────────────
if [[ -z "${DOMAIN}" ]]; then
    log_error "--domain is required. Example: --domain jarvis.example.com"
    exit 1
fi

# ── Main Installation ─────────────────────────────────────────

check_os

# Step 1: System packages
log_step "Installing system packages"
apt-get update -qq
apt-get install -y -qq \
    python3 python3-venv python3-pip python3-dev \
    build-essential git curl wget \
    ca-certificates gnupg lsb-release

# Check Python version
PYTHON_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
if python3 -c "import sys; exit(0 if sys.version_info >= (3, 12) else 1)"; then
    log_info "Python ${PYTHON_VER} OK (>= ${PYTHON_MIN})"
else
    log_error "Python ${PYTHON_VER} is too old. Need >= ${PYTHON_MIN}"
    exit 1
fi

# Step 2: Create system user
log_step "Creating system user '${SERVICE_USER}'"
if id "${SERVICE_USER}" &>/dev/null; then
    log_info "User '${SERVICE_USER}' already exists"
else
    useradd --system --create-home --home-dir "${DATA_DIR}" \
            --shell /usr/sbin/nologin --group "${SERVICE_USER}"
    log_info "Created user '${SERVICE_USER}' with home ${DATA_DIR}"
fi

# Step 3: Install Cognithor
log_step "Installing Cognithor to ${INSTALL_DIR}"
mkdir -p "${INSTALL_DIR}"

# Copy source (if running from repo)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "${SCRIPT_DIR}/pyproject.toml" ]]; then
    log_info "Installing from local source: ${SCRIPT_DIR}"
    cp -r "${SCRIPT_DIR}/src" "${INSTALL_DIR}/"
    cp "${SCRIPT_DIR}/pyproject.toml" "${INSTALL_DIR}/"
    cp "${SCRIPT_DIR}/README.md" "${INSTALL_DIR}/" 2>/dev/null || true
else
    log_error "Cannot find pyproject.toml. Run this script from the repository root."
    exit 1
fi

# Create venv and install
log_info "Creating virtual environment..."
python3 -m venv "${VENV_DIR}"
"${VENV_DIR}/bin/pip" install --upgrade pip -q
"${VENV_DIR}/bin/pip" install -q "${INSTALL_DIR}[all]"
log_info "Cognithor installed: $(${VENV_DIR}/bin/jarvis --version)"

# Create data directories
mkdir -p "${DATA_DIR}"/{logs,memory,workspace,cache}
chown -R "${SERVICE_USER}:${SERVICE_GROUP}" "${DATA_DIR}"

# Create .env if not exists
if [[ ! -f "${DATA_DIR}/.env" ]]; then
    cat > "${DATA_DIR}/.env" <<'ENVEOF'
# Cognithor environment (auto-generated by install-server.sh)
JARVIS_HOME=/var/lib/cognithor
JARVIS_LOGGING_LEVEL=INFO
# JARVIS_API_TOKEN=change-me
# JARVIS_TELEGRAM_TOKEN=
# JARVIS_TELEGRAM_ALLOWED_USERS=
ENVEOF
    chown "${SERVICE_USER}:${SERVICE_GROUP}" "${DATA_DIR}/.env"
    chmod 600 "${DATA_DIR}/.env"
    log_info "Created ${DATA_DIR}/.env (edit to configure)"
fi

# Step 4: Ollama
if $INSTALL_OLLAMA; then
    log_step "Installing Ollama"
    if command -v ollama &>/dev/null; then
        log_info "Ollama already installed: $(ollama --version)"
    else
        INSTALL_SCRIPT=$(mktemp)
        curl -fsSL --max-time 60 https://ollama.com/install.sh -o "$INSTALL_SCRIPT"
        if [ $? -eq 0 ]; then
            log_warn "Downloaded Ollama installer to $INSTALL_SCRIPT — executing..."
            sh "$INSTALL_SCRIPT"
        fi
        rm -f "$INSTALL_SCRIPT"
        log_info "Ollama installed"
    fi
    # Ensure ollama service is running
    systemctl enable --now ollama 2>/dev/null || true
    log_info "Pulling default models (this may take a while)..."
    ollama pull qwen3:8b || log_warn "Failed to pull qwen3:8b — pull manually later"
    ollama pull nomic-embed-text || log_warn "Failed to pull nomic-embed-text — pull manually later"
else
    log_info "Skipping Ollama installation (--no-ollama)"
fi

# Step 5: Systemd services
log_step "Installing systemd services"

cat > /etc/systemd/system/cognithor.service <<EOF
[Unit]
Description=Cognithor Agent OS
Documentation=https://github.com/team-soellner/jarvis
After=network-online.target ollama.service
Wants=network-online.target

[Service]
Type=simple
User=${SERVICE_USER}
Group=${SERVICE_GROUP}
ExecStart=${VENV_DIR}/bin/jarvis --no-cli --api-host 0.0.0.0
WorkingDirectory=${DATA_DIR}
EnvironmentFile=-${DATA_DIR}/.env

Restart=on-failure
RestartSec=10
StartLimitIntervalSec=300
StartLimitBurst=5
WatchdogSec=300

# Security hardening
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=${DATA_DIR} /tmp
PrivateTmp=true

# Resource limits
MemoryMax=4G
TasksMax=64

# Logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier=cognithor

[Install]
WantedBy=multi-user.target
EOF

cat > /etc/systemd/system/cognithor-webui.service <<EOF
[Unit]
Description=Cognithor Web UI (FastAPI + WebSocket)
Documentation=https://github.com/team-soellner/jarvis
After=cognithor.service
BindsTo=cognithor.service

[Service]
Type=simple
User=${SERVICE_USER}
Group=${SERVICE_GROUP}
ExecStart=${VENV_DIR}/bin/python -m uvicorn jarvis.channels.webui:create_app \
    --host 0.0.0.0 \
    --port 8080 \
    --factory \
    --log-level warning
WorkingDirectory=${DATA_DIR}
EnvironmentFile=-${DATA_DIR}/.env

Restart=on-failure
RestartSec=5

# Security hardening
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=${DATA_DIR} /tmp
PrivateTmp=true

# Resource limits
MemoryMax=1G
TasksMax=32

# Logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier=cognithor-webui

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable cognithor.service
systemctl enable cognithor-webui.service
log_info "Systemd services installed and enabled"

# Step 6: Nginx
if $INSTALL_NGINX; then
    log_step "Installing Nginx"
    apt-get install -y -qq nginx

    # TLS certificates
    CERT_DIR="/etc/cognithor/certs"
    mkdir -p "${CERT_DIR}"

    if $SELF_SIGNED; then
        log_info "Generating self-signed certificate for ${DOMAIN}"
        openssl req -x509 -nodes -days 365 \
            -newkey rsa:2048 \
            -keyout "${CERT_DIR}/privkey.pem" \
            -out "${CERT_DIR}/fullchain.pem" \
            -subj "/CN=${DOMAIN}" \
            2>/dev/null
        log_warn "Self-signed certificate generated. Browsers will show a warning."
    elif [[ -n "${EMAIL}" ]]; then
        log_info "Requesting Let's Encrypt certificate for ${DOMAIN}"
        apt-get install -y -qq certbot python3-certbot-nginx
        certbot --nginx -d "${DOMAIN}" --email "${EMAIL}" --agree-tos --non-interactive || {
            log_warn "Let's Encrypt failed. Falling back to self-signed."
            openssl req -x509 -nodes -days 365 \
                -newkey rsa:2048 \
                -keyout "${CERT_DIR}/privkey.pem" \
                -out "${CERT_DIR}/fullchain.pem" \
                -subj "/CN=${DOMAIN}" \
                2>/dev/null
        }
    else
        log_warn "No --email and no --self-signed. Generating self-signed certificate."
        openssl req -x509 -nodes -days 365 \
            -newkey rsa:2048 \
            -keyout "${CERT_DIR}/privkey.pem" \
            -out "${CERT_DIR}/fullchain.pem" \
            -subj "/CN=${DOMAIN}" \
            2>/dev/null
    fi

    # Write nginx config
    cat > /etc/nginx/sites-available/cognithor <<NGINXEOF
# Auto-generated by install-server.sh
upstream cognithor_webui {
    server 127.0.0.1:8080;
}

upstream cognithor_api {
    server 127.0.0.1:8741;
}

server {
    listen 80;
    server_name ${DOMAIN};

    location /health {
        proxy_pass http://cognithor_webui/api/v1/health;
    }

    location / {
        return 301 https://\$host\$request_uri;
    }
}

server {
    listen 443 ssl;
    http2 on;
    server_name ${DOMAIN};

    ssl_certificate     ${CERT_DIR}/fullchain.pem;
    ssl_certificate_key ${CERT_DIR}/privkey.pem;
    ssl_protocols       TLSv1.2 TLSv1.3;
    ssl_ciphers         HIGH:!aNULL:!MD5;

    client_max_body_size 55m;
    proxy_read_timeout 300s;

    add_header X-Frame-Options DENY always;
    add_header X-Content-Type-Options nosniff always;
    add_header Strict-Transport-Security "max-age=63072000; includeSubDomains" always;

    location / {
        proxy_pass http://cognithor_webui;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }

    location /ws/ {
        proxy_pass http://cognithor_webui;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_read_timeout 3600s;
    }

    location /control/ {
        rewrite ^/control/(.*) /\$1 break;
        proxy_pass http://cognithor_api;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
}
NGINXEOF

    # Enable site
    ln -sf /etc/nginx/sites-available/cognithor /etc/nginx/sites-enabled/
    rm -f /etc/nginx/sites-enabled/default

    nginx -t && systemctl reload nginx
    log_info "Nginx configured for ${DOMAIN}"
else
    log_info "Skipping Nginx installation (--no-nginx)"
fi

# Step 7: Firewall (ufw)
log_step "Configuring firewall (ufw)"
if command -v ufw &>/dev/null; then
    ufw allow 22/tcp comment "SSH" 2>/dev/null || true
    ufw allow 80/tcp comment "HTTP" 2>/dev/null || true
    ufw allow 443/tcp comment "HTTPS" 2>/dev/null || true
    # Do NOT expose 8080 or 8741 directly — Nginx handles it
    ufw --force enable 2>/dev/null || true
    log_info "Firewall configured (SSH, HTTP, HTTPS)"
else
    log_warn "ufw not found, skipping firewall configuration"
fi

# Step 8: Start services
log_step "Starting Cognithor services"
systemctl start cognithor.service
sleep 3
systemctl start cognithor-webui.service

# Wait for health
log_info "Waiting for services to become healthy..."
for i in $(seq 1 15); do
    if curl -sf http://127.0.0.1:8741/api/v1/health > /dev/null 2>&1; then
        log_info "Cognithor API is healthy"
        break
    fi
    sleep 2
done

for i in $(seq 1 10); do
    if curl -sf http://127.0.0.1:8080/api/v1/health > /dev/null 2>&1; then
        log_info "Cognithor WebUI is healthy"
        break
    fi
    sleep 2
done

# ── Summary ───────────────────────────────────────────────────
log_step "Installation Complete"
echo ""
echo "  Domain:     ${DOMAIN}"
echo "  Install:    ${INSTALL_DIR}"
echo "  Data:       ${DATA_DIR}"
echo "  Config:     ${DATA_DIR}/.env"
echo "  Venv:       ${VENV_DIR}"
echo ""
echo "  Services:"
echo "    systemctl status cognithor"
echo "    systemctl status cognithor-webui"
echo "    journalctl -u cognithor -f"
echo ""
if $INSTALL_NGINX; then
    echo "  Web UI:     https://${DOMAIN}/"
    echo "  Health:     https://${DOMAIN}/health"
    echo "  API:        https://${DOMAIN}/control/api/v1/health"
else
    echo "  Web UI:     http://${DOMAIN}:8080/"
    echo "  API:        http://${DOMAIN}:8741/api/v1/health"
fi
echo ""
echo "  Next steps:"
echo "    1. Edit ${DATA_DIR}/.env (set JARVIS_API_TOKEN)"
echo "    2. Pull Ollama models: ollama pull qwen3:32b"
echo "    3. Restart: systemctl restart cognithor"
echo ""
