# 05 · Deployment

Three ways to run Cognithor in production — local, Docker, or systemd.

**Prerequisites**
- Completed: [01 · First Crew](01-first-crew.en.md)

**Time:** 5 minutes
**End state:** You know the three deployment paths and can pick the right one.

---

## 1. Local development

Default mode — CLI + REST API + Flutter Command Center on `localhost`:

```bash
cognithor
# → starts CLI + local API on http://localhost:8741
```

Useful flags:

```bash
cognithor --api-port 9000       # alternative port
cognithor --no-cli              # headless: API only, no interactive shell
cognithor --log-level DEBUG     # verbose logging
cognithor --lite                # minimal startup (no optional deps)
cognithor --config /path/to/config.yaml  # custom config
```

## 2. Docker Compose

Production-capable setup with Ollama + Redis + Cognithor in one network:

```bash
git clone https://github.com/Alex8791-cyber/cognithor.git
cd cognithor
cp .env.example .env
# edit .env — COGNITHOR_API_PORT=8741, etc.

docker compose up -d
docker compose logs -f cognithor
```

**Ports:**
- `8741` — Cognithor REST API
- `11434` — Ollama
- `6379` — Redis (optional, for distributed locks)

**Volumes:**
- `./data/` → `/home/cognithor/.cognithor/` (persistent memory, vault, audit chain)

## 3. systemd (Linux server)

Create `/etc/systemd/system/cognithor.service`:

```ini
[Unit]
Description=Cognithor Agent OS
After=network-online.target ollama.service
Wants=ollama.service

[Service]
Type=simple
User=cognithor
Group=cognithor
Environment="COGNITHOR_API_PORT=8741"
Environment="OLLAMA_HOST=http://127.0.0.1:11434"
ExecStart=/usr/local/bin/cognithor --no-cli
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Enable:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now cognithor.service
sudo systemctl status cognithor
sudo journalctl -u cognithor -f
```

## 4. Configuration via env vars

Every key in `config.yaml` is also settable as a `COGNITHOR_*` env var. Cascade (highest priority first):

1. `COGNITHOR_*` env vars
2. `config.yaml` from `--config` or `~/.cognithor/config.yaml`
3. Package defaults

Examples:

```bash
export COGNITHOR_API_PORT=9000
export COGNITHOR_LLM_PROVIDER=ollama
export COGNITHOR_MODEL_PLANNER=qwen3:32b
export COGNITHOR_MODEL_EXECUTOR=qwen3:8b
```

## 5. Health-check endpoints

For load balancers / orchestrators:

```bash
curl http://localhost:8741/health
# → {"status":"ok","version":"0.93.0","uptime_s":1234}

curl http://localhost:8741/metrics
# → Prometheus format
```

## 6. Pre-production security checklist

- [ ] `COGNITHOR_API_PORT` bound only locally, or behind a reverse proxy (TLS!)
- [ ] Credential-vault key (`~/.cognithor/vault.key`) backed up separately
- [ ] Gatekeeper policy at **ORANGE by default** — whitelist explicitly, don't open
- [ ] Event bus / audit log regularly exported (SHA-256 chain)
- [ ] Ollama not exposed publicly (default: `127.0.0.1:11434`)

---

**Next:** [06 · Next Steps](06-next-steps.en.md)
