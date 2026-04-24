# 05 · Deployment

Drei Wege, Cognithor in Produktion zu betreiben — lokal, Docker oder systemd.

**Voraussetzungen**
- Abgeschlossen: [01 · Erste Crew](01-first-crew.md)

**Zeitbedarf:** 5 Minuten
**Endzustand:** Du kennst die drei Deployment-Wege und kannst entscheiden, welcher für dich passt.

---

## 1. Lokales Development

Standard-Modus — CLI + REST-API + Flutter-Command-Center auf `localhost`:

```bash
cognithor
# → Startet CLI + lokales API auf http://localhost:8741
```

Nützliche Flags:

```bash
cognithor --api-port 9000       # Alternativer Port
cognithor --no-cli              # Headless: nur API, keine interaktive Shell
cognithor --log-level DEBUG     # Verbose logging
cognithor --lite                # Minimaler Start (ohne optional deps)
cognithor --config /pfad/zu/config.yaml  # Custom config
```

## 2. Docker Compose

Produktions-taugliches Setup mit Ollama + Redis + Cognithor in einem Netzwerk:

```bash
git clone https://github.com/Alex8791-cyber/cognithor.git
cd cognithor
cp .env.example .env
# .env editieren — COGNITHOR_API_PORT=8741, etc.

docker compose up -d
docker compose logs -f cognithor
```

**Ports:**
- `8741` — Cognithor REST API
- `11434` — Ollama
- `6379` — Redis (optional, für Distributed Locks)

**Volumes:**
- `./data/` → `/home/cognithor/.cognithor/` (persistierte Memory, Vault, Audit-Chain)

## 3. systemd (Linux Server)

Erzeuge `/etc/systemd/system/cognithor.service`:

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

Aktivieren:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now cognithor.service
sudo systemctl status cognithor
sudo journalctl -u cognithor -f
```

## 4. Konfiguration via Env-Vars

Alle Keys aus `config.yaml` sind auch als `COGNITHOR_*` Env-Vars setzbar. Kaskade (höchste Priorität zuerst):

1. `COGNITHOR_*` Env-Vars
2. `config.yaml` aus `--config` oder `~/.cognithor/config.yaml`
3. Package-Defaults

Beispiele:

```bash
export COGNITHOR_API_PORT=9000
export COGNITHOR_LLM_PROVIDER=ollama
export COGNITHOR_MODEL_PLANNER=qwen3:32b
export COGNITHOR_MODEL_EXECUTOR=qwen3:8b
```

## 5. Health-Check-Endpoints

Für Loadbalancer / Orchestrators:

```bash
curl http://localhost:8741/health
# → {"status":"ok","version":"0.93.0","uptime_s":1234}

curl http://localhost:8741/metrics
# → Prometheus-Format
```

## 6. Security-Checkliste vor Production

- [ ] `COGNITHOR_API_PORT` nur lokal gebunden oder hinter Reverse-Proxy (TLS!)
- [ ] Credential-Vault-Key (`~/.cognithor/vault.key`) separat gesichert
- [ ] Gatekeeper-Policy auf **ORANGE by default** — explizit freigeben, nicht öffnen
- [ ] Event-Bus / Audit-Log regelmäßig extern sichern (SHA-256-Chain)
- [ ] Ollama nicht öffentlich exponieren (Standard: `127.0.0.1:11434`)

---

**Next:** [06 · Nächste Schritte](06-next-steps.md)
