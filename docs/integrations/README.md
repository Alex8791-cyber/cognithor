# Integrations

Die Cognithor-Integrationen sind **MCP-Tools** — offenes Protokoll, self-hostable,
kein Vendor-Lock-In. Die Liste unten wird automatisch aus dem Repo generiert —
kein Vapourware.

**Catalog:** [catalog.json](catalog.json)
**Generator:** `scripts/generate_integrations_catalog.py`
**CI-Verifikation:** `.github/workflows/integrations-catalog.yml` (fails bei Drift)

## Kategorien

Siehe `catalog.json` für die vollständige Liste. Hauptkategorien:

- `filesystem` — Datei-Operationen
- `web` — HTTP / Web-Scraping / Search
- `documents` — PDF, DOCX, Excel
- `browser` — Playwright-basierte Browser-Automation
- `memory` — Zugriff auf das 6-Tier Cognitive Memory
- `identity` — Ed25519-Key-Management
- `shell` — Sandboxed Shell-Execution
- `sevdesk` — **DACH:** sevDesk-Buchhaltung (v1.0 Launch)

## MCP-Protokoll

Alle Integrations folgen dem Model Context Protocol. Eigene Integrations bauen:
siehe `docs/quickstart/02-first-tool.md`.
