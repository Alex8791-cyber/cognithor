# `config_routes.py` Split — Plan

**Status:** PR 1 (Backbone) gemergt, PRs 2-8 ausstehend.
**Datum:** 2026-04-29.
**Basis:** Architekten-Blueprint vom 2026-04-28 (siehe Session-Transcript in `~/.claude/projects/D--Jarvis/`).

## Ausgangslage

`src/cognithor/channels/config_routes.py` war ein 6 628-LOC-Monolith mit `create_config_routes()` als Factory + 24 internen `_register_*_routes()`-Helfern, die zusammen 303 Endpoints registrieren. CLAUDE.md flaggt: "Plan before refactoring large files. Don't split them in a single drive-by commit."

## Ziel

Datei in ein Paket `src/cognithor/channels/config_routes/` mit thematisch gruppierten Sub-Modulen aufteilen. **Zero behaviour change** — jeder Endpoint behält Pfad, Auth, Response-Shape. Public API bleibt `from cognithor.channels.config_routes import create_config_routes`.

## Drift-Bremse

`tests/test_channels/fixtures/route_inventory.json` hält die exakten 303 Endpoint-Pfade fest. `test_route_inventory_unchanged` in `tests/test_channels/test_config_routes.py` schlägt an, sobald sich Pfade hinzufügen, verschwinden oder umbenennen — egal ob versehentlich oder bewusst. Bei einer **bewussten** Endpoint-Änderung muss die Fixture in derselben PR aktualisiert werden.

## Datei-Layout (Ziel)

```
src/cognithor/channels/
    config_routes/
        __init__.py          ← Re-Export von create_config_routes
        _factory.py          ← create_config_routes() + _hub_holder closure
        system.py            ← _register_system_routes
        config.py            ← _register_config_routes
        session.py           ← _register_session_routes + _register_memory_routes
        skills.py            ← _register_skill_routes + _register_skill_registry_routes + _register_hermes_routes
        monitoring.py        ← _register_monitoring_routes + _register_prometheus_routes
        security.py          ← _register_security_routes
        governance.py        ← _register_governance_routes
        evolution.py         ← _register_prompt_evolution_routes + _register_self_improvement_routes + _register_gepa_evolution_routes + _proposal_to_dict + _trace_to_dict
        infrastructure.py    ← _register_infrastructure_routes + _register_portal_routes + _register_backend_routes
        ui.py                ← _register_ui_routes (inkl. _load_yaml/_save_yaml)
        workflows.py         ← _register_workflow_graph_routes
        learning.py          ← _register_learning_routes + _register_ingest_routes
        autonomous.py        ← _register_autonomous_routes + _register_feedback_routes
        social.py            ← _register_social_routes
```

Begründung Cluster: Helfer mit gemeinsamer Domäne und unter ~800 LOC werden zusammengefügt; ab 800 LOC eigene Datei. `security.py` und `governance.py` sind getrennt, obwohl thematisch verwandt — beide sind je >200 LOC und haben unterschiedliche Stakeholder (Sicherheit vs. Compliance/Reputation).

## PR-Sequenz

| PR | Inhalt | Diff (LOC) | Status |
|---|---|---:|---|
| **1** | **Backbone** — `config_routes.py` → `config_routes/_factory.py`, `__init__.py` mit Re-Export, Route-Inventory-Fixture, Drift-Test, Test-Adapter für die zwei Source-String-Probes (`test_wiring_all_four`, `test_f025`) | ~100 | ✅ |
| 2 | `system.py` + `config.py` extrahieren | ~1 250 | offen |
| 3 | `session.py` + `monitoring.py` extrahieren | ~750 | offen |
| 4 | `security.py` + `governance.py` extrahieren | ~1 200 | offen |
| 5 | `skills.py` extrahieren (skill + skill_registry + hermes) | ~650 | offen |
| 6 | `ui.py` extrahieren (allein wegen Größe) | ~970 | offen |
| 7 | `evolution.py` + `infrastructure.py` + `workflows.py` extrahieren | ~1 030 | offen |
| 8 | `learning.py` + `autonomous.py` + `social.py` extrahieren | ~1 100 | offen |

Nach PR 8: `_factory.py` enthält nur noch `create_config_routes()` (Aufruf-Sequenz) plus `_hub_holder`/`_get_hub` Closure — ca. 100 LOC.

## Ablauf je PR (PR 2 ff.)

1. Sub-Modul-Datei mit demselben Header-Pattern wie `_factory.py` anlegen (`from __future__ import annotations`, `Any`-Imports, lazy-import-Pattern bewahren).
2. Die `_register_*_routes()`-Funktion(en) per **`git mv`-äquivalentem Cut/Paste** aus `_factory.py` in die neue Datei verschieben. Body unverändert.
3. In `_factory.py`: Funktion entfernen, am Anfang den Import aus dem Sub-Modul ergänzen.
4. **Kritisch:** Aufruf-Reihenfolge in `create_config_routes()` exakt erhalten — FastAPI matcht Routen in Registrierungs-Reihenfolge. Spezifische Pfade (`/skills/hermes/export/{name}`) müssen vor generischen (`/skill-registry/{slug}`) registriert werden.
5. `pytest tests/test_channels/test_config_routes.py::TestRouteRegistration::test_route_inventory_unchanged` als erster Check.
6. `pytest` über alle 9 Konsumenten-Test-Files (Liste in PR-1-Beschreibung).
7. `ruff check` + `ruff format --check`.

## Risiken

1. **`_hub_holder` Closure** (mittel): bleibt in `_factory.py`, wird als Callable an `monitoring.py` übergeben (heutiges Pattern). Sub-Module halten **keine** eigene Hub-Referenz.
2. **Reihenfolge-Abhängigkeit** (mittel): FastAPI matcht in Registrierungs-Reihenfolge. Aufruf-Reihenfolge in `_factory.py` muss byte-genau bleiben. Drift-Test fängt fehlende/zusätzliche Pfade, **nicht** aber Reihenfolge-Vertauschungen — daher pro PR mental gegen den Diff prüfen.
3. **`from __future__ import annotations`** (niedrig): jedes Sub-Modul braucht das, sonst FastAPI-Parser-Fehler bei Typen in Endpoint-Signaturen.
4. **Lazy-Imports innerhalb Handler** (niedrig): bleiben unverändert; nicht hochziehen.
5. **`_proposal_to_dict` / `_trace_to_dict`** (niedrig): wandern mit ihrem einzigen Aufrufer in `evolution.py`. Kein `_shared.py` nötig.
6. **Source-String-Probes** (niedrig, in PR 1 gefixt): `test_wiring_all_four` nutzt einen Helper, der bei einem Paket alle `.py`-Dateien zusammenfasst; `test_f025` greift fallback-fähig direkt auf `_factory` oder das Paket zu.

## Public API — darf NICHT brechen

`from cognithor.channels.config_routes import create_config_routes` muss in allen 11 Konsumenten weiter funktionieren:

- `src/cognithor/__main__.py` (2x)
- `src/cognithor/channels/webui.py` (2x)
- 9 Test-Files: `tests/test_channels/{test_config_routes,test_workflow_graph_routes,test_knowledge_graph_routes}.py`, `tests/test_config_manager.py`, `tests/test_ui_api_integration.py`, `tests/test_gateway/test_prompt_evolution_api.py`, `tests/test_integration/test_wiring_all_four.py`, `tests/test_security/{test_f003_auth_token_masking,test_f005_config_schema_validation,test_f025_config_routes_path_traversal}.py`
