# Skill Lifecycle + ARC-AGI-3 Skill — Implementierungsplan

**Datum:** 2026-03-30
**Status:** Geplant, nächste Session
**Priorität:** HOCH — Skills werden erstellt aber nie aktiviert (broken loop)

---

## Problem

1. `create_skill` MCP Tool schreibt `.md` + `.py` nach `~/.jarvis/skills/generated/`
2. `SkillRegistry.__init__` scannt dieses Verzeichnis NICHT
3. Erstellte Skills tauchen nie in der Tool-Liste auf
4. Cognithor weiß nicht dass er Skills hat die er selbst erstellt hat
5. Keine Qualitätskontrolle, kein Repair, keine Skill-Ideation

## Scope (4 Teile)

### Teil 1: Hot-Loading (sofort nach Erstellung verfügbar)

**Dateien:**
- `src/jarvis/mcp/skill_tools.py` — nach `create_skill()`: Registry-Eintrag sofort hinzufügen
- `src/jarvis/skills/registry.py` — `register_skill(path)` Methode hinzufügen (existiert evtl. als `_register`)

**Was passiert:**
```
User: "Erstelle einen Skill für Wettervorhersagen"
→ Planner: create_skill(name="weather_forecast", ...)
→ skill_tools.py: schreibt Dateien + ruft registry.register_skill() auf
→ Skill sofort in der Liste verfügbar
→ Planner kann Skill im selben Gespräch nutzen
```

### Teil 2: Startup-Loading (generierte Skills beim Start laden)

**Dateien:**
- `src/jarvis/skills/registry.py` — `load_from_directories()` erweitern um `generated/` Pfad
- Oder: neues `_load_generated_skills()` in `__init__`

**Was passiert:**
```
Cognithor startet
→ SkillRegistry.__init__()
→ Lädt built-in Skills (src/jarvis/skills/builtin/)
→ Lädt User Skills (~/.jarvis/skills/*.md)
→ NEU: Lädt generierte Skills (~/.jarvis/skills/generated/*.md)
→ Log: "skill_registry_loaded categories=7 total=22 (16 builtin + 6 generated)"
```

### Teil 3: Skill Lifecycle (Audit + Repair + Ideation)

**Dateien:**
- `src/jarvis/skills/lifecycle.py` — NEU
- Integration in Evolution Engine oder als eigener Cron-Job

**SkillLifecycleManager:**
- `audit_all()` — jeden Skill laden, Syntax prüfen, Test ausführen
- `repair_skill(name)` — defekten Skill mit LLM reparieren
- `suggest_skills()` — basierend auf häufigen User-Anfragen + Wissenslücken neue Skills vorschlagen
- `prune_unused(days=30)` — Skills die 30 Tage nicht genutzt wurden markieren

**Cron-Integration:**
```yaml
# In config.yaml
skill_lifecycle:
  audit_interval_hours: 24
  auto_repair: true
  suggest_new: true
```

**Was passiert:**
```
Alle 24h (oder im Idle):
→ SkillLifecycleManager.audit_all()
→ "weather_forecast: OK, code_review: BROKEN (SyntaxError line 15)"
→ auto_repair: LLM fixt den SyntaxError
→ suggest_skills: "User fragt oft nach PDF-Erstellung — Skill fehlt"
→ Optional: create_skill("pdf_generator", ...)
```

### Teil 4: ARC-AGI-3 als Skill

**Dateien:**
- `~/.jarvis/skills/generated/arc_agi3_play.md` — Skill-Definition
- Nutzt existierendes `src/jarvis/arc/` Modul

**Skill-Definition:**
```markdown
---
name: arc-agi3-play
trigger_keywords: ["arc", "arc-agi", "benchmark", "puzzle", "spiel"]
tools_required: ["arc_play", "arc_status", "arc_replay"]
description: "Spielt ARC-AGI-3 Benchmark-Games und analysiert die Ergebnisse"
---

# ARC-AGI-3 Spielmodus

Wenn der User ein ARC-AGI-3 Game spielen möchte:
1. Nutze `arc_play` mit der gewünschten Game-ID
2. Berichte den Score und die Strategie
3. Bei "benchmark": alle Games durchspielen
4. Bei "analyse": Ergebnisse interpretieren
```

**Was passiert:**
```
User: "Spiel ARC-AGI-3 Game ls20"
→ Planner erkennt Skill "arc-agi3-play" via trigger_keywords
→ Skill wird in den Kontext injiziert
→ Planner nutzt arc_play Tool
→ Ergebnis wird berichtet
```

## Implementierungsreihenfolge

| Phase | Was | Aufwand | Abhängigkeit |
|-------|-----|---------|-------------|
| 1 | Hot-Loading + Startup-Loading | 30min | Keine |
| 2 | ARC-AGI-3 Skill erstellen | 10min | Phase 1 |
| 3 | Skill Lifecycle Manager | 2h | Phase 1 |
| 4 | Cron-Integration + Config | 30min | Phase 3 |
| 5 | Tests | 1h | Alles |

## Dateien-Übersicht

**Erstellen:**
- `src/jarvis/skills/lifecycle.py` — Audit, Repair, Suggest, Prune
- `tests/test_skills/test_lifecycle.py`

**Modifizieren:**
- `src/jarvis/skills/registry.py` — generated dir laden, register_skill() public
- `src/jarvis/mcp/skill_tools.py` — Hot-Loading nach create_skill
- `src/jarvis/config.py` — SkillLifecycleConfig
- `src/jarvis/gateway/gateway.py` oder Cron — Lifecycle-Job

**Erstellen (Skill-Datei):**
- `~/.jarvis/skills/generated/arc_agi3_play.md`

## Risiko

- **GERING**: Änderungen an registry.py und skill_tools.py sind additiv
- **KEIN** Eingriff in PGE-Kern, Gatekeeper, Executor
- Skills sind Markdown-Dateien die als Context injiziert werden — kein Code-Execution-Risiko
