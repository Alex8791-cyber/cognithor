# 06 · Nächste Schritte

Du hast den Quickstart abgeschlossen — hier sind die besten weiterführenden Pfade.

**Voraussetzungen**
- Abgeschlossen: [05 · Deployment](05-deployment.md)

**Zeitbedarf:** 2 Minuten
**Endzustand:** Du hast dir 2–3 Themen ausgesucht, die du als nächstes vertiefen willst.

---

## Wichtige Vertiefungen

| Thema                  | Startpunkt                                                                             |
|------------------------|----------------------------------------------------------------------------------------|
| **6-Tier Memory**      | [`docs/memory.md`](../memory.md) — Core / Episodic / Semantic / Procedural / Working   |
| **Voice Mode**         | [`docs/voice.md`](../voice.md) — STT + TTS + Wakeword-Detection                        |
| **Computer Use**       | [`docs/computer-use.md`](../computer-use.md) — Desktop-Automation, Screenshots, Klicks |
| **MCP-Tool-Katalog**   | [`docs/integrations/README.md`](../integrations/README.md) — alle 145+ Tools           |
| **Agent Pack System**  | [`docs/packs.md`](../packs.md) — Plugin-Architektur + installierbare Packs             |
| **Community Skills**   | [`docs/community-skills.md`](../community-skills.md) — veröffentlichen + abonnieren    |
| **Guardrails — Advanced** | [`docs/guardrails.md`](../guardrails.md) — JSON-Schema, Halluzinations-Check, eigene   |
| **Hashline Guard**     | [`docs/hashline-guard.md`](../hashline-guard.md) — SHA-256 Audit-Chain                 |
| **CLI Reference**      | [`docs/CLI.md`](../CLI.md) — alle Flags + Sub-Commands                                 |
| **Config Reference**   | [`docs/CONFIG_REFERENCE.md`](../../CONFIG_REFERENCE.md) — alle Config-Keys             |

## Crew-Spezifisch

- **Templates** — `cognithor init <name> --template research` — siehe [Templates-Liste](../../src/cognithor/crew/templates/)
- **YAML-Konfigurierte Crews** — `cognithor.crew.yaml_loader` für `agents.yaml` + `tasks.yaml`
- **Hierarchical Process** — `process=CrewProcess.HIERARCHICAL` mit Manager-LLM für dynamische Delegation
- **Custom Integrations** — [`docs/integrations/`](../integrations/) für DACH-Connectoren (sevDesk etc.)

## Architektur-Deep-Dives

- [`docs/SYSTEM_ARCHITECTURE.md`](../SYSTEM_ARCHITECTURE.md) — Big Picture aller Module
- [`docs/superpowers/specs/2026-04-23-cognithor-crew-v1-adoption.md`](../superpowers/specs/2026-04-23-cognithor-crew-v1-adoption.md) — Crew-Layer Spec

## Community & Support

- [GitHub Issues](https://github.com/Alex8791-cyber/cognithor/issues) — Bug-Reports, Feature-Requests
- [cognithor.ai/packs](https://cognithor.ai/packs) — Kommerzielle + Community-Packs
- [CHANGELOG.md](../../CHANGELOG.md) — Release Notes

---

**Next:** [07 · Troubleshooting](07-troubleshooting.md)
