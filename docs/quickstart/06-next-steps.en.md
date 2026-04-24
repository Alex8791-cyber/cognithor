# 06 · Next Steps

You've completed the Quickstart — here are the best follow-up paths.

**Prerequisites**
- Completed: [05 · Deployment](05-deployment.en.md)

**Time:** 2 minutes
**End state:** You've picked 2–3 topics to dive deeper into.

---

## Deeper dives

| Topic                     | Starting point                                                                            |
|---------------------------|-------------------------------------------------------------------------------------------|
| **6-Tier Memory**         | [`docs/memory.md`](../memory.md) — Core / Episodic / Semantic / Procedural / Working      |
| **Voice Mode**            | [`docs/voice.md`](../voice.md) — STT + TTS + wake-word detection                          |
| **Computer Use**          | [`docs/computer-use.md`](../computer-use.md) — desktop automation, screenshots, clicks    |
| **MCP tool catalog**      | [`docs/integrations/README.md`](../integrations/README.md) — all 145+ tools               |
| **Agent Pack system**     | [`docs/packs.md`](../packs.md) — plugin architecture + installable packs                  |
| **Community Skills**      | [`docs/community-skills.md`](../community-skills.md) — publish + subscribe                |
| **Guardrails — Advanced** | [`docs/guardrails.md`](../guardrails.md) — JSON Schema, hallucination check, custom       |
| **Hashline Guard**        | [`docs/hashline-guard.md`](../hashline-guard.md) — SHA-256 audit chain                    |
| **CLI reference**         | [`docs/CLI.md`](../CLI.md) — all flags + sub-commands                                     |
| **Config reference**      | [`docs/CONFIG_REFERENCE.md`](../../CONFIG_REFERENCE.md) — every config key                |

## Crew-specific

- **Templates** — `cognithor init <name> --template research` — see [templates list](../../src/cognithor/crew/templates/)
- **YAML-configured crews** — `cognithor.crew.yaml_loader` for `agents.yaml` + `tasks.yaml`
- **Hierarchical process** — `process=CrewProcess.HIERARCHICAL` with a manager LLM for dynamic delegation
- **Custom integrations** — [`docs/integrations/`](../integrations/) for DACH connectors (sevDesk, etc.)

## Architecture deep dives

- [`docs/SYSTEM_ARCHITECTURE.md`](../SYSTEM_ARCHITECTURE.md) — big picture of all modules
- [`docs/superpowers/specs/2026-04-23-cognithor-crew-v1-adoption.md`](../superpowers/specs/2026-04-23-cognithor-crew-v1-adoption.md) — Crew-Layer spec

## Community & support

- [GitHub Issues](https://github.com/Alex8791-cyber/cognithor/issues) — bug reports, feature requests
- [cognithor.ai/packs](https://cognithor.ai/packs) — commercial + community packs
- [CHANGELOG.md](../../CHANGELOG.md) — release notes

---

**Next:** [07 · Troubleshooting](07-troubleshooting.en.md)
