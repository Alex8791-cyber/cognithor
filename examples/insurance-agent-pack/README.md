# insurance-agent-pack — Cognithor Reference Pack

A standalone-installable, **§34d-NEUTRAL** reference example for Cognithor.
Demonstrates a 4-agent PGE-Trinity pre-advisory crew (NeedsAssessor,
PolicyAnalyst, ComplianceGatekeeper, ReportGenerator) for the DACH
insurance domain.

> **This is a demo, not a product.** Read [`docs/DISCLAIMER.md`](./docs/DISCLAIMER.md)
> before you do anything with it.

## Why this pack exists

[ADR 0001](../../docs/adr/0001-pge-trinity-vs-group-chat.md) explains why
Cognithor uses Planner / Gatekeeper / Executor instead of a free-form
GroupChat. This pack makes that visible: every Crew turn passes through
the `ComplianceGatekeeper` agent before reaching `ReportGenerator`.
You can watch the Hashline-Guard audit chain build up.

## Install

```bash
pip install ./examples/insurance-agent-pack
```

## Usage

```bash
# Interactive interview (Konsolen-Session)
insurance-agent-pack run --interview

# Custom model — any Cognithor model_router spec works
insurance-agent-pack run --interview --model "ollama/qwen3:32b"

# Or with hosted backends if you have keys configured
insurance-agent-pack run --interview --model "openai/gpt-4o-mini"
```

## Architecture

See [`docs/architecture.md`](./docs/architecture.md) for the PGE-Trinity
flow diagram.

## Connection to v0.93.0 templates

Conceptually related to `cognithor init --template versicherungs-vergleich`
shipped in v0.93.0. WP3 focuses specifically on:
- **PolicyAnalyst** with PDF tool-use (new vs the v0.93.0 template).
- **ComplianceGatekeeper** as a *visible* PGE-demo agent (new).
- **Standalone-pip-installability** (the v0.93.0 template scaffolds *into*
  a project; this pack ships *as* a Python package).

## Pack-system note

This pack is **NOT** registered with `cognithor.packs`. That loader system
is reserved for private commerce-packs from the `cognithor-packs` repo
(EULA-gated, license-key validated). This is a public Apache-2.0 reference
implementation — pure `pip install`.

## Demo recording

[![asciicast](docs/demo_walkthrough.md)](./docs/demo_walkthrough.md)
(asciinema recording link added once captured.)

## Cross-links

- [Cognithor main repo](https://github.com/Alex8791-cyber/cognithor)
- [ADR 0001 — PGE Trinity vs Group Chat](../../docs/adr/0001-pge-trinity-vs-group-chat.md)
- [`cognithor.compat.autogen` migration guide](../../src/cognithor/compat/autogen/README.md)
- [`cognithor-bench`](../../cognithor_bench/README.md) — runs this pack as a benchmark scenario.

## License

Apache 2.0. See repo-root [LICENSE](../../LICENSE).
