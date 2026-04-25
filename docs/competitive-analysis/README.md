# Competitive Analysis — Cognithor in the Multi-Agent Framework Landscape

This directory documents how Cognithor compares with adjacent multi-agent
frameworks. The intent is sober comparison, not advocacy: every claim about
a competing project should be backed by a public-source link.

## Documents

- [`autogen.md`](./autogen.md) — Cognithor vs Microsoft AutoGen (Python `0.7.5`,
  Maintenance Mode since Q4 2025).
- [`microsoft-agent-framework.md`](./microsoft-agent-framework.md) — Cognithor
  vs Microsoft Agent Framework (MAF, GA April 2026).
- [`decision-matrix.md`](./decision-matrix.md) — Side-by-side feature matrix
  across Cognithor, AutoGen, MAF, LangGraph, CrewAI.

## Scope

These documents inform marketing material, technical decisions, and the
v0.94.0 AutoGen-Compatibility-Shim (`cognithor.compat.autogen`). They are
deliberately conservative — no performance claims without a benchmark, no
"X is dead" rhetoric.

## Related

- [ADR 0001 — PGE Trinity vs Group Chat](../adr/0001-pge-trinity-vs-group-chat.md)
- [`cognithor.compat.autogen` migration guide](../../src/cognithor/compat/autogen/README.md) (added in PR 3)
