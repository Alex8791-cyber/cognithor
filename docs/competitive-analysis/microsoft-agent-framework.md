# Cognithor vs Microsoft Agent Framework (MAF)

> **Status of this document:** 2026-04-25. Sources cited inline.

## 1. What MAF Is

Microsoft Agent Framework is the supported successor to AutoGen. GA was
April 2026[^1]. License: MIT. Languages: Python and .NET. Programming
model: **graph-based** workflow orchestration with `@workflow`,
`@activity`, and explicit edges between agent nodes.

MAF ships first-class integrations with Azure AI Foundry, Microsoft
Sentinel, and the broader Azure observability story. The `@tool` decorator
replaces AutoGen's `FunctionTool` workbench abstraction.

[^1]: https://learn.microsoft.com/en-us/agent-framework/

## 2. Programming-Model Shift

The most consequential change between AutoGen and MAF is **conversation →
graph**:

| AutoGen (`autogen-agentchat`) | MAF |
|-------------------------------|-----|
| Conversational chat history shared by team | DAG/graph with explicit nodes and edges |
| `RoundRobinGroupChat`, `SelectorGroupChat` | `@workflow` with conditional edges |
| `FunctionTool` / `Workbench` | `@tool` decorator on async functions |
| Termination via `MaxMessageTermination`, `TextMentionTermination` | Termination via end-nodes in the graph |
| State implicit in chat history | State explicit in the workflow context |

For tens of thousands of existing AutoGen users, this is a **hard
migration** — graph thinking differs structurally from chat thinking.

## 3. Why Cognithor Still Exists

MAF is excellent for Azure-centric Enterprise customers who want a vendor-
supported framework with first-class observability inside the Microsoft
stack. Cognithor's positioning is complementary, not competitive:

- **EU-Sovereignty**: No implicit Azure dependency. `OLLAMA_HOST` is the
  default, not an "alternative client". The DACH-region documentation
  layer (`cognithor init --template versicherungs-vergleich`) ships with
  the framework, not as a sample.
- **No Azure account required**: `pip install cognithor` plus a local
  Ollama gives a working agent in <5 minutes.
- **DSGVO-relevant features**: PII-redaction guardrails (`no_pii()`),
  Hashline-Guard audit chain, strict role separation via PGE-Trinity, all
  documented as compliance primitives — not afterthoughts.
- **Local inference first-class**: 16 providers including local Ollama,
  vLLM, llama.cpp; no managed-service preference.
- **Public Apache 2.0**: MAF is also MIT/permissive, but MAF's commercial
  motion is tightly bound to Azure. Cognithor's commercial layer
  (`cognithor.packs`) is opt-in, EULA-gated, and orthogonal to the core.

## 4. Not a Framework War

Cognithor does not position itself as "the alternative MAF" or claim
feature-superiority for graph-orchestration use-cases. For workflows that
**need** explicit DAG semantics (e.g., approval chains with conditional
branches, finance-style state machines), MAF or LangGraph is the right
tool. Cognithor is the right tool when:

- DSGVO/EU-residency is a hard requirement.
- Local inference is preferred (or required by policy).
- The team wants a chat-style multi-agent abstraction with a built-in
  Gatekeeper layer for safety.
- Vendor independence matters more than Azure-native observability.

## 5. References

- MAF documentation: https://learn.microsoft.com/en-us/agent-framework/
- MAF migration from AutoGen: https://learn.microsoft.com/en-us/agent-framework/migration-guide/from-autogen/
- Cognithor PGE-Trinity ADR: [`docs/adr/0001-pge-trinity-vs-group-chat.md`](../adr/0001-pge-trinity-vs-group-chat.md)
