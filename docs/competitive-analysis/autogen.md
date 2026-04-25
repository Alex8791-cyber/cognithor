# Cognithor vs Microsoft AutoGen

> **Status of this document:** 2026-04-25. Sources cited inline.

## 1. Status of AutoGen (Q2 2026)

The `microsoft/autogen` repository carries an explicit Maintenance-Mode
notice since approximately October 2025[^1]. The last actively-developed
Python release is `autogen-agentchat==0.7.5` (September 2025). Microsoft
actively redirects new users to the **Microsoft Agent Framework (MAF)** as
the supported successor[^2]. `agbench` and Magentic-One are tagged
"reference applications" rather than production-ready releases.

[^1]: https://github.com/microsoft/autogen — README banner.
[^2]: https://learn.microsoft.com/en-us/agent-framework/migration-guide/from-autogen/

## 2. Architecture Summary (3-Layer Design)

AutoGen ships as three Python packages:

- **`autogen-core`** — Actor-Model runtime (`SingleThreadedAgentRuntime`,
  `RoutedAgent`, `@message_handler`). Low-level concurrency primitive.
- **`autogen-agentchat`** — High-level conversational API
  (`AssistantAgent`, teams `RoundRobinGroupChat` / `SelectorGroupChat` /
  `Swarm` / `MagenticOneGroupChat`, message types `TextMessage` /
  `HandoffMessage` / `ToolCallSummaryMessage`).
- **`autogen-ext`** — Provider/Tool extensions (`OpenAIChatCompletionClient`,
  function-tool wrappers).

The **conversational programming model** treats agent-to-agent
communication as a sequence of messages on a shared chat history; teams
schedule turns. This contrasts with **graph-based** orchestration adopted
by MAF and LangGraph.

## 3. Common Ground with Cognithor

- Multi-Agent first-class (Cognithor's `cognithor.crew` since v0.93.0).
- MCP client support (Cognithor: 145 tools across 14 modules).
- Local-model story (AutoGen via Ollama-compatible clients; Cognithor:
  Ollama is the default, not an integration).
- Tool/function-call abstraction at the agent level.

## 4. Differences (honest)

**Where AutoGen is stronger:**
- Cross-language (Python + .NET).
- Larger English-speaking community.
- More extensive end-user documentation and tutorials.

**Where Cognithor is stronger:**
- **PGE-Trinity** as enforced role-separation (Planner-Gatekeeper-Executor)
  with Hashline-Guard audit chain — see
  [ADR 0001](../adr/0001-pge-trinity-vs-group-chat.md).
- **DSGVO-First defaults**: PII redaction, EU-provider documentation,
  offline-capable defaults. Mentioned only obliquely in AutoGen.
- **Vendor neutrality**: 16 LLM providers out-of-the-box; no implicit Azure
  preference.
- **Local inference first-class**: Ollama is the default execution path,
  not an opt-in extension.
- **6-Tier cognitive memory** (`cognithor.memory`) integrated with the
  Planner; AutoGen leaves memory to the user.
- **Deep Research v2** as a dedicated subsystem (`deep_research_v2.py`)
  rather than a sample notebook.

## 5. Conceptual Migration Path

AutoGen's `autogen-agentchat` Python API has a stable 1-shot path —
`AssistantAgent.run(task=...)` — that maps cleanly to
`cognithor.crew.Crew(agents=[a], tasks=[t]).kickoff_async()`. Cognithor
v0.94.0 ships `cognithor.compat.autogen` as a thin source-compatibility
shim covering this surface. See
[`cognithor.compat.autogen` migration guide](../../src/cognithor/compat/autogen/README.md)
(added in PR 3 of v0.94.0) for the supported subset and search-and-replace
import recipe.

The shim deliberately does **not** support `SelectorGroupChat`, `Swarm`,
or `MagenticOneGroupChat`; the rationale is in
[ADR 0001](../adr/0001-pge-trinity-vs-group-chat.md).

## 6. References

- AutoGen GitHub: https://github.com/microsoft/autogen
- AutoGen AgentChat reference: https://microsoft.github.io/autogen/stable//reference/python/autogen_agentchat.agents.html
- Magentic-One paper: https://arxiv.org/abs/2411.04468
- MAF migration guide: https://learn.microsoft.com/en-us/agent-framework/migration-guide/from-autogen/
