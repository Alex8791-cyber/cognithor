# `cognithor.compat.autogen` — Migration Guide

> **What this is.** A source-compatibility shim for
> [`autogen-agentchat==0.7.5`](https://github.com/microsoft/autogen)
> (Microsoft, MIT). It lets you run a useful subset of AutoGen-AgentChat
> code on Cognithor by changing **only the import paths**.

> **What this is not.** A reimplementation of AutoGen, MAF, or
> `autogen-core`. It does not replicate the GroupChat patterns that
> conflict with Cognithor's PGE-Trinity safety model — see
> [ADR 0001](../../../docs/adr/0001-pge-trinity-vs-group-chat.md).

## Quickstart — Search-and-Replace

```diff
- from autogen_agentchat.agents import AssistantAgent
- from autogen_agentchat.teams import RoundRobinGroupChat
- from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
- from autogen_ext.models.openai import OpenAIChatCompletionClient
+ from cognithor.compat.autogen import (
+     AssistantAgent, RoundRobinGroupChat,
+     MaxMessageTermination, TextMentionTermination,
+     OpenAIChatCompletionClient,
+ )
```

If your code uses only those symbols, that's the full migration. The 30-line
AutoGen hello-world example runs verbatim once imports are changed.

## Side-by-Side: AutoGen Hello-World

**AutoGen (original):**

```python
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent

async def main():
    client = OpenAIChatCompletionClient(model="gpt-4o-mini")
    agent = AssistantAgent("assistant", model_client=client)
    result = await agent.run(task="Say hello.")
    print(result.messages[-1].content)
```

**Cognithor compat (after search-and-replace):**

```python
from cognithor.compat.autogen import OpenAIChatCompletionClient, AssistantAgent

async def main():
    client = OpenAIChatCompletionClient(model="ollama/qwen3:8b")  # or any 16 providers
    agent = AssistantAgent("assistant", model_client=client)
    result = await agent.run(task="Say hello.")
    print(result.messages[-1].content)
```

The only meaningful change is the model spec — Cognithor accepts the full
model-router DSL, not just OpenAI model IDs. Pass `model="gpt-4o-mini"` if
you have an OpenAI key configured; pass `model="ollama/qwen3:8b"` for local.

## Supported Subset

| AutoGen Class | Status | Notes |
|---|---|---|
| `AssistantAgent` | ✅ Full 17-field signature parity | Internally delegates to `cognithor.crew` |
| `AssistantAgent.run` | ✅ | 1-shot, returns `TaskResult` |
| `AssistantAgent.run_stream` | ✅ | Async generator, AutoGen-shaped events |
| `RoundRobinGroupChat` | ✅ | Multi-round via `_RoundRobinAdapter` |
| `MaxMessageTermination` | ✅ | Counts messages |
| `TextMentionTermination` | ✅ | Substring match on last message |
| `MaxMessageTermination & TextMentionTermination` | ✅ | `__and__` overload |
| `MaxMessageTermination \| TextMentionTermination` | ✅ | `__or__` overload |
| `TextMessage`, `ToolCallSummaryMessage`, `HandoffMessage`, `StructuredMessage` | ✅ | AutoGen-shaped fields |
| `OpenAIChatCompletionClient` | ✅ | Backed by `cognithor.core.model_router` (16 providers) |
| `FunctionTool` / `Workbench` | ⚠️ Bridged via MCP | Custom tools need MCP registration |
| `SelectorGroupChat` | ❌ Not supported | LLM as security boundary — see [ADR 0001](../../../docs/adr/0001-pge-trinity-vs-group-chat.md) |
| `Swarm` | ❌ Not supported | HandoffMessage freedom conflicts with PGE-Trinity |
| `MagenticOneGroupChat` | ❌ Not supported | Separate workstream |
| `autogen_core` (`RoutedAgent`, `@message_handler`) | ❌ Out of scope | Actor-model, too low-level |

## Why are SelectorGroupChat / Swarm not supported?

Selector / Swarm patterns delegate the question "who speaks next?" to an LLM
or to free-form `HandoffMessage` exchanges between agents. Cognithor places
the Gatekeeper between every action and its execution — the Gatekeeper is
**rule-based** and inspectable. Letting an LLM bypass that boundary
breaks the safety model.

Detailed rationale: [ADR 0001 — PGE Trinity vs Group Chat](../../../docs/adr/0001-pge-trinity-vs-group-chat.md).

## When should you migrate off the compat layer?

The shim is a **temporary bridge**, not a destination. Once your code is
running on Cognithor and you've hit production stability, migrate to
native `cognithor.crew`:

- More idiomatic for Cognithor's PGE-Trinity (declarative `Crew`, explicit
  `kickoff_async`).
- First-class Hashline-Guard audit chain (no compat-layer wrapping).
- First-class guardrails (`no_pii()`, `chain()`, `StringGuardrail`).
- Better error messages — the shim's "AutoGen-shape" is sometimes a lossy
  translation.

A native rewrite of the hello-world above:

```python
from cognithor.crew import Crew, CrewAgent, CrewTask

async def main():
    agent = CrewAgent(role="assistant", goal="Greet the user", llm="ollama/qwen3:8b")
    task = CrewTask(description="Say hello.", expected_output="A short greeting.", agent=agent)
    crew = Crew(agents=[agent], tasks=[task])
    result = await crew.kickoff_async({})
    print(result.raw)
```

## Deprecation Warning

Importing `cognithor.compat.autogen` emits a `DeprecationWarning` pointing
back to this guide. The warning does not affect runtime behaviour. To
silence it for known-shim code:

```python
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"cognithor\.compat\.autogen")
```

## License Note

This shim is Apache 2.0. The API shape is concept-inspired from AutoGen
(MIT). No AutoGen source code is included verbatim. The repo-root
`NOTICE` carries the AutoGen-MIT attribution under "Third-party
attributions".
