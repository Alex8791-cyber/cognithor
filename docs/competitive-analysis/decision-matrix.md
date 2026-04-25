# Decision Matrix — Multi-Agent Frameworks

> **Status of this document:** 2026-04-25. Each cell sourced from public
> documentation; corrections welcome via issue.

This matrix compares feature surfaces, not performance. Performance
benchmarks are deliberately deferred — see [`cognithor_bench/README.md`](../../cognithor_bench/README.md)
(added in v0.94.0 PR 2).

| Dimension | Cognithor | AutoGen 0.7.5 | MAF 1.0 | LangGraph | CrewAI |
|-----------|-----------|---------------|---------|-----------|--------|
| Core License | Apache 2.0 | MIT | MIT | MIT | MIT |
| Host-Region (Default) | Local / EU | n/a (library) | Azure-leaning | n/a (library) | n/a (library) |
| Local Inference First-Class | Yes (Ollama default) | Via `OpenAIChatCompletionClient` | Possible, not default | Yes | Yes |
| LLM Providers OOTB | 16 | 1 (OpenAI-compat) + extensions | Azure AI + OpenAI | LangChain providers | LangChain providers |
| MCP Client | Yes (145 tools across 14 modules) | Yes | Yes | Via LangChain | Yes |
| A2A Protocol | Yes (`cognithor.a2a`) | Partial | Yes | No | No |
| Multi-Agent Pattern | PGE-Trinity (forced role separation) | Conversation (chat history) | Graph (DAG) | Graph (DAG) | Conversation (Crews) |
| DSGVO Compliance Claim | Explicit (PII redaction, EU-provider docs) | Not addressed | Implicit (Azure EU) | Not addressed | Not addressed |
| Audit Chain | Hashline Guard (xxhash chain) | No | Azure observability | LangSmith | No |
| Commercial Coupling | None (Apache core; opt-in commerce packs) | None (Microsoft project) | Microsoft / Azure | LangChain Inc. | CrewAI Inc. (Pro) |
| Active Maintenance Status | Active (v0.94.0 in flight) | Maintenance Mode | Active | Active | Active |

## How to read this matrix

- "Yes" / "No" answers reflect what's documented in the upstream framework
  as of 2026-04-25. They do not indicate quality or maturity.
- "First-Class" means a feature is treated as the default, not an opt-in
  extension.
- For runtime performance, see `cognithor_bench/` once GAIA/WebArena
  scenarios are integrated (post-v0.94.0).

## References

- AutoGen: https://github.com/microsoft/autogen
- MAF: https://learn.microsoft.com/en-us/agent-framework/
- LangGraph: https://langchain-ai.github.io/langgraph/
- CrewAI: https://docs.crewai.com/
