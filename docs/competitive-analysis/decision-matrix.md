# Decision Matrix — Multi-Agent Frameworks

> **Status of this document:** 2026-05-06. Each cell sourced from public
> documentation; corrections welcome via issue.

This matrix compares feature surfaces, not performance. Performance
benchmarks are deliberately deferred — see [`cognithor_bench/README.md`](../../cognithor_bench/README.md)
(added in v0.94.0 PR 2).

| Dimension | Cognithor | AutoGen 0.7.5 | MAF 1.0 | LangGraph | CrewAI |
|-----------|-----------|---------------|---------|-----------|--------|
| Core License | Apache 2.0 | MIT | MIT | MIT | MIT |
| Host-Region (Default) | Local / EU | n/a (library) | Azure-leaning | n/a (library) | n/a (library) |
| Local Inference First-Class | Yes (Ollama default; vLLM + llama-cpp opt-in) | Via `OpenAIChatCompletionClient` | Possible, not default | Yes | Yes |
| LLM Providers OOTB | 19 (Ollama, LM Studio, vLLM, llama-cpp + 15 cloud + Claude Code) | 1 (OpenAI-compat) + extensions | Azure AI + OpenAI | LangChain providers | LangChain providers |
| MCP Client | Yes (141 tools across 30 modules — auto-generated `docs/integrations/catalog.json`) | Yes | Yes | Via LangChain | Yes |
| A2A Protocol | Yes (`cognithor.a2a`) | Partial | Yes | No | No |
| Multi-Agent Pattern | PGE-Trinity (forced role separation) | Conversation (chat history) | Graph (DAG) | Graph (DAG) | Conversation (Crews) |
| DSGVO Compliance Claim | Explicit (PII redaction, EU-provider docs, Right-to-Erasure flow) | Not addressed | Implicit (Azure EU) | Not addressed | Not addressed |
| Audit Chain | HMAC-SHA-256 hash-chain (`prev_hash` over canonical NFC-JSON), `cognithor audit verify` | No | Azure observability | LangSmith | No |
| Operational-Trust Ledgers (TRUST-1..10) | 6 ledgers (Provenance, Permission-Scopes, Tool-Fingerprints, Cloud-Escalation, Cost-Micro-USD, Migration) + signed run-receipts | No | Partial (Azure side-channel) | Partial (LangSmith) | No |
| Resumable Batch Workflows | Yes — CRWE: JSONL streaming + atomic checkpoint + SIGINT/SIGTERM safe + manifest-tamper detection (v0.99.0) | No (chat-message centric) | Workflow primitive | Graph state checkpoint | No |
| Pack/Skill Registry Signing | Yes — TUF-Light Ed25519 + SHA-256 (offline Root + online Targets) | n/a | n/a | n/a | n/a |
| Cost Budgeting (per-run) | Yes — Cost-Micro-USD ledger, IDE gutter decoration, hard caps | No | No | No | No |
| Commercial Coupling | None (Apache core; opt-in commerce packs) | None (Microsoft project) | Microsoft / Azure | LangChain Inc. | CrewAI Inc. (Pro) |
| Active Maintenance Status | **Active — v0.99.0 GA on PyPI 2026-05-06** | Maintenance Mode | Active | Active | Active |

## How to read this matrix

- "Yes" / "No" answers reflect what's documented in the upstream framework
  as of 2026-05-06. They do not indicate quality or maturity.
- "First-Class" means a feature is treated as the default, not an opt-in
  extension.
- For runtime performance, see `cognithor_bench/` (GAIA/WebArena scenarios
  remain on the roadmap as of v0.99.0).

## References

- AutoGen: https://github.com/microsoft/autogen
- MAF: https://learn.microsoft.com/en-us/agent-framework/
- LangGraph: https://langchain-ai.github.io/langgraph/
- CrewAI: https://docs.crewai.com/
