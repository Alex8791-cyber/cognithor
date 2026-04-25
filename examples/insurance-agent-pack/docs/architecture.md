# Architecture — PGE-Trinity Visibility

```text
            ┌─────────────────────────────────────────────────┐
            │            insurance-agent-pack                  │
            │                                                  │
   user ──▶ │  CLI (--interview)                               │
            │     │                                            │
            │     ▼                                            │
            │  Crew(agents=[NA, PA, CG, RG], SEQUENTIAL)        │
            │     │                                            │
            │     ▼                                            │
            │  ┌──────────────────────────────────────────┐    │
            │  │ NeedsAssessor   — interview → profile     │    │
            │  │ PolicyAnalyst   — PDF extraction          │    │
            │  │ ComplianceGate  — pre-advisory check      │  ← visible PGE Gatekeeper
            │  │ ReportGenerator — markdown output         │    │
            │  └──────────────────────────────────────────┘    │
            │                                                  │
            │  Each agent's CrewTask runs through Cognithor's   │
            │  PGE-Trinity:                                    │
            │     Planner → Gatekeeper(framework) → Executor   │
            │                                                  │
            │  And ComplianceGatekeeper is an additional       │
            │  in-Crew check on top of the framework one.      │
            └─────────────────────────────────────────────────┘
```

The framework `Gatekeeper` (in `src/cognithor/core/gatekeeper.py`) handles
DSGVO PII redaction and tool allow-list classification (GREEN/YELLOW/
ORANGE/RED). The pack's `ComplianceGatekeeper` adds **domain-specific**
classification: pre-advisory vs legal-advice vs concrete-recommendation-
demand. Two layers, distinct concerns; both inspectable.

See:
- [ADR 0001](../../../docs/adr/0001-pge-trinity-vs-group-chat.md)
- [`docs/hashline-guard.md`](../../../docs/hashline-guard.md)
