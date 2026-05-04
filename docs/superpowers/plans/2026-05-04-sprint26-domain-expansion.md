# Sprint-26 — PSE Domain-Expansion (Owner-Committed Plan)

**Status:** Ready to start
**Owner:** Alexander Söllner (Solo)
**Source:** `Owner Decision Memo & Sprint-Kickoff` (2026-05-04)
**Memory:** [project_sprint26_owner_committed](../../../C:/Users/ArtiCall/.claude/projects/D--Jarvis/memory/project_sprint26_owner_committed.md) — see ~/.claude/projects/D--Jarvis/memory/

## Vision

Cognithor PSE wird das einzige lokale Open-Source-System, das Programme **über Domain-Grenzen hinweg** synthetisiert — JSON parsen → Datum extrahieren → SQL-Query bauen → Output validieren — alles deterministisch, replayable, mit signiertem Receipt.

## Owner-Decisions (committed, nicht verhandelbar)

| # | Frage | Entscheidung |
|---|---|---|
| D1 | Public Scorecard | JA, Soft-Launch nach 14d sobald Spider-easy ≥ 15 %. Ab Live: Regression-PR-Block. |
| D2 | Sprint-Länge | 4 Wochen Kalenderzeit, **Mittwoch + Sonntag Pause** (nicht verhandelbar) |
| D3 | Domain-Reihenfolge | SQL → JSON → Datetime → AST → BinaryData → Float → Image-Boost |
| D4 | LLM-Prior | EIN Modell (Qwen3.6:27B) + System-Prompt-Switching + Few-Shot-Banks |
| D5 | Bridge-Operatoren | Whitelist 12 Pairs (siehe unten). Lernende Discovery → Sprint-28. |
| D6 | Cost-Tracker | VORGEZOGEN in 26.1. Verkaufsargument für Insurance-Pack. |
| D7 | DoD | Soft-Bar 6/10 Domain-Ziele, Hard-Bar 4/10 + Cross-Domain-Demo |

## Bridge-Whitelist (12 Pairs)

```
json     → datetime    json     → number      json     → string
string   → datetime    string   → number      string   → json
datetime → sql_literal datetime → string
number   → sql_literal number   → string
bytes    → string      bytes    → number
```

Alles andere wird vom Verifier abgelehnt.

## Phasen-Plan

### 26.1 — Foundation + Cost-Tracker (Woche 1)

**Goal:** Infrastruktur steht, Cost-Tracker liefert bereits Wert (auch wenn Sprint hier abbricht).

**Tasks:**
- [ ] `pse_engine/domains/registry.py` — `DomainRegistry` mit Plugin-Loader
- [ ] `pse_engine/domains/base.py` — `Domain` Protocol: `(catalog, type_tags, verifier, llm_prior_path, property_suite)`
- [ ] `pse_engine/llm_prior/domain_aware.py` — System-Prompt-Switcher + Few-Shot-Bank-Loader
- [ ] `prompts/pse/<domain>/{system.md, examples.jsonl}` — Few-Shot-Bank pro Domain
- [ ] `pse_engine/verifier/property.py` — `PropertyVerifier` mit `hypothesis`
- [ ] `.github/workflows/pse-scorecard-nightly.yml` — Nightly CI (intern only)
- [ ] `docs/pse/scorecard.json` — Schema + erste Baseline (pre-Sprint-26 Werte)
- [ ] `pse_engine/telemetry/cost_per_domain.py` — pro Domain Token + Wall-Time
- [ ] `tests/test_pse/test_domain_registry.py` — Plugin-Loading + Isolation
- [ ] `tests/test_pse/test_property_verifier.py` — Property-Suite-Smoke

**CUT-OFF 1:** Foundation + Cost-Tracker live. Kann hier sauber pausieren.

### 26.2 — SQL + JSON + Bridge (Woche 2)

**Goal:** Erste 2 neue Domains live + erste Bridge funktional. Public Scorecard geht live.

**Tasks:**
- [ ] `pse_engine/domains/sql/catalog.py` — ~30 Primitiven (select/where/join/group/window/cte/dates)
- [ ] `pse_engine/domains/sql/types.py` — Type-Tags `Table`, `Column`, `SqlExpr`, `WindowSpec`, `JoinType`
- [ ] `pse_engine/domains/sql/verifier.py` — `sqlglot` parse + `duckdb` execute + result-set-equality
- [ ] `pse_engine/domains/sql/properties.py` — `query_idempotent`, `parameterised`, `no_full_scan`
- [ ] `pse_engine/domains/json/catalog.py` — ~20 Primitiven (path/filter/transform/combine)
- [ ] `pse_engine/domains/json/verifier.py` — `jsonschema` + cross-check vs `jq`
- [ ] `pse_engine/bridges/whitelist.py` — 12 typed Bridge-Operators
- [ ] `cognithor_bench/spider/` — Spider dev-easy harness (HF dataset import)
- [ ] `cognithor_bench/jq_cookbook/` — 150-task jq-cookbook harness
- [ ] CI integration für beide Benchmarks
- [ ] **Public Scorecard live** auf `cognithor.ai/pse/scorecard`

**CUT-OFF 2:** Sprint-26-Lite-Endzustand möglich (Foundation + SQL + JSON + erste Bridge).

### 26.3 — Datetime + AST + Float (Woche 3)

**Goal:** Code-Synthesis live (HumanEval-Plus), Datetime präzise, Float-Edge-Cases sauber.

**Tasks:**
- [ ] `pse_engine/domains/datetime/catalog.py` — ~25 Primitiven (parse/format/arithmetic/compare/tz/calendar)
- [ ] `pse_engine/domains/datetime/properties.py` — `tz_roundtrip`, `monotonic`, `dst_safe`
- [ ] `cognithor_bench/datetime_200/` — Custom 200-task suite + TempEval-3 wiring
- [ ] `pse_engine/domains/ast/catalog.py` — ~40 Primitiven (control/op/builtin/recursion)
- [ ] `pse_engine/domains/ast/sandbox.py` — Sandbox-Execution `timeout=2s, mem=128MB, no-net`
- [ ] `pse_engine/domains/ast/properties.py` — `terminates_within`, `pure_function`, `output_type`
- [ ] `cognithor_bench/humaneval_plus/` — HumanEval-Plus harness
- [ ] `pse_engine/domains/float/catalog.py` — ~15 Primitiven (nearly_equal/clamp/safe_div/kahan_sum)
- [ ] `cognithor_bench/float_100/` — Custom 100-task suite

### 26.4 — BinaryData + Image + Polish (Woche 4)

**Goal:** Sprint-Ende, Cross-Domain-Demo live, Reddit-Post raus.

**Tasks:**
- [ ] `pse_engine/domains/bytes/catalog.py` — ~25 Primitiven (read/write/encoding/hash/bitfield)
- [ ] `cognithor_bench/binary_200/` — 200-task suite (ZIP/GIF/ELF/JPEG marker walks)
- [ ] `pse_engine/domains/image/catalog_v2.py` — 12+ neue Pixel-Primitiven (symmetrie/anchor/conditional-fill/pattern/self-tile)
- [ ] `cognithor_bench/cross_domain/` — JSON→Datetime→SQL Demo-Tasks (10 cases)
- [ ] `docs/pse/scorecard.json` — Final-Werte
- [ ] `cognithor.ai/pse/scorecard` Page-Polish
- [ ] Reddit-Post auf `r/CognithorAgentOS` mit ehrlichen Zahlen
- [ ] Sprint-Retro `docs/superpowers/retros/2026-XX-sprint26.md`

## Externe Benchmark-Ziele

Siehe Memo. Soft-Bar = 6/10, Hard-Bar = 4/10 + Cross-Domain-Demo.

## Cut-Off-Regeln

- **Cut-Off 1** (Ende W1): Foundation + Cost-Tracker liefert Wert für Insurance-Pack-Marketing. Sprint kann hier pausieren.
- **Cut-Off 2** (Ende W2): Sprint-26-Lite (Foundation + SQL + JSON + 1 Bridge + Public Scorecard).
- **Pflicht-Pausen:** Mittwoch + Sonntag, nicht verhandelbar.

## Differenzierung — Public-Scorecard-Argument

Gegenüber CrewAI / AutoGen / II-Agent: dort werben Vibes, hier verifizierbare Zahlen aus reproduzierbaren Runs auf akzeptierten externen Benchmarks. Lokal-first-Open-Source mit Public Scorecard = stärkstes Vertrauenssignal.

## Stretch-Goals (nur falls 26.4 vor Plan)

- Auto-Discovery von DSL-Primitiven aus wiederkehrenden Composite-Patterns
- Symbolic Disagreement Sampling für ambiguous Inputs
- PSE-as-MCP für andere Channels (`synthesize me a regex/SQL`)
- Reverse-Engineering-Mode: aus 500 (input, output)-Logs Pipeline rekonstruieren
