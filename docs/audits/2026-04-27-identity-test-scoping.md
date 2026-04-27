# Identity/ test-coverage scoping — 2026-04-27

Sequel to `docs/audits/2026-04-27-stale-module-triage.md`. The triage flagged
`src/cognithor/identity/` as undertested (15 test refs over 9 270 LOC). Before
writing tests, this doc maps the actual structure so future test-writing
sessions can target the right files in the right order — and skip the ones
where autonomous test generation would produce pseudo-coverage.

## Summary

| Layer | Files | LOC | Existing tests | Coverage gap |
|---|---|---|---|---|
| top-level (`adapter.py`, `llm_bridge.py`, `__init__.py`) | 3 | 636 | 264 LOC across 2 files | 88 % untested |
| `cognitio/` (cognitive simulation) | 19 | 7 270 | 0 | complete |
| `storage/` (decentralised persistence) | 5 | 1 330 | 0 | complete |
| **TOTAL** | **27** | **9 270** | **264** | **97 % untested** |

## Per-file inventory

| File | Path | LOC | Public API surface | Test difficulty | Priority |
|---|---|---|---|---|---|
| `__init__.py` | identity/ | 23 | `IdentityLayer` | PURE-UNIT | P0 |
| `adapter.py` | identity/ | 437 | `IdentityLayer`, `enrich_context`, `process_interaction`, `reflect`, `save`, `load` | INTEGRATION-LIGHT | P0 |
| `llm_bridge.py` | identity/ | 199 | `CognithorLLMBridge`, `complete`, `chat`, `complete_json`, `health_check` | INTEGRATION-LIGHT | P0 |
| `attention.py` | cognitio/ | 295 | `MultiHeadAttention`, `HeadWeights` | PURE-UNIT | P1 |
| `biases.py` | cognitio/ | 442 | `BiasEngine`, `ConfirmationBias`, `AnchoringBias`, `AvailabilityBias`, `NegativeityBias` | DOMAIN-PHILOSOPHICAL | DEFER |
| `character.py` | cognitio/ | 441 | `CharacterManager`, `CognitiveState`, `PersonalityVector` | INTEGRATION-LIGHT | P1 |
| `dream.py` | cognitio/ | 375 | `DreamCycle` | DOMAIN-PHILOSOPHICAL | DEFER |
| `embeddings.py` | cognitio/ | 132 | `EmbeddingEngine`, `embed_text`, `embed_batch` | INTEGRATION-LIGHT | P1 |
| `emotion_shield.py` | cognitio/ | 373 | `EmotionShield`, `_detect_gaslighting`, `evaluate` | DOMAIN-PHILOSOPHICAL | DEFER |
| `engine.py` | cognitio/ | 1 660 | `CognitioEngine`, `_run_checkpoint`, `get_memory_by_id`, `reflect_on_interaction` | INTEGRATION-HEAVY | P0 |
| `epistemic.py` | cognitio/ | 166 | `EpistemicMap`, `add_belief`, `query_belief`, `contradiction` | PURE-UNIT | P1 |
| `existential.py` | cognitio/ | 191 | `ExistentialLayer`, `get_self_model_hint`, `checkin_event` | DOMAIN-PHILOSOPHICAL | DEFER |
| `garbage_collector.py` | cognitio/ | 393 | `GarbageCollector`, `collect_expired`, `collect_contradicted`, `consolidate` | INTEGRATION-LIGHT | P1 |
| `input_sanitizer.py` | cognitio/ | 84 | `InputSanitizer`, `sanitize`, `sanitize_batch` | PURE-UNIT | P1 |
| `memory.py` | cognitio/ | 267 | `MemoryRecord`, `MemoryType`, `MemoryStatus`, `MemoryValence`, `MemoryStore` | PURE-UNIT | P1 |
| `narrative.py` | cognitio/ | 293 | `NarrativeSelf`, `add_episode`, `get_narrative_arc`, `reflect` | DOMAIN-PHILOSOPHICAL | DEFER |
| `predictive.py` | cognitio/ | 209 | `PredictiveEngine`, `predict_next_query`, `predict_persona_drift` | DOMAIN-PHILOSOPHICAL | DEFER |
| `reality_check.py` | cognitio/ | 638 | `RealityCheck`, `validate_claim`, `contradiction_score`, `source_trust` | INTEGRATION-LIGHT | P0 |
| `somatic.py` | cognitio/ | 139 | `SomaticState`, `heart_rate_signal`, `arousal_level` | DOMAIN-PHILOSOPHICAL | DEFER |
| `temporal.py` | cognitio/ | 441 | `TemporalDensityTracker`, `add_event`, `density_at_time`, `recency_weight` | PURE-UNIT | P1 |
| `vector_store.py` | cognitio/ | 260 | `VectorStore`, `add`, `query`, `delete`, `count` | INTEGRATION-LIGHT | P0 |
| `working_memory.py` | cognitio/ | 464 | `WorkingMemory`, `add_interaction`, `get_recent`, `consolidate` | INTEGRATION-LIGHT | P0 |
| `arweave_store.py` | storage/ | 384 | `ArweaveStore`, `save_snapshot`, `retrieve_snapshot` | INTEGRATION-HEAVY | DEFER |
| `blockchain_anchor.py` | storage/ | 552 | `BlockchainAnchor`, `anchor_memory`, `verify_anchor` | INTEGRATION-HEAVY | DEFER |
| `ipfs_store.py` | storage/ | 129 | `IPFSStore`, `save_snapshot`, `retrieve_snapshot` | INTEGRATION-HEAVY | DEFER |
| `local_store.py` | storage/ | 165 | `LocalStore`, `save_snapshot`, `retrieve_snapshot` | INTEGRATION-LIGHT | P2 |
| `merkle_batcher.py` | storage/ | 94 | `MerkleBatcher`, `add_hash`, `flush`, `get_root` | PURE-UNIT | P2 |

## Recommended test-writing order

### Session 1 — Foundation, ~2 h, ~35 tests
- `memory.py` (15 tests): enum surface + `MemoryRecord` field defaults.
- `epistemic.py` (12 tests): belief add/query/contradiction round-trip.
- `input_sanitizer.py` (8 tests): regex edge cases (zero-width, control chars, unicode).

Outcome: ~475 LOC at >85 % branch coverage, P0/P1 PURE-UNIT cleared.

### Session 2 — Adapter layer, ~2.5 h, ~26 tests
- `llm_bridge.py` (14 tests): async bridging, `_parse_json_safe` permutations, thread safety, health-check failure modes.
- `adapter.py` partial (12 tests): facade init, `_empty_enrichment`, `enrich_context` happy + degraded paths. Full `process_interaction` deferred to Session 4.

Outcome: ~541 LOC of the integration glue verified.

### Session 3 — Vector + working memory, ~2.5 h, ~28 tests
- `vector_store.py` (16 tests): chromadb fixture, add/query/delete with `tmp_path`.
- `working_memory.py` partial (12 tests): SQLite fixture, `add_interaction`, `get_recent`, `consolidate`.

Outcome: persistence layers testable in isolation, unblocking engine.

### Session 4 — Cognitio engine, ≥ 3 h, ~40 tests (phased)
`engine.py` is 1 660 LOC. Sub-divide:
- Init + genesis anchors (8 tests)
- `add_interaction` flow + checkpoint triggers (12 tests)
- Memory retrieval + attention weighting (10 tests)
- Error recovery + state consistency (10 tests)

Outcome: P0 INTEGRATION-HEAVY entry-point covered.

### Session 5 — P1 logic + low-risk storage, ~3 h, ~25 tests
- `attention.py` (16 tests): salience math, head weights.
- `temporal.py` (8 tests): density_at_time, recency_weight (use freezegun).
- `local_store.py` (10 tests): tmp_path JSON round-trip.
- `merkle_batcher.py` (8 tests): pure-Python tree construction.

After Session 5: ~150 tests, ~3 000 LOC covered (~33 % of module). All P0 + most P1 cleared.

## Tests NOT to write autonomously

### DOMAIN-PHILOSOPHICAL — needs design spec first
`biases.py`, `dream.py`, `emotion_shield.py`, `existential.py`, `narrative.py`,
`predictive.py`, `somatic.py`.

These encode psychological / cognitive concepts. Without an oracle ("by what
measure is a confirmation-bias coefficient correct?", "what stimulus should
trigger the emotion shield?"), tests collapse into mock-the-internal-and-
assert-shape, which is **pseudo-coverage** — the test passes but doesn't
verify the cognitive contract.

The right move is integration-style tests that spawn a real engine, inject
known stimuli, and assert observable emergent behaviour. That's a separate
effort once a cognitive design doc / threat model exists.

### INTEGRATION-HEAVY — needs real test-net or contract-verified mocks
`arweave_store.py`, `blockchain_anchor.py`, `ipfs_store.py`.

Without an Arweave bundler, IPFS pinning service, or forked Base/Arbitrum
testnet, mocks can verify input-validation logic but not the actual
durability / anchor / pinning contracts. Mocks that just return the
expected dict shape are theatre.

The right move is unit-test the input validation + happy-path local logic,
then end-to-end test against staging services in CI (expensive but
authoritative).

## Threat-model questions blocking DEFER items

Before tests for `biases`, `dream`, `emotion_shield`, `existential`,
`narrative`, `predictive`, `somatic`:

1. **Cognitive spec** — how should the 12-layer system behave end-to-end?
2. **Threat model** — which cognitive failure modes are security-critical?
   ("Persona-drift after N adversarial inputs" vs "emotion-manipulation
   bypass"?)
3. **Validation oracle** — for philosophical modules, what makes a test
   correct?
4. **Integration boundaries** — isolation-tested or full-engine-only?

Once defined: integration tests on a real engine, no internal mocking.

## Realistic next-session plan

**Session 1** (foundation, ~2 h, ~35 tests). After Session 1 the module
moves from 3 % covered to ~8 %, and the next sessions can build on a
verified foundation. If only one session of effort is available, that's the
one with the highest ratio of certainty-gained-per-hour.

This doc itself is the artefact of effort that would otherwise have gone
into writing speculative tests that cover the wrong things. Identity test
coverage is now scoped, not just flagged.
