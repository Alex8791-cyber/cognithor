# Cognithor Synthesis Engine — Sprint 1 Plan

**Implementierungs-Kickoff für Phase-2-Spec v1.4**

---

| Feld | Wert |
|---|---|
| Sprint-Dauer | 2–3 Wochen |
| Voraussetzung | Phase 1 funktional, v1.4 Spec freigegeben |
| Ergebnis | Lauffähiges MVP der Synthesis-Engine, durchgängig vom CLI bis SynthesisResult |
| Test-Coverage-Ziel | 80 % zu Sprint-Ende |
| Benchmark-Ziel | Baseline-Score auf 20-Task-Subset des Leak-Free-Held-Out-Sets |

---

## Sprint-1-Philosophie

**Vertikal vor horizontal.** Lieber eine Ende-zu-Ende-Pipeline mit reduziertem Funktionsumfang, als jedes Modul einzeln perfekt. Das stellt früh sicher, dass die Schnittstellen passen und die Module zusammen lauffähig sind.

**Test-Driven für Daten-Strukturen.** Datentypen (`TaskSpec`, `ProgramState`, `VerifierResult`, `Budget`) zuerst, mit vollständigen Tests, bevor die Logik darüber gebaut wird. Hashing, Serialisierung, Mutual-Exclusion-Assertions sind Pflicht-Tests.

**Heuristik-Werte aus Config, nicht hardcoded.** Jeder Multiplier, jeder Threshold, jede Bound liest aus `configs/synthesis/heuristics.yaml`. Sprint-1-Pflicht.

**Interaktions-Tests als Frühwarnsystem.** Die drei Wechselwirkungs-Tests (E1×E2, E3×E1, E6×E7) werden in Sprint 1 geschrieben und ausgeführt, auch wenn die Reserve-Fixes (Few-Demos-Dampening, Argument-Quality) noch nicht aktiviert sind. Tests können dann grün, gelb (zu kalibrieren) oder rot (Reserve-Fix aktivieren) sein.

---

## Aufgaben-Reihenfolge

### Aufgabe 1 — Konfigurations-Infrastruktur

**Zeit:** 0.5 Tage
**Spec:** Implementierungs-Hinweis ChatGPT Runde 4, alle Heuristik-Werte aus `heuristics.yaml`
**Abhängigkeiten:** keine

**Inhalt:**
- `configs/synthesis/heuristics.yaml` ins Repo
- `cognithor/core/synthesis/config.py` — Pydantic-basierter Loader mit Schema-Validierung
- Validierung: Budget-Partition summiert auf 1.0, Mode-Schwellen monoton, Multiplier > 0
- CLI-Flag `--config-override` für Tests

**Akzeptanzkriterien:**
- [ ] `Config.load()` parst `heuristics.yaml` ohne Fehler
- [ ] Schema-Validierung schlägt bei mutierter Datei fehl (z.B. `mcts: 0.71` statt `0.70`)
- [ ] 100 % Test-Coverage auf `config.py`
- [ ] Alle nachfolgenden Module nutzen ausschließlich `Config`-Instanzen, keine Magic-Numbers

---

### Aufgabe 2 — Datentypen und Datenstrukturen

**Zeit:** 1 Tag
**Spec:** §9 (alle Versionen v1.0–v1.4)
**Abhängigkeiten:** Aufgabe 1

**Inhalt:**
- `cognithor/core/synthesis/types.py` — alle frozen Dataclasses
  - `TaskSpec`, `Demo`, `Program`, `ProgramState`
  - `FeatureWithConfidence` (mit Sample-Size-Property)
  - `DSLPrimitive` (mit `is_high_impact` + `is_structural_abstraction` + Mutual-Exclusion-Assertion)
  - `VerifierResult` (mit triviality + suspicion)
  - `Budget`, `PartitionedBudget`
  - `MCTSState`, `MCTSNode`
  - `SynthesisResult`, `CacheEntry`, `MixedPolicy`
- Property-Tests via Hypothesis für jede Datenstruktur

**Akzeptanzkriterien:**
- [ ] Alle frozen-Dataclasses sind hashbar
- [ ] `DSLPrimitive` rejected Konstruktion mit beiden Flags `True`
- [ ] `PartitionedBudget` summiert immer auf 1.0
- [ ] `FeatureWithConfidence.confidence` korrekt skaliert via Sample-Size
- [ ] Hypothesis-Test-Suite 100 % grün

---

### Aufgabe 3 — Phase-1-Komponenten einbinden (Adapter)

**Zeit:** 1 Tag
**Spec:** §7 (Phase-1-Übergabe)
**Abhängigkeiten:** Aufgabe 2

**Inhalt:**
- Phase-1-Module als Adapter ins Phase-2-Layout mounten:
  - ARC-DSL → `synthesis/dsl/arc_dsl.py` (mit High-Impact + Structural-Abstraction Klassifikation aus Config)
  - Feature-Extractor → mit `FeatureWithConfidence` als Output
  - Sandbox-Executor → unverändert
  - Enumerative Search → als Fast-Path verfügbar
  - Verifier (Phase-1-Stages 1-5) → erweitert um `partial_pixel_match`
- Smoke-Test: Phase-1-Pipeline läuft end-to-end auf 5 Test-Tasks

**Akzeptanzkriterien:**
- [ ] Alle Phase-1-Tests laufen weiter grün
- [ ] Feature-Extractor liefert `FeatureWithConfidence`-Werte mit korrekten `n_demos`
- [ ] DSL-Registry führt `is_high_impact`-Whitelist (7 Primitive) und `is_structural_abstraction`-Whitelist (5 Primitive)
- [ ] Verifier liefert graduiert (`partial_pixel_match` ∈ [0, 1])

---

### Aufgabe 4 — Symbolic-Prior

**Zeit:** 1.5 Tage
**Spec:** §4.4
**Abhängigkeiten:** Aufgaben 2, 3

**Inhalt:**
- `synthesis/prior/symbolic/prior.py` — `SymbolicPrior`-Klasse mit Heuristik-Katalog
- `synthesis/prior/symbolic/rules.py` — die ~20 Regeln aus §4.4.2
- `synthesis/prior/symbolic/confidence.py` — Sample-Size-Dämpfung
- Initiale Regel-Gewichte uniform aus Config; Coordinate-Ascent-Tuning kommt in Sprint 2

**Akzeptanzkriterien:**
- [ ] Symbolic-Prior allein liefert valide Policy für 50 Test-Tasks
- [ ] Sample-Size-Dämpfung getestet für n_demos ∈ {1, 2, 3, 4, 5}
- [ ] Property-Test: Σ probs = 1.0 (mit Toleranz 1e-6)
- [ ] Heuristik-Regel-Beiträge logged (DEBUG-Level)

---

### Aufgabe 5 — LLM-Prior-Service

**Zeit:** 2 Tage
**Spec:** §4.2-§4.7
**Abhängigkeiten:** Aufgaben 1, 2

**Inhalt:**
- `synthesis/prior/llm/service.py` — `LLMPriorService` mit Batched-Async-Queue
- `synthesis/prior/llm/backends/llama_cpp.py` — llama.cpp-Backend mit GBNF
- `synthesis/prior/llm/prompts/` — Jinja2-Templates (policy, repair_stage1_default, repair_stage1_alternative, repair_stage2)
- `synthesis/prior/llm/logit_mask.py` — semantische Constraints (Type, Range, Anti-Redundancy)
- `synthesis/prior/llm/cache.py` — TwoLevelCache, speichert nur LLM-Komponente
- `synthesis/prior/llm/calibration.py` — Temperature-Scaling-Kalibrierungs-Skript

**Wichtige Implementierungsdetails:**
- Determinismus-Modus: `--deterministic` Flag → batch_size=1, deterministic CUDA
- GBNF-Grammatik aus `dsl/grammars/arc.gbnf` lesen
- Logit-Masking *nach* GBNF (im Wrapper, nicht im Kernel)

**Akzeptanzkriterien:**
- [ ] LLM-Prior liefert Top-K-Liste mit Logits in valider DSL für Test-Spec
- [ ] GBNF + Logit-Mask: 10 000 Random-Specs → 100 % syntaktisch und semantisch valide Outputs
- [ ] Batched-Service liefert > 4× Throughput vs. sequenziell
- [ ] Cache speichert ohne α/Symbolic-Felder
- [ ] Determinismus-Test bytegleich über 10 Wiederholungen mit Seed=42

---

### Aufgabe 6 — Prior-Mixer + α-Controller

**Zeit:** 1.5 Tage
**Spec:** §4.4.4 (multiplikatives α)
**Abhängigkeiten:** Aufgaben 4, 5

**Inhalt:**
- `synthesis/prior/dual_prior.py` — `PriorMixer` mit `α = α_entropy · α_performance`
- `synthesis/prior/alpha_controller.py` — `AlphaController` mit Hysterese
- `synthesis/prior/performance_tracker.py` — `PriorPerformanceTracker` mit Sliding-Window

**Wichtige Tests:**
- α ∈ [0.25, 0.85] über 10 000 Random-Inputs (Property-Test)
- Hysterese: oszillierender Input → α stabil
- Cold-Start: keine Daten → α = 0.85 (Default)
- Cache-Hit → α wird live neu berechnet, nicht aus Cache gelesen

**Akzeptanzkriterien:**
- [ ] Multiplikative α-Formel implementiert wie Spec §4.4.4
- [ ] Hysterese-Window aus Config (5)
- [ ] Sample-Size-Dämpfung wirkt korrekt (Test mit 1, 2, 4 Demos)
- [ ] A/B-Test: künstlich-verschlechterter LLM → α sinkt korrekt unter 0.4 nach Window-Iterationen

---

### Aufgabe 7 — MCTS-Controller

**Zeit:** 2 Tage
**Spec:** §5
**Abhängigkeiten:** Aufgaben 2, 3, 6

**Inhalt:**
- `synthesis/mcts/controller.py` — `MCTSController` mit Anytime-Loop
- `synthesis/mcts/node.py` — `MCTSNode` mit PUCT-Score-Property
- `synthesis/mcts/puct.py` — PUCT-Formel mit `c_puct(d)`-Variante (depth-scaling opt-in)
- `synthesis/mcts/pruning.py` — Observational Equivalence + Type-Mismatch + Cost-Bound
- `synthesis/mcts/parallel.py` — Virtual-Loss + 4 Workers
- `synthesis/mcts/fallback.py` — `FallbackController` mit `min_node_depth_mean`

**Wichtige Tests:**
- PUCT-Formel mathematisch korrekt
- Selection→Expansion→Simulation→Backpropagation Reihenfolge
- Anytime: KeyboardInterrupt liefert `best_so_far`
- Fallback nicht ausgelöst bei `node_depth_mean < 2`
- Property-Test: Σ N(s,a) = N(s)

**Akzeptanzkriterien:**
- [ ] MCTS findet ≥ 30 % korrekte Lösungen auf 20-Task-Subset (sprint-1 baseline)
- [ ] Fallback-Trigger korrekt bei Score < 0.2 + Plateau
- [ ] Tree-Export zu JSON für Debugging
- [ ] 4 Workers stabil über 100 Test-Tasks (kein Race-Condition)

---

### Aufgabe 8 — Verifier-Erweiterungen (Triviality + Suspicion)

**Zeit:** 1 Tag
**Spec:** §7.3
**Abhängigkeiten:** Aufgabe 3

**Inhalt:**
- `synthesis/verifier/triviality.py` — 5 regelbasierte Tests
- `synthesis/verifier/suspicion.py` — metrik-basiert mit High-Impact + Structural-Abstraction Multipliers
- `synthesis/verifier/scoring.py` — Score-Aggregation mit Gewichten aus Config

**Adversarial-Test-Korpus:**
- 50 triviale Programme (Identity, Constant, etc.) → triviality ≤ 0.3
- 30 legitime Single-High-Impact → suspicion ≥ 0.85
- 20 Single-Structural-Abstraction (10 legitim, 10 verdächtig)
- 200 legitime Programme → triviality ≥ 0.85 UND suspicion ≥ 0.85

**Akzeptanzkriterien:**
- [ ] 50 Adversarial-Programme alle als ≤ 0.3 triviality klassifiziert
- [ ] 30 legitime Single-High-Impact-Programme alle suspicion ≥ 0.85
- [ ] 200 legitime Programme: keine False-Positive über 5 %
- [ ] Triviality + Suspicion gewichten korrekt im Final-Score (12 % + 12 %)

---

### Aufgabe 9 — Critic & Refiner mit Drei-Zonen-Mode

**Zeit:** 2 Tage
**Spec:** §6
**Abhängigkeiten:** Aufgaben 5, 6, 8

**Inhalt:**
- `synthesis/refiner/diff_analyzer.py` — Pixel/Struktur/Farb-Diff
- `synthesis/refiner/trace_replay.py` — Schritt-für-Schritt-Lokalisation
- `synthesis/refiner/local_edit.py` — deterministische Mutationen
- `synthesis/refiner/llm_repair_two_stage.py` — Two-Stage mit Retry
- `synthesis/refiner/symbolic_repair.py` — 5 Heuristik-Regeln aus §6.5.2
- `synthesis/refiner/hybrid_repair.py` — parallele Symbolic + Single-Stage-LLM
- `synthesis/refiner/mode_controller.py` — `RefinerModeController` mit 3-Call-Hysterese
- `synthesis/refiner/cegis.py` — CEGIS mit Budget-Cutoff
- `synthesis/refiner/escalation.py` — Stufen-Logik

**Wichtige Tests:**
- Drei-Zonen-Selection bei α ∈ {0.3, 0.4, 0.5}
- Hysterese: α-Sequenz [0.46, 0.44, 0.46, 0.44] → bleibt 3 Calls im ersten Mode
- Two-Stage-Retry bei systematischem Verifier-Fail
- CEGIS terminiert garantiert in ≤ 5 Iter UND Budget-Limit

**Akzeptanzkriterien:**
- [ ] Drei-Zonen-Mode-Selection deterministisch
- [ ] Hybrid-Mode liefert *bessere* Latenz als Full-LLM (P50-Vergleich)
- [ ] Refiner-Pipeline fügt mind. 8 % Erfolgsrate (Sprint-1-Ziel; Spec verlangt 12 % bis Phase-2-Ende)
- [ ] CEGIS-Budget hart eingehalten

---

### Aufgabe 10 — Synthesis-Engine Top-Level

**Zeit:** 1 Tag
**Spec:** §3.2 Kontrollfluss
**Abhängigkeiten:** alle

**Inhalt:**
- `synthesis/engine.py` — `SynthesisEngine.synthesize()` Top-Level-API
- Routing zwischen Phase-1-Fast-Path und Phase-2-MCTS
- Budget-Partition + Reclamation
- Early-Stop-Logik
- Memory-Hooks (Tactical-Memory-Schreiben)

**Akzeptanzkriterien:**
- [ ] Smoke-Test: 5 ARC-Tasks komplett synthetisiert (egal ob erfolgreich)
- [ ] Budget-Partition strikt eingehalten
- [ ] Early-Stop emittiert Telemetrie
- [ ] Cache-Write nach erfolgreicher Synthese

---

### Aufgabe 11 — Wechselwirkungs-Tests (E1×E2, E3×E1, E6×E7)

**Zeit:** 1 Tag
**Spec:** §12.2
**Abhängigkeiten:** Aufgabe 10

**Inhalt:**
- `tests/interactions/test_e1_e2_alpha_asymmetry.py`
- `tests/interactions/test_e3_e1_few_demos_llm_dominance.py`
- `tests/interactions/test_e6_e7_high_impact_reward_hacking.py`

**Vorgehen:**
1. Tests schreiben, ausführen
2. Falls grün: Reserve-Fixes bleiben deaktiviert
3. Falls gelb (knapp): Schwellen anpassen, dokumentieren
4. Falls rot: Reserve-Fixes aktivieren via Config-Flag (`reserved_fixes.few_demos_dampening.enabled: true` oder `reserved_fixes.argument_quality_factor.enabled: true`)

**Akzeptanzkriterien:**
- [ ] Alle 12 Wechselwirkungs-Tests laufen
- [ ] Gelb/Rot-Ergebnisse dokumentiert
- [ ] Reserve-Fixes (falls aktiviert) konfigurierbar
- [ ] Sprint-Ende-Report enthält Wechselwirkungs-Status

---

### Aufgabe 12 — Benchmark + CI

**Zeit:** 1 Tag
**Spec:** §12.3, §12.4
**Abhängigkeiten:** alle

**Inhalt:**
- `benchmarks/arc_agi_3/runner.py` — automatisierter Benchmark
- `benchmarks/arc_agi_3/leak_free_set/` — 20-Task-Subset für Sprint 1 (komplette 200-Task-Kuration in Sprint 2)
- GitHub-Actions-Workflow (täglich) auf festem Held-Out
- Streamlit-Dashboard für Score-Verlauf

**Akzeptanzkriterien:**
- [ ] Benchmark läuft End-to-End
- [ ] Sprint-1-Baseline-Score auf 20 Tasks ermittelt
- [ ] CI-Workflow grün bei neuem Push
- [ ] Dashboard zeigt P50/P95-Latenz, Score, Cache-Hit-Rate

---

## Sprint-Ende-Deliverables

1. **Lauffähige Engine:** `python -m cognithor.synthesis.engine --task <task.json>` liefert SynthesisResult.
2. **Test-Coverage 80 %+** über alle Module.
3. **Wechselwirkungs-Status-Report:** Welche Reserve-Fixes wurden aktiviert?
4. **Baseline-Benchmark-Score:** auf 20-Task-Subset des Leak-Free-Sets.
5. **Telemetrie-Dashboard:** Score-Verlauf, Latenz, α-Verlauf, Mode-Verteilung.
6. **Calibration-Report:** ECE-Werte für LLM und Symbolic, optimale τ-Werte in Config eingetragen.

---

## Risiken in Sprint 1

| Risiko | Wahrscheinlichkeit | Mitigation |
|---|---|---|
| llama.cpp + Qwen3.6 Setup unter Windows brüchig | Mittel | Tag 1 als reines Setup-Kapitel; WSL2-Fallback dokumentiert |
| GBNF-Grammar zu restriktiv → keine validen Outputs | Niedrig | Test mit minimaler DSL zuerst; Grammar inkrementell erweitern |
| MCTS-Implementierung subtle bugs (PUCT, virtual loss) | Mittel | Property-Tests + Vergleich gegen Referenz-Paper |
| Wechselwirkungs-Tests durchweg rot → mehrere Reserve-Fixes nötig | Niedrig | Reserve-Fixes sind als Config-Flags vorbereitet, kein Code-Refactor nötig |
| Sprint-Latenz-Budget nicht erreicht (P50 > 18s) | Mittel | Sprint-2 hat Performance-Optimierung als dediziertes Thema |

---

## Sprint-2-Vorschau (nicht Teil von Sprint 1)

- Coordinate-Ascent-Tuning der Symbolic-Prior-Regel-Gewichte
- Vollständige Leak-Free-Held-Out-Kuration (200 Tasks)
- Performance-Optimierung (Latenz-Budget-Profiling)
- ECE-Validierung mit größeren Datensätzen
- A2A-Channel-Integration in Cognithor
- Tactical-Memory-Hooks vollständig
- Hashline-Guard-Capability-Token-Validierung

---

## Nach Sprint 1: optionale fünfte Spec-Review

Wenn Sprint 1 abgeschlossen ist und reale Daten vorliegen, kann eine optionale fünfte Review-Runde mit ChatGPT durchgeführt werden — aber dann *daten-gestützt*, nicht spec-getrieben. Fragen wie:

- Sind die heuristischen Werte (3×, 1.5×, 0.35/0.45) empirisch trennscharf?
- Welcher Wechselwirkungs-Test war kritisch — und welche Reserve-Fixes wurden tatsächlich aktiviert?
- Welche `c_puct`-Wahl gewinnt im Grid-Search?
- Lohnt sich Q5_K_M wirklich gegenüber Q4_K_M?

Diese Review wäre dann der Übergang von "implementierungsreif" zu "empirisch validiert".

---

**Status: Bereit für Sprint-1-Start.**
