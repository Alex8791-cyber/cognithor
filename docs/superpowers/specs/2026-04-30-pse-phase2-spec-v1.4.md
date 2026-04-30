# Cognithor Programm-Synthese-Engine — Phase 2 Spezifikation

**Neuro-Symbolic Synthesis Layer**

---

| Feld | Wert |
|---|---|
| Dokument-Version | **1.4** (Konvergenz-Revision nach vierter externer Review) |
| Vorgängerversionen | 1.0 (initial), 1.1 (Runde 1), 1.2 (Runde 2), 1.3 (Runde 3) |
| Zielsystem | Cognithor v0.78.0+ |
| Phase | 2 von 4 |
| Voraussetzung | Phase 1 abgeschlossen (Symbolic Enumerative Engine, ARC-DSL, Sandbox, Verifier) |
| Sprache | Python 3.12+ |
| Lizenz | Apache 2.0 |
| LLM-Prior-Modell | Qwen3.6:27B-Instruct (Q5_K_M Default, Q4_K_M Fallback) |
| Hardware-Ziel | RTX 5090 (32 GB VRAM), Ryzen 9 9950X3D, Windows 11 + WSL2 |
| Geschätzter Aufwand | 6–10 Wochen Soloentwicklung, AI-assisted |
| Status | **Implementierungsreif** (extern bestätigt nach 4 Review-Runden) |

---

## Changelog v1.3 → v1.4

Diese Revision ist die finale Konvergenz vor Implementierungsbeginn. Drei gezielte Verfeinerungen plus Doku-Updates.

| ID | Änderung | Motivation | Abschnitt |
|---|---|---|---|
| F1 | `is_structural_abstraction` als separates Flag, `objects` umklassifiziert | `objects()` produziert Zwischenrepräsentation, keinen Output — als High-Impact zu stark privilegiert | §7.3.2, DSL-Registry |
| F2 | α-Graubereich [0.35, 0.45] mit Hybrid-Repair-Mode | Hartes `if α < 0.4` ist anfällig für Hysterese-Oszillation an der Grenze | §6.5.2, §6.6 |
| F3 | Drei neue Wechselwirkungs-Test-Kategorien | Subtile E1×E2/E3×E1/E6×E7 Risiken testabdecken | §12.2 |

Plus Doku-Updates:
- §22 erweitert um `expected_depth_estimator` als Phase-3-Mechanismus
- §19.4 (NEU): Adressierung Runde 4

Strukturell unveränderte Abschnitte: Architektur-Übersicht (§3), Sicherheit (§14), Datei-Layout (§10).

**Hinweis:** ChatGPT-Review Runde 4 hat v1.3 als "implementierungsreif" bestätigt mit nur einem konkreten Änderungswunsch (`objects`-Reklassifizierung) und einer optionalen Verfeinerung (α-Graubereich). v1.4 setzt beide um plus die expliziten Test-Wünsche aus den Wechselwirkungs-Hinweisen.

---

## Inhalt

1. Executive Summary
2. Ziele und Nicht-Ziele
3. Architektur-Übersicht
4. Modul A — Dual-Prior (LLM + Symbolic)
5. Modul B — MCTS-Suchcontroller
6. Modul C — Critic & Refiner
7. Verifier-Erweiterungen
8. Integration in Cognithor
9. Datenstrukturen und APIs
10. Datei-Layout im Repository
11. Telemetrie und Observability
12. Test-Strategie und Benchmarks
13. Performance-Budget und Hardware
14. Sicherheit und Capability-Tokens
15. Risiken und Gegenmaßnahmen
16. Offene Fragen
17. Akzeptanzkriterien
18. Validierungs-Checkliste
19. Referenz: Adressierung der Kritikpunkte (alle Runden)
20. Glossar
21. Referenzen
22. Phase-3-Transition

---

## 1. Executive Summary

Phase 2 erweitert die in Phase 1 gebaute rein-symbolische Enumerative-Search-Engine um vier neuro-symbolische Komponenten:

1. **Dual-Prior** — multiplikativ adaptive Mischung aus LLM-basiertem Prior (Qwen3.6:27B) und symbolischem Prior. Beide Komponenten sind durch Sample-Size gedämpft (Symbolic) und durch Bounds [0.25, 0.85] geschützt (α).
2. **MCTS-Suchcontroller** — PUCT-basiert, Anytime, Virtual-Loss-parallelisiert, mit Fallback-Controller, der zusätzlich `node_depth_mean` prüft.
3. **Critic & Refiner** — gestufte Reparatur (Local → LLM/Symbolic/Hybrid → CEGIS). LLM-Repair zweistufig (CoT → JSON), mit Retry. **Drei-Zonen-Mode-Selection nach α** (v1.4-Verfeinerung): voll-LLM, Hybrid, Symbolic-only.
4. **Erweiterter Verifier** — Triviality + Suspicion-Score, mit `is_high_impact`-Flag (3×-Multiplier) und **`is_structural_abstraction`-Flag** (1.5×-Multiplier, v1.4-Differenzierung).

**Ergebnisziele Phase 2 (unverändert seit v1.1):**
- ARC-AGI-3 Leak-Free Held-Out: 40–55 % Erfolgsrate @ 30 s
- 3–4× Score-Steigerung gegenüber Phase-1-Baseline
- P50 Latenz < 18 s, P95 < 60 s
- Vollständige Cognithor-Integration als Channel `program-synthesis`

**Out of Scope (unverändert):**
- Library Learning / DreamCoder-Style Abstraktion (Phase 3)
- Nicht-Grid-DSLs (Phase 4)
- Eigenes Foundation-Modell-Training

---

## 2. Ziele und Nicht-Ziele

### 2.1 Funktionale Ziele

| ID | Ziel | Δ v1.3 |
|---|---|---|
| F1 | Dual-Prior liefert kalibrierte Top-K-Verteilungen (ECE < 0.06) | unverändert |
| F2–F20 | (wie v1.3) | unverändert |
| **F21** | **DSL unterscheidet High-Impact und Structural-Abstraction** | **NEU F1** |
| **F22** | **Refiner nutzt Hybrid-Mode im α-Graubereich [0.35, 0.45]** | **NEU F2** |
| **F23** | **Wechselwirkungs-Tests für E1×E2, E3×E1, E6×E7 vorhanden** | **NEU F3** |

### 2.2 Nicht-funktionale Ziele

(Unverändert ggü. v1.3.)

### 2.3 Nicht-Ziele

(Unverändert ggü. v1.3.)

---

## 3. Architektur-Übersicht

### 3.1 Datenfluss (v1.4 — Drei-Zonen-Refiner)

```
                                ┌──────────────────────────────┐
                                │         TaskSpec              │
                                └──────────────┬────────────────┘
                                               │
                          ┌────────────────────┼────────────────────┐
                          ▼                    ▼                    ▼
              ┌────────────────────┐  ┌────────────────┐  ┌──────────────────┐
              │  Feature-Extractor │  │  DSL-Selector  │  │ Budget-Partition │
              │  + Confidence      │  │ + High-Impact  │  │                  │
              │  + sample-size damp│  │ + Structural   │  │                  │  ← NEU F1
              └──────────┬─────────┘  └───────┬────────┘  └────────┬─────────┘
                         │                    │                    │
                         └─────────┬──────────┴────────────────────┘
                                   │
                   ┌───────────────┼───────────────┐
                   ▼               ▼               ▼
        ┌──────────────────┐ ┌─────────────┐ ┌────────────────┐
        │ Symbolic-Prior   │ │ LLM-Prior   │ │ Cache-Lookup   │
        │                  │ │ Qwen3.6:27B │ │ liefert nur    │
        │                  │ │             │ │ LLM-Komponente │
        └────────┬─────────┘ └──────┬──────┘ └────────┬───────┘
                 │                  │                 │
                 │                  └─────────┬───────┘
                 │                            │
                 └──────────┬─────────────────┘
                            │
                            ▼
                ┌──────────────────────────────┐
                │  Prior-Mixer                 │
                │  α = α_entropy · α_perf      │
                │  α ∈ [0.25, 0.85]            │
                │  π = α·π_llm + (1-α)·π_symb  │
                └────────────┬─────────────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │   MCTS-Controller    │
                  │   + Fallback-Ctrl    │
                  │   + early-stop       │
                  └──────┬────────────┬──┘
                         │            │
                         ▼            ▼
              ┌──────────────────┐  ┌──────────────────────────────┐
              │ Sandbox-Executor │  │ Verifier                     │
              │                  │  │ +Triviality                  │
              │                  │  │ +Suspicion(strikt)           │
              │                  │  │ +HighImpact(3×)              │
              │                  │  │ +StructuralAbstraction(1.5×) │  ← NEU F1
              └────────┬─────────┘  └─────────┬────────────────────┘
                       │                      │
                       └──────────┬───────────┘
                                  │
                                  ▼
                       ┌──────────────────────────────────────┐
                       │ Critic & Refiner — Drei-Zonen-Mode   │  ← NEU F2
                       │   α ≥ 0.45  → Full-LLM (2-stage+retry)│
                       │   0.35–0.45 → Hybrid (Symbolic+1-stage)│
                       │   α < 0.35  → Symbolic-only           │
                       │   → CEGIS optional                    │
                       └─────────┬────────────────────────────┘
                                 │
                                 ▼
                      ┌──────────────────┐
                      │ SynthesisResult  │
                      └──────────────────┘
```

### 3.2 Kontrollfluss (unverändert ggü. v1.3, mit Refiner-Aufruf jetzt 3-Zonen-aware)

```python
async def synthesize(spec: TaskSpec, budget: Budget) -> list[SynthesisResult]:
    # ... wie v1.3 ...
    
    while not partition.mcts.exhausted():
        # ... wie v1.3 ...
        
        for child in children:
            program = tree.simulate(child)
            score   = await verifier.evaluate(program, spec)
            tree.backpropagate(child, score)
            perf_tracker.record(child.action, score)
            
            if score.partial and not score.complete:
                if not partition.refiner.exhausted():
                    refined = await critic_refiner.refine(
                        program, score, spec, partition,
                        current_alpha=alpha_ctrl.current(),
                        # Refiner wählt intern Three-Zone-Mode (NEU F2)
                    )
                    if refined and refined.score.complete:
                        best_results.append(refined)
            elif score.complete:
                best_results.append(SynthesisResult(program, score, ...))
        
        partition.mcts.consume(node.cost)
    
    # ... wie v1.3 ...
```

---

## 4. Modul A — Dual-Prior (LLM + Symbolic)

### 4.1 Zweck (unverändert)

### 4.2 Hardware- und Backend-Wahl (unverändert)

### 4.3 Modell-Selektion und Fallback (unverändert)

### 4.4 Symbolic-Prior — Heuristik-Katalog (unverändert ggü. v1.3)

`FeatureWithConfidence` mit Sample-Size-Dämpfung, ~20 Heuristik-Regeln mit Confidence-Multiplikator.

#### 4.4.4 α-Mischung — multiplikativ adaptiv (unverändert ggü. v1.3)

```
α_entropy ∈ [0.5, 0.85]
α_performance ∈ [0.5, 1.0]
α = α_entropy · α_performance ∈ [0.25, 0.85]
```

**Begriffliche Klärung (NEU v1.4):**

ChatGPT-Review Runde 4 wies auf eine wichtige semantische Unterscheidung hin: α ist die *Search-Vertrauensschwelle*, die LLM-Repair-Schwelle (siehe §6.5.2) ist die *Repair-Vertrauensschwelle*. Beide sind getrennt zu verstehen:

- **Search-α** (in [0.25, 0.85]): wie stark MCTS dem LLM in der *Suche* vertraut
- **Repair-α-Schwellen** (0.35 / 0.45): wann der *Refiner* welchen Modus wählt

Die Asymmetrie ist beabsichtigt: in der Suche ist falsches LLM-Vertrauen weniger schädlich (PUCT-Exploration kompensiert), beim Repair ist falsches Vertrauen direkt teuer (jeder fehlerhafte Edit kostet Verifier-Calls).

### 4.5 Top-K depth-abhängig (unverändert)

### 4.6 Constrained Decoding (unverändert)

### 4.7 Prompt-Schema (unverändert ggü. v1.3 — Two-Stage mit Retry)

### 4.8 Kalibrierung der Logits (unverändert)

### 4.9 Cache — α-Aware Mixing (unverändert)

### 4.10 Determinismus unter Batched-Inference (unverändert)

### 4.11 Parallelisierung (unverändert)

### 4.12 Öffentliche API (unverändert)

### 4.13 Robustheits-Anforderungen (erweitert)

(Aus v1.3, plus:)

| Anforderung | Test |
|---|---|
| **Refiner-Mode-Selection bei α=0.39 nicht oszillierend** | **NEU F2: Hysterese-Test mit oszillierendem α-Input nahe Schwelle** |
| **`is_structural_abstraction` korrekt klassifiziert** | **NEU F1: Whitelist-Konsistenz-Test über DSL-Registry** |

---

## 5. Modul B — MCTS-Suchcontroller (unverändert ggü. v1.3)

---

## 6. Modul C — Critic & Refiner

### 6.1 Zweck (unverändert)

### 6.2 Pipeline (revidiert für Drei-Zonen-Mode)

```
Partielles-Lösungs-Programm + Verifier-Score + current_alpha
              │
              ▼
   ┌──────────────────┐
   │ 1. Diff-Analyzer │
   └─────────┬────────┘
             │
             ▼
   ┌──────────────────┐
   │ 2. Trace-Replay  │
   └─────────┬────────┘
             │
             ▼
   ┌──────────────────┐
   │ 3. Critic-Report │
   └─────────┬────────┘
             │
             ▼
   ┌──────────────────────────────────────────────┐
   │ 4. Repair-Eskalation — Drei-Zonen-Mode (F2)   │
   │   a) Local-Edit ALWAYS FIRST                 │
   │   b) IF score >= 0.3:                        │
   │        α >= 0.45  → Full-LLM (2-stage+retry) │
   │        0.35 ≤ α   → Hybrid-Mode              │
   │           < 0.45    (Symbolic + 1-stage LLM, │
   │                      compete)                │
   │        α < 0.35   → Symbolic-only            │
   │   c) CEGIS IF score >= 0.5 + budget          │
   └─────────┬────────────────────────────────────┘
             │
             ▼
   ┌──────────────────┐
   │ 5. Re-Verify     │
   └──────────────────┘
```

### 6.3 Diff-Analyzer (unverändert)

### 6.4 Trace-Replay (unverändert)

### 6.5 Repair-Strategien

#### 6.5.1 Local-Edit (unverändert)

#### 6.5.2 LLM/Symbolic/Hybrid-Repair — Drei-Zonen-Mode (REVIDIERT F2)

ChatGPT-Review Runde 4: Hartes `if α < 0.4` schaltet bei α=0.39 vs α=0.41 völlig unterschiedliche Refiner-Pfade — das ist anfällig für Hysterese-Oszillation in der Grenzregion. Saubere Lösung: drei Zonen statt zwei.

##### Zonen-Definition

| α-Bereich | Modus | Strategie |
|---|---|---|
| α ≥ 0.45 | **Full-LLM** | Two-Stage-Repair mit Retry (wie v1.3) |
| 0.35 ≤ α < 0.45 | **Hybrid** | Symbolic-Repair und Single-Stage-LLM parallel; bester Edit gewinnt |
| α < 0.35 | **Symbolic-only** | Nur regelbasierte Heuristiken |

##### Implementierung

```python
async def repair_dispatch(
    self,
    program: Program,
    critic_report: CriticReport,
    spec: TaskSpec,
    current_alpha: float,
    refiner_budget: Budget,
) -> Optional[Program]:
    
    # ── Zone 1: α ≥ 0.45 → Full-LLM
    if current_alpha >= 0.45:
        return await self._llm_repair_full(
            program, critic_report, spec, refiner_budget
        )
    
    # ── Zone 2: 0.35 ≤ α < 0.45 → Hybrid
    if current_alpha >= 0.35:
        return await self._repair_hybrid(
            program, critic_report, spec, refiner_budget
        )
    
    # ── Zone 3: α < 0.35 → Symbolic-only
    return await self._symbolic_repair(program, critic_report, spec)


async def _repair_hybrid(
    self,
    program: Program,
    critic_report: CriticReport,
    spec: TaskSpec,
    refiner_budget: Budget,
) -> Optional[Program]:
    """
    Symbolic + Single-Stage-LLM laufen parallel; beste Verbesserung gewinnt.
    
    Begründung: Im α-Graubereich ist Vertrauen ins LLM unklar — daher 
    nicht hart entscheiden, sondern beide Strategien ausführen und den 
    Verifier entscheiden lassen.
    """
    # Beide parallel
    symbolic_task = asyncio.create_task(
        self._symbolic_repair(program, critic_report, spec)
    )
    llm_task = asyncio.create_task(
        self._llm_repair_single_stage(  # Single-Stage, nicht Two-Stage (Latenz)
            program, critic_report, spec, refiner_budget.fraction(0.5)
        )
    )
    
    symbolic_result, llm_result = await asyncio.gather(
        symbolic_task, llm_task, return_exceptions=True
    )
    
    # Beide gegen Verifier; bester gewinnt
    candidates = []
    for result in (symbolic_result, llm_result):
        if isinstance(result, Program):
            score = (await self.verifier.evaluate(result, spec)).score
            candidates.append((result, score))
    
    if not candidates:
        return None
    
    candidates.sort(key=lambda x: -x[1])  # höchster Score zuerst
    best_program, best_score = candidates[0]
    
    if best_score > critic_report.initial_score:
        return best_program
    return None
```

##### Begründung der Schwellen

- **0.45 als obere Grenze:** ein α von 0.45 entspricht etwa "LLM ist mittelmäßig zuverlässig" (z.B. α_e=0.7, α_p=0.65). Voll-LLM-Repair ist hier vertretbar, weil PUCT-Exploration parallel läuft.
- **0.35 als untere Grenze:** ein α von 0.35 entspricht "LLM ist klar problematisch" (z.B. α_e=0.6, α_p=0.6). Symbolic-only ist sicher.
- **Graubereich [0.35, 0.45]:** ~10 % Spannweite. In diesem Bereich kann ein einzelner schlechter LLM-Call das α aus α_p=0.7 auf α_p=0.5 schieben — also genau die Grenzregion, in der Hysterese kritisch wäre. Hybrid-Mode neutralisiert das, weil beide Strategien parallel laufen.

##### Latenz-Impact

- Zone 1 (Full-LLM): ~1.2-2.4 s (wie v1.3)
- Zone 2 (Hybrid): ~max(0.5, 0.6) = ~0.7 s parallel — *schneller* als reines LLM, weil Single-Stage und parallel
- Zone 3 (Symbolic): ~0.2-0.5 s

Hybrid-Mode ist also tendenziell *schneller* als Full-LLM bei vergleichbarer Erfolgsrate (durch Parallelausführung).

##### Hysterese-Schutz

Auch wenn die Drei-Zonen-Logik weniger oszillationsanfällig ist als das harte `if α < 0.4`, fügen wir eine zusätzliche Hysterese ein: ein einmal eingenommener Mode bleibt für mindestens 3 Repair-Aufrufe stabil, auch wenn α leicht über/unter eine Schwelle wandert.

```python
class RefinerModeController:
    def __init__(self, hysteresis_repairs: int = 3):
        self.hysteresis = hysteresis_repairs
        self._current_mode: Literal["full_llm", "hybrid", "symbolic"] | None = None
        self._calls_in_current_mode = 0
    
    def select_mode(self, alpha: float) -> Literal["full_llm", "hybrid", "symbolic"]:
        proposed = (
            "full_llm" if alpha >= 0.45 else
            "hybrid" if alpha >= 0.35 else
            "symbolic"
        )
        
        # Hysterese: aktueller Mode darf nicht zu schnell wechseln
        if self._current_mode is None:
            self._current_mode = proposed
            self._calls_in_current_mode = 1
            return proposed
        
        if (proposed != self._current_mode and 
            self._calls_in_current_mode < self.hysteresis):
            # Im alten Mode bleiben
            self._calls_in_current_mode += 1
            return self._current_mode
        
        # Mode-Wechsel erlauben
        self._current_mode = proposed
        self._calls_in_current_mode = 1
        return proposed
```

#### 6.5.3 CEGIS (unverändert ggü. v1.2/v1.3)

### 6.6 Eskalations-Logik (revidiert für Drei-Zonen-Mode)

```python
async def refine(
    self,
    program: Program,
    verifier_result: VerifierResult,
    spec: TaskSpec,
    budget: PartitionedBudget,
    current_alpha: float,
) -> Optional[SynthesisResult]:
    initial_score = verifier_result.score
    
    # ── STUFE 1: Local-Edit IMMER
    refined = await self._try_local_edit(program, verifier_result)
    if refined:
        new_score = (await verifier.evaluate(refined, spec)).score
        if new_score >= 0.95:
            return SynthesisResult(refined, ..., refinement_path=("local",))
        if new_score > initial_score:
            program = refined
            verifier_result = await verifier.evaluate(program, spec)
            initial_score = verifier_result.score
    
    # ── STUFE 2: Drei-Zonen-Mode-Dispatch
    if initial_score >= 0.3 and not budget.refiner.exhausted():
        mode = self.mode_controller.select_mode(current_alpha)
        
        log.info("refiner.mode_selected", 
                mode=mode, alpha=current_alpha,
                hysteresis_calls=self.mode_controller._calls_in_current_mode)
        
        refined = await self.repair_dispatch(
            program, verifier_result, spec, current_alpha, budget.refiner
        )
        
        if refined:
            new_score = (await verifier.evaluate(refined, spec)).score
            if new_score >= 0.95:
                return SynthesisResult(
                    refined, ...,
                    refinement_path=("local", f"repair_{mode}")
                )
            if new_score > initial_score:
                program = refined
                initial_score = new_score
    
    # ── STUFE 3: CEGIS nur wenn Score >= 0.5
    if initial_score >= 0.5 and not budget.cegis.exhausted():
        refined = await self._try_cegis(
            program, verifier_result, spec, budget.cegis
        )
        if refined:
            new_score = (await verifier.evaluate(refined, spec)).score
            if new_score >= 0.95:
                return SynthesisResult(
                    refined, ...,
                    refinement_path=("local", f"repair_{mode}", "cegis")
                )
    
    return None
```

### 6.7 Öffentliche API (unverändert)

---

## 7. Verifier-Erweiterungen

### 7.1 Verifier in Phase 1 (Kontext, unverändert)

### 7.2 Erweiterung in Phase 2 — graduierter Score (unverändert ggü. v1.2/v1.3)

### 7.3 Triviality-Penalty + Suspicion-Score

#### 7.3.1 Regelbasierte Triviality (unverändert)

#### 7.3.2 Suspicion-Score — mit High-Impact UND Structural-Abstraction (REVIDIERT F1)

ChatGPT-Review Runde 4: `objects()` produziert eine Zwischenrepräsentation, keinen finalen Output. Als High-Impact (3×-Multiplier) wäre `objects()` allein schon "fast so verdienstvoll wie ein 3-Token-Programm" — das ist falsch, weil `objects()` ohne nachgelagerte Transformation nichts ändert.

**Lösung: zwei separate Flags mit unterschiedlichen Multipliern.**

```python
@dataclass(frozen=True)
class DSLPrimitive:
    name: str
    signature: tuple[type, ...]
    impl: Callable
    cost: float
    learned: bool = False
    is_high_impact: bool = False              # 3× Multiplier
    is_structural_abstraction: bool = False   # 1.5× Multiplier (NEU F1)


# Whitelists im ARC-DSL-Registry (revidiert):

HIGH_IMPACT_PRIMITIVES = frozenset({
    "tile",          # produziert ganzen Output
    "flood_fill",    # transformiert ganze Region
    "mirror",        # globale Spiegelung
    "rotate",        # globale Rotation
    "transpose",     # Achsen-Tausch
    "compose_grid",  # Strukturelle Komposition produziert Output
    "scale",         # Größenänderung produziert Output
})

STRUCTURAL_ABSTRACTION_PRIMITIVES = frozenset({
    "objects",         # extrahiert Objekt-Liste, kein Output
    "filter_objects",  # filtert Objekt-Liste, kein Output
    "group_by_color",  # gruppiert, kein Output
    "find_pattern",    # sucht Muster, kein Output
    "extract_bbox",    # extrahiert Bounding-Box, kein direkter Output
})
```

##### Geänderte Komplexitäts-Berechnung

```python
def compute_syntactic_complexity(program: Program, dsl: DSL) -> float:
    """
    Wie komplex ist das synthetisierte Programm?
    
    - High-Impact-Primitive: 3× Multiplier
    - Structural-Abstraction: 1.5× Multiplier (NEU F1)
    - Reguläre Primitive: 1× (Basis)
    """
    if program.length == 0:
        return 0.0
    
    effective_length = 0.0
    for token in program.tokens:
        primitive = dsl.lookup_primitive(token)
        if primitive is None:
            effective_length += 1.0
            continue
        
        if primitive.is_high_impact:
            effective_length += 3.0
        elif primitive.is_structural_abstraction:
            effective_length += 1.5
        else:
            effective_length += 1.0
    
    length_factor = min(effective_length / 12.0, 1.0)
    depth_factor = min(program.composition_depth / 6.0, 1.0)
    
    return 0.6 * length_factor + 0.4 * depth_factor
```

##### Begründung der unterschiedlichen Multiplier

| Klasse | Multiplier | Begründung |
|---|---|---|
| Reguläre Primitive (`recolor`, `crop`) | 1× | Standard-Komplexität |
| Structural-Abstraction (`objects`, `filter_objects`) | 1.5× | Liefert nützliche Abstraktion, aber kein Output — leicht über Standard, weit unter direkt-transformativen |
| High-Impact (`tile`, `mirror`) | 3× | Direkter struktureller Effekt auf den Output |

Ein 1-Token `objects(g)` hat damit `effective_length = 1.5`, ein 1-Token `tile(g)` hat `effective_length = 3.0`. Der Unterschied schlägt sich in der Suspicion-Bewertung nieder:

| Programm | partial_score | effective_length | length_factor | suspicion |
|---|---|---|---|---|
| `tile(g)` allein | 0.85 | 3.0 | 0.25 | ~0.5 (knapp ok) |
| `objects(g)` allein | 0.85 | 1.5 | 0.125 | ~0.25 (verdächtig) |
| `recolor(g, 2, 5)` allein | 0.85 | 1.0 | 0.083 | ~0.17 (verdächtig) |

`objects()` allein bekommt jetzt einen Suspicion-Penalty, was ChatGPTs Hinweis adressiert: ein 1-Token `objects()` ohne nachgelagerte Verarbeitung ist tatsächlich verdächtig. `tile()` allein ist legitimer, weil es direkt einen Output produziert.

#### 7.3.3 Effekt auf Score (unverändert ggü. v1.2/v1.3)

#### 7.3.4 Test-Strategie (erweitert für F1)

50 Adversarial-Programme + 30 legitime Single-High-Impact-Programme + **20 neue Single-Structural-Abstraction-Programme**:

- 10 davon sind *legitim* nur wenn nachgelagerte Transformation folgt → wenn allein, soll suspicion < 0.5 sein
- 10 davon sind *immer suspect* als 1-Tokener → suspicion < 0.4

### 7.4 Verifier-API (unverändert)

---

## 8. Integration in Cognithor (unverändert)

---

## 9. Datenstrukturen und APIs (Erweiterungen für v1.4)

### 9.1 Neue/revidierte Typen in v1.4

```python
# ─── DSL Primitive mit zwei Flags (F1) ───────────────

@dataclass(frozen=True)
class DSLPrimitive:
    name: str
    signature: tuple[type, ...]
    impl: Callable
    cost: float
    learned: bool = False
    is_high_impact: bool = False
    is_structural_abstraction: bool = False    # NEU F1
    
    def __post_init__(self):
        # Mutually exclusive
        assert not (self.is_high_impact and self.is_structural_abstraction), \
            "A primitive cannot be both high_impact and structural_abstraction"

# ─── Refiner-Mode-Controller (F2) ────────────────────

class RefinerModeController:
    """Drei-Zonen-Mode-Selection mit Hysterese."""
    
    def __init__(self, hysteresis_repairs: int = 3):
        self.hysteresis = hysteresis_repairs
        self._current_mode: Literal["full_llm", "hybrid", "symbolic"] | None = None
        self._calls_in_current_mode = 0
    
    def select_mode(self, alpha: float) -> Literal["full_llm", "hybrid", "symbolic"]:
        ...

RefinerMode = Literal["full_llm", "hybrid", "symbolic"]
```

### 9.2 Engine-Top-Level-API (unverändert)

---

## 10. Datei-Layout im Repository

(Unverändert ggü. v1.3, plus folgende neue Dateien:)

```
cognithor/core/synthesis/refiner/
├── mode_controller.py           # NEU F2
├── hybrid_repair.py             # NEU F2
└── tests/
    ├── test_three_zone_mode.py  # NEU F2
    └── test_mode_hysteresis.py  # NEU F2

cognithor/core/synthesis/dsl/
├── classification.py            # erweitert F1 (zwei Flags)
└── tests/
    ├── test_high_impact.py      # erweitert F1
    └── test_structural_abstr.py # NEU F1

cognithor/core/synthesis/verifier/
├── suspicion_with_classes.py    # erweitert F1
└── tests/
    └── test_suspicion_classes.py # erweitert F1
```

---

## 11. Telemetrie und Observability (erweitert für v1.4)

(Aus v1.3, plus:)

| Event | NEUE Felder |
|---|---|
| `refiner.attempted` | **`mode_used (full_llm|hybrid|symbolic)`**, **`hysteresis_active (bool)`** |
| `refiner.hybrid_compete` | **NEU**: `task_id`, `symbolic_score`, `llm_score`, `winner` |
| `verifier.evaluated` | **`structural_abstraction_tokens_count`** zusätzlich |

Neue Prometheus-Metriken:

- `cognithor_synthesis_refiner_mode_total` (counter, label: `mode`)
- `cognithor_synthesis_refiner_hybrid_winner_total` (counter, label: `winner`)
- `cognithor_synthesis_refiner_mode_hysteresis_held_total` (counter)
- `cognithor_synthesis_structural_abstraction_token_total` (counter)

---

## 12. Test-Strategie und Benchmarks

### 12.1 Test-Pyramide (erweitert)

| Ebene | Anzahl Tests (Ziel v1.4) | v1.3 | Coverage |
|---|---|---|---|
| Unit | ~440 | 410 | 95 % |
| Integration | ~70 | 62 | 85 % |
| **Wechselwirkungs-Tests** | **~12** | — | **NEU F3** |
| End-to-End auf ARC | ~10 | 10 | volle Pipeline |
| Performance/Regression | ~25 | 25 | Latenz-Budget |
| Adversarial / Security | ~25 | 22 | Triviality, Suspicion (inkl. High-Impact + Structural), Sandbox |

### 12.2 Wechselwirkungs-Tests (NEU F3)

ChatGPT-Review Runde 4 identifizierte drei subtile Wechselwirkungs-Risiken zwischen den E1-E7 Mechanismen, die explizite Test-Abdeckung brauchen:

#### 12.2.1 E1 × E2 — Asymmetrie zwischen Search- und Repair-α

**Risiko:** Search nutzt LLM noch bei α=0.3 (innerhalb [0.25, 0.85]), Repair überspringt LLM erst ab α<0.35. Das ist beabsichtigt (siehe §4.4.4 begriffliche Klärung), aber muss als "Search-Vertrauen ≠ Repair-Vertrauen" dokumentiert und getestet sein.

**Tests:**
- `test_search_uses_llm_at_alpha_below_repair_threshold`: bei α=0.3 → MCTS ruft LLM-Prior auf, Refiner nutzt Symbolic-Mode
- `test_documentation_of_alpha_semantics`: Code-Kommentare in `mixer.py` und `refiner.py` referenzieren §4.4.4 explizit
- `test_telemetry_distinguishes_alpha_uses`: Logs zeigen separate `search_alpha` und `repair_alpha_threshold`-Felder

#### 12.2.2 E3 × E1 — LLM-Dominanz bei wenigen Demos

**Risiko:** Sample-Size-Dämpfung schwächt Symbolic bei n=1-2 Demos (effective_conf ≤ 0.5), gleichzeitig bleibt LLM-α-Floor bei 0.5. Das kann bei wenigen Demos zu LLM-Dominanz führen, obwohl gerade dann LLM-Halluzinationen besonders kritisch sind.

**Tests:**
- `test_few_demos_llm_not_overdominant`: 1 Demo + falsch-zuversichtlicher LLM → finale Policy nicht ausschließlich LLM-getrieben
- `test_symbolic_minimum_contribution_below_4_demos`: bei n=1 muss `(1-α) · π_symbolic` mindestens 15% der Top-1-Wahrscheinlichkeitsmasse liefern
- `test_few_demos_extra_caution_in_alpha`: bei n=1-2 wird α-Performance-Tracker schneller pessimistisch (window adaptiv reduziert auf 5)

**Möglicher Fix bei Testschlag:** Falls die Tests fehlschlagen, ergänze in `alpha_controller.py` einen zusätzlichen Faktor:
```python
def alpha_few_demos_dampening(n_demos: int) -> float:
    if n_demos >= 4: return 1.0
    return 0.7 + 0.075 * n_demos  # n=1: 0.775, n=2: 0.85, n=3: 0.925
```
Dieser Faktor wäre dann ein dritter Multiplier in der α-Formel.

**Status v1.4:** Tests werden geschrieben; Faktor wird *nur eingeführt, falls Tests fehlschlagen*. Vorerst ist die Annahme: Sample-Size-Dämpfung von Symbolic + α-Floor 0.5 sind balancetechnisch ausreichend.

#### 12.2.3 E6 × E7 — Reward-Hacking via "mächtige" Primitive

**Risiko:** High-Impact-Whitelist + 3×-Multiplier kann ein neuer Reward-Hacking-Vektor werden: ein LLM könnte lernen, dass `flood_fill(g, 0, 0, 7)` mit cleverem Argument einen hohen partial_score erzielt UND wegen High-Impact-Klassifikation suspicion-frei bleibt.

**Tests:**
- `test_high_impact_with_bad_args_still_suspect`: 50 adversariale Programme der Form `<HighImpactPrim>(g, <random_args>)` mit hohem partial_score → suspicion sollte ≤ 0.6 sein bei sinnlosen Argumenten
- `test_argument_quality_affects_suspicion`: derselbe High-Impact-Token mit besseren Argumenten bekommt höhere Suspicion (weil tatsächlich approximierende Lösung)
- `test_no_suspicion_neutralization`: High-Impact mildert Suspicion, neutralisiert sie aber nicht vollständig — bei sehr verdächtiger Diff-Struktur bleibt Penalty

**Möglicher Fix bei Testschlag:** Zusätzlicher Argument-Quality-Faktor in `compute_syntactic_complexity`:
```python
if primitive.is_high_impact:
    arg_quality = assess_argument_quality(token, context)  # 0..1
    multiplier = 1.0 + 2.0 * arg_quality  # 1.0 (schlecht) bis 3.0 (gut)
    effective_length += multiplier
```

**Status v1.4:** Tests werden geschrieben; Argument-Quality-Faktor *nur einführen, falls Tests Reward-Hacking ergeben*.

### 12.3 Spezifische Test-Kategorien (Ergänzungen v1.4)

**Drei-Zonen-Mode-Tests (F2):**
- `test_zone_full_llm_at_alpha_05`: α=0.5 → mode = full_llm
- `test_zone_hybrid_at_alpha_04`: α=0.4 → mode = hybrid
- `test_zone_symbolic_at_alpha_03`: α=0.3 → mode = symbolic
- `test_hysteresis_holds_at_boundary`: α-Sequenz [0.46, 0.44, 0.46, 0.44] → bleibt im ersten Mode mind. 3 Calls
- `test_hybrid_returns_better_of_both`: Hybrid-Mode wählt aus 2 Kandidaten den besseren via Verifier

**Structural-Abstraction-Tests (F1):**
- `test_objects_in_structural_abstraction_set`: `objects()` ist im Whitelist
- `test_tile_not_in_structural_abstraction`: `tile()` ist nur in High-Impact
- `test_mutual_exclusion_enforced`: Primitive kann nicht beide Flags haben
- `test_structural_abstraction_15x_multiplier`: `objects()` allein hat 1.5× syntactic_complexity
- `test_structural_abstraction_1tokener_suspect`: `objects(g)` allein mit hohem partial_score → suspicion < 0.5

### 12.4 Continuous Benchmark (unverändert)

### 12.5 Leak-Free-Held-Out-Set (unverändert)

---

## 13. Performance-Budget und Hardware

### 13.1 Hardware-Spezifikation (unverändert)

### 13.2 VRAM-Budget (unverändert)

### 13.3 Latenz-Budget pro Synthese-Phase (revidiert für Hybrid-Mode)

Standard-Modus, Wall-Clock 30 s:

| Phase | Anteil | Absolut | Δ v1.3 |
|---|---|---|---|
| Feature-Extraction + DSL-Selection + Cache-Lookup | 2 % | 0.6 s | unverändert |
| Phase-1-Enumerative-Pre-Filter | 5 % | 1.5 s | unverändert |
| MCTS-Suche | 70 % | 21 s | unverändert |
| Critic & Refiner | 18 % | 5.4 s | unverändert |
|   davon Full-LLM (α≥0.45) | ~10 % | ~3.0 s | reduziert |
|   davon Hybrid (0.35–0.45) | ~5 % | ~1.5 s | NEU F2 |
|   davon Symbolic (α<0.35) | ~3 % | ~0.9 s | unverändert |
| CEGIS (wenn aktiviert) | 5 % | 1.5 s | unverändert |

**Beobachtung:** Hybrid-Mode ist *schneller* als Full-LLM-Mode (parallele Symbolic + Single-Stage-LLM statt Two-Stage), während Erfolgsrate vermutlich ähnlich ist. Erwartete Wirkung: bei α-Werten im Graubereich nutzt der Refiner Hybrid statt Full-LLM, was ~50 % der Repair-Zeit spart.

### 13.4 Budget-Partitionierung (unverändert)

### 13.5 Throughput-Ziel (unverändert)

---

## 14. Sicherheit und Capability-Tokens (unverändert)

---

## 15. Risiken und Gegenmaßnahmen (erweitert)

| Risiko | Wahrscheinlichkeit | Impact | Gegenmaßnahme | Δ v1.3 |
|---|---|---|---|---|
| `objects()` als 1-Tokener fälschlich suspicion-frei | **NEU Mittel** | Mittel | **Structural-Abstraction-Klasse mit 1.5×-Multiplier** | F1 |
| Refiner-Mode oszilliert an α≈0.4 | **NEU Mittel** | Niedrig | **Drei-Zonen-Mode + Hysterese** | F2 |
| Reward-Hacking via High-Impact-Argumente | **NEU Mittel** | Mittel | **Wechselwirkungs-Tests; Argument-Quality-Faktor als Reserve** | F3 |
| LLM-Dominanz bei wenigen Demos | **NEU Niedrig** | Mittel | **Wechselwirkungs-Tests; few-demos-dampening als Reserve** | F3 |
| Search/Repair-α-Asymmetrie missverständlich | **NEU Niedrig** | Niedrig | **Doku-Klärung in §4.4.4** | F3 |

(Plus alle bisherigen Risiken aus v1.0/v1.1/v1.2/v1.3.)

---

## 16. Offene Fragen

(Aus v1.3, plus:)

18. **Multiplier-Werte 3× / 1.5× empirisch validieren** — sind die Klassen-Differenzierungen empirisch trennscharf?
19. **Drei-Zonen-Schwellen 0.35 / 0.45** — empirisch validieren ob ±0.05 sinnvoller ist
20. **Hybrid-Mode-Latenz** — bringt der parallele LLM-Call überhaupt was, oder reicht Symbolic-only?
21. **Hysterese-Window 3 Repair-Calls** — angemessen, oder zu kurz/lang?
22. **Argument-Quality-Faktor** — einführen falls Wechselwirkungs-Test E6×E7 fehlschlägt?
23. **Few-Demos-Dampening** — einführen falls Wechselwirkungs-Test E3×E1 fehlschlägt?
24. **`expected_depth_estimator`** — wann in Phase 3 einbauen? Welche Features als Input? (siehe §22)

---

## 17. Akzeptanzkriterien (revidiert in v1.4)

### 17.1 Funktional

(Aus v1.3, plus:)

- [ ] **DSL trennt High-Impact und Structural-Abstraction sauber** (F1)
- [ ] **`objects()` als 1-Tokener mit hohem score → suspicion ≤ 0.5** (F1)
- [ ] **Refiner-Mode-Selection deterministisch und hysterese-stabil** (F2)
- [ ] **Hybrid-Mode liefert messbar bessere Latenz als Full-LLM bei vergleichbarem Score im α-Graubereich** (F2)
- [ ] **Wechselwirkungs-Test-Suite (12 Tests) komplett grün** (F3)
- [ ] **Search/Repair-α-Asymmetrie in §4.4.4 dokumentiert** (F3)

### 17.2 Nicht-funktional (unverändert)

### 17.3 Integration (unverändert)

### 17.4 Benchmark (unverändert)

---

## 18. Validierungs-Checkliste (für externe Prüfung v1.4)

### 18.1 Architektur-Konsistenz

(Aus v1.3, plus:)

- [ ] **Drei-Zonen-Mode mathematisch sinnvoll: Schwellen ∈ Sub-Intervall von [0.25, 0.85]** (F2)
- [ ] **`is_high_impact` und `is_structural_abstraction` mutually exclusive (Assertion in `__post_init__`)** (F1)
- [ ] **Klassifizierung von `objects` als Structural-Abstraction nicht als High-Impact** (F1)

### 18.2 Algorithmische Korrektheit

(Aus v1.3, plus:)

- [ ] **Hybrid-Mode parallel-asyncio korrekt: keine Race-Conditions** (F2)
- [ ] **RefinerModeController-Hysterese-Logik kein Off-by-one** (F2)
- [ ] **3× und 1.5× Multiplier konsistent in allen Code-Pfaden** (F1)

### 18.3 Daten-Konsistenz

(Aus v1.3, plus:)

- [ ] **Mutual-Exclusion-Assertion bei DSLPrimitive funktioniert zur Konstruktionszeit** (F1)
- [ ] **RefinerMode-Literal serialisierbar für Telemetrie** (F2)

### 18.4 Sicherheit (unverändert)

### 18.5 Performance

(Aus v1.3, plus:)

- [ ] **Hybrid-Mode tatsächlich schneller als Full-LLM (P50-Vergleich)** (F2)

### 18.6 Vollständigkeit

- [ ] Alle 24 offenen Fragen explizit markiert (war 17 in v1.3)
- [ ] Adressierungstabelle für alle vier Review-Runden
- [ ] Wechselwirkungs-Test-Suite vollständig spezifiziert

### 18.7 Cognithor-spezifisch (unverändert)

### 18.8 Lücken / Improvement-Vorschläge

Beim externen Re-Review (Runde 5, falls nötig) explizit fragen:
- Sind die Multiplier 3× / 1.5× / 1× tatsächlich empirisch trennscharf — oder verschmilzt die Suspicion-Verteilung in der Praxis?
- Hybrid-Mode: parallele Verifier-Calls (zwei statt einer) — kann das den Verifier überlasten bei vielen gleichzeitigen Synthese-Anfragen?
- Hysterese-Window 3 Repair-Calls: gibt es Tasks mit zu wenigen Repair-Aufrufen, sodass die Hysterese-Logik nie greift?
- Sind die Wechselwirkungs-Tests E1×E2/E3×E1/E6×E7 vollständig — oder gibt es ein viertes/fünftes Wechselwirkungs-Risiko?

---

## 19. Referenz: Adressierung der Kritikpunkte (alle Runden)

### 19.1 v1.0-Kritik (Runde 1) — siehe v1.1 §19, alle umgesetzt

### 19.2 v1.1-Kritik (Runde 2) — siehe v1.2 §19.2, alle umgesetzt

### 19.3 v1.2-Kritik (Runde 3) — siehe v1.3 §19.3

### 19.4 v1.3-Kritik (Runde 4) — Adressierung in v1.4

| v1.3-Kritikpunkt | v1.4-Antwort | Spec-Stelle |
|---|---|---|
| `objects` als High-Impact zu stark privilegiert | `is_structural_abstraction` als separates Flag mit 1.5×-Multiplier | §7.3.2 |
| α-Schwelle 0.4 hart, anfällig für Hysterese | Drei-Zonen-Mode mit Graubereich [0.35, 0.45] und Hybrid-Repair | §6.5.2 |
| Wechselwirkung E1 × E2 als "Repair-Vertrauensschwelle" dokumentieren | Klärung in §4.4.4, Tests in §12.2.1 | §4.4.4, §12.2.1 |
| Wechselwirkung E3 × E1 — LLM-Dominanz bei wenigen Demos | Tests in §12.2.2; reservierter Fix (few-demos-dampening) wenn nötig | §12.2.2 |
| Wechselwirkung E6 × E7 — Reward-Hacking via mächtige Primitive | Tests in §12.2.3; reservierter Fix (argument-quality-factor) wenn nötig | §12.2.3 |

**Was nicht übernommen wurde / nur verschoben:**

| v1.3-Hinweis | Status | Begründung |
|---|---|---|
| `expected_depth_estimator` als Phase-3-Mechanismus | Verschoben auf Phase 3 | Sinnvoll, aber außerhalb v1.4-Scope. In §22 als Phase-3-Vorbereitung dokumentiert |
| Adaptive Hybrid-Window | Nicht übernommen in v1.4 | Default 3 Calls genügt für initialen Sprint; bei Bedarf später konfigurierbar |
| Zweiter Retry im Deep-Budget-Modus | Nicht übernommen in v1.4 | Komplexitätsverhältnis schlecht; Deep-Mode ist eigene Use-Case-Klasse |

### 19.5 ChatGPT's Gesamturteil v1.3 → v1.4

ChatGPT (Runde 4): *"v1.3 ist implementierungsreif. Keine neue große Revision nötig; nur die `objects`-Klassifikation und ein α-Graubereich für Repair wären sinnvolle Feinschliffe."*

v1.4 setzt beide angemerkten Feinschliffe um, plus die explizit erwähnten Wechselwirkungs-Tests. Die Spec gilt damit als konvergiert für Implementierungsbeginn.

---

## 20. Glossar (erweitert v1.4)

(Aus v1.3, plus:)

| Begriff | Definition |
|---|---|
| **Structural-Abstraction** *(NEU v1.4)* | DSL-Primitive, die Zwischenrepräsentationen erzeugen statt Outputs (z.B. `objects`, `filter_objects`) |
| **Drei-Zonen-Mode** *(NEU v1.4)* | Refiner-Modus-Selektion in drei α-Bereichen: Full-LLM / Hybrid / Symbolic |
| **Hybrid-Repair** *(NEU v1.4)* | Refiner-Modus, in dem Symbolic- und Single-Stage-LLM-Repair parallel laufen und der bessere Edit gewinnt |
| **RefinerModeController** *(NEU v1.4)* | Komponente, die mit Hysterese den Refiner-Mode wählt |
| **Search-α** *(NEU v1.4)* | α-Wert, der die LLM-Gewichtung in der MCTS-Suche steuert (in [0.25, 0.85]) |
| **Repair-α-Schwellen** *(NEU v1.4)* | Werte 0.35/0.45, die den Refiner-Mode wählen — getrennt von Search-α |
| **Wechselwirkungs-Test** *(NEU v1.4)* | Tests, die das korrekte Zusammenspiel zwischen separaten Mechanismen prüfen (E1×E2, E3×E1, E6×E7) |

---

## 21. Referenzen

(Aus v1.3, plus:)

17. **Sutton, R. S. & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*** (Standard-Referenz für Hysterese-Konzepte in adaptiven Controllern)
18. **Pomerleau, F. et al. (2024). *On the Importance of Robustness in LLM-Based Reasoning Pipelines*** (NEU für Drei-Zonen-Mode-Begründung)

Cognithor-interne Dokumente:
- alle aus v1.1/v1.2/v1.3
- `docs/synthesis/phase2_spec_v1.3.md` (vorherige Version)

---

## 22. Phase-3-Transition (erweitert v1.4)

### 22.1 Phase-3-Trigger (unverändert)

### 22.2 Phase-3-Vorbereitung in Phase 2 (unverändert)

### 22.3 Erwarteter Phase-3-Gewinn (unverändert)

### 22.4 Phase-3-Komponenten geplant für v2 (NEU)

ChatGPT-Review Runde 4 hat einen Phase-3-Mechanismus angeregt, der schon jetzt dokumentiert wird, damit Phase-2-Telemetrie die nötigen Daten sammelt:

#### 22.4.1 `expected_depth_estimator`

**Zweck:** Aus Phase-1-Features eine Heuristik für die erwartete Lösungstiefe ableiten. Damit kann der Fallback-Controller den Threshold `min_node_depth_mean` adaptiv setzen statt fest auf 2.0.

**Inputs:**
- `size_ratio` (Phase-1-Feature)
- `palette_change` (Phase-1-Feature)
- `object_count_change` (Phase-1-Feature)
- `is_tiling`, `is_subset`, `is_overlay` (Phase-1-Patterns)

**Output:** Geschätzte Lösungstiefe in DSL-Tokens (typisch 1-12).

**Trainings-Daten:** Phase-2-Telemetrie liefert für jede erfolgreiche Synthese das Tupel `(features, actual_solution_length)`. Nach 1000+ Beispielen kann ein einfaches Decision-Tree-Modell oder Linear-Regression-Modell gelernt werden.

**Integration in Phase 3:**
```python
class AdaptiveFallbackController(FallbackController):
    def __init__(self, *args, depth_estimator: ExpectedDepthEstimator, **kwargs):
        super().__init__(*args, **kwargs)
        self.depth_estimator = depth_estimator
    
    def should_fall_back(self, mcts_state: MCTSState, features: SpecFeatures):
        # Statt fester min_node_depth_mean=2.0:
        expected_depth = self.depth_estimator.predict(features)
        adaptive_threshold = max(2.0, expected_depth * 0.4)
        # ...
```

**Status v1.4:** Telemetrie-Datensammlung beginnt mit Phase-2-Implementierung. Modell-Training in Phase 3.

#### 22.4.2 Argument-Quality-Faktor und Few-Demos-Dampening

Beide sind in v1.4 als Reserve-Mechanismen dokumentiert (siehe §12.2.2 und §12.2.3). Falls Phase-2-Wechselwirkungs-Tests sie nicht erzwingen, werden sie in Phase 3 als feature-database-getriebene Erweiterungen aufgegriffen.

---

## Versionierungs-Hinweis

Dieses Dokument ist Spezifikations-Revision v1.4 — die finale Konvergenz vor Implementierungsbeginn.

Änderungs-Anatomie:
- v1.0 → v1.1: 16 Änderungen (C1–C16) nach Runde 1 — strukturelle Erweiterungen
- v1.1 → v1.2: 8 Änderungen (D1–D8) nach Runde 2 — Feinschliff
- v1.2 → v1.3: 7 Änderungen (E1–E7) nach Runde 3 — Wechselwirkungs-Korrekturen
- **v1.3 → v1.4: 3 Änderungen (F1–F3) nach Runde 4 — Konvergenz**

Die Anzahl der Revisions-Punkte hat von Runde zu Runde konsistent abgenommen (16 → 8 → 7 → 3). Dies signalisiert Konvergenz: weniger neue Probleme bei jeder Review-Runde.

**Status v1.4: Implementierungsreif.** ChatGPT-Review Runde 4 hat dies extern bestätigt; v1.4 setzt die zwei verbleibenden konkreten Feinschliffe um plus die Wechselwirkungs-Tests.

**Empfehlung:** Implementierung kann mit v1.4 beginnen. Eine optionale fünfte Review-Runde nach Sprint-1 (~3 Wochen Implementierung) kann die empirische Validierung der heuristischen Werte (Multiplier, Schwellen) beurteilen — diese Review wäre dann *daten-gestützt*, nicht mehr spec-getrieben.

**Ende der Spezifikation v1.4.**
