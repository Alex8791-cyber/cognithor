# ARC-AGI-3 Live Validation Runbook

**Hardware:** RTX 5090 (32 GB VRAM), Linux + CUDA 12+ recommended.
**Model:** `Qwen/Qwen3.6-27B-FP8` (no `Instruct` variant exists upstream — base + FP8 are the canonical Qwen3.6-27B repos).

This runbook drives the first end-to-end Sprint-11 validation: three Cognithor agents (Random / DSL / LLM) playing 3-5 ARC-AGI-3 games, with frozen scorecards committed back as `.ci/arc_agi3_scorecards/*.json` regression baselines.

## Step 1 — vLLM server

```bash
# Install vLLM (OpenAI-compatible server)
pip install vllm>=0.6.3

# Launch the FP8 model on the RTX 5090. Reserve 92 % of VRAM for
# weights + KV-cache; reduce to 0.85 if you also need the GPU for
# something else simultaneously.
vllm serve Qwen/Qwen3.6-27B-FP8 \
    --port 8000 \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.92 \
    --dtype auto
```

Verify in another shell:

```bash
curl http://localhost:8000/v1/models
# expected: list with Qwen/Qwen3.6-27B-FP8

curl -X POST http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "Qwen/Qwen3.6-27B-FP8",
      "messages": [{"role": "user", "content": "Reply with the single word OK."}],
      "max_tokens": 8
    }'
# expected: a JSON response with content "OK" (or similar)
```

If the model has no chat template baked into its tokenizer (base models sometimes don't), add `--chat-template <path>` pointing to a Qwen2-style chat template. Most modern Qwen base models ship the template; check the HF model card.

## Step 2 — ARC-AGI-3 harness setup

```bash
git clone https://github.com/arcprize/ARC-AGI-3-Agents.git
cd ARC-AGI-3-Agents
uv sync

# .env
cp .env.example .env
echo "ARC_API_KEY=<your_key_from_three.arcprize.org>" >> .env

# Install Cognithor as an editable dep
uv pip install -e /path/to/cognithor
```

## Step 3 — Cognithor agent registration

Drop this into `agents/templates/cognithor_agents.py`:

```python
"""Cognithor PSE agents (Sprint-11) registered for the ARC-AGI-3 harness."""
from cognithor.channels.program_synthesis.arc_agi3 import (
    LLMReasoningAgent,
    RandomActionAgent,
    Sprint10DSLAgent,
    build_vllm_choice_fn,
    cognithor_agent_factory,
)
from cognithor.core.vllm_backend import VLLMBackend


# Lower-bound baseline.
CognithorRandom = cognithor_agent_factory(delegate=RandomActionAgent(seed=42))

# Sprint-10 DSL heuristic — deterministic, no GPU needed.
CognithorDSL = cognithor_agent_factory(delegate=Sprint10DSLAgent())

# LLM-driven, requires the running vLLM from Step 1.
_backend = VLLMBackend(base_url="http://localhost:8000/v1")
_choice_fn = build_vllm_choice_fn(
    backend=_backend,
    model_name="Qwen/Qwen3.6-27B-FP8",
    temperature=0.3,
)
CognithorLLM = cognithor_agent_factory(
    delegate=LLMReasoningAgent(choice_fn=_choice_fn),
)
```

Then add to `agents/__init__.py`'s `AVAILABLE_AGENTS`:

```python
from agents.templates.cognithor_agents import (
    CognithorRandom, CognithorDSL, CognithorLLM,
)

AVAILABLE_AGENTS = {
    # ... existing entries ...
    "cognithor_random": CognithorRandom,
    "cognithor_dsl": CognithorDSL,
    "cognithor_llm": CognithorLLM,
}
```

## Step 4 — Execute the validation runs

Pick 3-5 representative games from the ARC-AGI-3 catalogue. The locksmith / ls20 family is a good starter set (small grids, clear win conditions).

```bash
# Random baseline (the lower bound)
uv run main.py --agent=cognithor_random --game=ls20      | tee /tmp/random_ls20.log
uv run main.py --agent=cognithor_random --game=locksmith | tee /tmp/random_locksmith.log

# Sprint-10 DSL heuristic
uv run main.py --agent=cognithor_dsl --game=ls20      | tee /tmp/dsl_ls20.log
uv run main.py --agent=cognithor_dsl --game=locksmith | tee /tmp/dsl_locksmith.log

# LLM-reasoning agent (vLLM/Qwen3.6-27B-FP8 — slower per action)
uv run main.py --agent=cognithor_llm --game=ls20      | tee /tmp/llm_ls20.log
uv run main.py --agent=cognithor_llm --game=locksmith | tee /tmp/llm_locksmith.log
```

Each run prints a scorecard URL + dumps the scorecard JSON to stdout.

## Step 5 — Freeze baselines

Capture the scorecards into Cognithor's `.ci/`:

```bash
mkdir -p /path/to/cognithor/.ci/arc_agi3_scorecards
# Save each agent's combined scorecard. The harness writes them
# under recordings/ — pick them up:
cp recordings/*cognithor_random*.scorecard.json \
   /path/to/cognithor/.ci/arc_agi3_scorecards/random.json
cp recordings/*cognithor_dsl*.scorecard.json \
   /path/to/cognithor/.ci/arc_agi3_scorecards/dsl.json
cp recordings/*cognithor_llm*.scorecard.json \
   /path/to/cognithor/.ci/arc_agi3_scorecards/llm.json
```

## Step 6 — Aggregate + report

```bash
cd /path/to/cognithor
python -c "
from cognithor.channels.program_synthesis.arc_agi3 import (
    parse_scorecard, summarise,
)
import json, pathlib
for f in sorted(pathlib.Path('.ci/arc_agi3_scorecards').glob('*.json')):
    results = parse_scorecard(json.loads(f.read_text()))
    s = summarise(results)
    print(f'{f.stem:>10}: n={s.n_games}, won={s.n_won}, '
          f'win_rate={s.win_rate:.0%}, '
          f'mean_progress={s.mean_progress_ratio:.0%}, '
          f'total_actions={s.total_actions}')
"
```

Expected shape — Random ≈ 0 % win, DSL marginal, LLM hopefully 1-2 wins on simple games. Anything below Random is a regression; the LLM-vs-DSL gap is the real signal.

## Step 7 — Commit + open a Sprint-12 PR

```bash
cd /path/to/cognithor
git checkout -b feat/sprint12-arc-agi3-live-baselines
git add .ci/arc_agi3_scorecards/
git add docs/superpowers/reports/sprint12_live_validation.md  # write a short report
git commit -m "feat(pse): Sprint-12 — first live ARC-AGI-3 scorecards on RTX 5090"
git push -u origin feat/sprint12-arc-agi3-live-baselines
gh pr create --title "feat(pse): Sprint-12 — live ARC-AGI-3 baselines (Random/DSL/LLM)" --body "..."
```

## Failure modes + remediation

- **vLLM crashes on launch with OOM**: drop `--gpu-memory-utilization` to 0.85, or `--max-model-len` to 8192.
- **Agent throws `ImportError: ARC-AGI-3-Agents harness is not installed`**: you're running outside the harness venv. `uv run` from inside the cloned harness directory.
- **LLM falls back to DSL on every frame** (telemetry: `LLMActionDecoder fallback`): inspect the LLM response — usually invalid JSON. Try lowering temperature to 0.1 or adjusting the system prompt in `arc_agi3/llm_agent.py`.
- **All three agents score 0 on every game**: the games may need richer state-tracking than Sprint-11 ships. That's Sprint-12+ DSL-program-synthesis territory.

## Hardware constraint summary

- 27B BF16: ~54 GB → does not fit on 5090
- 27B FP8 (this runbook): ~27 GB → fits with KV-cache headroom on 32 GB VRAM
- 27B AWQ-Q4: ~14 GB → fits with lots of room (use this if VRAM is contested)
