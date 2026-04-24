# External Quickstart Review — Checklist

Target: a Python-literate dev new to Cognithor completes `00` (Installation) + `01` (First Crew) in ≤ 30 minutes WITHOUT the author's help.

This checklist pairs with [`EXTERNAL_REVIEW_RESULTS.md`](EXTERNAL_REVIEW_RESULTS.md), where reviewers log their runs.

---

## Setup checklist

- [ ] Tester has Python 3.12+ and Ollama ≥ 0.4.0 installed (or is willing to install via `pip install cognithor[all]` + the Ollama one-liner).
- [ ] Tester has NOT seen the plan, spec, or internal design docs for Cognithor Crew / Quickstart Feature 2.
- [ ] Tester has a timer running (wall-clock from "clone" to "first successful run").

## Review run

- [ ] Clone repo, `cd cognithor`
- [ ] `pip install -e ".[dev]"` → should succeed in < 2 min
- [ ] Read `docs/quickstart/README.md` → understand the 8-page flow from the index alone
- [ ] Follow `00-installation.md` Option B → `cognithor --version` prints `0.93.0`
- [ ] Follow `01-first-crew.md` → scaffold + run the research example
- [ ] Optional: try a second page (02 / 03 / 04) without author input
- [ ] Report: was anything unclear? Did any step fail? Note time taken per step.

## What to record per run

- **OS:** e.g. Ubuntu 24.04 / Windows 11 / macOS 14
- **Install path:** Option A / B / C
- **Time in each step:** 00, 01, optional 02+
- **Unclear moments:** quote the exact heading/paragraph
- **Failures / errors:** full stack trace
- **Suggestions:** what would have unblocked you faster

## Verdict

Acceptance at end of run. One of:

- **`PASS`** — flow completed, no blockers, no follow-ups.
- **`PASS_WITH_FOLLOWUPS`** — completed, but noted N follow-ups. List them as GitHub issues tagged `quickstart-0.93.1`. Each must be non-blocking.
- **`FAIL`** — could not complete 00+01 in ≤ 30 min without help. **Blocks 0.93.0 release.**

---

**Verdict (check one):**

- [ ] PASS
- [ ] PASS_WITH_FOLLOWUPS
- [ ] FAIL

**Reviewer:** _________________
**Date:** _________________
**OS / Python version:** _________________
**Total time taken:** ___ min
**Follow-ups (if any):** _________________

---

_After finishing, append a new `### Run N` block to [`EXTERNAL_REVIEW_RESULTS.md`](EXTERNAL_REVIEW_RESULTS.md) with your findings._
