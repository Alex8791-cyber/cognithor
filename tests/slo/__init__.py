"""SLO p95/p99 latency-regression suite — Sprint 3.1.

Uses pytest-benchmark to capture per-call latency on hot paths and
compares against a stored baseline. CI fails on > 15 % median or
> 25 % p99 regression.

Hot paths covered:
  * AuditLogger append (file-write critical path)
  * MemoryManager.search (read path everyone hits)
  * Gatekeeper.classify_risk (every tool call goes through this)
  * VlmRouter.select_profile_explained (every video call)
  * Crew.kickoff (entry point for batch flows)

Run::

    pytest tests/slo/ --benchmark-only \\
        --benchmark-storage=./benchmarks \\
        --benchmark-compare=last \\
        --benchmark-compare-fail=median:15%,mean:15%
"""
