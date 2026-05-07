"""AI output regression suite — Sprint 2.4.

Pinned golden prompts, judged by an LLM-as-judge with structured
rubric. Goal: detect quality drift when prompts, models, or the
personality engine change.

Runs nightly (needs API key — Anthropic Claude is the default judge).
PR-blocking only on high-quality-drop detections (judge confidence
high, rubric mismatch on must-have keywords).
"""
