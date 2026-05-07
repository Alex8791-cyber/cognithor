"""LLM-/agent-adversarial test harness — Sprint 1.2.

Runs a curated corpus of prompt-injection / jailbreak / tool-hijack
attacks against the Gatekeeper risk classifier and Planner-stub, and
records pass/fail per attack with the matched rule_id.

Goal: regression gate. The corpus only ever grows; once an attack is
in, it must keep being caught. New attacks classify the system's state
on the day they're added — the test harness records baseline as
*expected failure tolerated* (with a TTL) so a grace period for the
fix is built in.
"""
