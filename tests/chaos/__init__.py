"""Chaos engineering harness — Sprint 2.1.

The PGE-Trinity loop has many ways to fail; this suite injects them
deliberately and asserts that recovery is graceful AND the audit chain
remains intact afterwards.

Fault primitives live in :mod:`tests.chaos.faults`. Each chaos test
imports a primitive, applies it inside a context manager, runs the
target operation, and verifies recovery + audit-chain integrity.
"""
