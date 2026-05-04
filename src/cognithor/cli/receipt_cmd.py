"""CLI commands for ``cognithor receipt show`` / ``... verify``.

Operator-facing front door to the TRUST-1 audit run-receipt API +
the TRUST-5..10 trust bundle (#402, #403). Lets an operator dump a
receipt for a given session_id (optionally including the trust
bundle and an HMAC signature) and later verify a persisted bundle
without spinning up the full agent.

Kept intentionally thin — the real work lives in
:meth:`cognithor.audit.AuditLogger.run_receipt` and
:meth:`cognithor.audit.AuditLogger.verify_receipt_signature`.
"""

from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING, Any

from cognithor.audit import AuditLogger

if TYPE_CHECKING:
    from pathlib import Path


def cmd_show(
    *,
    session_id: str,
    log_dir: Path | None = None,
    include_trust: bool = False,
    signing_key: str | None = None,
    out: Path | None = None,
) -> int:
    """Print a TRUST-1 receipt to stdout (or write it to ``out``).

    Parameters
    ----------
    session_id:
        Session / run id to aggregate. Must be non-empty.
    log_dir:
        Directory holding ``audit_*.jsonl`` files. ``None`` → the
        logger's in-memory ring buffer only (useful for in-process
        introspection; usually you want to point at the on-disk
        audit log so the post-mortem covers restarted processes).
    include_trust:
        Fold the TRUST-5..10 ledger bundle into the receipt under a
        top-level ``"trust"`` key (#403).
    signing_key:
        HMAC-SHA-256 signing key. ``None`` leaves the ``signature``
        field empty.
    out:
        Optional path to write the bundle. When omitted, the bundle
        prints to stdout.

    Returns
    -------
    0 on success, 2 on bad arguments. Empty receipts (ghost session)
    still return 0 — the structured "no match" payload is the
    answer the operator asked for.
    """
    if not session_id:
        print("error: session_id is required", file=sys.stderr)
        return 2
    logger = AuditLogger(log_dir=log_dir)
    bundle = logger.run_receipt(
        session_id,
        signing_key=signing_key,
        include_trust=include_trust,
    )
    payload = json.dumps(bundle, indent=2, ensure_ascii=False)
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(payload, encoding="utf-8")
        print(f"wrote {out}")
    else:
        print(payload)
    return 0


def cmd_verify(*, bundle_path: Path, signing_key: str) -> int:
    """Verify the HMAC-SHA-256 signature on a persisted bundle.

    Returns
    -------
    0  if the signature is valid.
    1  if the bundle has no signature, or the signature does not match.
    2  on bad arguments / unreadable / non-JSON bundle.
    """
    if not signing_key:
        print("error: --key is required for verify", file=sys.stderr)
        return 2
    try:
        raw = bundle_path.read_text(encoding="utf-8")
    except (OSError, FileNotFoundError) as exc:
        print(f"error: cannot read {bundle_path}: {exc}", file=sys.stderr)
        return 2
    try:
        bundle: dict[str, Any] = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f"error: invalid JSON in {bundle_path}: {exc}", file=sys.stderr)
        return 2
    if not isinstance(bundle, dict):
        print(
            f"error: bundle at {bundle_path} is not a JSON object",
            file=sys.stderr,
        )
        return 2
    if not bundle.get("signature"):
        print("invalid: bundle has no signature", file=sys.stderr)
        return 1
    if AuditLogger.verify_receipt_signature(bundle, signing_key):
        print("ok: signature valid")
        return 0
    print("invalid: signature does not match the provided key", file=sys.stderr)
    return 1


__all__ = ["cmd_show", "cmd_verify"]
