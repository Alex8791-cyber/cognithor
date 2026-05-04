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


def cmd_export_all(
    *,
    log_dir: Path,
    out_dir: Path,
    include_trust: bool = False,
    signing_key: str | None = None,
) -> int:
    """Export one signed receipt JSON per session_id under ``out_dir``.

    Walks the same audit_*.jsonl files :func:`cmd_list` enumerates,
    then emits one ``<session_id>.json`` per session. Sessions
    without an explicit ``session_id`` bucket under ``_unscoped.json``.
    Files are written with deterministic JSON formatting so a later
    diff can spot post-export tampering.

    Parameters
    ----------
    log_dir:
        Directory holding ``audit_*.jsonl``. Required.
    out_dir:
        Target directory for receipt files. Created if missing.
        Existing files in the directory are overwritten without
        prompt — this is a bulk-export, not an incremental sync.
    include_trust:
        Fold the TRUST-5..10 ledger bundle into each receipt.
    signing_key:
        Optional HMAC-SHA-256 signing key applied to every emitted
        receipt.

    Returns
    -------
    0 on success, 2 on bad arguments. Empty audit log writes a
    ``manifest.json`` listing zero sessions and returns 0.
    """
    if not log_dir or not log_dir.exists():
        print("error: --log-dir must point to an existing directory", file=sys.stderr)
        return 2

    sessions: set[str] = set()
    for jsonl in sorted(log_dir.glob("audit_*.jsonl")):
        try:
            for raw in jsonl.read_text(encoding="utf-8").splitlines():
                line = raw.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(data, dict):
                    sessions.add(str(data.get("session_id", "")))
        except OSError:
            continue

    out_dir.mkdir(parents=True, exist_ok=True)
    logger = AuditLogger(log_dir=log_dir)
    manifest: list[dict[str, Any]] = []
    for sid in sorted(sessions):
        bundle = logger.run_receipt(
            sid,
            signing_key=signing_key,
            include_trust=include_trust,
        )
        filename = (sid or "_unscoped").replace("/", "_").replace("\\", "_")
        out_path = out_dir / f"{filename}.json"
        payload = json.dumps(bundle, indent=2, ensure_ascii=False, sort_keys=True)
        out_path.write_text(payload, encoding="utf-8")
        manifest.append(
            {
                "session_id": sid,
                "file": out_path.name,
                "entry_count": int(bundle.get("entry_count", 0)),
                "signed": bool(bundle.get("signature")),
                "include_trust": include_trust,
            }
        )

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {"sessions": manifest, "count": len(manifest)},
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(f"exported {len(manifest)} session(s) to {out_dir}")
    return 0


def cmd_list(
    *,
    log_dir: Path | None = None,
    limit: int = 50,
) -> int:
    """List every distinct ``session_id`` present in the audit log.

    Walks ``audit_*.jsonl`` files under ``log_dir`` and emits one line
    per session: ``<session_id> <entry_count> <first_seen>``. Sorted
    most-recent-first by the first-seen timestamp.

    Sessions without an explicit ``session_id`` (boot-time, scheduler,
    GC) are bucketed under ``""`` and surface as ``(unscoped)`` so the
    operator can spot un-tagged activity.

    Parameters
    ----------
    log_dir:
        Directory holding ``audit_*.jsonl`` files. ``None`` → no file
        scan (returns just the in-memory ring buffer of a freshly
        constructed AuditLogger, which is empty in CLI invocations).
    limit:
        Max number of sessions to print. Newest-first; older sessions
        get truncated. Must be >= 1.

    Returns
    -------
    0 on success, 2 on bad arguments.
    """
    if limit < 1:
        print("error: --limit must be >= 1", file=sys.stderr)
        return 2

    sessions: dict[str, dict[str, Any]] = {}
    if log_dir is not None and log_dir.exists():
        for jsonl in sorted(log_dir.glob("audit_*.jsonl")):
            try:
                for raw in jsonl.read_text(encoding="utf-8").splitlines():
                    line = raw.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(data, dict):
                        continue
                    sid = str(data.get("session_id", ""))
                    timestamp = str(data.get("timestamp", ""))
                    bucket = sessions.setdefault(
                        sid,
                        {"count": 0, "first_seen": timestamp, "last_seen": timestamp},
                    )
                    bucket["count"] = int(bucket.get("count", 0)) + 1
                    if timestamp and (not bucket["first_seen"] or timestamp < bucket["first_seen"]):
                        bucket["first_seen"] = timestamp
                    if timestamp and (not bucket["last_seen"] or timestamp > bucket["last_seen"]):
                        bucket["last_seen"] = timestamp
            except OSError:
                continue

    if not sessions:
        print("(no audit entries found)")
        return 0

    ordered = sorted(
        sessions.items(),
        key=lambda kv: str(kv[1].get("last_seen", "")),
        reverse=True,
    )[:limit]

    print(f"{'session_id':<40} {'count':>7}  last_seen")
    for sid, meta in ordered:
        display = sid or "(unscoped)"
        print(f"{display:<40} {int(meta.get('count', 0)):>7}  {meta.get('last_seen', '')}")
    return 0


def cmd_diff(*, a_path: Path, b_path: Path) -> int:
    """Diff two TRUST-1 receipt bundles.

    Surfaces the operationally-meaningful differences between two
    receipts: per-section entry-count delta, total-cost delta,
    escalation-count delta, fingerprint-divergence, new permission
    scopes, and migration-chain head movement. Designed for "we ran
    the same workflow twice — what changed?" post-mortems.

    Output is human-readable, not JSON: one section per kind of
    delta, blank when no changes. Returns:

    * 0 — receipts loaded successfully (regardless of whether they
      differ; an unchanged-pair still returns 0).
    * 2 — bad arguments / unreadable / non-JSON / non-object bundle.
    """
    bundles: list[dict[str, Any]] = []
    for label, path in (("a", a_path), ("b", b_path)):
        try:
            raw = path.read_text(encoding="utf-8")
        except (OSError, FileNotFoundError) as exc:
            print(f"error: cannot read {label}={path}: {exc}", file=sys.stderr)
            return 2
        try:
            bundle = json.loads(raw)
        except json.JSONDecodeError as exc:
            print(f"error: invalid JSON in {label}={path}: {exc}", file=sys.stderr)
            return 2
        if not isinstance(bundle, dict):
            print(
                f"error: bundle at {label}={path} is not a JSON object",
                file=sys.stderr,
            )
            return 2
        bundles.append(bundle)
    a_bundle, b_bundle = bundles[0], bundles[1]

    a_sid = a_bundle.get("session_id", "?")
    b_sid = b_bundle.get("session_id", "?")
    print(f"diff: a={a_sid!r} → b={b_sid!r}")

    # ── Audit-aggregate deltas ─────────────────────────────────────
    a_count = int(a_bundle.get("entry_count", 0))
    b_count = int(b_bundle.get("entry_count", 0))
    if a_count != b_count:
        print(f"  entry_count: {a_count} → {b_count}  (delta {b_count - a_count:+d})")

    a_agg = _safe_dict(a_bundle.get("aggregate"))
    b_agg = _safe_dict(b_bundle.get("aggregate"))
    for key in ("success_count", "failure_count", "pii_count"):
        av = int(a_agg.get(key, 0))
        bv = int(b_agg.get(key, 0))
        if av != bv:
            print(f"  aggregate.{key}: {av} → {bv}  (delta {bv - av:+d})")

    # ── Trust-bundle deltas (only if BOTH receipts have one) ───────
    a_has_trust = "trust" in a_bundle
    b_has_trust = "trust" in b_bundle
    if not a_has_trust and not b_has_trust:
        return 0
    if a_has_trust != b_has_trust:
        print(
            f"  trust block: {'present' if a_has_trust else 'absent'} → "
            f"{'present' if b_has_trust else 'absent'}"
        )
        return 0
    a_trust = _safe_dict(a_bundle.get("trust"))
    b_trust = _safe_dict(b_bundle.get("trust"))

    # Cost delta (USD micro on the summary).
    a_cost = _safe_dict(_safe_dict(a_trust.get("cost")).get("summary"))
    b_cost = _safe_dict(_safe_dict(b_trust.get("cost")).get("summary"))
    a_micro = int(a_cost.get("total_cost_usd_micro", 0))
    b_micro = int(b_cost.get("total_cost_usd_micro", 0))
    if a_micro != b_micro:
        delta = b_micro - a_micro
        print(
            f"  trust.cost: {a_micro / 1_000_000:.6f} → {b_micro / 1_000_000:.6f} USD "
            f"(delta {delta / 1_000_000:+.6f})"
        )

    # Escalation event-count delta.
    a_esc = _safe_dict(_safe_dict(a_trust.get("escalations")).get("summary"))
    b_esc = _safe_dict(_safe_dict(b_trust.get("escalations")).get("summary"))
    a_ev = int(a_esc.get("event_count", 0))
    b_ev = int(b_esc.get("event_count", 0))
    if a_ev != b_ev:
        print(f"  trust.escalations.event_count: {a_ev} → {b_ev}  (delta {b_ev - a_ev:+d})")

    # Fingerprint divergence.
    a_fps = _safe_list(_safe_dict(a_trust.get("fingerprints")).get("all"))
    b_fps = _safe_list(_safe_dict(b_trust.get("fingerprints")).get("all"))
    a_hashes = {str(fp.get("content_hash", "")) for fp in a_fps if isinstance(fp, dict)}
    b_hashes = {str(fp.get("content_hash", "")) for fp in b_fps if isinstance(fp, dict)}
    new_in_b = sorted(b_hashes - a_hashes)
    gone_from_a = sorted(a_hashes - b_hashes)
    if new_in_b or gone_from_a:
        print(f"  trust.fingerprints: +{len(new_in_b)} -{len(gone_from_a)}")
        for h in new_in_b[:3]:
            print(f"    + {h[:12]}…")
        for h in gone_from_a[:3]:
            print(f"    - {h[:12]}…")

    # Permission-scope diff.
    a_scopes = {
        f"{s.get('axis')}:{s.get('identity')}"
        for s in _safe_list(a_trust.get("permission_scopes"))
        if isinstance(s, dict)
    }
    b_scopes = {
        f"{s.get('axis')}:{s.get('identity')}"
        for s in _safe_list(b_trust.get("permission_scopes"))
        if isinstance(s, dict)
    }
    if a_scopes != b_scopes:
        added = sorted(b_scopes - a_scopes)
        removed = sorted(a_scopes - b_scopes)
        print(f"  trust.permission_scopes: +{len(added)} -{len(removed)}")
        for s in added:
            print(f"    + {s}")
        for s in removed:
            print(f"    - {s}")

    # Migration-head movement per domain.
    a_heads = _safe_dict(_safe_dict(a_trust.get("migrations")).get("head_version"))
    b_heads = _safe_dict(_safe_dict(b_trust.get("migrations")).get("head_version"))
    domains = sorted(set(a_heads) | set(b_heads))
    for domain in domains:
        a_head = a_heads.get(domain)
        b_head = b_heads.get(domain)
        if a_head != b_head:
            print(f"  trust.migrations.{domain}: {a_head!r} → {b_head!r}")
    return 0


def _safe_dict(obj: object) -> dict[str, Any]:
    return obj if isinstance(obj, dict) else {}


def _safe_list(obj: object) -> list[Any]:
    return obj if isinstance(obj, list) else []


__all__ = [
    "cmd_diff",
    "cmd_export_all",
    "cmd_list",
    "cmd_show",
    "cmd_verify",
]
