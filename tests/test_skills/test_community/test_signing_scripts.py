"""End-to-end smoke for the four operator scripts in scripts/registry_signing/.

PACK-4 acceptance criterion §11: "All four signing scripts execute end-to-end
on a clean checkout."

The test:
  1. Runs ``generate_root_key.py``    → ``./root/`` with PEM + b64
  2. Runs ``generate_targets_key.py`` → ``./targets/`` with PEM + b64
  3. Runs ``sign_root.py``            → ``./root.json``
  4. Runs ``sign_payload.py``         → ``./signed_registry.json``
  5. Loads root.json + signed_registry.json through ``RegistryVerifier``
     using the freshly-minted Root pubkey, and asserts the chain holds
     end-to-end.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pytest

from cognithor.skills.community import _pinned_keys
from cognithor.skills.community.signing import RegistryVerifier

if TYPE_CHECKING:
    from pathlib import Path

SCRIPTS = "scripts/registry_signing"


def _run(*argv: str) -> None:
    proc = subprocess.run(
        [sys.executable, *argv],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise AssertionError(
            f"script exited {proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )


@pytest.mark.slow
def test_full_signing_chain_e2e(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # 1 + 2: mint both keypairs.
    root_dir = tmp_path / "root"
    targets_dir = tmp_path / "targets"
    _run(f"{SCRIPTS}/generate_root_key.py", "--out-dir", str(root_dir))
    _run(f"{SCRIPTS}/generate_targets_key.py", "--out-dir", str(targets_dir))

    root_pub_b64 = (root_dir / "root_public.b64").read_text(encoding="utf-8").strip()
    assert (root_dir / "root_private.pem").exists()
    assert (targets_dir / "targets_private.pem").exists()
    assert (targets_dir / "targets_public.b64").exists()

    # 3: sign root.json delegating to the Targets key.
    root_json_path = tmp_path / "root.json"
    _run(
        f"{SCRIPTS}/sign_root.py",
        "--root-key",
        str(root_dir / "root_private.pem"),
        "--targets-pubkey",
        str(targets_dir / "targets_public.b64"),
        "--version",
        "1",
        "--valid-days",
        "365",
        "--min-client-version",
        "0.97.0",
        "--out",
        str(root_json_path),
    )
    assert root_json_path.exists()

    # 4: build an unsigned registry.json then sign it with the Targets key.
    now = datetime.now(UTC)
    unsigned_registry = {
        "_type": "registry",
        "version": 1,
        "issued_at": now.isoformat(),
        "valid_until": (now + timedelta(days=14)).isoformat(),
        "skills": [{"name": "demo", "publisher": "alice"}],
    }
    unsigned_path = tmp_path / "unsigned_registry.json"
    unsigned_path.write_text(json.dumps(unsigned_registry), encoding="utf-8")
    signed_path = tmp_path / "signed_registry.json"
    _run(
        f"{SCRIPTS}/sign_payload.py",
        "--in",
        str(unsigned_path),
        "--key",
        str(targets_dir / "targets_private.pem"),
        "--out",
        str(signed_path),
    )
    assert signed_path.exists()

    # 5: verify the full chain with the Cognithor verifier, pinning the
    # freshly-minted Root pubkey via monkeypatch.
    monkeypatch.setattr(_pinned_keys, "ROOT_PUBLIC_KEY_B64", root_pub_b64)
    monkeypatch.setattr(_pinned_keys, "REQUIRE_SIGNED_REGISTRY", True)

    verifier = RegistryVerifier(state_path=tmp_path / "state.json")
    root_payload = verifier.verify_root(root_json_path.read_bytes(), now=now)
    assert root_payload.version == 1
    assert root_payload.body["targets"]["public_key"] == (
        (targets_dir / "targets_public.b64").read_text(encoding="utf-8").strip()
    )

    registry_payload = verifier.verify_targets_payload(
        signed_path.read_bytes(),
        expected_type="registry",
        channel_key="registry",
        now=now,
    )
    assert registry_payload.version == 1
    assert registry_payload.body["skills"][0]["name"] == "demo"
