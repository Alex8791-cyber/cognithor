"""Cross-repo guard: the Flutter frontend's `kFrontendVersion` constant and
its `pubspec.yaml` version MUST stay in sync with the Python backend's
`__version__` (major.minor).

Without this, the 0.91 → 0.92 bump forgot both Flutter spots and every
user on the Windows installer hit a "Version Mismatch" overlay that
refused to let them use the app (reported on issue #131 by
@PCAssistSoftware). The overlay can only be cleared by re-installing,
which the user cannot do — the next installer carries the same stale
constant. This test exists so the CI breaks loudly if a release bump
ever forgets Flutter again.
"""

from __future__ import annotations

import re
from pathlib import Path

import cognithor

REPO_ROOT = Path(__file__).resolve().parents[1]
FLUTTER_APP = REPO_ROOT / "flutter_app"
CONNECTION_PROVIDER = FLUTTER_APP / "lib" / "providers" / "connection_provider.dart"
PUBSPEC = FLUTTER_APP / "pubspec.yaml"


def _major_minor(v: str) -> str:
    return ".".join(v.split(".")[:2])


class TestFlutterVersionInSyncWithBackend:
    def test_kfrontendversion_matches_backend_major_minor(self):
        src = CONNECTION_PROVIDER.read_text(encoding="utf-8")
        match = re.search(r"kFrontendVersion\s*=\s*'([^']+)'", src) or re.search(
            r'kFrontendVersion\s*=\s*"([^"]+)"', src
        )
        assert match, (
            f"Could not find kFrontendVersion in {CONNECTION_PROVIDER}. "
            f"The release checklist / this cross-check needs updating."
        )
        flutter_version = match.group(1)
        assert _major_minor(flutter_version) == _major_minor(cognithor.__version__), (
            f"Flutter kFrontendVersion={flutter_version!r} is out of sync with "
            f"cognithor.__version__={cognithor.__version__!r}. The 'Version "
            f"Mismatch' overlay will block every user. Bump "
            f"{CONNECTION_PROVIDER.relative_to(REPO_ROOT)} and "
            f"{PUBSPEC.relative_to(REPO_ROOT)} before release."
        )

    def test_pubspec_version_matches_backend_major_minor(self):
        text = PUBSPEC.read_text(encoding="utf-8")
        match = re.search(r"^version:\s*([0-9]+\.[0-9]+\.[0-9]+)", text, re.MULTILINE)
        assert match, f"Could not find 'version:' in {PUBSPEC}."
        pubspec_version = match.group(1)
        assert _major_minor(pubspec_version) == _major_minor(cognithor.__version__), (
            f"flutter_app/pubspec.yaml version={pubspec_version!r} is out of sync "
            f"with cognithor.__version__={cognithor.__version__!r}."
        )
