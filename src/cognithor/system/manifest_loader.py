"""Layer 3 — Manifest loader + verifier.

Responsibilities:
1. Load `manifest/v2/{tiers,models,pricing}.yaml` from one of:
   a) `~/.cognithor/manifest_cache/v2/` (online-refreshed cache)
   b) `<repo>/manifest/v2/` (embedded fallback shipped with the wheel)
2. Verify Ed25519 signature against MANIFEST_TARGETS_KEY (TUF-Light).
3. Cross-validate + check active recalls.
4. Return a fully-merged `Manifest` for L4 to consume.

Design decisions:
- Embedded fallback ALWAYS works offline (ships with wheel).
- Online refresh is best-effort, never blocks boot.
- Recall-list is checked at every load; recalled manifests hard-fail.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import yaml

from cognithor.system.manifest_models import (
    Manifest,
    Model,
    PricingManifest,
    RecallList,
    Tier,
)
from cognithor.utils.logging import get_logger

log = get_logger(__name__)

__all__ = [
    "DEFAULT_MANIFEST_BASE_URL",
    "ManifestLoader",
    "ManifestRecalledError",
    "ManifestRefreshError",
    "ManifestSource",
]

# Source-of-truth URL for online refresh. The embedded copy in `<repo>/manifest/v2/`
# is the offline fallback shipped with each cognithor wheel.
DEFAULT_MANIFEST_BASE_URL = (
    "https://raw.githubusercontent.com/Alex8791-cyber/cognithor/main/manifest"
)


class ManifestRefreshError(Exception):
    pass


class ManifestRecalledError(Exception):
    pass


@dataclass(frozen=True)
class ManifestSource:
    base_path: Path  # absolute path to manifest/v2/
    origin: str  # "embedded" | "cache" | "online"
    manifest_version: str  # from tiers.yaml
    expires_utc: str | None
    signature_verified: bool  # True iff Ed25519 sig matched


def _embedded_manifest_path() -> Path:
    """Return absolute path to manifest/v2/ shipped alongside the source tree.

    Cognithor source layout: `<repo>/src/cognithor/system/manifest_loader.py`,
    Manifest at `<repo>/manifest/v2/`. We walk up to the repo root.
    """
    here = Path(__file__).resolve()
    # .../src/cognithor/system/manifest_loader.py → .../<repo>
    repo_root = here.parent.parent.parent.parent
    return repo_root / "manifest"


def _user_cache_path() -> Path:
    return Path.home() / ".cognithor" / "manifest_cache"


class ManifestLoader:
    """Loads + verifies the Manifest. Stateful: caches the parsed result."""

    def __init__(
        self,
        *,
        embedded_root: Path | None = None,
        cache_root: Path | None = None,
        base_url: str = DEFAULT_MANIFEST_BASE_URL,
        max_cache_age_s: int = 30 * 86400,
    ) -> None:
        self._embedded_root = embedded_root or _embedded_manifest_path()
        self._cache_root = cache_root or _user_cache_path()
        self._base_url = base_url
        self._max_cache_age_s = max_cache_age_s
        self._loaded: Manifest | None = None
        self._loaded_pricing: PricingManifest | None = None
        self._loaded_source: ManifestSource | None = None

    def load(
        self,
        *,
        prefer_online: bool = False,
        force_refresh: bool = False,
    ) -> tuple[Manifest, ManifestSource]:
        """Load + verify the Manifest. Cached after first call."""
        if self._loaded and self._loaded_source and not force_refresh:
            return self._loaded, self._loaded_source

        source: ManifestSource | None = None

        # 1. Try cache (if fresh AND not forced refresh)
        if not force_refresh:
            try:
                source = self._try_cache()
            except Exception as exc:
                log.debug("manifest_cache_load_failed", error=str(exc))

        # 2. Try online (if requested or cache miss)
        if prefer_online or force_refresh or source is None:
            try:
                source = self._try_online()
            except ManifestRefreshError as exc:
                log.warning("manifest_online_refresh_failed", error=str(exc))

        # 3. Fallback to embedded
        if source is None:
            source = self._load_embedded()

        # Recall check — hard fail if active
        self._enforce_recalls(source)

        manifest = self._parse_manifest(source.base_path)
        self._loaded = manifest
        self._loaded_source = source
        return manifest, source

    def load_pricing(self) -> PricingManifest | None:
        """Load pricing.yaml — non-fatal if missing."""
        if self._loaded_pricing is not None:
            return self._loaded_pricing
        if self._loaded_source is None:
            self.load()
        assert self._loaded_source is not None
        pricing_path = self._loaded_source.base_path / "v2" / "pricing.yaml"
        if not pricing_path.exists():
            return None
        try:
            data = yaml.safe_load(pricing_path.read_text(encoding="utf-8"))
            self._loaded_pricing = PricingManifest.model_validate(data)
            return self._loaded_pricing
        except Exception as exc:
            log.warning("pricing_manifest_load_failed", error=str(exc))
            return None

    # ── Source resolvers ──────────────────────────────────────────────

    def _try_cache(self) -> ManifestSource | None:
        cache = self._cache_root
        tiers_path = cache / "v2" / "tiers.yaml"
        if not tiers_path.exists():
            return None
        age = self._file_age_s(tiers_path)
        if age > self._max_cache_age_s:
            log.info("manifest_cache_stale_age_s", age_s=age, max=self._max_cache_age_s)
            return None
        version, expires = self._extract_version(tiers_path)
        return ManifestSource(
            base_path=cache,
            origin="cache",
            manifest_version=version,
            expires_utc=expires,
            signature_verified=self._verify_signature(cache),
        )

    def _try_online(self) -> ManifestSource:
        """Best-effort online refresh — writes to cache_root and returns it."""
        import urllib.error
        import urllib.request

        files = ("v2/tiers.yaml", "v2/models.yaml", "v2/pricing.yaml", "recalls/active.json")
        cache = self._cache_root
        cache.mkdir(parents=True, exist_ok=True)

        for relpath in files:
            url = f"{self._base_url}/{relpath}"
            target = cache / relpath
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "cognithor-manifest/1"})
                with urllib.request.urlopen(req, timeout=8) as resp:
                    target.write_bytes(resp.read())
            except (urllib.error.URLError, OSError, ValueError) as exc:
                raise ManifestRefreshError(f"fetch {url}: {exc}") from exc

        version, expires = self._extract_version(cache / "v2" / "tiers.yaml")
        return ManifestSource(
            base_path=cache,
            origin="online",
            manifest_version=version,
            expires_utc=expires,
            signature_verified=self._verify_signature(cache),
        )

    def _load_embedded(self) -> ManifestSource:
        embedded = self._embedded_root
        version, expires = self._extract_version(embedded / "v2" / "tiers.yaml")
        return ManifestSource(
            base_path=embedded,
            origin="embedded",
            manifest_version=version,
            expires_utc=expires,
            # If a manifest.sig is present, verify it. If not, embedded is
            # trusted via the wheel's own signing (Phase-1 fallback).
            signature_verified=self._verify_signature(embedded),
        )

    # ── Parser ────────────────────────────────────────────────────────

    def _parse_manifest(self, base: Path) -> Manifest:
        tiers_data = yaml.safe_load((base / "v2" / "tiers.yaml").read_text(encoding="utf-8"))
        models_data = yaml.safe_load((base / "v2" / "models.yaml").read_text(encoding="utf-8"))

        # Build a dict[id, Model] so tier.model_set ids resolve cheaply
        models_dict: dict[str, Model] = {}
        for raw_model in models_data.get("models", []):
            m = Model.model_validate(raw_model)
            models_dict[m.id] = m

        tiers_tuple = tuple(Tier.model_validate(t) for t in tiers_data.get("tiers", []))

        return Manifest(
            schema_version=tiers_data["schema_version"],
            manifest_version=tiers_data["manifest_version"],
            expires_utc=tiers_data.get("expires_utc"),
            tiers=tiers_tuple,
            models=models_dict,
        )

    # ── Recall enforcement ────────────────────────────────────────────

    def _enforce_recalls(self, source: ManifestSource) -> None:
        recall_path = source.base_path / "recalls" / "active.json"
        if not recall_path.exists():
            return
        try:
            data = json.loads(recall_path.read_text(encoding="utf-8"))
            recalls = RecallList.model_validate(data)
        except Exception as exc:
            log.warning("recall_parse_failed", error=str(exc))
            return
        for r in recalls.recalls:
            if r.manifest_version == source.manifest_version:
                raise ManifestRecalledError(
                    f"Manifest {r.manifest_version} is recalled: {r.reason} "
                    f"(severity={r.severity}, at={r.recalled_at_utc}). "
                    f"Refresh manifest before continuing."
                )

    # ── Signature verification ────────────────────────────────────────

    def _verify_signature(self, base: Path) -> bool:
        """TUF-Light Ed25519 verify of the embedded/cache manifest.

        Payload format (must match scripts/sign_manifest.py):
          tiers.yaml + b"\\n--MANIFEST-DELIM--\\n" + models.yaml +
          b"\\n--PRICING-SHA256:" + sha256_hex(pricing.yaml).

        For Phase-1 we accept unsigned manifests with a WARN. Phase-2
        (after targets-key mint) makes signature mandatory.
        """
        import base64
        import hashlib

        sig_path = base / "v2" / "manifest.sig"
        if not sig_path.exists():
            log.debug("manifest_unsigned", base=str(base))
            return False
        try:
            from cognithor.system._pinned_keys import HARDWARE_MANIFEST_TARGETS_KEY
        except ImportError:
            log.debug("manifest_targets_key_module_missing")
            return False
        if HARDWARE_MANIFEST_TARGETS_KEY is None:
            log.debug("manifest_targets_key_not_pinned_yet")
            return False
        if not HARDWARE_MANIFEST_TARGETS_KEY.startswith("ed25519:"):
            log.warning("manifest_targets_key_invalid_format")
            return False

        try:
            from cryptography.hazmat.primitives.asymmetric.ed25519 import (
                Ed25519PublicKey,
            )
        except ImportError:
            log.warning("cryptography_module_unavailable")
            return False

        try:
            pub_b64 = HARDWARE_MANIFEST_TARGETS_KEY.split(":", 1)[1]
            pub = Ed25519PublicKey.from_public_bytes(base64.b64decode(pub_b64))
            tiers = (base / "v2" / "tiers.yaml").read_bytes()
            models = (base / "v2" / "models.yaml").read_bytes()
            pricing_hash = (
                hashlib.sha256((base / "v2" / "pricing.yaml").read_bytes())
                .hexdigest()
                .encode("ascii")
            )
            payload = (
                tiers + b"\n--MANIFEST-DELIM--\n" + models + b"\n--PRICING-SHA256:" + pricing_hash
            )
            sig = base64.b64decode(sig_path.read_text(encoding="utf-8").strip())
            pub.verify(sig, payload)
            return True
        except Exception as exc:
            log.warning("manifest_verify_error", error=str(exc))
            return False

    # ── Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _extract_version(path: Path) -> tuple[str, str | None]:
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
            return data.get("manifest_version", "unknown"), data.get("expires_utc")
        except Exception:
            return "unknown", None

    @staticmethod
    def _file_age_s(path: Path) -> int:
        import time

        try:
            return int(time.time() - path.stat().st_mtime)
        except OSError:
            return 10**9
