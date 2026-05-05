"""Pack interface — the stable contract between packs and Core.

Every pack provides a ``pack_manifest.json`` validated against ``PackManifest``
and a ``pack.py`` module that exposes a top-level ``Pack`` class extending
``AgentPack``. The loader validates the manifest, imports the module,
instantiates ``Pack(manifest)``, and calls ``register(context)``.

Deliberately narrow:

- Packs receive only a ``PackContext`` facade, never the raw Gateway.
- Packs declare their contributions (tools, lead sources) in the manifest
  AND in their ``register()`` body — the manifest fields are informational
  for the catalog and loader, the real wiring happens in ``register()``.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+(?:[-+][0-9A-Za-z\.-]+)?$")
_NS_PACK_RE = re.compile(r"^[a-z][a-z0-9-]{0,63}$")
_SHA256_RE = re.compile(r"^[a-f0-9]{64}$")
_VALID_RISK_LEVELS = frozenset({"green", "yellow", "orange", "red"})


class Publisher(BaseModel):
    """Publisher metadata. First-party packs use ``id="cognithor-official"``.

    Multi-publisher ready: the Q4 2026 community marketplace will register
    independent publishers without any schema change.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    display_name: str
    website: str | None = None
    contact_email: str | None = None
    payout_provider: str | None = None  # "lemonsqueezy" | "stripe-connect" (Phase 2)


class RevenueShare(BaseModel):
    """Revenue split between creator and platform. Default: 70/30."""

    model_config = ConfigDict(extra="forbid")

    creator: int = Field(default=70, ge=0, le=100)
    platform: int = Field(default=30, ge=0, le=100)

    @model_validator(mode="after")
    def _sum_to_100(self) -> RevenueShare:
        if self.creator + self.platform != 100:
            raise ValueError("creator + platform must sum to 100")
        return self


class PricingTier(BaseModel):
    """One tier (indie / commercial / ...) used for price anchoring on the site."""

    model_config = ConfigDict(extra="forbid")

    list_price: int = Field(ge=0)
    launch_price: int = Field(ge=0)
    post_launch_price: int = Field(ge=0)
    launch_cap: int = Field(ge=1)
    currency: str = "USD"


class PackManifest(BaseModel):
    """Validated pack manifest stored as ``pack_manifest.json`` at pack root."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    namespace: str
    pack_id: str
    version: str
    display_name: str
    description: str
    license: str
    min_cognithor_version: str
    max_cognithor_version: str | None = None
    entrypoint: str = "pack.py"
    eula_sha256: str
    publisher: Publisher
    revenue_share: RevenueShare = Field(default_factory=RevenueShare)

    # Declarative contributions (informational; real wiring is in register()).
    lead_sources: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)

    # Risk classification for the tools this pack exposes. Mapping of
    # tool_name -> risk_level ("green" | "yellow" | "orange" | "red").
    # Consumed by the PackLoader to populate the Gatekeeper's tool registry so
    # pack tools get correct risk classification without requiring packs to
    # register each tool via MCP. Tools not listed here fall through to the
    # Gatekeeper's default ORANGE (unknown tools are treated as risky).
    tool_risks: dict[str, str] = Field(default_factory=dict)

    # Commerce.
    checkout_url: str | None = None
    commercial_checkout_url: str | None = None
    pricing: dict[str, PricingTier] = Field(default_factory=dict)

    @field_validator("namespace")
    @classmethod
    def _validate_namespace(cls, v: str) -> str:
        if "/" in v or not _NS_PACK_RE.match(v):
            raise ValueError(
                f"namespace must match {_NS_PACK_RE.pattern!r} (lowercase, no slashes)"
            )
        return v

    @field_validator("pack_id")
    @classmethod
    def _validate_pack_id(cls, v: str) -> str:
        if "/" in v or not _NS_PACK_RE.match(v):
            raise ValueError(f"pack_id must match {_NS_PACK_RE.pattern!r} (lowercase, no slashes)")
        return v

    @field_validator("version")
    @classmethod
    def _validate_version(cls, v: str) -> str:
        if not _SEMVER_RE.match(v):
            raise ValueError(f"version must be semver X.Y.Z, got {v!r}")
        return v

    @field_validator("eula_sha256")
    @classmethod
    def _validate_eula_hash(cls, v: str) -> str:
        if not _SHA256_RE.match(v):
            raise ValueError("eula_sha256 must be 64 lowercase hex chars")
        return v

    @field_validator("entrypoint")
    @classmethod
    def _validate_entrypoint(cls, v: str) -> str:
        # PASS-4 SEC-HIGH: ``entrypoint`` was free-form ``str`` and the
        # loader resolved it as ``pack_dir / manifest.entrypoint``. A
        # manifest claiming ``entrypoint: "../../core/planner.py"`` would
        # then ``exec_module`` core code instead of the pack's own. Lock
        # to a plain filename ending in ``.py`` so the resolved path is
        # always a single child of the pack directory.
        if "/" in v or "\\" in v or ".." in v:
            raise ValueError(
                "entrypoint must be a plain filename inside the pack directory "
                f"(no path separators or '..'), got {v!r}"
            )
        if not v.endswith(".py"):
            raise ValueError(f"entrypoint must end with .py, got {v!r}")
        return v

    @field_validator("tool_risks")
    @classmethod
    def _validate_tool_risks(cls, v: dict[str, str]) -> dict[str, str]:
        for tool, level in v.items():
            if level not in _VALID_RISK_LEVELS:
                raise ValueError(
                    f"tool_risks[{tool!r}] = {level!r} must be one of {sorted(_VALID_RISK_LEVELS)}"
                )
        return v

    @model_validator(mode="after")
    def _pricing_required_for_paid(self) -> PackManifest:
        if self.license == "proprietary" and not self.pricing:
            raise ValueError("pricing is required for proprietary-licensed packs")
        return self

    @model_validator(mode="after")
    def _tool_risks_subset_of_tools(self) -> PackManifest:
        if self.tool_risks and self.tools:
            declared = set(self.tools)
            risked = set(self.tool_risks)
            extra = risked - declared
            if extra:
                raise ValueError(
                    f"tool_risks keys {sorted(extra)} not in declared tools={self.tools}"
                )
        return self

    @property
    def qualified_id(self) -> str:
        """Globally unique id: ``namespace/pack_id``."""
        return f"{self.namespace}/{self.pack_id}"


class PackContext(BaseModel):
    """Facade passed to ``AgentPack.register()``.

    A narrow, stable subset of the Gateway so Gateway internals can be
    refactored without breaking installed packs.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    gateway: Any = None
    config: Any = None
    mcp_client: Any = None
    leads: Any = None  # cognithor.leads.LeadService


class AgentPack(ABC):
    """Abstract base class all packs inherit from.

    Concrete subclasses live in each pack's ``pack.py`` as a top-level class
    named ``Pack`` (case-sensitive; the loader looks for this exact name).
    """

    def __init__(self, manifest: PackManifest) -> None:
        self.manifest = manifest

    @abstractmethod
    def register(self, context: PackContext) -> None:
        """Wire the pack into a running Cognithor instance.

        Called once at startup after the loader has validated the manifest.
        Typical implementations register ``LeadSource`` instances via
        ``context.leads.register_source(...)``, MCP tools via
        ``context.mcp_client.register_tool(...)``, or REST routes via
        ``context.gateway._api.add_route(...)``.

        Must be idempotent: the loader may call it again on reload.
        """

    def unregister(self, context: PackContext) -> None:
        """Cleanup on unload. Default: no-op.

        Override in subclasses that hold resources (background tasks,
        database connections, subprocess handles).
        """
