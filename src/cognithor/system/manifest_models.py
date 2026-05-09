"""Layer 3 — Pydantic schemas for the external Manifest.

Defines the in-memory shape of:
- `manifest/v2/tiers.yaml`
- `manifest/v2/models.yaml`
- `manifest/v2/pricing.yaml`
- `manifest/recalls/active.json`

All field names match the YAML keys so loading is a straight
`Manifest.model_validate(yaml_data)`.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

__all__ = [
    "BackendConfig",
    "Manifest",
    "Model",
    "PricingEntry",
    "PricingManifest",
    "Recall",
    "RecallList",
    "Tier",
]

QualityTier = Literal["S", "M", "L", "XL"]
SpeedTier = Literal["S", "M", "L"]


class RoleFitness(BaseModel):
    planner: float = 0.0
    executor: float = 0.0
    coder: float = 0.0
    formulate: float = 0.0
    embedding: float = 0.0
    fast_path_validator: float = 0.0


class Model(BaseModel):
    """Single model definition from `models.yaml`."""

    id: str
    display_name: str
    license: str
    backend_ids: dict[str, str | None] = Field(default_factory=dict)
    requires_capabilities: tuple[str, ...] = ()
    disk_gb: float = 0.0
    ram_gb_min: float = 0.0
    vram_gb_min: float = 0.0
    context_length: int = 0
    role_fitness: RoleFitness = Field(default_factory=RoleFitness)
    quality_tier: QualityTier = "M"
    speed_tier: SpeedTier = "M"
    deprecated_after_utc: str | None = None
    superseded_by: str | None = None
    quirks: tuple[str, ...] = ()


class ModelSet(BaseModel):
    """Per-tier model role assignments. Each value is a model `id`."""

    planner: str
    executor: str
    coder: str
    embedding: str
    formulate: str
    fast_path_validator: str


class BackendConfig(BaseModel):
    """Backend-specific knobs propagated into config.yaml on apply."""

    docker_image: str | None = None
    base_url: str | None = None
    requires_api_key: str | None = None
    gpu_memory_utilization: float | None = None
    enforce_eager: bool | None = None
    cpu_offload_gb: int | None = None
    max_model_len: int | None = None
    num_speculative_tokens: int | None = None
    enable_prefix_caching: bool | None = None


class PerformanceEstimates(BaseModel):
    planner_tok_s_p50: float = 0.0
    executor_tok_s_p50: float = 0.0
    formulate_tok_s_p50: float = 0.0
    first_token_ms_p50: int = 0


class Tier(BaseModel):
    """Single tier definition from `tiers.yaml`."""

    id: str
    display_name: str
    rationale_de: str = ""
    rationale_en: str = ""
    requires_capabilities: tuple[str, ...] = ()
    requires_cognithor: str = ">=0.99.0"
    backend: str
    backend_config: BackendConfig = Field(default_factory=BackendConfig)
    model_set: ModelSet
    estimated_setup_minutes: int = 5
    estimated_disk_gb: float = 0.0
    performance_estimates: PerformanceEstimates = Field(default_factory=PerformanceEstimates)


class Manifest(BaseModel):
    """Top-level Manifest combining tiers + models + pricing."""

    schema_version: int
    manifest_version: str
    expires_utc: str | None = None
    tiers: tuple[Tier, ...]
    models: dict[str, Model] = Field(default_factory=dict)


class PricingEntry(BaseModel):
    input_eur_per_mtok: float = 0.0
    output_eur_per_mtok: float = 0.0
    cache_read_eur_per_mtok: float = 0.0
    cache_write_eur_per_mtok: float = 0.0
    valid_until_utc: str | None = None


class PricingManifest(BaseModel):
    schema_version: int
    manifest_version: str
    expires_utc: str | None = None
    providers: dict[str, dict[str, PricingEntry]] = Field(default_factory=dict)
    local_inference: dict[str, Any] = Field(default_factory=dict)
    default_usage_profile: dict[str, Any] = Field(default_factory=dict)


class Recall(BaseModel):
    manifest_version: str
    reason: str
    recalled_at_utc: str
    severity: Literal["critical", "high", "medium", "low"]


class RecallList(BaseModel):
    schema_version: int
    comment: str = ""
    recalls: tuple[Recall, ...] = ()
