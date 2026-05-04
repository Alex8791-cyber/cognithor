"""Sprint-26 Domain-Expansion infrastructure.

The Domain layer sits **above** the existing primitive registry
(``cognithor.channels.program_synthesis.dsl.registry``) and groups
related primitives plus the metadata the synthesis pipeline needs to
route, score, and verify a synthesis request:

* a primitive **catalog** scoped to the domain (≈20-40 primitives)
* domain-specific **type tags** (``Table``, ``JsonPath``, ``Datetime``…)
* a **verifier** that knows how to execute synthesised programs
* a **property suite** for hypothesis-based generative testing
* a **few-shot bank** path used by the LLM prior
* an externally-tracked **benchmark name** for the public scorecard

Public API:

>>> from cognithor.channels.program_synthesis.domains import (
...     Domain, DomainMetadata, DomainRegistry, DOMAIN_REGISTRY,
... )

The canonical, process-local registry instance is ``DOMAIN_REGISTRY``.
Tests build fresh registries via ``DomainRegistry()``.
"""

from __future__ import annotations

from cognithor.channels.program_synthesis.domains._register_all import (
    SPRINT26_DOMAIN_NAMES,
    register_all_sprint26_domains,
)
from cognithor.channels.program_synthesis.domains.base import (
    Domain,
    DomainCapability,
    DomainMetadata,
)
from cognithor.channels.program_synthesis.domains.cost_tracker import (
    DomainCostRecord,
    DomainCostTracker,
)
from cognithor.channels.program_synthesis.domains.llm_prior import (
    DomainAwareLLMPrior,
    FewShotBank,
    FewShotExample,
)
from cognithor.channels.program_synthesis.domains.property_verifier import (
    PropertyResult,
    PropertyVerifier,
)
from cognithor.channels.program_synthesis.domains.registry import (
    DOMAIN_REGISTRY,
    DomainRegistry,
)
from cognithor.channels.program_synthesis.domains.scorecard import (
    Scorecard,
    ScorecardEntry,
)

__all__ = [
    "DOMAIN_REGISTRY",
    "SPRINT26_DOMAIN_NAMES",
    "Domain",
    "DomainAwareLLMPrior",
    "DomainCapability",
    "DomainCostRecord",
    "DomainCostTracker",
    "DomainMetadata",
    "DomainRegistry",
    "FewShotBank",
    "FewShotExample",
    "PropertyResult",
    "PropertyVerifier",
    "Scorecard",
    "ScorecardEntry",
    "register_all_sprint26_domains",
]
