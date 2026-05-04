"""Cross-domain synthesis demo cases (Sprint-26.4).

10 demo tasks combining JSON → Datetime → SQL bridges. Used by the
Public Scorecard as the "cross-domain capability" line. Each case
is a fixture with input examples, the expected program shape, and
the bridges it exercises.
"""

from __future__ import annotations

from cognithor_bench.cross_domain.cases import (
    CrossDomainCase,
    load_cross_domain_cases,
)

__all__ = ["CrossDomainCase", "load_cross_domain_cases"]
