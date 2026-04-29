# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""ARC-DSL: typed grid-transformation primitives.

Phase 1 ships ~50 base primitives + 5 higher-order primitives + 12
predicate constructors. See spec §7 for the full catalog.
"""

from __future__ import annotations

from cognithor.channels.program_synthesis.dsl.signatures import Signature

__all__ = ["Signature"]
