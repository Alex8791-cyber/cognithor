# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Version constants for the PSE channel and its DSL.

The DSL version is part of every cache key (see spec §14.1) so that DSL
changes invalidate cached synthesis results automatically.

Semver:
    Major   primitive removed or signature changed (cache break).
    Minor   primitive added (cache compatible).
    Patch   cost / docs only.
"""

from __future__ import annotations

PSE_VERSION = "1.2.0-draft"
DSL_VERSION = "1.2.0"
