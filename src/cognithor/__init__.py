"""Cognithor · Agent OS -- Local-first autonomous agent operating system."""

__version__ = "0.92.7"
__author__ = "Alexander Söllner"

# ── Centralized branding — single source of truth for banner + version ──

BANNER_ASCII = r"""
   ____  ___   ____ _   _ ___ _____ _   _  ___  ____
  / ___|| _ \ / ___| \ | |_ _|_   _| | | |/ _ \|  _ \
 | |   || | | | |  _|  \| || |  | | | |_| | | | | |_) |
 | |___|| |_| | |_| | |\  || |  | | |  _  | |_| |  _ <
  \____||___/ \____|_| \_|___| |_| |_| |_|\___/|_| \_\
""".strip("\n")

PRODUCT_NAME = "COGNITHOR"
PRODUCT_FULL = f"Cognithor · Agent OS v{__version__}"

# ── Public Crew-Layer surface (Spec §1.2) ──
# Re-export the crew subpackage so `import cognithor; cognithor.crew.Crew(...)`
# works, not just `from cognithor.crew import Crew`. This is a spec requirement,
# not a convenience — the §1.4 PKV example documents both import styles.
from cognithor import crew as crew
