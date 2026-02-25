"""
Jarvis · Shared Test-Fixtures.

Alle Tests nutzen ein temporäres Verzeichnis statt ~/.jarvis/.
So sind Tests isoliert und reproduzierbar.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from jarvis.config import JarvisConfig, ensure_directory_structure

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def tmp_jarvis_home(tmp_path: Path) -> Path:
    """Temporäres Jarvis-Home-Verzeichnis."""
    return tmp_path / ".jarvis"


@pytest.fixture
def config(tmp_jarvis_home: Path) -> JarvisConfig:
    """JarvisConfig mit temporärem Home-Verzeichnis."""
    return JarvisConfig(jarvis_home=tmp_jarvis_home)


@pytest.fixture
def initialized_config(config: JarvisConfig) -> JarvisConfig:
    """JarvisConfig mit erstellter Verzeichnisstruktur."""
    ensure_directory_structure(config)
    return config
