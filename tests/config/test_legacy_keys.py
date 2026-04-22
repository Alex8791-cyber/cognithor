from __future__ import annotations

import logging

import pytest
from pydantic import ValidationError

from cognithor.config import CognithorConfig, load_config


class TestLegacyTopLevelKeysAreTolerated:
    """Legacy YAML keys must not crash `load_config` — issue #131.

    A user upgrading from an older Jarvis version can have a `config.yaml`
    with fields the current `CognithorConfig` no longer recognizes
    (e.g. `max_agents`, `memory_limit_mb`, `rag`). `extra="forbid"` on
    the model used to turn these into an unrecoverable launch crash.
    """

    def test_unknown_top_level_keys_are_stripped_with_warning(self, tmp_path, caplog):
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(
            "max_agents: 1\n"
            "max_concurrent: 1\n"
            "memory_limit_mb: 2048\n"
            "rag:\n"
            "  enabled: false\n"
            "language: de\n",
            encoding="utf-8",
        )

        with caplog.at_level(logging.WARNING, logger="cognithor.config"):
            cfg = load_config(cfg_file)

        assert cfg is not None
        assert cfg.language == "de"
        warnings = " ".join(r.getMessage() for r in caplog.records)
        for key in ("max_agents", "max_concurrent", "memory_limit_mb", "rag"):
            assert key in warnings, (
                f"Expected key '{key}' in deprecation warning, got: {warnings!r}"
            )

    def test_valid_config_without_unknown_keys_still_loads_clean(self, tmp_path, caplog):
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("language: en\nowner_name: Tester\n", encoding="utf-8")

        with caplog.at_level(logging.WARNING, logger="cognithor.config"):
            cfg = load_config(cfg_file)

        assert cfg.language == "en"
        assert cfg.owner_name == "Tester"
        unknown_warnings = [r for r in caplog.records if "Unbekannte Felder" in r.getMessage()]
        assert unknown_warnings == []

    def test_programmatic_construction_still_rejects_unknown_fields(self):
        """The safety net on the model itself is preserved — dev/test code
        paths that construct `CognithorConfig(**kwargs)` directly must keep
        catching typos, only the on-disk YAML read path is tolerant."""
        with pytest.raises(ValidationError):
            CognithorConfig(definitely_not_a_field=1)

    def test_real_errors_still_surface_unchanged(self, tmp_path):
        """When the YAML has a *real* error (wrong type on a known field),
        `load_config` must still raise — we only swallow extra_forbidden."""
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("language: 123\n", encoding="utf-8")
        with pytest.raises(ValidationError):
            load_config(cfg_file)
