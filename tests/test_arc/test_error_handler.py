"""Tests for jarvis.arc.error_handler."""

import pytest
import numpy as np


class TestSafeFrameExtract:
    def test_none_returns_fallback(self):
        from jarvis.arc.error_handler import safe_frame_extract

        result = safe_frame_extract(None)
        assert result.shape == (64, 64)
        assert np.all(result == 0)

    def test_valid_frame_attribute(self):
        from jarvis.arc.error_handler import safe_frame_extract

        obs = type("O", (), {"frame": [[[0] * 64 for _ in range(64)]]})()
        result = safe_frame_extract(obs)
        assert result.shape == (64, 64)

    def test_frame_3d_squeeze(self):
        """Real SDK format: (1, 64, 64) int8 → should squeeze to (64, 64)."""
        from jarvis.arc.error_handler import safe_frame_extract

        frame_data = np.ones((1, 64, 64), dtype=np.int8) * 5
        obs = type("O", (), {"frame": frame_data})()
        result = safe_frame_extract(obs)
        assert result.shape == (64, 64)
        assert result[0, 0] == 5

    def test_unknown_format_returns_fallback(self):
        from jarvis.arc.error_handler import safe_frame_extract

        obs = type("O", (), {"unknown_field": 42})()
        result = safe_frame_extract(obs)
        assert result.shape == (64, 64)

    def test_flat_array_reshaped(self):
        from jarvis.arc.error_handler import safe_frame_extract

        flat = np.ones(64 * 64, dtype=np.int8)
        obs = type("O", (), {"frame": flat})()
        result = safe_frame_extract(obs)
        assert result.shape == (64, 64)

    def test_2d_grid_passthrough(self):
        from jarvis.arc.error_handler import safe_frame_extract

        grid = np.zeros((64, 64), dtype=np.int8)
        obs = type("O", (), {"frame": grid})()
        result = safe_frame_extract(obs)
        assert result.shape == (64, 64)


class TestRetryOnError:
    def test_succeeds_on_first_try(self):
        from jarvis.arc.error_handler import retry_on_error

        @retry_on_error(max_retries=2, delay_seconds=0)
        def ok():
            return 42

        assert ok() == 42

    def test_retries_on_failure(self):
        from jarvis.arc.error_handler import retry_on_error

        call_count = 0

        @retry_on_error(max_retries=2, delay_seconds=0, exceptions=(ValueError,))
        def flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("fail")
            return "ok"

        assert flaky() == "ok"
        assert call_count == 3

    def test_raises_after_max_retries(self):
        from jarvis.arc.error_handler import retry_on_error

        @retry_on_error(max_retries=1, delay_seconds=0, exceptions=(ValueError,))
        def always_fails():
            raise ValueError("permanent")

        with pytest.raises(ValueError, match="permanent"):
            always_fails()


class TestGameRunGuard:
    def test_suppresses_exceptions(self):
        from jarvis.arc.error_handler import GameRunGuard

        class FakeArcade:
            def make(self, game_id):
                return type(
                    "Env",
                    (),
                    {
                        "reset": lambda s: None,
                        "step": lambda s, a: None,
                    },
                )()

            def get_scorecard(self):
                return None

        with GameRunGuard(FakeArcade(), "test") as guard:
            raise RuntimeError("boom")
        assert len(guard.errors) == 1
        assert "boom" in guard.errors[0]["error"]

    def test_env_creation_failure(self):
        from jarvis.arc.error_handler import GameRunGuard, EnvironmentConnectionError

        class BadArcade:
            def make(self, game_id):
                return None

            def get_scorecard(self):
                return None

        with pytest.raises(EnvironmentConnectionError):
            with GameRunGuard(BadArcade(), "bad_game"):
                pass

    def test_successful_run(self):
        from jarvis.arc.error_handler import GameRunGuard

        class FakeArcade:
            def make(self, game_id):
                return type("Env", (), {})()

            def get_scorecard(self):
                return type("SC", (), {"score": 0.5})()

        with GameRunGuard(FakeArcade(), "test") as guard:
            pass  # No error
        assert len(guard.errors) == 0


class TestExceptions:
    def test_exception_hierarchy(self):
        from jarvis.arc.error_handler import (
            ArcAgentError,
            EnvironmentConnectionError,
            FrameExtractionError,
        )

        assert issubclass(FrameExtractionError, ArcAgentError)
        assert issubclass(EnvironmentConnectionError, ArcAgentError)
        assert issubclass(ArcAgentError, Exception)
