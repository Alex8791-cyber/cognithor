"""Tests fuer ErrorClusterer."""

import pytest
from jarvis.telemetry.error_clustering import ErrorClusterer, _levenshtein_ratio


class TestLevenshteinRatio:
    def test_identical(self):
        assert _levenshtein_ratio("hello", "hello") == 1.0

    def test_empty_both(self):
        assert _levenshtein_ratio("", "") == 1.0

    def test_empty_one(self):
        assert _levenshtein_ratio("", "hello") == 0.0

    def test_similar(self):
        ratio = _levenshtein_ratio("hello world", "hello wrld")
        assert ratio > 0.8

    def test_different(self):
        ratio = _levenshtein_ratio("abc", "xyz")
        assert ratio < 0.5


class TestErrorClusterer:
    def setup_method(self):
        self.clusterer = ErrorClusterer(similarity_threshold=0.6)

    def test_add_error(self):
        self.clusterer.add_error("TimeoutError", "timeout after 30s")
        assert self.clusterer.total_errors == 1

    def test_cluster_similar_errors(self):
        for i in range(5):
            self.clusterer.add_error("TimeoutError", f"timeout after {30+i}s")
        clusters = self.clusterer.get_clusters()
        assert len(clusters) == 1
        assert clusters[0]["count"] == 5

    def test_separate_different_types(self):
        self.clusterer.add_error("TimeoutError", "timeout")
        self.clusterer.add_error("ConnectionError", "connection refused")
        clusters = self.clusterer.get_clusters()
        assert len(clusters) == 2

    def test_top_errors(self):
        for _ in range(10):
            self.clusterer.add_error("TimeoutError", "timeout 30s")
        for _ in range(3):
            self.clusterer.add_error("ValueError", "invalid input")
        top = self.clusterer.get_top_errors(n=2)
        assert len(top) == 2
        assert top[0]["count"] == 10

    def test_severity(self):
        for _ in range(10):
            self.clusterer.add_error("TimeoutError", "timeout 30s")
        clusters = self.clusterer.get_clusters()
        assert clusters[0]["severity"] == "high"

    def test_clear(self):
        self.clusterer.add_error("E", "msg")
        self.clusterer.clear()
        assert self.clusterer.total_errors == 0

    def test_max_entries_limit(self):
        clusterer = ErrorClusterer(max_entries=5)
        for i in range(10):
            clusterer.add_error("E", f"msg {i}")
        assert clusterer.total_errors == 5

    def test_empty_clusters(self):
        assert self.clusterer.get_clusters() == []
