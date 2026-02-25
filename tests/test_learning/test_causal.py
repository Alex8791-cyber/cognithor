"""Tests fuer CausalAnalyzer."""

import pytest

from jarvis.learning.causal import CausalAnalyzer


class TestCausalAnalyzer:
    def setup_method(self):
        self.analyzer = CausalAnalyzer()  # in-memory

    def teardown_method(self):
        self.analyzer.close()

    def test_record_sequence(self):
        self.analyzer.record_sequence("s1", ["read_file", "write_file"], 0.9)
        assert self.analyzer.get_total_sequences() == 1

    def test_record_empty_sequence_ignored(self):
        self.analyzer.record_sequence("s1", [], 0.5)
        assert self.analyzer.get_total_sequences() == 0

    def test_get_sequence_scores(self):
        # Record same pattern multiple times
        for i in range(5):
            self.analyzer.record_sequence(
                f"s{i}", ["read_file", "exec_command", "write_file"], 0.8,
            )
        scores = self.analyzer.get_sequence_scores(min_occurrences=3)
        assert len(scores) > 0
        # Should find 2er subsequences like (read_file, exec_command)
        subseqs = [s.subsequence for s in scores]
        assert ("read_file", "exec_command") in subseqs

    def test_sequence_scores_min_occurrences(self):
        self.analyzer.record_sequence("s1", ["a", "b"], 0.8)
        self.analyzer.record_sequence("s2", ["a", "b"], 0.9)
        # Only 2 occurrences, min is 3
        scores = self.analyzer.get_sequence_scores(min_occurrences=3)
        assert len(scores) == 0

    def test_sequence_scores_sorted_by_score(self):
        for i in range(5):
            self.analyzer.record_sequence(f"good{i}", ["a", "b"], 0.9)
        for i in range(5):
            self.analyzer.record_sequence(f"bad{i}", ["c", "d"], 0.2)

        scores = self.analyzer.get_sequence_scores(min_occurrences=3)
        assert len(scores) >= 2
        # Higher score first
        assert scores[0].avg_score >= scores[-1].avg_score

    def test_suggest_tools(self):
        # Create clear pattern: read_file -> exec_command -> write_file
        for i in range(10):
            self.analyzer.record_sequence(
                f"s{i}", ["read_file", "exec_command", "write_file"], 0.9,
            )

        suggestions = self.analyzer.suggest_tools(["read_file", "exec_command"])
        assert len(suggestions) > 0
        assert "write_file" in suggestions

    def test_suggest_tools_empty_sequence(self):
        assert self.analyzer.suggest_tools([]) == []

    def test_suggest_tools_no_data(self):
        suggestions = self.analyzer.suggest_tools(["read_file"])
        assert suggestions == []

    def test_suggest_tools_fallback_to_last_tool(self):
        # Patterns that start differently but end with exec_command -> write_file
        for i in range(10):
            self.analyzer.record_sequence(
                f"s{i}", ["search_memory", "exec_command", "write_file"], 0.8,
            )

        # Current sequence doesn't match exactly, but last tool matches
        suggestions = self.analyzer.suggest_tools(["exec_command"])
        assert len(suggestions) > 0

    def test_confidence_calculation(self):
        for i in range(10):
            self.analyzer.record_sequence(
                f"s{i}", ["a", "b", "c"], 0.8,
            )

        scores = self.analyzer.get_sequence_scores(min_occurrences=3)
        for s in scores:
            assert 0 < s.confidence <= 1.0
            assert s.occurrence_count >= 3

    def test_three_tool_subsequences(self):
        for i in range(5):
            self.analyzer.record_sequence(
                f"s{i}", ["a", "b", "c", "d"], 0.7,
            )

        scores = self.analyzer.get_sequence_scores(min_occurrences=3, max_subseq_len=3)
        subseqs = [s.subsequence for s in scores]
        # Should include 3er subsequences
        assert ("a", "b", "c") in subseqs
        assert ("b", "c", "d") in subseqs
