"""Tests for ``DomainAwareLLMPrior`` + ``FewShotBank``."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from cognithor.channels.program_synthesis.domains.base import (
    DomainCapability,
    DomainMetadata,
)
from cognithor.channels.program_synthesis.domains.llm_prior import (
    DomainAwareLLMPrior,
    FewShotBank,
    FewShotExample,
)


def _meta(name: str = "sql") -> DomainMetadata:
    return DomainMetadata(
        name=name,
        display_name=name.upper(),
        description=f"Synthesis for {name}.",
        capabilities=frozenset({DomainCapability.SYNTHESISE}),
    )


class TestFewShotExample:
    def test_from_dict_canonical(self) -> None:
        ex = FewShotExample.from_dict(
            {
                "input_examples": [{"input": "x", "output": "X"}],
                "target_program": "uppercase",
                "rationale": "trivial mapping",
            }
        )
        assert ex.target_program == "uppercase"
        assert ex.rationale == "trivial mapping"
        assert ex.input_examples[0]["input"] == "x"

    def test_from_dict_aliases(self) -> None:
        ex = FewShotExample.from_dict(
            {
                "examples": [{"input": "a", "output": "b"}],
                "program": "x",
            }
        )
        assert ex.target_program == "x"
        assert ex.input_examples[0]["output"] == "b"

    def test_from_dict_rejects_non_list_examples(self) -> None:
        with pytest.raises(ValueError, match="input_examples"):
            FewShotExample.from_dict({"input_examples": "not a list", "target_program": "x"})

    def test_from_dict_rejects_non_str_program(self) -> None:
        with pytest.raises(ValueError, match="target_program"):
            FewShotExample.from_dict({"input_examples": [], "target_program": 42})


class TestFewShotBank:
    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        bank = FewShotBank(tmp_path / "nope.jsonl")
        assert bank.load() == ()

    def test_loads_jsonl(self, tmp_path: Path) -> None:
        path = tmp_path / "examples.jsonl"
        path.write_text(
            json.dumps({"input_examples": [{"i": 1}], "target_program": "id"})
            + "\n"
            + json.dumps({"input_examples": [{"i": 2}], "target_program": "id2"})
            + "\n",
            encoding="utf-8",
        )
        bank = FewShotBank(path)
        out = bank.load()
        assert len(out) == 2
        assert out[0].target_program == "id"

    def test_skips_malformed_lines(self, tmp_path: Path) -> None:
        path = tmp_path / "examples.jsonl"
        path.write_text(
            json.dumps({"input_examples": [], "target_program": "ok"})
            + "\n"
            + "{not json\n"
            + json.dumps({"input_examples": [], "target_program": "also-ok"})
            + "\n",
            encoding="utf-8",
        )
        bank = FewShotBank(path)
        assert len(bank.load()) == 2

    def test_skips_blank_and_comment_lines(self, tmp_path: Path) -> None:
        path = tmp_path / "examples.jsonl"
        path.write_text(
            "\n// comment\n" + json.dumps({"input_examples": [], "target_program": "ok"}) + "\n",
            encoding="utf-8",
        )
        bank = FewShotBank(path)
        assert len(bank.load()) == 1

    def test_load_is_cached(self, tmp_path: Path) -> None:
        path = tmp_path / "examples.jsonl"
        path.write_text(
            json.dumps({"input_examples": [], "target_program": "x"}) + "\n",
            encoding="utf-8",
        )
        bank = FewShotBank(path)
        first = bank.load()
        path.write_text("", encoding="utf-8")  # mutate after load
        second = bank.load()
        assert first is second  # cached, no re-read


class TestDomainAwareLLMPrior:
    def test_fallback_system_prompt(self, tmp_path: Path) -> None:
        prior = DomainAwareLLMPrior(prompts_root=tmp_path)
        prompt = prior.system_prompt(_meta("sql"))
        assert "SQL" in prompt
        assert "deterministic" in prompt.lower()

    def test_filesystem_system_prompt_takes_precedence(self, tmp_path: Path) -> None:
        domain_dir = tmp_path / "sql"
        domain_dir.mkdir()
        (domain_dir / "system.md").write_text("CUSTOM SQL PROMPT", encoding="utf-8")
        prior = DomainAwareLLMPrior(prompts_root=tmp_path)
        assert prior.system_prompt(_meta("sql")) == "CUSTOM SQL PROMPT"

    def test_build_messages_no_few_shot(self, tmp_path: Path) -> None:
        prior = DomainAwareLLMPrior(prompts_root=tmp_path)
        msgs = prior.build_messages(
            _meta("sql"),
            [{"input": [1, 2], "output": [2, 1]}],
        )
        assert msgs[0]["role"] == "system"
        # last message is the live user request
        assert msgs[-1]["role"] == "user"
        payload = json.loads(msgs[-1]["content"])
        assert payload["examples"][0]["output"] == [2, 1]

    def test_build_messages_includes_few_shots(self, tmp_path: Path) -> None:
        domain_dir = tmp_path / "sql"
        domain_dir.mkdir()
        (domain_dir / "examples.jsonl").write_text(
            json.dumps(
                {
                    "input_examples": [{"input": "a", "output": "A"}],
                    "target_program": "upper",
                    "rationale": "Map letter to upper",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        prior = DomainAwareLLMPrior(prompts_root=tmp_path)
        msgs = prior.build_messages(_meta("sql"), [{"input": "x"}])
        # system + (user + assistant) few-shot + user live = 4
        assert len(msgs) == 4
        roles = [m["role"] for m in msgs]
        assert roles == ["system", "user", "assistant", "user"]
        assert "<reasoning>" in msgs[2]["content"]

    def test_max_few_shot_limit(self, tmp_path: Path) -> None:
        domain_dir = tmp_path / "sql"
        domain_dir.mkdir()
        lines = [
            json.dumps({"input_examples": [], "target_program": f"prog_{i}"}) for i in range(10)
        ]
        (domain_dir / "examples.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
        prior = DomainAwareLLMPrior(prompts_root=tmp_path, max_few_shot=2)
        msgs = prior.build_messages(_meta("sql"), [])
        # system + 2 * (user + assistant) + live user = 6
        assert len(msgs) == 6
