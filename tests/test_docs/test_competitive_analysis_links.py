# tests/test_docs/test_competitive_analysis_links.py
"""Verify the WP1 + WP5 docs all exist and cross-references resolve."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_competitive_analysis_files_exist() -> None:
    for name in ("README.md", "autogen.md", "microsoft-agent-framework.md", "decision-matrix.md"):
        path = REPO_ROOT / "docs" / "competitive-analysis" / name
        assert path.exists(), f"missing {path}"


def test_adr_files_exist() -> None:
    assert (REPO_ROOT / "docs" / "adr" / "README.md").exists()
    assert (REPO_ROOT / "docs" / "adr" / "0001-pge-trinity-vs-group-chat.md").exists()


def test_autogen_md_minimum_length() -> None:
    body = (REPO_ROOT / "docs" / "competitive-analysis" / "autogen.md").read_text(encoding="utf-8")
    assert len(body.split()) >= 400, f"autogen.md is below 400 words ({len(body.split())} words)"


def test_maf_md_minimum_length() -> None:
    body = (REPO_ROOT / "docs" / "competitive-analysis" / "microsoft-agent-framework.md").read_text(
        encoding="utf-8"
    )
    assert len(body.split()) >= 400, f"MAF doc is below 400 words ({len(body.split())} words)"


def test_adr_mentions_three_groupchat_patterns() -> None:
    body = (REPO_ROOT / "docs" / "adr" / "0001-pge-trinity-vs-group-chat.md").read_text(
        encoding="utf-8"
    )
    for name in ("RoundRobinGroupChat", "SelectorGroupChat", "Swarm"):
        assert name in body, f"ADR 0001 must mention {name} by name"


def test_root_readme_links_competitive_analysis() -> None:
    body = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert "docs/competitive-analysis/" in body, "root README must link competitive-analysis"
    assert "docs/adr/0001-pge-trinity-vs-group-chat.md" in body, "root README must link ADR 0001"
