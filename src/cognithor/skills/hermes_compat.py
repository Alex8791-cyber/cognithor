"""Hermes Agent / agentskills.io SKILL.md format compatibility.

Enables Cognithor to import skills from the agentskills.io ecosystem
and export Cognithor skills to SKILL.md format.

SKILL.md Format (agentskills.io):
---
name: skill_name
description: What the skill does
author: author_name
version: 1.0.0
tags: [tag1, tag2]
inputs:
  - name: param1
    type: string
    description: Parameter description
outputs:
  - name: result
    type: string
    description: Output description
---
# Skill Name

## Instructions
Step-by-step instructions for the agent...

## Examples
Example usage...
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from pathlib import Path

from cognithor.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class HermesSkill:
    """A skill in the agentskills.io SKILL.md format."""

    name: str
    description: str = ""
    author: str = ""
    version: str = "1.0.0"
    tags: list[str] = field(default_factory=list)
    inputs: list[dict[str, str]] = field(default_factory=list)
    outputs: list[dict[str, str]] = field(default_factory=list)
    instructions: str = ""
    examples: str = ""


class HermesCompatLayer:
    """Import/export skills in agentskills.io SKILL.md format."""

    @staticmethod
    def parse_skill_md(content: str) -> HermesSkill:
        """Parse a SKILL.md file content into a HermesSkill."""
        # Extract YAML frontmatter
        match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)", content, re.DOTALL)
        if not match:
            raise ValueError("No YAML frontmatter found")

        frontmatter = yaml.safe_load(match.group(1))
        body = match.group(2)

        # Extract sections from body
        instructions = ""
        examples = ""
        current_section = ""
        for line in body.split("\n"):
            if line.startswith("## Instructions"):
                current_section = "instructions"
                continue
            elif line.startswith("## Examples"):
                current_section = "examples"
                continue
            elif line.startswith("## "):
                current_section = ""
                continue

            if current_section == "instructions":
                instructions += line + "\n"
            elif current_section == "examples":
                examples += line + "\n"

        return HermesSkill(
            name=frontmatter.get("name", ""),
            description=frontmatter.get("description", ""),
            author=frontmatter.get("author", ""),
            version=frontmatter.get("version", "1.0.0"),
            tags=frontmatter.get("tags", []),
            inputs=frontmatter.get("inputs", []),
            outputs=frontmatter.get("outputs", []),
            instructions=instructions.strip(),
            examples=examples.strip(),
        )

    @staticmethod
    def to_skill_md(skill: HermesSkill) -> str:
        """Export a HermesSkill to SKILL.md format."""
        frontmatter = {
            "name": skill.name,
            "description": skill.description,
            "author": skill.author,
            "version": skill.version,
            "tags": skill.tags,
        }
        if skill.inputs:
            frontmatter["inputs"] = skill.inputs
        if skill.outputs:
            frontmatter["outputs"] = skill.outputs

        md = f"---\n{yaml.dump(frontmatter, default_flow_style=False)}---\n\n"
        md += f"# {skill.name}\n\n"
        if skill.instructions:
            md += f"## Instructions\n\n{skill.instructions}\n\n"
        if skill.examples:
            md += f"## Examples\n\n{skill.examples}\n\n"
        return md

    @staticmethod
    def import_from_file(path: Path) -> HermesSkill:
        """Import a SKILL.md file."""
        content = path.read_text(encoding="utf-8")
        return HermesCompatLayer.parse_skill_md(content)

    @staticmethod
    def export_to_file(skill: HermesSkill, path: Path) -> None:
        """Export a skill to SKILL.md file."""
        content = HermesCompatLayer.to_skill_md(skill)
        path.write_text(content, encoding="utf-8")

    @staticmethod
    def cognithor_to_hermes(cognithor_skill: dict[str, Any]) -> HermesSkill:
        """Convert a Cognithor skill dict to HermesSkill."""
        return HermesSkill(
            name=cognithor_skill.get("name", ""),
            description=cognithor_skill.get("description", ""),
            author=cognithor_skill.get("author", "cognithor"),
            version=cognithor_skill.get("version", "1.0.0"),
            tags=cognithor_skill.get("tags", []),
            instructions=cognithor_skill.get("prompt", ""),
        )

    @staticmethod
    def hermes_to_cognithor(hermes: HermesSkill) -> dict[str, Any]:
        """Convert a HermesSkill to Cognithor skill dict."""
        return {
            "name": hermes.name,
            "description": hermes.description,
            "author": hermes.author,
            "version": hermes.version,
            "tags": hermes.tags,
            "prompt": hermes.instructions,
            "source": "hermes",
            "format": "skill.md",
        }
