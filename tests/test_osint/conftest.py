"""Test fixtures for HIM OSINT tests."""

from __future__ import annotations

import json
import pytest

TERRY_CASE = {
    "request": {
        "target_name": "dinnar1407-code",
        "target_github": "dinnar1407-code",
        "claims": [
            "works at Anthropic",
            "most robust A2A implementation",
            "pre-alpha refinement phase",
        ],
        "target_type": "person",
        "depth": "standard",
        "requester_justification": "Received collaboration request via GitHub, verifying credentials",
    },
    "expected": {
        "trust_score_range": [30, 70],
        "trust_label": "mixed",
        "must_have_red_flags": True,
    },
    "mock_github_response": {
        "login": "dinnar1407-code",
        "name": "Terry",
        "company": None,
        "bio": "Building Agent Nexus",
        "public_repos": 3,
        "followers": 12,
        "following": 5,
        "created_at": "2024-01-15T00:00:00Z",
        "html_url": "https://github.com/dinnar1407-code",
        "location": None,
    },
}


@pytest.fixture
def terry_case():
    return TERRY_CASE
