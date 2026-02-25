"""Tests für Gmail Sync."""
import pytest
from .skill import GmailSyncSkill

class TestGmailSyncSkill:
    def test_name(self) -> None:
        assert GmailSyncSkill.NAME == "gmail_sync"
