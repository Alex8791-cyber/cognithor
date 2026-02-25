"""Tests."""
from .skill import BackupSkill

def test_cron() -> None:
    assert BackupSkill.CRON is not None
