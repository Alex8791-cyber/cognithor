"""Jarvis Skill: Gmail Sync (API)."""

import httpx
from jarvis.skills.base import BaseSkill


class GmailSyncSkill(BaseSkill):
    NAME = "gmail_sync"
    REQUIRES_NETWORK = True
    API_BASE = "https://api.example.com/v1"

    async def execute(self, params: dict) -> dict:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{{self.API_BASE}}/endpoint")
            return {{"data": resp.json()}}
