"""LLM Bridge — adapts Cognithor's UnifiedLLMClient to Immortal Mind's LLMClient interface.

Immortal Mind's CognitioEngine expects an llm_client with:
    - complete(prompt, system_prompt=None, max_tokens=1024, temperature=0.7) -> str
    - chat(messages, system_prompt=None, max_tokens=1024, temperature=0.7) -> str
    - complete_json(prompt, expected_keys=None, ...) -> dict[str, Any]
    - health_check() -> bool

This bridge adapts Cognithor's async LLM backend to this synchronous interface.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from cognithor.core.unified_llm import UnifiedLLMClient

logger = logging.getLogger("cognithor.identity.llm_bridge")


class CognithorLLMBridge:
    """Adapts Cognithor's UnifiedLLMClient to Immortal Mind's LLMClient interface.

    Cognithor's LLM is async; Immortal Mind's CognitioEngine calls LLM
    synchronously from background threads (consolidation worker).
    This bridge uses asyncio.run_coroutine_threadsafe() for thread-safe access.
    """

    def __init__(
        self,
        llm_client: UnifiedLLMClient,
        model: str = "",
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        self._llm = llm_client
        self._model = model
        self._loop = loop

    def _run_async(self, coro: Any) -> Any:
        """Run an async coroutine from a sync context (thread-safe).

        Must be called from a worker thread, never from the event loop thread.
        """
        loop = self._loop
        if loop is None:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                # No running loop — create one (fallback for tests)
                return asyncio.run(coro)

        if loop.is_running():
            # Guard: detect if we're ON the event loop thread (would deadlock)
            if threading.current_thread() is threading.main_thread():
                import warnings

                warnings.warn(
                    "LLMBridge._run_async called from the main thread while "
                    "the event loop is running — this would deadlock. "
                    "Falling back to a new event loop in a thread.",
                    RuntimeWarning,
                    stacklevel=3,
                )
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor(1) as pool:
                    return pool.submit(asyncio.run, coro).result(timeout=120)
            # Called from a different thread (e.g., consolidation worker)
            future = asyncio.run_coroutine_threadsafe(coro, loop)
            return future.result(timeout=120)
        else:
            return loop.run_until_complete(coro)

    def complete(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> str:
        """Single-shot text completion."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        async def _call() -> str:
            resp = await self._llm.chat(
                model=self._model,
                messages=messages,
                temperature=temperature,
            )
            if isinstance(resp, dict):
                return cast("str", resp.get("content", resp.get("message", {}).get("content", "")))

            return str(resp)

        return cast("str", self._run_async(_call()))

    def chat(
        self,
        messages: list[dict[str, Any]],
        system_prompt: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> str:
        """Multi-turn chat completion."""
        all_msgs = []
        if system_prompt:
            all_msgs.append({"role": "system", "content": system_prompt})
        all_msgs.extend(messages)

        async def _call() -> str:
            resp = await self._llm.chat(
                model=self._model,
                messages=all_msgs,
                temperature=temperature,
            )
            if isinstance(resp, dict):
                return cast("str", resp.get("content", resp.get("message", {}).get("content", "")))

            return str(resp)

        return cast("str", self._run_async(_call()))

    def complete_json(
        self,
        prompt: str,
        expected_keys: list[str] | None = None,
        system_prompt: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.2,
    ) -> dict[str, Any]:
        """LLM request expecting JSON output with repair logic."""
        json_instruction = (
            "\n\nREPLY ONLY IN VALID JSON FORMAT. "
            "Do NOT add explanations, markdown, or code blocks."
        )
        try:
            raw = self.complete(
                prompt + json_instruction,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return self._parse_json_safe(raw, expected_keys)
        except Exception as e:
            logger.warning("complete_json failed: %s", e)
            return {k: None for k in (expected_keys or [])}

    def health_check(self) -> bool:
        """Check if the LLM backend is accessible."""
        try:
            result = self.complete("Respond with 'ok'.", max_tokens=5)
            return bool(result)
        except Exception:
            return False

    @staticmethod
    def _parse_json_safe(text: str, expected_keys: list[str] | None = None) -> dict[str, Any]:
        """Extract JSON from LLM output with repair attempts."""
        defaults = {k: None for k in (expected_keys or [])}

        # Direct parse
        try:
            data = json.loads(text.strip())
            if isinstance(data, dict):
                data.update({k: data.get(k) for k in (expected_keys or [])})
                return data
        except json.JSONDecodeError:
            pass

        # Markdown code block
        match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
        if match:
            try:
                data = json.loads(match.group(1))
                if isinstance(data, dict):
                    data.update({k: data.get(k) for k in (expected_keys or [])})
                    return data
            except json.JSONDecodeError:
                pass

        # First { ... } block
        idx = text.find("{")
        if idx != -1 and len(text) <= 8192:
            try:
                data, _ = json.JSONDecoder().raw_decode(text, idx)
                if isinstance(data, dict):
                    data.update({k: data.get(k) for k in (expected_keys or [])})
                    return data
            except json.JSONDecodeError:
                pass

        return defaults
