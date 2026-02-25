"""Jarvis · Unified Error Hierarchy.

Provides a structured exception hierarchy for the entire Jarvis Agent OS.
All custom exceptions inherit from JarvisError, which carries an error_code
and optional details dict for programmatic handling.

Usage::

    from jarvis.core.errors import ConfigError, LLMError

    raise ConfigError("Invalid model name", error_code="CONFIG_INVALID_MODEL")
    raise LLMError("Provider timeout", details={"provider": "openai", "timeout_s": 30})
"""

from __future__ import annotations


class JarvisError(Exception):
    """Base exception for all Jarvis errors."""

    def __init__(
        self,
        message: str,
        error_code: str = "JARVIS_ERROR",
        details: dict | None = None,
    ) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.details = details or {}


class ConfigError(JarvisError):
    """Configuration-related errors (loading, validation, missing keys)."""

    def __init__(
        self,
        message: str,
        error_code: str = "CONFIG_ERROR",
        details: dict | None = None,
    ) -> None:
        super().__init__(message, error_code=error_code, details=details)


class LLMError(JarvisError):
    """LLM backend errors (timeouts, rate limits, invalid responses)."""

    def __init__(
        self,
        message: str,
        error_code: str = "LLM_ERROR",
        details: dict | None = None,
    ) -> None:
        super().__init__(message, error_code=error_code, details=details)


class ToolExecutionError(JarvisError):
    """Errors during tool/skill execution."""

    def __init__(
        self,
        message: str,
        error_code: str = "TOOL_EXECUTION_ERROR",
        details: dict | None = None,
    ) -> None:
        super().__init__(message, error_code=error_code, details=details)


class SandboxError(JarvisError):
    """Sandbox-related errors (creation, resource limits, violations)."""

    def __init__(
        self,
        message: str,
        error_code: str = "SANDBOX_ERROR",
        details: dict | None = None,
    ) -> None:
        super().__init__(message, error_code=error_code, details=details)


class ChannelError(JarvisError):
    """Communication channel errors (connection, authentication, send failures)."""

    def __init__(
        self,
        message: str,
        error_code: str = "CHANNEL_ERROR",
        details: dict | None = None,
    ) -> None:
        super().__init__(message, error_code=error_code, details=details)


class JarvisMemoryError(JarvisError):
    """Memory subsystem errors (storage, retrieval, corruption).

    Named JarvisMemoryError to avoid shadowing the builtin MemoryError.
    """

    def __init__(
        self,
        message: str,
        error_code: str = "MEMORY_ERROR",
        details: dict | None = None,
    ) -> None:
        super().__init__(message, error_code=error_code, details=details)


class JarvisSecurityError(JarvisError):
    """Security-related errors (encryption, access control, audit failures).

    Named JarvisSecurityError to avoid shadowing the builtin SecurityError
    (which does not exist in Python, but avoids confusion with common naming).
    """

    def __init__(
        self,
        message: str,
        error_code: str = "SECURITY_ERROR",
        details: dict | None = None,
    ) -> None:
        super().__init__(message, error_code=error_code, details=details)


class PolicyViolation(JarvisError):
    """Policy violations (gatekeeper rules, safety policies)."""

    def __init__(
        self,
        message: str,
        error_code: str = "POLICY_VIOLATION",
        details: dict | None = None,
    ) -> None:
        super().__init__(message, error_code=error_code, details=details)


class GatekeeperDenied(PolicyViolation):
    """Gatekeeper specifically denied an action."""

    def __init__(
        self,
        message: str,
        error_code: str = "GATEKEEPER_DENIED",
        details: dict | None = None,
    ) -> None:
        super().__init__(message, error_code=error_code, details=details)


class AuthenticationError(JarvisError):
    """Authentication failures (invalid tokens, expired sessions)."""

    def __init__(
        self,
        message: str,
        error_code: str = "AUTHENTICATION_ERROR",
        details: dict | None = None,
    ) -> None:
        super().__init__(message, error_code=error_code, details=details)


class RateLimitExceeded(JarvisError):
    """Rate limit exceeded for API calls or resource usage."""

    def __init__(
        self,
        message: str,
        error_code: str = "RATE_LIMIT_EXCEEDED",
        details: dict | None = None,
    ) -> None:
        super().__init__(message, error_code=error_code, details=details)
