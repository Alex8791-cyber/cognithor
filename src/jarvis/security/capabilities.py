"""Tool-Capabilities-Matrix mit feingranularem Policy-Matching."""

from __future__ import annotations

from typing import Any

from jarvis.models import ToolCapability, ToolCapabilitySpec, PolicyDecision
from jarvis.utils.logging import get_logger

log = get_logger(__name__)


class SandboxProfile:
    """Sandbox-Profil mit erlaubten/verbotenen Capabilities."""

    def __init__(
        self,
        name: str,
        allowed_capabilities: set[ToolCapability] | None = None,
        denied_capabilities: set[ToolCapability] | None = None,
        max_memory_mb: int = 512,
        max_timeout_seconds: int = 30,
    ) -> None:
        self.name = name
        self.allowed_capabilities = allowed_capabilities or set()
        self.denied_capabilities = denied_capabilities or set()
        self.max_memory_mb = max_memory_mb
        self.max_timeout_seconds = max_timeout_seconds

    def is_capability_allowed(self, cap: ToolCapability) -> bool:
        """Prueft ob eine Capability erlaubt ist."""
        if cap in self.denied_capabilities:
            return False
        if self.allowed_capabilities and cap not in self.allowed_capabilities:
            return False
        return True


# Default Sandbox Profiles
RESTRICTIVE = SandboxProfile(
    name="restrictive",
    allowed_capabilities={
        ToolCapability.FS_READ,
        ToolCapability.MEMORY_READ,
        ToolCapability.SYSTEM_INFO,
    },
    denied_capabilities={
        ToolCapability.FS_WRITE,
        ToolCapability.NETWORK_HTTP,
        ToolCapability.NETWORK_WS,
        ToolCapability.EXEC_PROCESS,
        ToolCapability.EXEC_SCRIPT,
        ToolCapability.CREDENTIAL_ACCESS,
    },
    max_memory_mb=256,
    max_timeout_seconds=15,
)

STANDARD = SandboxProfile(
    name="standard",
    allowed_capabilities={
        ToolCapability.FS_READ,
        ToolCapability.FS_WRITE,
        ToolCapability.MEMORY_READ,
        ToolCapability.MEMORY_WRITE,
        ToolCapability.NETWORK_HTTP,
        ToolCapability.SYSTEM_INFO,
    },
    denied_capabilities={
        ToolCapability.EXEC_PROCESS,
        ToolCapability.CREDENTIAL_ACCESS,
    },
    max_memory_mb=512,
    max_timeout_seconds=30,
)

PERMISSIVE = SandboxProfile(
    name="permissive",
    allowed_capabilities=set(ToolCapability),
    denied_capabilities=set(),
    max_memory_mb=2048,
    max_timeout_seconds=120,
)

# Profile registry
_DEFAULT_PROFILES: dict[str, SandboxProfile] = {
    "restrictive": RESTRICTIVE,
    "standard": STANDARD,
    "permissive": PERMISSIVE,
}


class CapabilityMatrix:
    """Registry aller Tool-Capability-Spezifikationen."""

    def __init__(self) -> None:
        self._specs: dict[str, ToolCapabilitySpec] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        """Registriert Default-Spezifikationen fuer bekannte Tools."""
        defaults = [
            ToolCapabilitySpec(
                tool_name="read_file",
                capabilities=frozenset({ToolCapability.FS_READ}),
                max_timeout_seconds=10,
            ),
            ToolCapabilitySpec(
                tool_name="write_file",
                capabilities=frozenset({ToolCapability.FS_WRITE}),
                max_timeout_seconds=10,
            ),
            ToolCapabilitySpec(
                tool_name="edit_file",
                capabilities=frozenset({ToolCapability.FS_READ, ToolCapability.FS_WRITE}),
                max_timeout_seconds=10,
            ),
            ToolCapabilitySpec(
                tool_name="list_directory",
                capabilities=frozenset({ToolCapability.FS_READ}),
                max_timeout_seconds=10,
            ),
            ToolCapabilitySpec(
                tool_name="delete_file",
                capabilities=frozenset({ToolCapability.FS_WRITE}),
                max_timeout_seconds=10,
            ),
            ToolCapabilitySpec(
                tool_name="exec_command",
                capabilities=frozenset({ToolCapability.EXEC_PROCESS, ToolCapability.FS_READ, ToolCapability.FS_WRITE}),
                max_memory_mb=1024,
                max_timeout_seconds=60,
            ),
            ToolCapabilitySpec(
                tool_name="web_search",
                capabilities=frozenset({ToolCapability.NETWORK_HTTP}),
                network_domains=["*"],
                max_timeout_seconds=30,
            ),
            ToolCapabilitySpec(
                tool_name="fetch_url",
                capabilities=frozenset({ToolCapability.NETWORK_HTTP}),
                network_domains=["*"],
                max_timeout_seconds=30,
            ),
            ToolCapabilitySpec(
                tool_name="search_memory",
                capabilities=frozenset({ToolCapability.MEMORY_READ}),
                max_timeout_seconds=10,
            ),
            ToolCapabilitySpec(
                tool_name="save_to_memory",
                capabilities=frozenset({ToolCapability.MEMORY_WRITE}),
                max_timeout_seconds=10,
            ),
            ToolCapabilitySpec(
                tool_name="email_send",
                capabilities=frozenset({ToolCapability.NETWORK_HTTP, ToolCapability.CREDENTIAL_ACCESS}),
                max_timeout_seconds=30,
            ),
        ]
        for spec in defaults:
            self._specs[spec.tool_name] = spec

    def register_tool(self, spec: ToolCapabilitySpec) -> None:
        """Registriert eine Tool-Spezifikation."""
        self._specs[spec.tool_name] = spec
        log.debug("capability_registered", tool=spec.tool_name)

    def get_spec(self, tool_name: str) -> ToolCapabilitySpec | None:
        """Holt die Spezifikation eines Tools."""
        return self._specs.get(tool_name)

    def check_allowed(self, tool_name: str, profile: SandboxProfile) -> bool:
        """Prueft ob ein Tool im gegebenen Profil erlaubt ist."""
        spec = self._specs.get(tool_name)
        if spec is None:
            # Unknown tools are not allowed in restrictive profiles
            return profile.name == "permissive"

        for cap in spec.capabilities:
            if not profile.is_capability_allowed(cap):
                return False

        return True

    def get_violations(self, tool_name: str, profile: SandboxProfile) -> list[str]:
        """Gibt Liste verletzter Capabilities zurueck."""
        spec = self._specs.get(tool_name)
        if spec is None:
            return [f"Unknown tool: {tool_name}"]

        violations: list[str] = []
        for cap in spec.capabilities:
            if not profile.is_capability_allowed(cap):
                violations.append(f"Capability '{cap.value}' not allowed in profile '{profile.name}'")

        if spec.max_memory_mb > profile.max_memory_mb:
            violations.append(
                f"Tool requires {spec.max_memory_mb}MB, profile allows {profile.max_memory_mb}MB"
            )
        if spec.max_timeout_seconds > profile.max_timeout_seconds:
            violations.append(
                f"Tool needs {spec.max_timeout_seconds}s timeout, profile allows {profile.max_timeout_seconds}s"
            )
        return violations

    @property
    def registered_tools(self) -> list[str]:
        return list(self._specs.keys())


class PolicyEvaluator:
    """Evaluiert Tool-Calls gegen ein SandboxProfile."""

    def __init__(self, matrix: CapabilityMatrix | None = None) -> None:
        self._matrix = matrix or CapabilityMatrix()
        self._profiles = dict(_DEFAULT_PROFILES)

    def register_profile(self, profile: SandboxProfile) -> None:
        """Registriert ein zusaetzliches Profil."""
        self._profiles[profile.name] = profile

    def get_profile(self, name: str) -> SandboxProfile | None:
        """Holt ein Profil nach Name."""
        return self._profiles.get(name)

    def evaluate(
        self,
        tool_name: str,
        params: dict[str, Any] | None = None,
        profile_name: str = "standard",
    ) -> PolicyDecision:
        """Evaluiert einen Tool-Call gegen ein Profil."""
        profile = self._profiles.get(profile_name)
        if profile is None:
            return PolicyDecision(
                allowed=False,
                violations=[f"Unknown profile: {profile_name}"],
                suggested_profile="standard",
            )

        violations = self._matrix.get_violations(tool_name, profile)

        if violations:
            # Suggest a more permissive profile
            suggested = ""
            if profile_name == "restrictive":
                suggested = "standard"
            elif profile_name == "standard":
                suggested = "permissive"

            return PolicyDecision(
                allowed=False,
                violations=violations,
                suggested_profile=suggested,
            )

        return PolicyDecision(allowed=True)

    @property
    def matrix(self) -> CapabilityMatrix:
        return self._matrix

    @property
    def available_profiles(self) -> list[str]:
        return list(self._profiles.keys())
