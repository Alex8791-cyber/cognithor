"""GDPR Compliance Engine — central runtime enforcement.

Every data processing operation must pass through this engine.
It enforces: consent requirements, legal basis validation,
purpose limitations, and privacy mode.
"""

from __future__ import annotations

from jarvis.security.consent import ConsentManager
from jarvis.security.gdpr import DataPurpose, ProcessingBasis
from jarvis.utils.logging import get_logger

log = get_logger(__name__)

__all__ = ["ComplianceEngine", "ComplianceViolation"]


class ComplianceViolation(Exception):
    """Raised when a processing operation violates GDPR policy."""


class ComplianceEngine:
    """Central GDPR policy enforcer. Called before every processing operation.

    Rules:
    1. Consent-based processing requires actual consent
    2. Privacy mode blocks all persistent storage except security
    3. OSINT requires explicit OSINT consent
    4. Legitimate interest bypasses consent (security monitoring, audit)
    """

    # System-internal channels exempt from user consent (use LEGITIMATE_INTEREST)
    _SYSTEM_CHANNELS = {"cron", "sub_agent", "system", "evolution", "heartbeat"}

    def __init__(
        self,
        consent_manager: ConsentManager | None = None,
        enabled: bool = True,
        consent_required: bool = True,
        policy_version: str = "1.0",
    ) -> None:
        self._consent = consent_manager
        self._enabled = enabled
        self._consent_required = consent_required
        self._policy_version = policy_version
        self._privacy_mode = False

    def set_privacy_mode(self, enabled: bool) -> None:
        self._privacy_mode = enabled
        log.info("privacy_mode_changed", enabled=enabled)

    @property
    def privacy_mode(self) -> bool:
        return self._privacy_mode

    def check(
        self,
        user_id: str,
        channel: str,
        legal_basis: ProcessingBasis,
        purpose: DataPurpose,
        data_types: list[str] | None = None,
    ) -> None:
        """Verify that the processing operation is GDPR-compliant.

        Raises ComplianceViolation if not allowed.
        Does nothing if engine is disabled (development mode).
        """
        if not self._enabled:
            return

        # Rule 1: Privacy mode blocks everything except security
        if self._privacy_mode and purpose != DataPurpose.SECURITY:
            raise ComplianceViolation(f"Privacy mode active — {purpose.value} processing blocked")

        # Rule 2: System-internal channels use LEGITIMATE_INTEREST, not consent
        if channel in self._SYSTEM_CHANNELS:
            if legal_basis != ProcessingBasis.LEGITIMATE_INTEREST:
                legal_basis = ProcessingBasis.LEGITIMATE_INTEREST
            # System channels are exempt from user consent
            return

        # Rule 3: Consent-based processing requires actual consent (FAIL-CLOSED)
        if legal_basis == ProcessingBasis.CONSENT and self._consent_required:
            if self._consent is None:
                # No consent store available — fail CLOSED, not open
                raise ComplianceViolation(
                    "Consent store unavailable — cannot verify consent. Processing blocked."
                )
            if not self._consent.has_consent(user_id, channel, policy_version=self._policy_version):
                raise ComplianceViolation(
                    f"No consent for {purpose.value} on channel {channel}. "
                    f"User {user_id[:8]} must accept the privacy notice first."
                )

        # Rule 4: OSINT requires explicit OSINT consent (above and beyond data_processing)
        if purpose == DataPurpose.OSINT:
            if self._consent is None:
                raise ComplianceViolation("Consent store unavailable for OSINT check.")
            if not self._consent.has_consent(user_id, channel, "osint"):
                raise ComplianceViolation(
                    f"OSINT investigation requires explicit osint consent from user {user_id[:8]}"
                )

        # Rule 5: Purpose-specific restriction (Art. 18/21)
        if self._consent and self._consent.is_restricted(user_id, channel, purpose.value):
            raise ComplianceViolation(
                f"User has restricted {purpose.value} processing on channel {channel}"
            )

        # Rule 6: Legitimate interest is allowed without consent
        # (security monitoring, audit trails, fraud detection)

        log.debug(
            "compliance_check_passed",
            user=user_id[:8],
            channel=channel,
            basis=legal_basis.value,
            purpose=purpose.value,
        )
