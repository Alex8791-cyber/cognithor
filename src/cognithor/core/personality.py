"""Personality Engine: Adds warmth, greetings, and empathy to Jarvis responses.

Provides configurable personality directives that are injected into the
SYSTEM_PROMPT, making Jarvis feel less robotic and more human.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from cognithor.config import PersonalityConfig


class PersonalityEngine:
    """Generates personality-enriched prompt fragments.

    Configurable via PersonalityConfig (warmth, humor, follow-up questions, etc.).
    """

    def __init__(self, config: PersonalityConfig | None = None) -> None:
        if config is None:
            from cognithor.config import PersonalityConfig

            config = PersonalityConfig()
        self._config = config

    @property
    def config(self) -> PersonalityConfig:
        return self._config

    def get_greeting_fragment(self) -> str:
        """Returns a time-of-day greeting fragment (German).

        Returns empty string if greetings are disabled.
        """
        if not self._config.greeting_enabled:
            return ""

        hour = datetime.now().hour

        if 5 <= hour < 12:
            return "Morgen! "
        elif 12 <= hour < 17:
            return ""  # nachmittags kein extra Gruss noetig
        elif 17 <= hour < 22:
            return "Hey, guten Abend! "
        else:
            return "Na, auch noch wach? "

    def get_personality_directives(self) -> str:
        """Returns personality directives for the SYSTEM_PROMPT.

        The directives are scaled by the warmth and humor config values.
        """
        directives: list[str] = []

        # Warmth (0.0 = sachlich, 1.0 = warmherzig)
        if self._config.warmth >= 0.3:
            directives.append("Sei freundlich -- nicht steif.")
        if self._config.warmth >= 0.6:
            directives.append("Zeig Verstaendnis wenn was nicht klappt.")
        if self._config.warmth >= 0.8:
            directives.append("Erkenne gute Ideen an. Sag auch mal 'clever!' oder 'guter Ansatz'.")

        # Humor (0.0 = ernst, 1.0 = locker)
        if self._config.humor >= 0.3:
            directives.append("Ab und zu ein lockerer Kommentar ist okay.")
        if self._config.humor >= 0.6:
            directives.append("Witz ist erlaubt wenn es passt -- aber nicht erzwingen.")

        # Erfolge feiern
        if self._config.success_celebration:
            directives.append(
                "Wenn was geklappt hat, sag es kurz: 'Laeuft!', 'Fertig!', 'Hat geklappt!'."
            )

        # Nachfragen
        if self._config.follow_up_questions:
            directives.append("Frag am Ende kurz ob noch was offen ist, wenn es Sinn macht.")

        return "\n".join(directives)

    def enhance_response(self, text: str, context: dict[str, Any] | None = None) -> str:
        """Post-process a response with personality touches.

        Currently a pass-through. Can be extended to add greeting
        fragments or success celebration when appropriate.
        """
        return text

    def build_personality_block(self) -> str:
        """Builds the complete personality block for SYSTEM_PROMPT injection.

        Returns empty string if all personality features are disabled.
        """
        directives = self.get_personality_directives()
        if not directives:
            return ""

        return f"\n## Persönlichkeit\n{directives}\n"
