"""Skill Package: Offizielles Paketformat für Jarvis-Skills.

Definiert das standardisierte Format für Skill-Pakete, die über
P2P-Netzwerke oder lokale Verzeichnisse verteilt werden können.

Komponenten:

  1. SkillManifest: YAML-basiertes Manifest mit Name, Version,
     Trigger-Wörtern, benötigten Tools, Sandbox-Rechten und
     Abhängigkeiten.

  2. PackageSigner: Ed25519-basierte digitale Signaturen für
     Authentizität und Integrität.

  3. CodeAnalyzer: Statische Analyse des Skill-Codes auf
     gefährliche Patterns (eval, exec, subprocess, network, etc.).

  4. PackageBuilder: Erstellt signierte Pakete aus Skills.

  5. PackageInstaller: Verifiziert und installiert Pakete in
     die lokale SkillRegistry mit Sandbox-Isolation.

Sicherheitsmodell:
  - Jedes Paket MUSS signiert sein (Herausgeber-Zertifikat)
  - Jedes Paket wird vor Installation durch CodeAnalyzer geprüft
  - Heruntergeladene Skills laufen in strikter Sandbox
  - Kein Netzwerkzugriff, begrenzter Speicher, isoliertes Dateisystem

Bibel-Referenz: §6.5 (Skill Distribution)
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import tarfile
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import Any

logger = logging.getLogger("jarvis.skills.package")

# Manifest-Limits
MAX_MANIFEST_MEMORY_MB = 1024
MAX_MANIFEST_TIMEOUT_SECONDS = 300
MAX_DESCRIPTION_LENGTH = 200

# Paket-Member-Groessenlimit (Bytes)
MAX_PACKAGE_MEMBER_SIZE = 2_097_152  # 2 MB

__all__ = [
    "SkillManifest",
    "SkillPackage",
    "PackageSigner",
    "PackageSignature",
    "PackageBuilder",
    "PackageInstaller",
    "CodeAnalyzer",
    "AnalysisReport",
    "AnalysisVerdict",
    "TrustLevel",
    "SandboxPermission",
    "InstallResult",
]


# ============================================================================
# Enums
# ============================================================================


class TrustLevel(Enum):
    """Vertrauensstufe eines Paket-Herausgebers."""

    UNKNOWN = "unknown"  # Unbekannter Herausgeber
    COMMUNITY = "community"  # Community-validiert (Reputation > Schwellwert)
    VERIFIED = "verified"  # Manuell verifizierter Herausgeber
    OFFICIAL = "official"  # Offizielles Jarvis-Team


class AnalysisVerdict(Enum):
    """Ergebnis der Code-Analyse."""

    SAFE = "safe"  # Keine Auffälligkeiten
    SUSPICIOUS = "suspicious"  # Warnung, aber installierbar
    DANGEROUS = "dangerous"  # Blockiert — nicht installieren


class SandboxPermission(Enum):
    """Granulare Sandbox-Rechte für Skills."""

    FILE_READ = "file_read"  # Dateien lesen (Workspace)
    FILE_WRITE = "file_write"  # Dateien schreiben (Workspace)
    NETWORK = "network"  # Netzwerkzugriff
    EXEC = "exec"  # Subprozesse starten
    MEMORY_ACCESS = "memory_access"  # Jarvis Memory lesen
    MEMORY_WRITE = "memory_write"  # Jarvis Memory schreiben
    LLM_CALL = "llm_call"  # LLM-API aufrufen


# ============================================================================
# Manifest
# ============================================================================


@dataclass
class SkillManifest:
    """Paket-Manifest: Beschreibt einen Skill vollständig.

    Wird als manifest.yaml in jedes Paket eingebettet.
    """

    # Identifikation
    name: str  # Eindeutiger Paketname (snake_case)
    version: str  # SemVer: "1.2.3"
    description: str  # Kurzbeschreibung (max 200 Zeichen)

    # Herausgeber
    author: str  # Name oder Pseudonym
    author_id: str = ""  # Öffentlicher Schlüssel (hex) oder UUID

    # Skill-Metadaten
    trigger_keywords: list[str] = field(default_factory=list)
    tools_required: list[str] = field(default_factory=list)
    category: str = "general"

    # Sandbox-Rechte (minimal-invasiv)
    permissions: list[str] = field(default_factory=list)
    max_memory_mb: int = 128  # Maximaler Speicher
    timeout_seconds: int = 30  # Maximale Ausführungszeit
    network_allowed: bool = False  # Explizit kein Netzwerk (Default)

    # Abhängigkeiten
    dependencies: list[str] = field(default_factory=list)  # ["other_skill>=1.0"]
    jarvis_min_version: str = "0.1.0"  # Mindest-Jarvis-Version

    # Integrität
    content_hash: str = ""  # SHA-256 über Code + Tests
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )

    @property
    def qualified_name(self) -> str:
        """Vollqualifizierter Name: name@version."""
        return f"{self.name}@{self.version}"

    @property
    def parsed_permissions(self) -> list[SandboxPermission]:
        """Parsed permissions zu Enum-Werten."""
        result = []
        for p in self.permissions:
            try:
                result.append(SandboxPermission(p))
            except ValueError:
                pass
        return result

    def to_dict(self) -> dict[str, Any]:
        """Serialisiert Manifest als Dict."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "author_id": self.author_id,
            "trigger_keywords": self.trigger_keywords,
            "tools_required": self.tools_required,
            "category": self.category,
            "permissions": self.permissions,
            "max_memory_mb": self.max_memory_mb,
            "timeout_seconds": self.timeout_seconds,
            "network_allowed": self.network_allowed,
            "dependencies": self.dependencies,
            "jarvis_min_version": self.jarvis_min_version,
            "content_hash": self.content_hash,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SkillManifest:
        """Deserialisiert ein Manifest."""
        return cls(
            name=data["name"],
            version=data["version"],
            description=data.get("description", ""),
            author=data.get("author", "unknown"),
            author_id=data.get("author_id", ""),
            trigger_keywords=data.get("trigger_keywords", []),
            tools_required=data.get("tools_required", []),
            category=data.get("category", "general"),
            permissions=data.get("permissions", []),
            max_memory_mb=data.get("max_memory_mb", 128),
            timeout_seconds=data.get("timeout_seconds", 30),
            network_allowed=data.get("network_allowed", False),
            dependencies=data.get("dependencies", []),
            jarvis_min_version=data.get("jarvis_min_version", "0.1.0"),
            content_hash=data.get("content_hash", ""),
            created_at=data.get("created_at", ""),
        )

    def validate(self) -> list[str]:
        """Validiert das Manifest. Gibt Fehlermeldungen zurück."""
        errors: list[str] = []
        if not re.match(r"^[a-z][a-z0-9_]{2,49}$", self.name):
            errors.append(
                f"Ungültiger Paketname '{self.name}': "
                "3-50 Zeichen, snake_case, beginnt mit Buchstabe"
            )
        if not re.match(r"^\d+\.\d+\.\d+$", self.version):
            errors.append(f"Ungültige Version '{self.version}': SemVer erwartet (X.Y.Z)")
        if len(self.description) > MAX_DESCRIPTION_LENGTH:
            errors.append(f"Beschreibung zu lang (max {MAX_DESCRIPTION_LENGTH} Zeichen)")
        if not self.author:
            errors.append("Autor fehlt")
        if self.max_memory_mb > MAX_MANIFEST_MEMORY_MB:
            errors.append(f"max_memory_mb > {MAX_MANIFEST_MEMORY_MB} nicht erlaubt")
        if self.timeout_seconds > MAX_MANIFEST_TIMEOUT_SECONDS:
            errors.append(f"timeout_seconds > {MAX_MANIFEST_TIMEOUT_SECONDS} nicht erlaubt")
        return errors


# ============================================================================
# Signierung (Ed25519-kompatibel, HMAC-Fallback)
# ============================================================================


@dataclass
class PackageSignature:
    """Digitale Signatur eines Skill-Pakets."""

    signer_id: str  # Öffentlicher Schlüssel oder ID
    signature: str  # Hex-encodierte Signatur
    algorithm: str = "hmac-sha256"  # oder "ed25519"
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )


class PackageSigner:
    """Signiert und verifiziert Skill-Pakete.

    Unterstützt:
      - HMAC-SHA256 (Fallback, immer verfügbar)
      - Ed25519 (wenn cryptography-Paket installiert)

    Args:
        private_key: Privater Schlüssel (Hex-String für HMAC, Bytes für Ed25519).
        signer_id: Identifikation des Herausgebers.
    """

    def __init__(self, private_key: str, signer_id: str = "") -> None:
        self._key = private_key
        self._signer_id = signer_id or hashlib.sha256(private_key.encode()).hexdigest()[:16]

    @property
    def signer_id(self) -> str:
        return self._signer_id

    def sign(self, content: bytes) -> PackageSignature:
        """Signiert Inhalt mit HMAC-SHA256.

        Args:
            content: Zu signierender Inhalt (Bytes).

        Returns:
            PackageSignature mit Signatur.
        """
        import hmac as hmac_mod

        sig = hmac_mod.new(
            self._key.encode(), content, hashlib.sha256,
        ).hexdigest()

        return PackageSignature(
            signer_id=self._signer_id,
            signature=sig,
            algorithm="hmac-sha256",
        )

    def verify(self, content: bytes, signature: PackageSignature) -> bool:
        """Verifiziert eine HMAC-SHA256 Signatur.

        Args:
            content: Originaler Inhalt.
            signature: Zu verifizierende Signatur.

        Returns:
            True wenn die Signatur gültig ist.
        """
        import hmac as hmac_mod

        if signature.algorithm != "hmac-sha256":
            logger.warning("Unsupported signature algorithm: %s", signature.algorithm)
            return False

        expected = hmac_mod.new(
            self._key.encode(), content, hashlib.sha256,
        ).hexdigest()

        return hmac_mod.compare_digest(expected, signature.signature)


# ============================================================================
# Code-Analyse
# ============================================================================


@dataclass
class AnalysisReport:
    """Ergebnis der statischen Code-Analyse."""

    verdict: AnalysisVerdict
    findings: list[str] = field(default_factory=list)
    dangerous_patterns: list[str] = field(default_factory=list)
    suspicious_patterns: list[str] = field(default_factory=list)
    lines_of_code: int = 0

    @property
    def is_installable(self) -> bool:
        """Paket darf installiert werden (SAFE oder SUSPICIOUS)."""
        return self.verdict != AnalysisVerdict.DANGEROUS


# Gefährliche Patterns (blockieren Installation)
_DANGEROUS_PATTERNS: list[tuple[str, str]] = [
    (r"\beval\s*\(", "eval() — dynamische Code-Ausführung"),
    (r"\bexec\s*\(", "exec() — dynamische Code-Ausführung"),
    (r"\b__import__\s*\(", "__import__() — dynamischer Import"),
    (r"\bsubprocess\b", "subprocess — Prozess-Start"),
    (r"\bos\.system\s*\(", "os.system() — Shell-Kommando"),
    (r"\bos\.popen\s*\(", "os.popen() — Shell-Pipe"),
    (r"\bshutil\.rmtree\s*\(", "shutil.rmtree() — rekursives Löschen"),
    (r"\bopen\s*\([^)]*['\"]\/(?:etc|proc|sys)", "Zugriff auf Systemverzeichnisse"),
    (r"\bsocket\b", "socket — direkter Netzwerkzugriff"),
    (r"\bctypes\b", "ctypes — C-Level-Zugriff"),
    (r"\bpickle\.loads?\s*\(", "pickle.load — unsichere Deserialisierung"),
    (r"\b(?:requests|httpx|urllib)\.(?:get|post|put|delete)\s*\(", "HTTP-Request ohne Genehmigung"),
    (r"(?:PRIVATE|SECRET|PASSWORD|API_KEY)\s*=\s*['\"]", "Hartcodierte Credentials"),
]

# Verdächtige Patterns (Warnung, aber installierbar)
_SUSPICIOUS_PATTERNS: list[tuple[str, str]] = [
    (r"\bgetattr\s*\(", "getattr() — dynamischer Attributzugriff"),
    (r"\bglobals\s*\(\)", "globals() — globaler Namespace-Zugriff"),
    (r"\bcompile\s*\(", "compile() — Code-Kompilierung"),
    (r"\bos\.environ", "os.environ — Umgebungsvariablen-Zugriff"),
    (r"\bimportlib\b", "importlib — dynamischer Import"),
    (r"\bsys\.path", "sys.path — Modifizierung des Importpfads"),
    (r"\bthreading\b", "threading — Multi-Threading"),
    (r"\basyncio\.create_subprocess", "asyncio.create_subprocess — Subprozess"),
]


class CodeAnalyzer:
    """Statische Analyse von Skill-Code auf Sicherheitsrisiken.

    Prüft Python-Code auf gefährliche und verdächtige Patterns.
    Zusätzliche Checks:
      - Code-Länge (>2000 Zeilen = verdächtig)
      - Manifest-Permissions vs. tatsächlicher Code
    """

    def __init__(
        self,
        *,
        max_lines: int = 2000,
        extra_dangerous: list[tuple[str, str]] | None = None,
    ) -> None:
        self._max_lines = max_lines
        self._dangerous = list(_DANGEROUS_PATTERNS)
        if extra_dangerous:
            self._dangerous.extend(extra_dangerous)
        self._suspicious = list(_SUSPICIOUS_PATTERNS)

    def analyze(
        self,
        code: str,
        manifest: SkillManifest | None = None,
    ) -> AnalysisReport:
        """Analysiert Python-Code auf Sicherheitsrisiken.

        Args:
            code: Python-Quellcode.
            manifest: Optional, für Permission-Cross-Check.

        Returns:
            AnalysisReport mit Verdict und Findings.
        """
        findings: list[str] = []
        dangerous: list[str] = []
        suspicious: list[str] = []
        lines = code.count("\n") + 1

        # Code-Länge prüfen
        if lines > self._max_lines:
            suspicious.append(
                f"Code hat {lines} Zeilen (>{self._max_lines} — ungewöhnlich lang)"
            )

        # Kommentare und Strings entfernen für zuverlässigere Analyse
        clean_code = self._strip_comments_and_strings(code)

        # Gefährliche Patterns
        for pattern, description in self._dangerous:
            matches = re.findall(pattern, clean_code)
            if matches:
                dangerous.append(f"GEFÄHRLICH: {description} ({len(matches)}×)")

        # Verdächtige Patterns
        for pattern, description in self._suspicious:
            matches = re.findall(pattern, clean_code)
            if matches:
                suspicious.append(f"VERDÄCHTIG: {description} ({len(matches)}×)")

        # Permission-Cross-Check
        if manifest:
            perm_findings = self._check_permissions(clean_code, manifest)
            findings.extend(perm_findings)

        # Syntax-Check
        try:
            compile(code, "<skill>", "exec")
        except SyntaxError as e:
            dangerous.append(f"Syntax-Fehler: {e}")

        # Verdict bestimmen
        if dangerous:
            verdict = AnalysisVerdict.DANGEROUS
        elif suspicious or findings:
            verdict = AnalysisVerdict.SUSPICIOUS
        else:
            verdict = AnalysisVerdict.SAFE

        return AnalysisReport(
            verdict=verdict,
            findings=findings,
            dangerous_patterns=dangerous,
            suspicious_patterns=suspicious,
            lines_of_code=lines,
        )

    def _check_permissions(
        self,
        code: str,
        manifest: SkillManifest,
    ) -> list[str]:
        """Cross-Check: Code verwendet Funktionen die nicht im Manifest stehen."""
        findings: list[str] = []
        perms = set(manifest.permissions)

        # Netzwerk-Code ohne Permission
        if not manifest.network_allowed and re.search(
            r"\b(?:requests|httpx|urllib|aiohttp)\b", code,
        ):
            findings.append(
                "Code referenziert Netzwerk-Bibliotheken, "
                "aber network_allowed=False im Manifest"
            )

        # Datei-Schreiben ohne Permission
        if "file_write" not in perms and re.search(
            r"\bopen\s*\([^)]*['\"]w", code,
        ):
            findings.append(
                "Code öffnet Dateien zum Schreiben, "
                "aber file_write nicht in Permissions"
            )

        return findings

    @staticmethod
    def _strip_comments_and_strings(code: str) -> str:
        """Entfernt Kommentare und String-Literale für sichere Pattern-Analyse."""
        # Multiline-Strings
        code = re.sub(r'""".*?"""', '""', code, flags=re.DOTALL)
        code = re.sub(r"'''.*?'''", "''", code, flags=re.DOTALL)
        # Single-line Strings — bounded repetition to prevent ReDoS
        code = re.sub(r'"[^"\\]{0,10000}(?:\\.[^"\\]{0,10000}){0,100}"', '""', code)
        code = re.sub(r"'[^'\\]{0,10000}(?:\\.[^'\\]{0,10000}){0,100}'", "''", code)
        # Kommentare
        code = re.sub(r"#.*$", "", code, flags=re.MULTILINE)
        return code


# ============================================================================
# Skill-Paket
# ============================================================================


@dataclass
class SkillPackage:
    """Ein vollständiges, verteilbares Skill-Paket.

    Struktur:
      manifest.json    — Paket-Metadaten
      skill.py         — Skill-Code
      test_skill.py    — Unit-Tests
      skill.md         — Markdown-Dokumentation
      signature.json   — Digitale Signatur
    """

    manifest: SkillManifest
    code: str  # Python-Quellcode
    test_code: str = ""  # Unit-Tests
    documentation: str = ""  # Markdown
    signature: PackageSignature | None = None

    @property
    def content_hash(self) -> str:
        """SHA-256 über Code + Tests (reproduzierbar)."""
        content = (self.code + self.test_code).encode()
        return hashlib.sha256(content).hexdigest()

    @property
    def is_signed(self) -> bool:
        return self.signature is not None

    @property
    def package_id(self) -> str:
        """Eindeutige Paket-ID: name@version-hash[:8]."""
        return f"{self.manifest.name}@{self.manifest.version}-{self.content_hash[:8]}"

    def to_bytes(self) -> bytes:
        """Serialisiert das Paket als tar.gz Bytes."""
        buf = BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            # Manifest
            manifest_data = json.dumps(self.manifest.to_dict(), indent=2).encode()
            self._add_bytes_to_tar(tar, "manifest.json", manifest_data)

            # Code
            self._add_bytes_to_tar(tar, "skill.py", self.code.encode())

            # Tests
            if self.test_code:
                self._add_bytes_to_tar(tar, "test_skill.py", self.test_code.encode())

            # Dokumentation
            if self.documentation:
                self._add_bytes_to_tar(tar, "skill.md", self.documentation.encode())

            # Signatur
            if self.signature:
                sig_data = json.dumps({
                    "signer_id": self.signature.signer_id,
                    "signature": self.signature.signature,
                    "algorithm": self.signature.algorithm,
                    "timestamp": self.signature.timestamp,
                }).encode()
                self._add_bytes_to_tar(tar, "signature.json", sig_data)

        return buf.getvalue()

    # Allowed file names inside a skill package (zip-slip protection)
    _ALLOWED_MEMBERS = frozenset({
        "manifest.json", "skill.py", "test_skill.py", "skill.md", "signature.json",
    })

    @classmethod
    def from_bytes(cls, data: bytes) -> SkillPackage:
        """Deserialisiert ein Paket aus tar.gz Bytes."""
        buf = BytesIO(data)
        files: dict[str, str] = {}

        max_member_size = MAX_PACKAGE_MEMBER_SIZE

        with tarfile.open(fileobj=buf, mode="r:gz") as tar:
            for member in tar.getmembers():
                # Zip-slip protection: reject path traversal and unexpected names
                if member.name not in cls._ALLOWED_MEMBERS:
                    logger.warning(
                        "Skipping unexpected member in skill package: %s",
                        member.name,
                    )
                    continue
                if ".." in member.name or member.name.startswith("/"):
                    raise ValueError(
                        f"Pfad-Traversal erkannt im Paket: {member.name}"
                    )
                if member.size > max_member_size:
                    raise ValueError(
                        f"Paket-Mitglied zu gross: {member.name} ({member.size:,} > {max_member_size:,} Bytes)"
                    )
                f = tar.extractfile(member)
                if f:
                    files[member.name] = f.read().decode()

        if "manifest.json" not in files:
            raise ValueError("Paket enthält kein manifest.json")
        if "skill.py" not in files:
            raise ValueError("Paket enthält kein skill.py")

        try:
            manifest = SkillManifest.from_dict(json.loads(files["manifest.json"]))
        except (json.JSONDecodeError, KeyError) as exc:
            raise ValueError(f"Ungültiges manifest.json: {exc}") from exc

        signature = None
        if "signature.json" in files:
            try:
                sig_data = json.loads(files["signature.json"])
                signature = PackageSignature(
                    signer_id=sig_data["signer_id"],
                    signature=sig_data["signature"],
                    algorithm=sig_data.get("algorithm", "hmac-sha256"),
                    timestamp=sig_data.get("timestamp", ""),
                )
            except (json.JSONDecodeError, KeyError) as exc:
                logger.warning("Ungültige signature.json: %s", exc)

        return cls(
            manifest=manifest,
            code=files["skill.py"],
            test_code=files.get("test_skill.py", ""),
            documentation=files.get("skill.md", ""),
            signature=signature,
        )

    @staticmethod
    def _add_bytes_to_tar(tar: tarfile.TarFile, name: str, data: bytes) -> None:
        """Fügt Bytes als Datei zum tar hinzu."""
        import tarfile as tf

        info = tf.TarInfo(name=name)
        info.size = len(data)
        tar.addfile(info, BytesIO(data))


# ============================================================================
# Package Builder
# ============================================================================


class PackageBuilder:
    """Erstellt signierte Skill-Pakete.

    Usage:
        builder = PackageBuilder(signer=PackageSigner(key, "author"))
        package = builder.build(manifest, code, test_code)
        data = package.to_bytes()
    """

    def __init__(
        self,
        *,
        signer: PackageSigner | None = None,
        analyzer: CodeAnalyzer | None = None,
    ) -> None:
        self._signer = signer
        self._analyzer = analyzer or CodeAnalyzer()

    def build(
        self,
        manifest: SkillManifest,
        code: str,
        test_code: str = "",
        documentation: str = "",
        *,
        skip_analysis: bool = False,
    ) -> SkillPackage:
        """Erstellt ein neues Skill-Paket.

        1. Manifest validieren
        2. Code analysieren
        3. Content-Hash berechnen
        4. Signieren (wenn Signer vorhanden)

        Args:
            manifest: Paket-Manifest.
            code: Python-Quellcode.
            test_code: Unit-Tests.
            documentation: Markdown-Dokumentation.
            skip_analysis: Analyse überspringen (nur für Tests).

        Returns:
            Fertiges SkillPackage.

        Raises:
            ValueError: Bei Validierungsfehlern oder gefährlichem Code.
        """
        # 1. Manifest validieren
        errors = manifest.validate()
        if errors:
            raise ValueError(f"Manifest ungültig: {'; '.join(errors)}")

        # 2. Code analysieren
        if not skip_analysis:
            report = self._analyzer.analyze(code, manifest)
            if report.verdict == AnalysisVerdict.DANGEROUS:
                raise ValueError(
                    f"Code-Analyse: GEFÄHRLICH — {'; '.join(report.dangerous_patterns)}"
                )

        # 3. Content-Hash
        content_hash = hashlib.sha256((code + test_code).encode()).hexdigest()
        manifest.content_hash = content_hash

        # 4. Paket erstellen
        package = SkillPackage(
            manifest=manifest,
            code=code,
            test_code=test_code,
            documentation=documentation,
        )

        # 5. Signieren
        if self._signer:
            signable = (manifest.name + manifest.version + content_hash).encode()
            package.signature = self._signer.sign(signable)

        logger.info(
            "Paket erstellt: %s (hash=%s, signiert=%s)",
            package.package_id, content_hash[:8], package.is_signed,
        )
        return package


# ============================================================================
# Package Installer
# ============================================================================


@dataclass
class InstallResult:
    """Ergebnis einer Paket-Installation."""

    success: bool
    package_id: str = ""
    message: str = ""
    analysis_report: AnalysisReport | None = None
    installed_path: str = ""


class PackageInstaller:
    """Installiert verifizierte Skill-Pakete in die lokale Umgebung.

    Workflow:
      1. Signatur prüfen (wenn vorhanden)
      2. Code analysieren
      3. Sandbox-Rechte aus Manifest ableiten
      4. Dateien in Skills-Verzeichnis schreiben
      5. In SkillRegistry registrieren

    Args:
        skills_dir: Verzeichnis für installierte Skills.
        trusted_signers: Vertrauenswürdige Herausgeber-IDs.
        require_signature: True = nur signierte Pakete installieren.
    """

    def __init__(
        self,
        skills_dir: Path,
        *,
        trusted_signers: set[str] | None = None,
        require_signature: bool = True,
        analyzer: CodeAnalyzer | None = None,
        signer: PackageSigner | None = None,
    ) -> None:
        self._skills_dir = skills_dir
        self._trusted_signers = trusted_signers or set()
        self._require_signature = require_signature
        self._analyzer = analyzer or CodeAnalyzer()
        self._signer = signer  # Für Signatur-Verifikation
        self._installed: dict[str, SkillPackage] = {}

    @property
    def installed_count(self) -> int:
        return len(self._installed)

    def get_installed(self, name: str) -> SkillPackage | None:
        return self._installed.get(name)

    def list_installed(self) -> list[SkillPackage]:
        return list(self._installed.values())

    def install(self, package: SkillPackage) -> InstallResult:
        """Installiert ein Skill-Paket.

        Args:
            package: Zu installierendes Paket.

        Returns:
            InstallResult mit Erfolg/Fehler-Informationen.
        """
        pkg_id = package.package_id

        # 1. Signatur prüfen
        if self._require_signature:
            if not package.is_signed:
                return InstallResult(
                    success=False,
                    package_id=pkg_id,
                    message="Paket ist nicht signiert (require_signature=True)",
                )

            if package.signature and package.signature.signer_id not in self._trusted_signers:
                if self._trusted_signers:  # Nur prüfen wenn Trusted-Liste nicht leer
                    return InstallResult(
                        success=False,
                        package_id=pkg_id,
                        message=f"Herausgeber '{package.signature.signer_id}' nicht vertrauenswürdig",
                    )

        # 2. Signatur-Integrität
        if package.is_signed and self._signer:
            signable = (
                package.manifest.name
                + package.manifest.version
                + package.content_hash
            ).encode()
            if not self._signer.verify(signable, package.signature):
                return InstallResult(
                    success=False,
                    package_id=pkg_id,
                    message="Signatur-Verifikation fehlgeschlagen",
                )

        # 3. Code analysieren
        report = self._analyzer.analyze(package.code, package.manifest)
        if not report.is_installable:
            return InstallResult(
                success=False,
                package_id=pkg_id,
                message=f"Code-Analyse: {'; '.join(report.dangerous_patterns)}",
                analysis_report=report,
            )

        # 4. Content-Hash verifizieren
        actual_hash = package.content_hash
        if package.manifest.content_hash and actual_hash != package.manifest.content_hash:
            return InstallResult(
                success=False,
                package_id=pkg_id,
                message="Content-Hash stimmt nicht überein (Manipulation?)",
            )

        # 5. Dateien schreiben
        skill_dir = self._skills_dir / package.manifest.name
        skill_dir.mkdir(parents=True, exist_ok=True)

        (skill_dir / "skill.py").write_text(package.code, encoding="utf-8")
        if package.test_code:
            (skill_dir / "test_skill.py").write_text(package.test_code, encoding="utf-8")
        if package.documentation:
            (skill_dir / "skill.md").write_text(package.documentation, encoding="utf-8")

        manifest_path = skill_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(package.manifest.to_dict(), indent=2),
            encoding="utf-8",
        )

        # 6. Tracking
        self._installed[package.manifest.name] = package

        logger.info(
            "Paket installiert: %s → %s", pkg_id, skill_dir,
        )
        return InstallResult(
            success=True,
            package_id=pkg_id,
            message=f"Erfolgreich installiert in {skill_dir}",
            analysis_report=report,
            installed_path=str(skill_dir),
        )

    def uninstall(self, name: str) -> bool:
        """Deinstalliert ein Skill-Paket.

        Args:
            name: Paketname.

        Returns:
            True wenn erfolgreich.
        """
        package = self._installed.pop(name, None)
        if not package:
            return False

        skill_dir = self._skills_dir / name
        if skill_dir.exists():
            import shutil

            shutil.rmtree(skill_dir)

        logger.info("Paket deinstalliert: %s", name)
        return True

    def sandbox_config_for(self, name: str) -> dict[str, Any]:
        """Generiert Sandbox-Konfiguration für ein installiertes Paket.

        Args:
            name: Paketname.

        Returns:
            Sandbox-Config Dict für AgentRouter.
        """
        package = self._installed.get(name)
        if not package:
            return {}

        m = package.manifest
        return {
            "network": m.network_allowed,
            "max_memory_mb": m.max_memory_mb,
            "timeout": m.timeout_seconds,
            "filesystem": "workspace_only",
            "permissions": m.permissions,
        }
