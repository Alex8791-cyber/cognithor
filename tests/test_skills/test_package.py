"""Tests für Skill Package: Manifest, Signierung, Code-Analyse, Build & Install.

Testet alle Sicherheitsaspekte der Skill-Distribution:
  - Manifest-Validierung (SemVer, Namen, Limits)
  - Digitale Signaturen (HMAC-SHA256, Verifikation)
  - Code-Analyse (gefährliche/verdächtige Patterns)
  - Paket-Serialisierung (tar.gz Round-Trip)
  - Package Builder (Build + Sign + Analyse)
  - Package Installer (Verify + Analyse + Install + Sandbox-Config)
"""

from __future__ import annotations

from pathlib import Path

import pytest

from jarvis.skills.package import (
    AnalysisReport,
    AnalysisVerdict,
    CodeAnalyzer,
    InstallResult,
    PackageBuilder,
    PackageInstaller,
    PackageSigner,
    PackageSignature,
    SandboxPermission,
    SkillManifest,
    SkillPackage,
    TrustLevel,
)


# ============================================================================
# SkillManifest
# ============================================================================


class TestSkillManifest:
    """Manifest-Datenmodell und Validierung."""

    def test_valid_manifest(self) -> None:
        m = SkillManifest(
            name="bu_vergleich",
            version="1.0.0",
            description="BU-Tarifvergleich",
            author="Alexander",
        )
        assert m.validate() == []
        assert m.qualified_name == "bu_vergleich@1.0.0"

    def test_invalid_name_special_chars(self) -> None:
        m = SkillManifest(name="BU-Vergleich!", version="1.0.0", description="x", author="a")
        errors = m.validate()
        assert len(errors) >= 1
        assert "Paketname" in errors[0]

    def test_invalid_name_too_short(self) -> None:
        m = SkillManifest(name="ab", version="1.0.0", description="x", author="a")
        assert len(m.validate()) >= 1

    def test_invalid_version(self) -> None:
        m = SkillManifest(name="valid_name", version="v1", description="x", author="a")
        errors = m.validate()
        assert any("Version" in e for e in errors)

    def test_description_too_long(self) -> None:
        m = SkillManifest(
            name="valid_name", version="1.0.0",
            description="x" * 201, author="a",
        )
        assert any("Beschreibung" in e for e in m.validate())

    def test_missing_author(self) -> None:
        m = SkillManifest(name="valid_name", version="1.0.0", description="x", author="")
        assert any("Autor" in e for e in m.validate())

    def test_memory_limit(self) -> None:
        m = SkillManifest(
            name="valid_name", version="1.0.0", description="x",
            author="a", max_memory_mb=2000,
        )
        assert any("max_memory_mb" in e for e in m.validate())

    def test_timeout_limit(self) -> None:
        m = SkillManifest(
            name="valid_name", version="1.0.0", description="x",
            author="a", timeout_seconds=500,
        )
        assert any("timeout_seconds" in e for e in m.validate())

    def test_serialization_roundtrip(self) -> None:
        m = SkillManifest(
            name="test_skill", version="2.1.0",
            description="Ein Test", author="Tester",
            trigger_keywords=["test", "prüfung"],
            tools_required=["memory_search"],
            permissions=["file_read", "memory_access"],
        )
        d = m.to_dict()
        m2 = SkillManifest.from_dict(d)
        assert m2.name == m.name
        assert m2.version == m.version
        assert m2.trigger_keywords == m.trigger_keywords
        assert m2.permissions == m.permissions

    def test_parsed_permissions(self) -> None:
        m = SkillManifest(
            name="test_skill", version="1.0.0", description="x",
            author="a", permissions=["file_read", "network", "invalid"],
        )
        perms = m.parsed_permissions
        assert SandboxPermission.FILE_READ in perms
        assert SandboxPermission.NETWORK in perms
        assert len(perms) == 2  # invalid wird ignoriert


# ============================================================================
# PackageSigner
# ============================================================================


class TestPackageSigner:
    """Ed25519-kompatible Signierung."""

    def test_sign_and_verify(self) -> None:
        signer = PackageSigner("my_secret_key_123", "author_1")
        content = b"Hello World Skill Code"

        sig = signer.sign(content)
        assert sig.signer_id == "author_1"
        assert sig.algorithm == "hmac-sha256"
        assert len(sig.signature) == 64  # SHA-256 hex

        assert signer.verify(content, sig) is True

    def test_verify_tampered_content(self) -> None:
        signer = PackageSigner("secret")
        content = b"Original"
        sig = signer.sign(content)

        assert signer.verify(b"Tampered", sig) is False

    def test_verify_wrong_key(self) -> None:
        signer1 = PackageSigner("key_1")
        signer2 = PackageSigner("key_2")

        content = b"Content"
        sig = signer1.sign(content)

        assert signer2.verify(content, sig) is False

    def test_auto_signer_id(self) -> None:
        signer = PackageSigner("auto_key")
        assert len(signer.signer_id) == 16

    def test_signature_timestamp(self) -> None:
        signer = PackageSigner("key")
        sig = signer.sign(b"data")
        assert sig.timestamp  # Nicht leer


# ============================================================================
# CodeAnalyzer
# ============================================================================


class TestCodeAnalyzer:
    """Statische Code-Analyse auf Sicherheitsrisiken."""

    @pytest.fixture
    def analyzer(self) -> CodeAnalyzer:
        return CodeAnalyzer()

    def test_safe_code(self, analyzer: CodeAnalyzer) -> None:
        code = '''
async def handler(query: str) -> str:
    """Einfacher Handler."""
    return f"Ergebnis für {query}"
'''
        report = analyzer.analyze(code)
        assert report.verdict == AnalysisVerdict.SAFE
        assert report.is_installable

    def test_dangerous_eval(self, analyzer: CodeAnalyzer) -> None:
        code = 'result = eval(user_input)'
        report = analyzer.analyze(code)
        assert report.verdict == AnalysisVerdict.DANGEROUS
        assert not report.is_installable
        assert any("eval" in p for p in report.dangerous_patterns)

    def test_dangerous_exec(self, analyzer: CodeAnalyzer) -> None:
        report = analyzer.analyze('exec("os.system(\'rm -rf /\')")')
        assert report.verdict == AnalysisVerdict.DANGEROUS

    def test_dangerous_subprocess(self, analyzer: CodeAnalyzer) -> None:
        report = analyzer.analyze('import subprocess\nsubprocess.run(["ls"])')
        assert report.verdict == AnalysisVerdict.DANGEROUS

    def test_dangerous_network(self, analyzer: CodeAnalyzer) -> None:
        report = analyzer.analyze('import socket\ns = socket.socket()')
        assert report.verdict == AnalysisVerdict.DANGEROUS

    def test_dangerous_pickle(self, analyzer: CodeAnalyzer) -> None:
        report = analyzer.analyze('import pickle\nobj = pickle.loads(data)')
        assert report.verdict == AnalysisVerdict.DANGEROUS

    def test_dangerous_credentials(self, analyzer: CodeAnalyzer) -> None:
        report = analyzer.analyze('API_KEY = "sk-12345abcde"')
        assert report.verdict == AnalysisVerdict.DANGEROUS

    def test_dangerous_http_request(self, analyzer: CodeAnalyzer) -> None:
        report = analyzer.analyze('import requests\nrequests.get("http://evil.com")')
        assert report.verdict == AnalysisVerdict.DANGEROUS

    def test_suspicious_getattr(self, analyzer: CodeAnalyzer) -> None:
        report = analyzer.analyze('val = getattr(obj, name)')
        assert report.verdict == AnalysisVerdict.SUSPICIOUS
        assert report.is_installable  # Verdächtig, aber installierbar

    def test_suspicious_globals(self, analyzer: CodeAnalyzer) -> None:
        report = analyzer.analyze('g = globals()')
        assert report.verdict == AnalysisVerdict.SUSPICIOUS

    def test_too_many_lines(self) -> None:
        analyzer = CodeAnalyzer(max_lines=10)
        code = "\n".join([f"line_{i} = {i}" for i in range(50)])
        report = analyzer.analyze(code)
        assert report.verdict == AnalysisVerdict.SUSPICIOUS

    def test_syntax_error(self, analyzer: CodeAnalyzer) -> None:
        report = analyzer.analyze("def broken(:\n  pass")
        assert report.verdict == AnalysisVerdict.DANGEROUS

    def test_permission_cross_check_network(self, analyzer: CodeAnalyzer) -> None:
        manifest = SkillManifest(
            name="test_skill", version="1.0.0", description="x",
            author="a", network_allowed=False,
        )
        # httpx ohne Netzwerk-Permission → im Comment, not in actual code patterns
        code = "import aiohttp\n# This code uses network"
        report = analyzer.analyze(code, manifest)
        # aiohttp reference triggers finding
        assert len(report.findings) >= 1

    def test_comments_not_flagged(self, analyzer: CodeAnalyzer) -> None:
        """Code in Kommentaren sollte nicht flagged werden."""
        code = '# eval() ist gefährlich\nresult = "safe"'
        report = analyzer.analyze(code)
        assert report.verdict == AnalysisVerdict.SAFE

    def test_strings_not_flagged(self, analyzer: CodeAnalyzer) -> None:
        """Strings die gefährliche Wörter enthalten sind sicher."""
        code = 'msg = "Bitte kein eval() verwenden"'
        report = analyzer.analyze(code)
        assert report.verdict == AnalysisVerdict.SAFE

    def test_lines_of_code_counted(self, analyzer: CodeAnalyzer) -> None:
        code = "a = 1\nb = 2\nc = 3"
        report = analyzer.analyze(code)
        assert report.lines_of_code == 3


# ============================================================================
# SkillPackage Serialization
# ============================================================================


class TestSkillPackage:
    """Paket-Serialisierung und Deserialisierung."""

    def test_roundtrip(self) -> None:
        manifest = SkillManifest(
            name="test_pkg", version="1.0.0",
            description="Test-Paket", author="Tester",
        )
        original = SkillPackage(
            manifest=manifest,
            code="async def handler(): return 'ok'",
            test_code="def test_ok(): assert True",
            documentation="# Test\nEin Test-Skill.",
        )

        data = original.to_bytes()
        restored = SkillPackage.from_bytes(data)

        assert restored.manifest.name == "test_pkg"
        assert restored.manifest.version == "1.0.0"
        assert restored.code == original.code
        assert restored.test_code == original.test_code
        assert restored.documentation == original.documentation

    def test_roundtrip_with_signature(self) -> None:
        signer = PackageSigner("key", "author")
        manifest = SkillManifest(
            name="signed_pkg", version="2.0.0",
            description="Signiert", author="author",
        )
        package = SkillPackage(
            manifest=manifest,
            code="pass",
            signature=signer.sign(b"content"),
        )

        data = package.to_bytes()
        restored = SkillPackage.from_bytes(data)

        assert restored.is_signed
        assert restored.signature.signer_id == "author"

    def test_content_hash_deterministic(self) -> None:
        manifest = SkillManifest(
            name="hash_test", version="1.0.0",
            description="x", author="a",
        )
        p1 = SkillPackage(manifest=manifest, code="code", test_code="test")
        p2 = SkillPackage(manifest=manifest, code="code", test_code="test")
        assert p1.content_hash == p2.content_hash

    def test_package_id_format(self) -> None:
        manifest = SkillManifest(
            name="my_skill", version="3.1.4",
            description="x", author="a",
        )
        p = SkillPackage(manifest=manifest, code="pass")
        assert p.package_id.startswith("my_skill@3.1.4-")

    def test_missing_manifest_raises(self) -> None:
        import io
        import tarfile

        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            data = b"print('hello')"
            info = tarfile.TarInfo(name="skill.py")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))

        with pytest.raises(ValueError, match="manifest.json"):
            SkillPackage.from_bytes(buf.getvalue())


# ============================================================================
# PackageBuilder
# ============================================================================


class TestPackageBuilder:
    """Paket-Erstellung mit Validierung und Signierung."""

    @pytest.fixture
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            name="build_test", version="1.0.0",
            description="Builder-Test", author="Builder",
        )

    def test_build_unsigned(self, manifest: SkillManifest) -> None:
        builder = PackageBuilder()
        code = "async def handler(): return 'ok'"
        pkg = builder.build(manifest, code)

        assert pkg.manifest.name == "build_test"
        assert not pkg.is_signed
        assert pkg.manifest.content_hash  # Hash wurde gesetzt

    def test_build_signed(self, manifest: SkillManifest) -> None:
        signer = PackageSigner("build_key", "builder")
        builder = PackageBuilder(signer=signer)
        pkg = builder.build(manifest, "pass")

        assert pkg.is_signed
        assert pkg.signature.signer_id == "builder"

    def test_build_rejects_invalid_manifest(self) -> None:
        builder = PackageBuilder()
        bad_manifest = SkillManifest(
            name="X", version="nope", description="x", author="",
        )
        with pytest.raises(ValueError, match="Manifest ungültig"):
            builder.build(bad_manifest, "pass")

    def test_build_rejects_dangerous_code(self, manifest: SkillManifest) -> None:
        builder = PackageBuilder()
        dangerous = "import subprocess\nsubprocess.call(['rm', '-rf', '/'])"
        with pytest.raises(ValueError, match="GEFÄHRLICH"):
            builder.build(manifest, dangerous)

    def test_build_skip_analysis(self, manifest: SkillManifest) -> None:
        builder = PackageBuilder()
        dangerous = "eval('bad')"
        # Mit skip_analysis sollte es klappen
        pkg = builder.build(manifest, dangerous, skip_analysis=True)
        assert pkg.code == dangerous


# ============================================================================
# PackageInstaller
# ============================================================================


class TestPackageInstaller:
    """Installation mit Signatur-Prüfung und Sandbox-Config."""

    @pytest.fixture
    def signer(self) -> PackageSigner:
        return PackageSigner("install_key", "trusted_author")

    @pytest.fixture
    def signed_package(self, signer: PackageSigner) -> SkillPackage:
        manifest = SkillManifest(
            name="install_test", version="1.0.0",
            description="Install-Test", author="Tester",
            permissions=["file_read"],
            max_memory_mb=64,
            timeout_seconds=10,
        )
        builder = PackageBuilder(signer=signer)
        return builder.build(manifest, "async def handler(): return 'ok'")

    def test_install_signed_package(
        self, tmp_path: Path, signer: PackageSigner, signed_package: SkillPackage,
    ) -> None:
        installer = PackageInstaller(
            tmp_path / "skills",
            trusted_signers={"trusted_author"},
            signer=signer,
        )
        result = installer.install(signed_package)

        assert result.success
        assert (tmp_path / "skills" / "install_test" / "skill.py").exists()
        assert (tmp_path / "skills" / "install_test" / "manifest.json").exists()

    def test_install_rejects_unsigned(self, tmp_path: Path) -> None:
        manifest = SkillManifest(
            name="unsigned_pkg", version="1.0.0",
            description="x", author="a",
        )
        pkg = SkillPackage(manifest=manifest, code="pass")

        installer = PackageInstaller(tmp_path / "skills", require_signature=True)
        result = installer.install(pkg)

        assert not result.success
        assert "nicht signiert" in result.message

    def test_install_rejects_untrusted_signer(
        self, tmp_path: Path, signed_package: SkillPackage,
    ) -> None:
        installer = PackageInstaller(
            tmp_path / "skills",
            trusted_signers={"other_author"},  # Nicht der Signer
        )
        result = installer.install(signed_package)
        assert not result.success
        assert "nicht vertrauenswürdig" in result.message

    def test_install_rejects_dangerous_code(
        self, tmp_path: Path, signer: PackageSigner,
    ) -> None:
        manifest = SkillManifest(
            name="evil_skill", version="1.0.0",
            description="Böse", author="Hacker",
        )
        # skip_analysis beim Build, aber Installer prüft nochmal
        pkg = SkillPackage(
            manifest=manifest,
            code="import subprocess\nsubprocess.call(['rm', '-rf', '/'])",
            signature=signer.sign(b"fake"),
        )

        installer = PackageInstaller(
            tmp_path / "skills",
            require_signature=False,
        )
        result = installer.install(pkg)
        assert not result.success

    def test_install_without_signature_requirement(self, tmp_path: Path) -> None:
        manifest = SkillManifest(
            name="no_sig_needed", version="1.0.0",
            description="x", author="a",
        )
        pkg = SkillPackage(manifest=manifest, code="x = 1")

        installer = PackageInstaller(tmp_path / "skills", require_signature=False)
        result = installer.install(pkg)
        assert result.success

    def test_uninstall(
        self, tmp_path: Path, signer: PackageSigner, signed_package: SkillPackage,
    ) -> None:
        installer = PackageInstaller(
            tmp_path / "skills",
            trusted_signers={"trusted_author"},
            signer=signer,
        )
        installer.install(signed_package)
        assert installer.installed_count == 1

        removed = installer.uninstall("install_test")
        assert removed
        assert installer.installed_count == 0
        assert not (tmp_path / "skills" / "install_test").exists()

    def test_sandbox_config(
        self, tmp_path: Path, signer: PackageSigner, signed_package: SkillPackage,
    ) -> None:
        installer = PackageInstaller(
            tmp_path / "skills",
            trusted_signers={"trusted_author"},
            signer=signer,
        )
        installer.install(signed_package)

        config = installer.sandbox_config_for("install_test")
        assert config["network"] is False
        assert config["max_memory_mb"] == 64
        assert config["timeout"] == 10
        assert "file_read" in config["permissions"]

    def test_sandbox_config_unknown(self, tmp_path: Path) -> None:
        installer = PackageInstaller(tmp_path / "skills", require_signature=False)
        assert installer.sandbox_config_for("nonexistent") == {}

    def test_list_installed(
        self, tmp_path: Path, signer: PackageSigner, signed_package: SkillPackage,
    ) -> None:
        installer = PackageInstaller(
            tmp_path / "skills",
            trusted_signers={"trusted_author"},
            signer=signer,
        )
        installer.install(signed_package)

        installed = installer.list_installed()
        assert len(installed) == 1
        assert installed[0].manifest.name == "install_test"
