#!/usr/bin/env python3
"""Jarvis · Smoke-Test – Validiert die Installation.

Prüft:
  1. Python-Version + kritische Imports
  2. Verzeichnisstruktur (~/.jarvis/)
  3. Config laden + validieren
  4. Ollama-Verbindung + Modelle
  5. Memory-System Initialisierung
  6. Gatekeeper + Policies laden
  7. MCP-Tools registrieren
  8. Credential-Store Verschlüsselung
  9. Audit-Trail schreiben + verifizieren
  10. Gateway-Instanziierung

Exit-Code: 0=OK, 1=Kritisch, 2=Warnungen
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import tempfile
import time
from pathlib import Path

# Farben
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def ok(msg: str) -> None:
    print(f"  {GREEN}✓{RESET} {msg}")


def warn(msg: str) -> None:
    print(f"  {YELLOW}⚠{RESET} {msg}")


def fail(msg: str) -> None:
    print(f"  {RED}✗{RESET} {msg}")


def header(msg: str) -> None:
    print(f"\n{BOLD}{CYAN}── {msg} ──{RESET}")


class SmokeTest:
    def __init__(self, jarvis_home: str, ollama_url: str, venv: str) -> None:
        self.jarvis_home = Path(jarvis_home)
        self.ollama_url = ollama_url
        self.venv = Path(venv) if venv else None
        self.passed = 0
        self.warned = 0
        self.failed = 0

    def _pass(self, msg: str) -> None:
        ok(msg)
        self.passed += 1

    def _warn(self, msg: str) -> None:
        warn(msg)
        self.warned += 1

    def _fail(self, msg: str) -> None:
        fail(msg)
        self.failed += 1

    def test_python_imports(self) -> None:
        header("1. Python + Imports")
        v = sys.version_info
        if v >= (3, 12):
            self._pass(f"Python {v.major}.{v.minor}.{v.micro}")
        else:
            self._fail(f"Python {v.major}.{v.minor} – mindestens 3.12 benötigt")
            return

        for module, desc in [
            ("pydantic", "Datenvalidierung"),
            ("httpx", "HTTP-Client"),
            ("yaml", "Config (PyYAML)"),
            ("structlog", "Logging"),
            ("rich", "Terminal-UI"),
            ("prompt_toolkit", "CLI-Input"),
        ]:
            try:
                __import__(module)
                self._pass(f"{module} ({desc})")
            except ImportError:
                self._fail(f"{module} fehlt")

        try:
            import jarvis

            self._pass(f"jarvis v{jarvis.__version__}")
        except ImportError:
            self._fail("jarvis nicht importierbar")

        for module, desc in [
            ("fastapi", "Web-UI"),
            ("cryptography", "Verschlüsselung"),
            ("numpy", "Embeddings"),
        ]:
            try:
                __import__(module)
                self._pass(f"{module} ({desc})")
            except ImportError:
                self._warn(f"{module} fehlt – {desc} nicht verfügbar")

    def test_directories(self) -> None:
        header("2. Verzeichnisstruktur")
        if not self.jarvis_home.exists():
            self._warn(f"{self.jarvis_home} fehlt – wird beim Start erstellt")
            return
        self._pass(f"JARVIS_HOME: {self.jarvis_home}")
        for d in [
            "memory",
            "memory/episodes",
            "memory/knowledge",
            "memory/procedures",
            "memory/index",
            "logs",
        ]:
            if (self.jarvis_home / d).exists():
                self._pass(f"  {d}/")
            else:
                self._warn(f"  {d}/ fehlt")
        if (self.jarvis_home / "config.yaml").exists():
            self._pass("config.yaml vorhanden")
        else:
            self._warn("config.yaml fehlt – Defaults werden verwendet")

    def test_config(self) -> None:
        header("3. Konfiguration")
        try:
            from jarvis.config import load_config

            config = load_config(self.jarvis_home / "config.yaml")
            self._pass(f"Config geladen (v{config.version})")
            self._pass(f"Ollama: {config.ollama.base_url}")
            self._pass(f"Planner: {config.models.planner.name}")
            self._pass(f"Executor: {config.models.executor.name}")
            self._pass(f"Embedding: {config.models.embedding.name}")
        except Exception as exc:
            self._fail(f"Config-Fehler: {exc}")

    def test_ollama(self) -> None:
        header("4. Ollama-Verbindung")
        import httpx

        try:
            resp = httpx.get(f"{self.ollama_url}/api/version", timeout=5)
            if resp.status_code == 200:
                self._pass(f"Ollama erreichbar (v{resp.json().get('version', '?')})")
            else:
                self._warn(f"Ollama Status {resp.status_code}")
                return
        except Exception:
            self._warn(f"Ollama nicht erreichbar auf {self.ollama_url}")
            return

        try:
            resp = httpx.get(f"{self.ollama_url}/api/tags", timeout=10)
            models = [m["name"] for m in resp.json().get("models", [])]
            self._pass(f"{len(models)} Modell(e) installiert")
            for m in models[:10]:
                self._pass(f"  {m}")
            for req in ["qwen3", "nomic-embed"]:
                if not any(req in m for m in models):
                    self._warn(f"Empfohlenes Modell fehlt: {req}*")
        except Exception as exc:
            self._warn(f"Modell-Abfrage fehlgeschlagen: {exc}")

    def test_memory(self) -> None:
        header("5. Memory-System")
        try:
            from jarvis.config import JarvisConfig, ensure_directory_structure
            from jarvis.memory.manager import MemoryManager

            config = JarvisConfig(jarvis_home=self.jarvis_home)
            ensure_directory_structure(config)
            manager = MemoryManager(config)
            stats = manager.initialize()
            self._pass(
                f"Memory initialisiert (Chunks: {stats.get('chunks', 0)}, Entities: {stats.get('entities', 0)})"
            )
            manager.close_sync()
        except Exception as exc:
            self._fail(f"Memory-Fehler: {exc}")

    def test_gatekeeper(self) -> None:
        header("6. Gatekeeper")
        try:
            from jarvis.config import JarvisConfig
            from jarvis.core.gatekeeper import Gatekeeper
            from jarvis.models import PlannedAction, SessionContext, GateStatus

            config = JarvisConfig(jarvis_home=self.jarvis_home)
            gk = Gatekeeper(config)
            gk.initialize()
            self._pass("Gatekeeper initialisiert")

            action = PlannedAction(
                tool="exec_command", params={"command": "rm -rf /"}, rationale="test"
            )
            session = SessionContext(user_id="smoke-test", channel="cli")
            decision = gk.evaluate(action, session)
            if decision.status == GateStatus.BLOCK:
                self._pass("Destruktiver Befehl korrekt geblockt")
            else:
                self._fail(f"rm -rf / NICHT geblockt: {decision.status}")
        except Exception as exc:
            self._fail(f"Gatekeeper-Fehler: {exc}")

    def test_mcp_tools(self) -> None:
        header("7. MCP-Tools")
        try:
            from jarvis.config import JarvisConfig
            from jarvis.mcp.client import JarvisMCPClient
            from jarvis.mcp.filesystem import register_fs_tools
            from jarvis.mcp.shell import register_shell_tools
            from jarvis.mcp.web import register_web_tools

            config = JarvisConfig(jarvis_home=self.jarvis_home)
            mcp = JarvisMCPClient(config)
            register_fs_tools(mcp, config)
            register_shell_tools(mcp, config)
            register_web_tools(mcp, config)
            tools = mcp.get_tool_list()
            self._pass(f"{len(tools)} Tools: {', '.join(sorted(tools))}")
        except Exception as exc:
            self._fail(f"MCP-Fehler: {exc}")

    def test_credentials(self) -> None:
        header("8. Credential-Store")
        try:
            from jarvis.security.credentials import CredentialStore

            with tempfile.NamedTemporaryFile(suffix=".enc", delete=False) as tmp:
                store = CredentialStore(store_path=Path(tmp.name), passphrase="smoke-test-2026")
                store.store("test", "key", "geheim123")
                result = store.retrieve("test", "key")
                if result == "geheim123":
                    self._pass("Verschlüsselung + Entschlüsselung OK")
                else:
                    self._fail("Entschlüsselung fehlgeschlagen")
                raw = Path(tmp.name).read_text(encoding="utf-8")
                if "geheim123" not in raw:
                    self._pass("Datei korrekt verschlüsselt")
                else:
                    self._fail("Credential im Klartext!")
                Path(tmp.name).unlink()
        except ImportError:
            self._warn("cryptography fehlt")
        except Exception as exc:
            self._fail(f"Credential-Fehler: {exc}")

    def test_audit(self) -> None:
        header("9. Audit-Trail")
        try:
            from jarvis.security.audit import AuditTrail
            from jarvis.models import AuditEntry, GateStatus, RiskLevel

            with tempfile.TemporaryDirectory() as tmpdir:
                audit = AuditTrail(log_dir=Path(tmpdir))
                for i in range(3):
                    audit.record(
                        AuditEntry(
                            session_id="smoke",
                            action_tool=f"t-{i}",
                            action_params_hash=hashlib.sha256(f"p{i}".encode()).hexdigest(),
                            decision_status=GateStatus.ALLOW,
                            decision_reason="test",
                            risk_level=RiskLevel.GREEN,
                        )
                    )
                valid, total, _ = audit.verify_chain()
                if valid and total == 3:
                    self._pass("3 Einträge geschrieben, Hash-Chain intakt")
                else:
                    self._fail(f"Chain defekt: valid={valid}, total={total}")
        except Exception as exc:
            self._fail(f"Audit-Fehler: {exc}")

    def test_gateway(self) -> None:
        header("10. Gateway")
        try:
            from jarvis.config import JarvisConfig
            from jarvis.gateway.gateway import Gateway

            config = JarvisConfig(jarvis_home=self.jarvis_home)
            Gateway(config)
            self._pass("Gateway instanziiert")
        except Exception as exc:
            self._fail(f"Gateway-Fehler: {exc}")

    def summary(self) -> int:
        total = self.passed + self.warned + self.failed
        print(f"\n{BOLD}{'=' * 50}{RESET}")
        print(f"  {GREEN}✓ {self.passed} bestanden{RESET}")
        if self.warned:
            print(f"  {YELLOW}⚠ {self.warned} Warnungen{RESET}")
        if self.failed:
            print(f"  {RED}✗ {self.failed} fehlgeschlagen{RESET}")
        print(f"  Gesamt: {total} Prüfungen")
        print(f"{BOLD}{'=' * 50}{RESET}")
        if self.failed:
            print(f"\n{RED}{BOLD}  ✗ Jarvis kann nicht starten.{RESET}\n")
            return 1
        elif self.warned:
            print(f"\n{YELLOW}{BOLD}  ⚠ Jarvis startet, aber eingeschränkt.{RESET}\n")
            return 2
        else:
            print(f"\n{GREEN}{BOLD}  ✓ Jarvis ist bereit! Starte mit: jarvis{RESET}\n")
            return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Jarvis Smoke-Test")
    parser.add_argument("--jarvis-home", default=str(Path.home() / ".jarvis"))
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--venv", default="")
    args = parser.parse_args()

    print(f"\n{BOLD}{CYAN}Jarvis · Smoke-Test{RESET}\n{'=' * 50}")
    st = SmokeTest(args.jarvis_home, args.ollama_url, args.venv)
    start = time.monotonic()
    st.test_python_imports()
    st.test_directories()
    st.test_config()
    st.test_ollama()
    st.test_memory()
    st.test_gatekeeper()
    st.test_mcp_tools()
    st.test_credentials()
    st.test_audit()
    st.test_gateway()
    print(f"\n  Dauer: {time.monotonic() - start:.1f}s")
    return st.summary()


if __name__ == "__main__":
    sys.exit(main())
