#!/usr/bin/env python3
"""Translate German comments, docstrings, and log messages to English in mcp/ files.

Rules:
- Translate ONLY comments (#), docstrings (triple-quoted), and log message strings
- DO NOT translate system prompts for the LLM
- DO NOT translate user-facing error messages displayed to the user
- DO NOT translate i18n strings (t() calls)
- DO NOT translate variable names
- Log event KEYS stay as-is; only translate human-readable descriptions
"""

import re
from pathlib import Path

# Map of German -> English translations for comments, docstrings, and log strings
# These are applied as simple string replacements (exact match within the file)
TRANSLATIONS: list[tuple[str, str]] = [
    # ══════════════════════════════════════════════════════════════
    # __init__.py - already done via Edit tool
    # ══════════════════════════════════════════════════════════════

    # ══════════════════════════════════════════════════════════════
    # api_hub.py - module docstring already done via Edit tool
    # ══════════════════════════════════════════════════════════════

    # api_hub.py - class/method docstrings and comments
    ('"""Fehler bei API-Hub-Operationen."""', '"""Error during API hub operations."""'),
    ('"""Prueft ob ein Request erlaubt ist und registriert ihn."""', '"""Check if a request is allowed and register it."""'),
    ('"""Verbleibende Requests im aktuellen Fenster."""', '"""Remaining requests in the current window."""'),
    ('"""Gibt den Pfad zur integrations.json zurueck."""', '"""Return the path to integrations.json."""'),
    ('"""Gibt den Pfad zum Fernet-Key zurueck."""', '"""Return the path to the Fernet key."""'),
    ('"""Erstellt oder laedt einen Fernet-Key fuer Verschluesselung.\n\n    Returns:\n        Fernet-Instanz oder None wenn cryptography nicht verfuegbar.\n    """',
     '"""Create or load a Fernet key for encryption.\n\n    Returns:\n        Fernet instance or None if cryptography is not available.\n    """'),
    ('"""Laedt Integrationen aus der JSON-Datei (ggf. entschluesselt)."""', '"""Load integrations from the JSON file (decrypting if needed)."""'),
    ('# Versuche Fernet-Entschluesselung', '# Try Fernet decryption'),
    ('pass  # Fallthrough zu Plaintext', 'pass  # Fall through to plaintext'),
    ('# Plaintext-Fallback', '# Plaintext fallback'),
    ('"""Speichert Integrationen (ggf. verschluesselt)."""', '"""Save integrations (encrypting if possible)."""'),
    ('# Plaintext-Fallback mit Warnung', '# Plaintext fallback with warning'),
    ('"""Liest das Credential aus der angegebenen Umgebungsvariable.\n\n    Returns:\n        Credential-String oder None.\n    """',
     '"""Read the credential from the specified environment variable.\n\n    Returns:\n        Credential string or None.\n    """'),
    ('"""Baut Authentifizierungs-Header basierend auf auth_type.\n\n    Returns:\n        Dict mit Auth-Headern.\n    """',
     '"""Build authentication headers based on auth_type.\n\n    Returns:\n        Dict with auth headers.\n    """'),
    ('"""Baut Query-Parameter fuer API-Key-Auth.\n\n    Returns:\n        Dict mit Auth-Query-Parametern.\n    """',
     '"""Build query parameters for API key auth.\n\n    Returns:\n        Dict with auth query parameters.\n    """'),
    ('"""Kuerzt Text auf max_chars mit Hinweis."""', '"""Truncate text to max_chars with a note."""'),
    ('"""Maskiert moegliche Credentials in Fehlermeldungen."""', '"""Mask possible credentials in error messages."""'),
    ('"""API Integration Hub -- Persistente Verbindungen zu externen APIs. [B§5.3]\n\n    Verwaltet API-Konfigurationen und fuehrt authentifizierte Requests\n    durch. Credentials werden NIE gespeichert, nur Env-Var-Namen.\n    """',
     '"""API Integration Hub -- Persistent connections to external APIs. [B§5.3]\n\n    Manages API configurations and performs authenticated requests.\n    Credentials are NEVER stored, only env var names.\n    """'),
    ('"""Holt oder erstellt einen Rate-Limiter fuer eine Integration."""', '"""Get or create a rate limiter for an integration."""'),
    ('"""Listet konfigurierte API-Integrationen auf.\n\n        Returns:\n            Formatierte Liste von Integrationen und verfuegbaren Templates.\n        """',
     '"""List configured API integrations.\n\n        Returns:\n            Formatted list of integrations and available templates.\n        """'),
    ('# Konfigurierte Integrationen', '# Configured integrations'),
    ('# Verfuegbare Templates', '# Available templates'),
    ('"""Konfiguriert eine API-Integration.\n\n        Args:\n            name: Integrations-Name (z.B. "github").\n            base_url: API-Basis-URL (optional, nutzt Template-Default).\n            auth_type: Authentifizierungs-Typ (bearer/api_key/basic).\n            credential_env: Name der Umgebungsvariable mit dem Token/Key.\n            headers: Zusaetzliche HTTP-Header.\n\n        Returns:\n            Erfolgs-/Fehlermeldung mit Integrations-Details.\n        """',
     '"""Configure an API integration.\n\n        Args:\n            name: Integration name (e.g. "github").\n            base_url: API base URL (optional, uses template default).\n            auth_type: Authentication type (bearer/api_key/basic).\n            credential_env: Name of the environment variable with the token/key.\n            headers: Additional HTTP headers.\n\n        Returns:\n            Success/error message with integration details.\n        """'),
    ('# Template laden falls vorhanden', '# Load template if available'),
    ('# Werte mit Template-Defaults zusammenfuehren', '# Merge values with template defaults'),
    ('# URL validieren', '# Validate URL'),
    ('# Integration-Konfiguration erstellen', '# Create integration configuration'),
    ('# Template-spezifische Felder uebernehmen', '# Copy template-specific fields'),
    ('# Speichern', '# Save'),
    ('# Ergebnis', '# Result'),
    ('"""Fuehrt einen authentifizierten API-Call aus.\n\n        Args:\n            integration: Name der Integration.\n            method: HTTP-Methode (GET/POST/PUT/DELETE/PATCH/HEAD).\n            endpoint: Pfad, wird an base_url angehaengt.\n            body: Request-Body (als JSON).\n            headers: Zusaetzliche Header.\n\n        Returns:\n            Status-Code + Response-Body.\n        """',
     '"""Execute an authenticated API call.\n\n        Args:\n            integration: Name of the integration.\n            method: HTTP method (GET/POST/PUT/DELETE/PATCH/HEAD).\n            endpoint: Path appended to base_url.\n            body: Request body (as JSON).\n            headers: Additional headers.\n\n        Returns:\n            Status code + response body.\n        """'),
    ('# Method validieren', '# Validate method'),
    ('# URL zusammenbauen', '# Build URL'),
    ('# Template fuer Auth-Konfiguration', '# Template for auth configuration'),
    ('# Auth-Header', '# Auth headers'),
    ('# Auth-Query-Parameter (fuer api_key Auth)', '# Auth query parameters (for api_key auth)'),
    ('# Alle Header zusammenfuehren', '# Merge all headers'),
    ('# Content-Type fuer POST/PUT/PATCH mit Body', '# Content-Type for POST/PUT/PATCH with body'),
    ('# HTTP-Request ausfuehren', '# Execute HTTP request'),
    ('# Antwort formatieren', '# Format response'),
    ('# JSON huebsch formatieren wenn moeglich', '# Pretty-print JSON if possible'),
    ('"""Entfernt eine API-Integration.\n\n        Args:\n            name: Integrations-Name.\n\n        Returns:\n            Bestaetigung.\n        """',
     '"""Remove an API integration.\n\n        Args:\n            name: Integration name.\n\n        Returns:\n            Confirmation.\n        """'),
    ('# Rate-Limiter entfernen', '# Remove rate limiter'),
    ('"""Fuehrt einen Health-Check gegen die API durch.\n\n        Returns:\n            Tuple (success, message).\n        """',
     '"""Perform a health check against the API.\n\n        Returns:\n            Tuple (success, message).\n        """'),
    ('"""Fuehrt den HTTP-Request via httpx aus.\n\n        Falls httpx nicht verfuegbar, Fallback auf urllib.\n\n        Returns:\n            Tuple (status_code, response_text).\n        """',
     '"""Execute the HTTP request via httpx.\n\n        Falls back to urllib if httpx is not available.\n\n        Returns:\n            Tuple (status_code, response_text).\n        """'),
    ('# Versuche httpx (bevorzugt)', '# Try httpx (preferred)'),
    ('# Fallback: urllib (synchron, in Executor)', '# Fallback: urllib (synchronous, in executor)'),
    ('"""HTTP-Request via httpx (async)."""', '"""HTTP request via httpx (async)."""'),
    ('"""HTTP-Request via urllib (synchron, Fallback)."""', '"""HTTP request via urllib (synchronous, fallback)."""'),
    ('"""Registriert API-Hub-Tools beim MCP-Client.\n\n    Returns:\n        APIHub-Instanz.\n    """',
     '"""Register API hub tools with the MCP client.\n\n    Returns:\n        APIHub instance.\n    """'),

    # ══════════════════════════════════════════════════════════════
    # bridge.py
    # ══════════════════════════════════════════════════════════════
    ('"""MCP Bridge: Verbindet bestehende Builtin-Tools mit dem MCP-Server.\n\nDieses Modul ist die zentrale Brücke zwischen dem bestehenden\nregister_builtin_handler()-System und dem neuen MCP-Server-Modus.\n\nARCHITEKTUR:\n  - Ohne MCP-Server: Tools laufen wie bisher über register_builtin_handler()\n  - Mit MCP-Server: Tools werden ZUSÄTZLICH über den MCP-Server exponiert\n  - Der MCP-Server ist rein additiv -- er ersetzt nichts\n\nVerantwortlich für:\n  1. Bestehende Builtin-Handler in MCPToolDefs konvertieren\n  2. Tool-Annotations hinzufügen (readOnly, destructive, etc.)\n  3. Resources und Prompts beim Server registrieren\n  4. Discovery/Agent-Card aufbauen\n  5. HTTP-Endpoints bereitstellen (wenn HTTP-Modus aktiv)\n\nBibel-Referenz: §5.5.5 (MCP Bridge)\n"""',
     '"""MCP Bridge: Connects existing builtin tools with the MCP server.\n\nThis module is the central bridge between the existing\nregister_builtin_handler() system and the new MCP server mode.\n\nARCHITECTURE:\n  - Without MCP server: Tools run as before via register_builtin_handler()\n  - With MCP server: Tools are ADDITIONALLY exposed via the MCP server\n  - The MCP server is purely additive -- it replaces nothing\n\nResponsible for:\n  1. Converting existing builtin handlers into MCPToolDefs\n  2. Adding tool annotations (readOnly, destructive, etc.)\n  3. Registering resources and prompts with the server\n  4. Building discovery/agent card\n  5. Providing HTTP endpoints (when HTTP mode is active)\n\nReference: §5.5.5 (MCP Bridge)\n"""'),
    ('# Welche Tools sind read-only (ändern nichts am System)?', '# Which tools are read-only (do not modify the system)?'),
    ('# Welche Tools sind destruktiv (können Daten löschen/überschreiben)?', '# Which tools are destructive (can delete/overwrite data)?'),
    ('# Welche Tools sind idempotent (mehrfach aufrufen = gleich)?', '# Which tools are idempotent (calling multiple times = same result)?'),
    ('# Tools die im MCP-Server-Modus fuer externe Clients (VSCode etc.) sicher sind.\n# Kein Shell-Exec, kein Computer-Use, kein Remote-Shell, kein Docker-Run.',
     '# Tools that are safe for external clients (VSCode etc.) in MCP server mode.\n# No shell exec, no computer use, no remote shell, no docker run.'),
    ('"""Erzeugt MCP-Annotations für ein Tool basierend auf seinem Namen."""', '"""Generate MCP annotations for a tool based on its name."""'),
    ('"""Zentrale Brücke zwischen Builtin-Handlers und MCP-Server.\n\n    Nutzung:\n        bridge = MCPBridge(config)\n        bridge.setup(mcp_client, memory_manager)\n        await bridge.start()  # Startet MCP-Server falls konfiguriert\n    """',
     '"""Central bridge between builtin handlers and MCP server.\n\n    Usage:\n        bridge = MCPBridge(config)\n        bridge.setup(mcp_client, memory_manager)\n        await bridge.start()  # Starts MCP server if configured\n    """'),
    ('"""Richtet den MCP-Server-Modus ein (falls konfiguriert).\n\n        Liest die MCP-Server-Config, konvertiert bestehende Builtin-Tools\n        in MCPToolDefs und registriert Resources + Prompts.\n\n        Args:\n            mcp_client: Der bestehende JarvisMCPClient mit registrierten Tools\n            memory: MemoryManager für Resource-Zugriff\n\n        Returns:\n            True wenn MCP-Server-Modus aktiviert wurde, False sonst.\n        """',
     '"""Set up the MCP server mode (if configured).\n\n        Reads the MCP server config, converts existing builtin tools\n        into MCPToolDefs and registers resources + prompts.\n\n        Args:\n            mcp_client: The existing JarvisMCPClient with registered tools\n            memory: MemoryManager for resource access\n\n        Returns:\n            True if MCP server mode was activated, False otherwise.\n        """'),
    ('# Server-Config aus Jarvis-Config laden', '# Load server config from Jarvis config'),
    ('# MCP-Server erstellen', '# Create MCP server'),
    ('# 1. Bestehende Builtin-Tools konvertieren', '# 1. Convert existing builtin tools'),
    ('# 2. Resources registrieren', '# 2. Register resources'),
    ('# 3. Prompts registrieren', '# 3. Register prompts'),
    ('"""Startet den MCP-Server (falls aktiviert)."""', '"""Start the MCP server (if enabled)."""'),
    ('"""Stoppt den MCP-Server."""', '"""Stop the MCP server."""'),
    ('"""Ist der MCP-Server-Modus aktiv?"""', '"""Is the MCP server mode active?"""'),
    ('"""Der MCP-Server (für HTTP-Endpoint-Registrierung)."""', '"""The MCP server (for HTTP endpoint registration)."""'),
    ('"""Der Discovery-Manager (für Agent-Card-Endpoint)."""', '"""The discovery manager (for agent card endpoint)."""'),
    ('"""Konvertiert bestehende Builtin-Handler in MCPToolDefs.\n\n        Liest alle registrierten Tools aus dem MCP-Client und\n        erstellt für jeden eine MCPToolDef mit Annotations.\n\n        Returns:\n            Anzahl konvertierter Tools.\n        """',
     '"""Convert existing builtin handlers into MCPToolDefs.\n\n        Reads all registered tools from the MCP client and\n        creates an MCPToolDef with annotations for each one.\n\n        Returns:\n            Number of converted tools.\n        """'),
    ('# Handler aus dem Client holen', '# Get handler from the client'),
    ('# MCPToolDef erstellen', '# Create MCPToolDef'),
    ('"""Lädt die MCP-Server-Konfiguration.\n\n        Prüft zuerst die Jarvis-Config, dann die MCP-Config-YAML.\n        Default: DISABLED.\n        """',
     '"""Load the MCP server configuration.\n\n        Checks the Jarvis config first, then the MCP config YAML.\n        Default: DISABLED.\n        """'),
    ('# Aus MCP-Config-YAML laden', '# Load from MCP config YAML'),
    ('"""Verarbeitet einen eingehenden MCP HTTP-Request.\n\n        Wird von config_routes.py aufgerufen.\n\n        Args:\n            body: JSON-RPC-Message(s)\n            auth_header: Authorization-Header-Wert\n\n        Returns:\n            JSON-RPC-Response(s)\n        """',
     '"""Process an incoming MCP HTTP request.\n\n        Called by config_routes.py.\n\n        Args:\n            body: JSON-RPC message(s)\n            auth_header: Authorization header value\n\n        Returns:\n            JSON-RPC response(s)\n        """'),
    ('# Token aus "Bearer xxx" extrahieren', '# Extract token from "Bearer xxx"'),
    ('"""Gibt die Agent Card zurück (für /.well-known/agent.json)."""', '"""Return the Agent Card (for /.well-known/agent.json)."""'),
    ('"""Gesamtstatistiken der MCP-Bridge."""', '"""Overall statistics of the MCP bridge."""'),
]

# This is getting extremely long. Let me use a more efficient approach.
# I'll define a comprehensive set of German->English pattern translations.

# Additional patterns for remaining files - using regex-based approach
REGEX_TRANSLATIONS: list[tuple[str, str]] = []

def apply_translations(filepath: Path, content: str) -> str:
    """Apply all translations to file content."""
    for old, new in TRANSLATIONS:
        if old in content:
            content = content.replace(old, new)
    return content

def main():
    mcp_dir = Path("src/jarvis/mcp")
    changed = 0
    for py_file in sorted(mcp_dir.glob("*.py")):
        original = py_file.read_text(encoding="utf-8")
        translated = apply_translations(py_file, original)
        if translated != original:
            py_file.write_text(translated, encoding="utf-8")
            changed += 1
            print(f"  Translated: {py_file.name}")
        else:
            print(f"  (no changes): {py_file.name}")
    print(f"\nFiles modified: {changed}")

if __name__ == "__main__":
    main()
