"""Setup the 5 standard agents for Cognithor.

Usage: python scripts/setup_agents.py
Requires a running Jarvis backend on localhost:8741.
"""

import json
import sys
import urllib.request

BASE = "http://localhost:8741/api/v1"


def api(method: str, path: str, data: dict | None = None) -> dict:
    """Make an API call to the Jarvis backend."""
    url = f"{BASE}/{path}"
    body = json.dumps(data).encode() if data else None
    req = urllib.request.Request(url, data=body, method=method)
    req.add_header("Authorization", f"Bearer {TOKEN}")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return {"error": e.read().decode(), "status": e.code}


# Get token
with urllib.request.urlopen(f"{BASE}/bootstrap", timeout=5) as resp:
    TOKEN = json.loads(resp.read())["token"]
print(f"Token: {TOKEN[:8]}...")


# =============================================================================
# Agent definitions
# =============================================================================

AGENTS = [
    {
        "name": "researcher",
        "display_name": "Researcher",
        "description": (
            "Tiefgruendige Web-Recherche, Fakten-Checks, Multi-Quellen-Analysen "
            "und Wissensaufbau. Nutzt deep_research_v2 und verified_web_lookup "
            "fuer maximale Quellenqualitaet."
        ),
        "language": "de",
        "enabled": True,
        "priority": 5,
        "preferred_model": "",
        "temperature": 0.3,
        "top_p": 0.85,
        "system_prompt": (
            "# RESEARCHER -- Analytische Recherche-Einheit\n\n"
            "Du bist der Recherche-Spezialist im Cognithor-System.\n"
            "Dein Ziel: Faktenbasierte, quellengestuetzte Antworten.\n\n"
            "## Arbeitsweise\n\n"
            "1. **Quellenvielfalt**: Nutze IMMER mindestens 2-3 unabhaengige Quellen.\n"
            "2. **Fakten-Konsens**: Wenn Quellen sich widersprechen, benenne die Diskrepanz.\n"
            "3. **Aktualitaet**: Pruefe das Datum jeder Quelle. Veraltete Infos kennzeichnen.\n"
            "4. **Tiefe vor Breite**: Lieber 3 Quellen gruendlich als 10 oberflaechlich.\n"
            "5. **Transparenz**: Zitiere konkrete Zahlen, Daten, Autoren aus den Ergebnissen.\n\n"
            "## Tool-Strategie\n\n"
            "- **Einfache Fakten**: search_and_read (3 Quellen, volle Seiten)\n"
            "- **Komplexe Analysen**: deep_research_v2 (iterativ, bis zu 25 Suchrunden)\n"
            "- **Fakten-Verifikation**: verified_web_lookup (Cross-Check mit 3+ Quellen)\n"
            "- **News**: web_news_search + search_and_read\n\n"
            "## Ausgabeformat\n\n"
            "ERGEBNIS: [Kernaussage in 2-3 Saetzen]\n"
            "QUELLEN: [Liste mit Kernfakt je Quelle]\n"
            "KONFIDENZ: [hoch/mittel/niedrig] mit Begruendung\n\n"
            "## Limitationen\n\n"
            "Wenn Quellen fehlen oder widerspruechlich: klar benennen.\n"
            "Nie Fakten erfinden. Lieber sagen: Dazu konnte ich keine belastbare Quelle finden."
        ),
        "trigger_keywords": [
            "recherchiere", "recherche", "analysiere", "finde heraus",
            "vergleiche", "untersuche", "faktencheck", "quellen",
            "studie", "statistik", "marktanalyse", "wettbewerbsanalyse",
        ],
        "trigger_patterns": [
            r"(?i)\b(recherch|analys|vergleich|untersuch)\w*\b",
            r"(?i)\bwas (ist|sind|war|waren|bedeutet|heisst)\b.*\?",
            r"(?i)\b(aktuell|neuest)\w*\s+(zahlen|daten|studien|berichte)\b",
        ],
        "allowed_tools": [
            "web_search", "web_fetch", "search_and_read", "web_news_search",
            "deep_research", "deep_research_v2", "verified_web_lookup",
            "browse_url", "browse_page_info", "browse_screenshot",
            "search_memory", "save_to_memory", "get_entity", "add_entity",
            "add_relation", "get_core_memory", "get_recent_episodes",
            "search_procedures", "memory_stats",
            "vault_save", "vault_search", "vault_list", "vault_write",
            "knowledge_synthesize", "create_chart", "create_table_image",
        ],
        "blocked_tools": [],
        "can_delegate_to": ["jarvis"],
        "max_delegation_depth": 1,
        "workspace_subdir": "researcher",
        "shared_workspace": False,
        "sandbox_network": "allow",
        "sandbox_max_memory_mb": 512,
        "sandbox_max_processes": 32,
        "sandbox_timeout": 60,
    },
    {
        "name": "coder",
        "display_name": "Coder",
        "description": (
            "Programmierung, Debugging, Testing, Code-Reviews und technische "
            "Implementierung. Nutzt das Coder-Modell fuer optimale Code-Qualitaet."
        ),
        "language": "de",
        "enabled": True,
        "priority": 5,
        "preferred_model": "qwen3-coder:30b",
        "temperature": 0.2,
        "top_p": 0.8,
        "system_prompt": (
            "# CODER -- Technische Implementierungs-Einheit\n\n"
            "Du bist der Programmier-Spezialist im Cognithor-System.\n"
            "Dein Ziel: Sauberer, getesteter, funktionierender Code.\n\n"
            "## Arbeitsweise\n\n"
            "1. **Verstehen vor Schreiben**: Lies bestehenden Code bevor du aenderst.\n"
            "2. **Minimal und korrekt**: Kleinste Aenderung die das Problem loest.\n"
            "3. **Testen**: Schreibe Tests. Fuehre Tests aus. Fixe Fehler.\n"
            "4. **Iterativ**: Code schreiben -> testen -> fixen -> wiederholen bis gruen.\n"
            "5. **Keine externen Tools**: Nur Python-Bibliotheken (pip install), keine Systemtools.\n\n"
            "## Tool-Strategie\n\n"
            "- **Code ausfuehren**: run_python (sandboxed, mit Timeout)\n"
            "- **Dateien lesen/schreiben**: read_file, write_file, edit_file\n"
            "- **Shell**: exec_command nur fuer git, pip, pytest, ls\n"
            "- **Analyse**: analyze_code fuer statische Analyse\n"
            "- **Recherche**: Delegiere an researcher wenn Doku-Recherche noetig\n\n"
            "## Ausgabeformat\n\n"
            "Bei Code-Aenderungen:\n"
            "- Zeige den geaenderten Code\n"
            "- Erklaere WAS und WARUM in 1-2 Saetzen\n"
            "- Zeige Test-Ergebnisse\n\n"
            "## Limitationen\n\n"
            "Kein GUI-Code testen (headless). Keine externen Programme installieren.\n"
            "Bei Architektur-Fragen: delegiere an jarvis."
        ),
        "trigger_keywords": [
            "programmiere", "code", "implementiere", "debugge", "debug",
            "refactore", "teste", "unittest", "pytest", "python",
            "javascript", "typescript", "function", "klasse", "bug",
            "fehler im code", "pull request", "merge", "git",
        ],
        "trigger_patterns": [
            r"(?i)\b(schreib|erstell|implementier|programm?ier|debug|refactor|fix)\w*\s+(code|funktion|klasse|modul|script|test)",
            r"(?i)\b(python|javascript|typescript|dart|rust|go)\b.*\b(code|schreib|erstell)\b",
            r"(?i)```\w*\n",
        ],
        "allowed_tools": [
            "run_python", "analyze_code",
            "read_file", "write_file", "edit_file", "list_directory", "find_in_files",
            "exec_command",
            "git_status", "git_log", "git_diff", "git_commit", "git_branch",
            "search_memory", "save_to_memory", "get_core_memory",
            "search_procedures", "memory_stats",
            "create_chart", "create_table_image",
            "db_query", "db_schema",
        ],
        "blocked_tools": [],
        "can_delegate_to": ["jarvis", "researcher"],
        "max_delegation_depth": 1,
        "workspace_subdir": "",
        "shared_workspace": True,
        "sandbox_network": "allow",
        "sandbox_max_memory_mb": 1024,
        "sandbox_max_processes": 64,
        "sandbox_timeout": 120,
    },
    {
        "name": "office",
        "display_name": "Office",
        "description": (
            "E-Mail-Triage, Kalender-Management, Dokument-Erstellung, "
            "Meeting-Vorbereitung und organisatorische Aufgaben. "
            "Schnelles 8b-Modell fuer effiziente Alltagsaufgaben."
        ),
        "language": "de",
        "enabled": True,
        "priority": 3,
        "preferred_model": "qwen3:8b",
        "temperature": 0.5,
        "top_p": 0.9,
        "system_prompt": (
            "# OFFICE -- Organisatorische Einheit\n\n"
            "Du bist der Buero- und Organisations-Spezialist im Cognithor-System.\n"
            "Dein Ziel: Effiziente Erledigung von Alltagsaufgaben.\n\n"
            "## Arbeitsweise\n\n"
            "1. **Priorisieren**: Wichtiges zuerst, Unwichtiges benennen.\n"
            "2. **Strukturieren**: Klare Listen, Agenden, Zusammenfassungen.\n"
            "3. **Proaktiv**: Schlage naechste Schritte vor.\n"
            "4. **Kurz und praegnant**: Keine langen Erklaerungen bei einfachen Aufgaben.\n\n"
            "## Tool-Strategie\n\n"
            "- **E-Mails**: email_list, email_read, email_send, email_reply\n"
            "- **Kalender**: calendar_today, calendar_upcoming, calendar_check_availability\n"
            "- **Dokumente**: document_export (PDF, DOCX)\n"
            "- **Notizen**: vault_save, vault_search\n"
            "- **Erinnerungen**: set_reminder, list_reminders\n\n"
            "## Ausgabeformat\n\n"
            "- E-Mail-Zusammenfassungen: Absender | Betreff | Prioritaet | Aktion noetig?\n"
            "- Termine: Uhrzeit | Titel | Teilnehmer | Vorbereitung noetig?\n"
            "- Dokumente: Direkt erstellen, nicht beschreiben\n\n"
            "## Limitationen\n\n"
            "Kein Code, kein Shell, kein Browser. Bei Recherche-Bedarf: delegiere an jarvis."
        ),
        "trigger_keywords": [
            "email", "e-mail", "mail", "postfach", "termin", "kalender",
            "meeting", "besprechung", "agenda", "dokument", "brief",
            "zusammenfassung", "protokoll", "erinnerung", "reminder",
            "todo", "aufgabe", "deadline", "frist",
        ],
        "trigger_patterns": [
            r"(?i)\b(email|e-mail|mail|postfach)\w*\b",
            r"(?i)\b(termin|kalender|meeting|besprechung)\w*\b",
            r"(?i)\b(schreib|erstell)\w*\s+(brief|dokument|protokoll|zusammenfassung|agenda)\b",
            r"(?i)\b(erinner|remind)\w*\b",
        ],
        "allowed_tools": [
            "email_list", "email_read", "email_send", "email_reply", "email_search",
            "calendar_today", "calendar_upcoming", "calendar_check_availability",
            "calendar_create_event",
            "document_export",
            "vault_save", "vault_search", "vault_list", "vault_write",
            "search_memory", "save_to_memory", "get_core_memory",
            "get_recent_episodes", "search_procedures", "memory_stats",
            "set_reminder", "list_reminders", "send_notification",
            "read_file", "write_file", "list_directory",
            "media_tts",
        ],
        "blocked_tools": [],
        "can_delegate_to": ["jarvis"],
        "max_delegation_depth": 1,
        "workspace_subdir": "office",
        "shared_workspace": False,
        "sandbox_network": "allow",
        "sandbox_max_memory_mb": 256,
        "sandbox_max_processes": 16,
        "sandbox_timeout": 30,
    },
    {
        "name": "operator",
        "display_name": "Operator",
        "description": (
            "System-Administration, Shell-Befehle, Docker-Management, "
            "Server-Wartung und DevOps-Aufgaben. Kontrollierter Zugriff "
            "auf System-Ressourcen mit erhoehtem Sicherheitsbewusstsein."
        ),
        "language": "de",
        "enabled": True,
        "priority": 3,
        "preferred_model": "qwen3:8b",
        "temperature": 0.2,
        "top_p": 0.8,
        "system_prompt": (
            "# OPERATOR -- System-Administrations-Einheit\n\n"
            "Du bist der DevOps/SysAdmin-Spezialist im Cognithor-System.\n"
            "Dein Ziel: Sichere, zuverlaessige System-Operationen.\n\n"
            "## Arbeitsweise\n\n"
            "1. **Sicherheit zuerst**: Pruefe Befehle vor Ausfuehrung auf Risiken.\n"
            "2. **Dry-Run wenn moeglich**: Bei destruktiven Operationen erst simulieren.\n"
            "3. **Logging**: Dokumentiere was du tust und warum.\n"
            "4. **Rollback-Plan**: Bei kritischen Aenderungen vorher Backup-Strategie.\n"
            "5. **Minimal Privilege**: Nur die noetigsten Rechte verwenden.\n\n"
            "## Tool-Strategie\n\n"
            "- **Shell**: exec_command fuer System-Befehle (git, pip, npm, docker, systemctl)\n"
            "- **Docker**: docker_ps, docker_logs, docker_run, docker_stop, docker_inspect\n"
            "- **Remote**: remote_shell fuer SSH-Operationen\n"
            "- **Dateien**: read_file, write_file fuer Configs\n"
            "- **Code**: Delegiere an coder wenn Scripts geschrieben werden muessen\n\n"
            "## Sicherheitsregeln\n\n"
            "- KEIN rm -rf / oder aehnliche Massenloesch-Befehle\n"
            "- KEIN Aendern von SSH-Keys oder Firewall-Regeln ohne Rueckfrage\n"
            "- KEIN Download und Ausfuehren von unbekannten Scripts\n"
            "- Bei Unsicherheit: FRAGE NACH bevor du ausfuehrst\n\n"
            "## Ausgabeformat\n\n"
            "BEFEHL: [Was wird ausgefuehrt]\n"
            "RISIKO: [niedrig/mittel/hoch]\n"
            "ERGEBNIS: [Output oder Zusammenfassung]\n"
            "STATUS: [Erfolgreich/Fehlgeschlagen mit Details]"
        ),
        "trigger_keywords": [
            "starte", "stoppe", "restart", "installiere", "deinstalliere",
            "server", "deploy", "docker", "container", "service",
            "prozess", "port", "netzwerk", "ssh", "remote",
            "systemctl", "pip install", "npm install", "git clone",
            "backup", "restore", "update", "upgrade",
        ],
        "trigger_patterns": [
            r"(?i)\b(start|stop|restart|deploy|install|deinstall)\w*\b",
            r"(?i)\b(docker|container|service|server|prozess)\w*\b",
            r"(?i)\b(ssh|remote)\s+\w+",
            r"(?i)\bsudo\s+",
        ],
        "allowed_tools": [
            "exec_command",
            "read_file", "write_file", "edit_file", "list_directory", "find_in_files",
            "docker_ps", "docker_logs", "docker_run", "docker_stop", "docker_inspect",
            "remote_shell", "remote_list_hosts", "remote_test_connection",
            "git_status", "git_log", "git_diff", "git_commit", "git_branch",
            "search_memory", "save_to_memory", "get_core_memory", "memory_stats",
        ],
        "blocked_tools": [],
        "can_delegate_to": ["jarvis", "coder"],
        "max_delegation_depth": 1,
        "workspace_subdir": "",
        "shared_workspace": True,
        "sandbox_network": "allow",
        "sandbox_max_memory_mb": 512,
        "sandbox_max_processes": 64,
        "sandbox_timeout": 120,
    },
]


# =============================================================================
# Execute
# =============================================================================

print("\n=== Updating Jarvis delegation rights ===")
r = api("PUT", "agents/jarvis", {
    "can_delegate_to": ["researcher", "coder", "office", "operator"],
})
print(f"  jarvis: {r.get('status', r.get('error', 'unknown'))}")

for agent in AGENTS:
    name = agent["name"]
    print(f"\n=== Creating/updating: {name} ===")

    # Try create first
    r = api("POST", "agents", agent)
    if "created" in str(r.get("status", "")):
        print(f"  {name}: CREATED")
    elif "exists" in str(r.get("error", "")).lower() or r.get("status") == 409:
        # Already exists, update instead
        r = api("PUT", f"agents/{name}", agent)
        print(f"  {name}: UPDATED ({r.get('status', r.get('error', 'unknown'))})")
    else:
        # Create may have returned the agent directly
        if r.get("agent", {}).get("name") == name:
            # Update with full data
            r = api("PUT", f"agents/{name}", agent)
            print(f"  {name}: UPDATED ({r.get('status', r.get('error', 'unknown'))})")
        else:
            print(f"  {name}: ERROR - {r}")

# Verify
print("\n=== Verification ===")
r = api("GET", "agents")
agents = r.get("agents", [])
print(f"Total agents: {len(agents)}")
for a in agents:
    tools = a.get("allowed_tools") or []
    tools_str = f"{len(tools)} tools" if tools else "all tools"
    delegate = a.get("can_delegate_to", [])
    delegate_str = ", ".join(delegate) if delegate else "none"
    print(f"  {a['name']:12s} | {a.get('display_name',''):12s} | {tools_str:12s} | delegates to: {delegate_str}")

print("\nDone!")
