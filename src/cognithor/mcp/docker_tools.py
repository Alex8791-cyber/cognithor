"""Docker-Tools fuer Cognithor -- Container-Management via CLI.

Ermoeglicht dem Agenten Docker-Container zu verwalten:
  - docker_ps: Container auflisten
  - docker_logs: Container-Logs abrufen
  - docker_inspect: Container/Image inspizieren
  - docker_run: Container starten (ORANGE -- Gatekeeper-Approval)
  - docker_stop: Container stoppen

Alle Befehle werden via asyncio.create_subprocess_exec ausgefuehrt.
Docker-CLI muss installiert sein; Modul wird stillschweigend uebersprungen
wenn Docker nicht verfuegbar ist.

Bibel-Referenz: §5.3 (MCP Tools)
"""

from __future__ import annotations

import asyncio
import contextlib
import re
import shlex
import subprocess
from typing import TYPE_CHECKING, Any

from cognithor.i18n import t
from cognithor.utils.logging import get_logger

if TYPE_CHECKING:
    from cognithor.config import CognithorConfig

log = get_logger(__name__)

__all__ = [
    "DockerError",
    "DockerTools",
    "register_docker_tools",
]

# ── Konstanten ─────────────────────────────────────────────────────────────

_MAX_OUTPUT_CHARS = 50_000
_DEFAULT_TIMEOUT = 60  # Sekunden pro Docker-Kommando
_CONTAINER_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]*$")
_PORT_MAPPING_RE = re.compile(r"^\d{1,5}:\d{1,5}$")

# System-Verzeichnisse, die NICHT als Bind-Mount erlaubt sind
_BLOCKED_MOUNT_PATHS_UNIX = frozenset(
    {
        "/etc",
        "/var",
        "/usr",
        "/bin",
        "/sbin",
        "/lib",
        "/lib64",
        "/boot",
        "/proc",
        "/sys",
        "/dev",
        "/root",
    }
)
_BLOCKED_MOUNT_PATHS_WIN = frozenset(
    {
        "c:\\windows",
        "c:\\program files",
        "c:\\program files (x86)",
        "c:\\programdata",
        "c:\\system volume information",
    }
)

# Gefaehrliche Docker-Flags
_BLOCKED_FLAGS = frozenset(
    {
        "--privileged",
        "--network=host",
        "--network host",
        "--pid=host",
        "--pid host",
        "--ipc=host",
        "--ipc host",
        "--cap-add=ALL",
        "--cap-add ALL",
    }
)


class DockerError(Exception):
    """Fehler bei Docker-Operationen."""


def _docker_available() -> bool:
    """Prueft ob Docker-CLI verfuegbar ist (synchron, fuer Init)."""
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=10,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False


def _truncate(text: str, max_chars: int = _MAX_OUTPUT_CHARS) -> str:
    """Kuerzt Text auf max_chars mit Hinweis."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n\n[... output truncated at {max_chars} chars]"


def _sanitize_container_name(name: str) -> str | None:
    """Validiert Container-Namen: alphanumerisch + Dash + Underscore + Punkt.

    Returns:
        Bereinigter Name oder None bei ungueltigem Input.
    """
    name = name.strip()
    if not name:
        return None
    if _CONTAINER_NAME_RE.match(name):
        return name
    return None


def _validate_port(port: str) -> bool:
    """Validiert Port-Mapping im Format 'HOST:CONTAINER'."""
    if not _PORT_MAPPING_RE.match(port):
        return False
    host_port, container_port = port.split(":")
    return 1 <= int(host_port) <= 65535 and 1 <= int(container_port) <= 65535


def _is_blocked_mount(path_str: str) -> bool:
    """Prueft ob ein Pfad ein blockiertes System-Verzeichnis ist."""
    normalized = path_str.replace("\\", "/").rstrip("/").lower()
    # Unix-Pfade
    for blocked in _BLOCKED_MOUNT_PATHS_UNIX:
        if normalized == blocked or normalized.startswith(blocked + "/"):
            return True
    # Windows-Pfade
    for blocked in _BLOCKED_MOUNT_PATHS_WIN:
        blocked_norm = blocked.replace("\\", "/").lower()
        if normalized == blocked_norm or normalized.startswith(blocked_norm + "/"):
            return True
    return False


def _parse_docker_error(stderr: str) -> str:
    """Parst Docker-CLI-Fehler in benutzerfreundliche Meldungen."""
    stderr_lower = stderr.lower()
    if "no such container" in stderr_lower:
        return t("docker.container_not_found")
    if "no such image" in stderr_lower:
        return t("docker.image_not_found")
    if "is already in use" in stderr_lower:
        return (
            "Ein Container mit diesem Namen existiert bereits. "
            "Waehle einen anderen Namen oder stoppe den bestehenden Container."
        )
    if "permission denied" in stderr_lower:
        return (
            "Docker-Zugriff verweigert. Ist der Docker-Daemon gestartet "
            "und hat der Benutzer die noetigen Rechte?"
        )
    if "cannot connect" in stderr_lower or "connection refused" in stderr_lower:
        return "Docker-Daemon nicht erreichbar. Ist Docker gestartet?"
    if "port is already allocated" in stderr_lower:
        return "Der angegebene Port ist bereits belegt. Waehle einen anderen Host-Port."
    if "pull access denied" in stderr_lower:
        return "Kein Zugriff auf das Image. Ist eine Authentifizierung noetig?"
    # Fallback: Original-Fehler (gekuerzt)
    return stderr[:500] if stderr else "Unbekannter Docker-Fehler."


async def _run_docker(*args: str, timeout: int = _DEFAULT_TIMEOUT) -> tuple[int, str, str]:
    """Fuehrt ein Docker-Kommando async aus.

    Returns:
        Tuple von (exit_code, stdout, stderr).
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            "docker",
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            proc.communicate(),
            timeout=timeout,
        )
        stdout = stdout_bytes.decode("utf-8", errors="replace") if stdout_bytes else ""
        stderr = stderr_bytes.decode("utf-8", errors="replace") if stderr_bytes else ""
        return proc.returncode or 0, stdout, stderr
    except TimeoutError:
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        return -1, "", f"Docker command timed out after {timeout}s"
    except FileNotFoundError:
        return -1, "", "Docker CLI not found. Is Docker installed?"
    except OSError as exc:
        return -1, "", f"Failed to execute Docker: {exc}"


class DockerTools:
    """Docker-Container-Management via CLI. [B§5.3]

    Alle Operationen verwenden asyncio.create_subprocess_exec
    fuer non-blocking I/O. Kein docker-py erforderlich.
    """

    def __init__(self, config: CognithorConfig) -> None:
        self._config = config
        self._workspace_dir = str(config.workspace_dir)
        log.info("docker_tools_init", workspace=self._workspace_dir)

    async def docker_ps(
        self,
        *,
        all: bool = False,
        filter: str | None = None,
    ) -> str:
        """Liste Docker-Container auf.

        Args:
            all: Auch gestoppte Container anzeigen.
            filter: Optionaler Filter (z.B. "name=web", "status=running").

        Returns:
            Formatierte Tabelle der Container.
        """
        args = [
            "ps",
            "--format",
            "table {{.ID}}\t{{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}",
        ]
        if all:
            args.append("-a")
        if filter:
            # Sanitize filter: only allow safe characters
            safe_filter = re.sub(r"[^a-zA-Z0-9=_.*/-]", "", filter)
            if safe_filter:
                args.extend(["--filter", safe_filter])

        exit_code, stdout, stderr = await _run_docker(*args)
        if exit_code != 0:
            return f"Error: {_parse_docker_error(stderr)}"
        if not stdout.strip():
            return "No containers found."
        return _truncate(stdout)

    async def docker_logs(
        self,
        *,
        container: str,
        tail: int = 100,
        since: str | None = None,
        follow: bool = False,
    ) -> str:
        """Ruft Container-Logs ab.

        Args:
            container: Container-Name oder ID.
            tail: Anzahl der letzten Zeilen (Default: 100).
            since: Zeitfilter (z.B. "1h", "2024-01-01").
            follow: NICHT unterstuetzt -- wird ignoriert.

        Returns:
            Log-Output (max 50000 Zeichen).
        """
        # follow wird IMMER ignoriert (streaming nicht unterstuetzt)
        _ = follow

        name = _sanitize_container_name(container)
        if name is None:
            return (
                "Error: Invalid container name. Use alphanumeric "
                "characters, dashes, underscores, and dots only."
            )

        # Tail-Limit validieren
        tail = max(1, min(tail, 10000))

        args = ["logs", "--tail", str(tail)]
        if since:
            # Sanitize since value
            safe_since = re.sub(r"[^a-zA-Z0-9.:T-]", "", since)
            if safe_since:
                args.extend(["--since", safe_since])
        args.append(name)

        exit_code, stdout, stderr = await _run_docker(*args)
        if exit_code != 0:
            return f"Error: {_parse_docker_error(stderr)}"
        # Docker logs schreibt oft auf stderr statt stdout
        output = stdout + stderr if stdout else stderr
        if not output.strip():
            return f"No logs found for container '{name}'."
        return _truncate(output)

    async def docker_inspect(
        self,
        *,
        target: str,
        format: str | None = None,
    ) -> str:
        """Inspiziert Container oder Image.

        Args:
            target: Container-Name/ID oder Image-Name.
            format: Optionales Go-Template (z.B. "{{.State.Status}}").

        Returns:
            Formatiertes JSON mit Key-Details.
        """
        name = _sanitize_container_name(target)
        # Image-Namen koennen Slashes enthalten (z.B. "library/nginx")
        if name is None:
            # Erlaubt Image-Namen mit Slash, Doppelpunkt
            if not re.match(r"^[a-zA-Z0-9][a-zA-Z0-9_./:@-]*$", target.strip()):
                return "Error: Invalid target name."
            name = target.strip()

        args = ["inspect"]
        if format:
            # PASS-3 SEC-HIGH: Go-template injection.
            # The previous comment claimed "sanitize" but only `.strip()`
            # was applied. Go templates support ``{{range .Mounts}}``,
            # ``{{call X}}``, ``{{with X}}`` and printf-family functions,
            # which can exfiltrate host mount paths or other sensitive
            # fields from the inspect output. Restrict to a strict
            # ``{{.Field}}`` / ``{{.Field.SubField}}`` allowlist with no
            # function calls, no range, no with, no pipes.
            safe_fmt = format.strip()
            if safe_fmt and not re.fullmatch(
                r"(?:\{\{\s*\.[A-Za-z_][A-Za-z0-9_.]*\s*\}\}\s*)+",
                safe_fmt,
            ):
                return (
                    "Error: --format template must be a sequence of "
                    "{{.Field}} or {{.Field.SubField}} accessors only "
                    "(no range/with/call/printf)."
                )
            if safe_fmt:
                args.extend(["--format", safe_fmt])
        args.append(name)

        exit_code, stdout, stderr = await _run_docker(*args)
        if exit_code != 0:
            return f"Error: {_parse_docker_error(stderr)}"
        return _truncate(stdout)

    async def docker_run(
        self,
        *,
        image: str,
        name: str | None = None,
        ports: list[str] | None = None,
        env: dict[str, str] | None = None,
        command: str | None = None,
        detach: bool = True,
        remove: bool = True,
    ) -> str:
        """Startet einen Docker-Container.

        SICHERHEIT: Blockiert --privileged, --network host, und
        Bind-Mounts zu System-Verzeichnissen.

        Args:
            image: Docker-Image (z.B. "nginx:latest").
            name: Optionaler Container-Name.
            ports: Port-Mappings (z.B. ["8080:80", "443:443"]).
            env: Umgebungsvariablen.
            command: Optionaler Befehl.
            detach: Im Hintergrund starten (Default: true).
            remove: Container nach Beenden loeschen (Default: true).

        Returns:
            Container-ID + Name bei Erfolg.
        """
        # Image validieren
        image = image.strip()
        if not image:
            return "Error: Image name is required."
        if not re.match(r"^[a-zA-Z0-9][a-zA-Z0-9_./:@-]*$", image):
            return "Error: Invalid image name."

        # Container-Name validieren
        if name is not None:
            validated_name = _sanitize_container_name(name)
            if validated_name is None:
                return (
                    "Error: Invalid container name. Use alphanumeric "
                    "characters, dashes, underscores, and dots only."
                )
            name = validated_name

        # Port-Mappings validieren
        if ports:
            for port in ports:
                if not _validate_port(port):
                    return (
                        f"Error: Invalid port mapping '{port}'. "
                        "Expected format: HOST_PORT:CONTAINER_PORT (1-65535)."
                    )

        # Env-Variablen auf Credential-Leaks pruefen (keine Werte loggen)
        # Keine Blockierung, nur Warnung

        # Docker-Argumente zusammenbauen
        args: list[str] = ["run"]

        if detach:
            args.append("-d")
        if remove:
            args.append("--rm")
        if name:
            args.extend(["--name", name])
        if ports:
            for port in ports:
                args.extend(["-p", port])
        if env:
            for key, value in env.items():
                # Sanitize key: alphanumeric + underscore only
                if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", key):
                    return f"Error: Invalid environment variable name: '{key}'."
                args.extend(["-e", f"{key}={value}"])

        # Security: Pruefe ob der konstruierte Befehl blockierte Flags enthaelt
        args_str = " ".join(args).lower()
        for blocked in _BLOCKED_FLAGS:
            if blocked in args_str:
                return f"Error: Docker flag '{blocked}' is blocked for security reasons."

        args.append(image)

        if command:
            # Security: Pruefe auf blockierte Flags im Command
            cmd_lower = command.lower()
            for blocked in _BLOCKED_FLAGS:
                if blocked in cmd_lower:
                    return f"Error: Blocked flag detected in command: '{blocked}'."
            # PASS-3 SEC-HIGH: ``command.split()`` is naive whitespace
            # splitting and silently drops shell quoting. ``shlex.split``
            # respects quotes the operator typed, so a value like
            # ``sh -c "ls -la"`` keeps the quoted argument intact instead
            # of being broken into ``["sh", "-c", "\"ls", "-la\""]``.
            # Additionally refuse any command whose first token launches
            # an inner shell with ``-c`` — that's the canonical container
            # shell-injection vector.
            try:
                cmd_tokens = shlex.split(command)
            except ValueError as exc:
                return f"Error: invalid command quoting: {exc}"
            if cmd_tokens:
                argv0 = cmd_tokens[0].rsplit("/", 1)[-1].lower()
                if argv0 in {"sh", "bash", "zsh", "ash", "dash", "ksh"} and "-c" in cmd_tokens:
                    return (
                        "Error: refusing to launch an inner shell with "
                        "-c inside the container; pass the program "
                        "argv directly."
                    )
            args.extend(cmd_tokens)

        log.info(
            "docker_run_start",
            image=image,
            name=name or "(auto)",
            detach=detach,
            ports=ports,
        )

        exit_code, stdout, stderr = await _run_docker(*args, timeout=_DEFAULT_TIMEOUT)
        if exit_code != 0:
            return f"Error: {_parse_docker_error(stderr)}"

        container_id = stdout.strip()[:12]
        result = f"Container started successfully.\nID: {container_id}"
        if name:
            result += f"\nName: {name}"
        return result

    async def docker_stop(
        self,
        *,
        container: str,
        timeout: int = 10,
    ) -> str:
        """Stoppt einen Container.

        Args:
            container: Container-Name oder ID.
            timeout: Sekunden bis SIGKILL (Default: 10).

        Returns:
            Bestaetigung.
        """
        name = _sanitize_container_name(container)
        if name is None:
            return (
                "Error: Invalid container name. Use alphanumeric "
                "characters, dashes, underscores, and dots only."
            )

        timeout = max(0, min(timeout, 300))

        args = ["stop", "-t", str(timeout), name]
        exit_code, stdout, stderr = await _run_docker(*args)
        if exit_code != 0:
            return f"Error: {_parse_docker_error(stderr)}"

        return f"Container '{name}' stopped successfully."


def register_docker_tools(
    mcp_client: Any,
    config: Any,
) -> DockerTools | None:
    """Registriert Docker-Tools beim MCP-Client.

    Prueft zuerst ob Docker verfuegbar ist. Ueberspringt die
    Registrierung stillschweigend wenn Docker nicht installiert ist.

    Returns:
        DockerTools-Instanz oder None wenn Docker nicht verfuegbar.
    """
    if not _docker_available():
        log.info("docker_not_available_skipping_registration")
        return None

    docker = DockerTools(config)

    # docker_ps — GREEN
    mcp_client.register_builtin_handler(
        "docker_ps",
        docker.docker_ps,
        description=(
            "List Docker containers. Shows ID, name, image, status, and ports. "
            "Use all=true to include stopped containers."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "all": {
                    "type": "boolean",
                    "description": "Include stopped containers (default: false)",
                    "default": False,
                },
                "filter": {
                    "type": "string",
                    "description": "Filter containers (e.g. 'name=web', 'status=running')",
                },
            },
            "required": [],
        },
    )

    # docker_logs — GREEN
    mcp_client.register_builtin_handler(
        "docker_logs",
        docker.docker_logs,
        description=(
            "Get logs from a Docker container. Supports tail lines and time-based filtering."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "container": {
                    "type": "string",
                    "description": "Container name or ID",
                },
                "tail": {
                    "type": "integer",
                    "description": "Number of lines from the end (default: 100, max: 10000)",
                    "default": 100,
                },
                "since": {
                    "type": "string",
                    "description": "Show logs since timestamp (e.g. '1h', '2024-01-01')",
                },
            },
            "required": ["container"],
        },
    )

    # docker_inspect — GREEN
    mcp_client.register_builtin_handler(
        "docker_inspect",
        docker.docker_inspect,
        description=(
            "Inspect a Docker container or image. Returns detailed JSON info "
            "including state, config, network settings, and mounts."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "target": {
                    "type": "string",
                    "description": "Container name/ID or image name to inspect",
                },
                "format": {
                    "type": "string",
                    "description": "Go template for output formatting (e.g. '{{.State.Status}}')",
                },
            },
            "required": ["target"],
        },
    )

    # docker_run — ORANGE (Gatekeeper-Approval erforderlich)
    mcp_client.register_builtin_handler(
        "docker_run",
        docker.docker_run,
        description=(
            "Start a Docker container from an image. Supports port mapping, "
            "environment variables, and custom commands. "
            "Security: --privileged and host network modes are blocked."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "image": {
                    "type": "string",
                    "description": "Docker image (e.g. 'nginx:latest', 'python:3.12-slim')",
                },
                "name": {
                    "type": "string",
                    "description": "Container name (optional)",
                },
                "ports": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Port mappings (e.g. ['8080:80', '443:443'])",
                },
                "env": {
                    "type": "object",
                    "additionalProperties": {"type": "string"},
                    "description": "Environment variables (e.g. {'NODE_ENV': 'production'})",
                },
                "command": {
                    "type": "string",
                    "description": "Command to run in the container",
                },
                "detach": {
                    "type": "boolean",
                    "description": "Run in background (default: true)",
                    "default": True,
                },
                "remove": {
                    "type": "boolean",
                    "description": "Remove container after exit (default: true)",
                    "default": True,
                },
            },
            "required": ["image"],
        },
    )

    # docker_stop — YELLOW
    mcp_client.register_builtin_handler(
        "docker_stop",
        docker.docker_stop,
        description=(
            "Stop a running Docker container gracefully. Sends SIGTERM, then SIGKILL after timeout."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "container": {
                    "type": "string",
                    "description": "Container name or ID",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Seconds to wait before SIGKILL (default: 10)",
                    "default": 10,
                },
            },
            "required": ["container"],
        },
    )

    log.info(
        "docker_tools_registered",
        tools=["docker_ps", "docker_logs", "docker_inspect", "docker_run", "docker_stop"],
    )
    return docker
