"""E-Mail-Tools fuer Cognithor: IMAP-Lesezugriff und SMTP-Versand.

Ermoeglicht dem Agenten E-Mail-Verwaltung ueber IMAP und SMTP.

Tools:
  - email_read_inbox: Letzte E-Mails abrufen
  - email_search: E-Mails durchsuchen
  - email_send: E-Mail versenden (ORANGE — erfordert Bestaetigung)
  - email_summarize: Posteingang zusammenfassen (algorithmisch)

Sicherheit:
  - SSL/TLS erzwungen (IMAP4_SSL, SMTP_SSL/STARTTLS)
  - Passwort nur aus Umgebungsvariable (NIE in Config gespeichert)
  - Anhang-Pfade gegen Workspace-Sandbox geprueft
  - Rate-Limit: max 10 Sendungen pro Stunde

Bibel-Referenz: §5.3 (jarvis-email Server)
"""

from __future__ import annotations

import asyncio
import contextlib
import email
import email.header
import email.utils
import imaplib
import re
import smtplib
import ssl
import time
from collections import defaultdict
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import TYPE_CHECKING, Any

from cognithor.i18n import t
from cognithor.utils.logging import get_logger

if TYPE_CHECKING:
    from cognithor.config import CognithorConfig

log = get_logger(__name__)

# ── Konstanten ─────────────────────────────────────────────────────────────

_EMAIL_REGEX = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_MAX_PREVIEW_CHARS = 500
_MAX_SEND_PER_HOUR = 10
_IMAP_TIMEOUT = 30  # seconds
_SMTP_TIMEOUT = 30  # seconds

__all__ = [
    "EmailError",
    "EmailTools",
    "register_email_tools",
]


class EmailError(Exception):
    """Fehler bei E-Mail-Operationen."""


def _strip_html(html: str) -> str:
    """Entfernt HTML-Tags und gibt reinen Text zurueck."""
    text = _HTML_TAG_RE.sub("", html)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _decode_header(raw: str | None) -> str:
    """Dekodiert einen MIME-codierten Header-Wert."""
    if not raw:
        return ""
    parts = email.header.decode_header(raw)
    decoded_parts: list[str] = []
    for data, charset in parts:
        if isinstance(data, bytes):
            decoded_parts.append(data.decode(charset or "utf-8", errors="replace"))
        else:
            decoded_parts.append(str(data))
    return " ".join(decoded_parts)


def _extract_body_preview(msg: email.message.Message) -> tuple[str, bool]:
    """Extrahiert eine Text-Vorschau und prueft auf Anhaenge.

    Returns:
        (preview_text, has_attachments)
    """
    has_attachments = False
    text_body = ""

    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            disposition = str(part.get("Content-Disposition", ""))

            if "attachment" in disposition.lower():
                has_attachments = True
                continue

            if content_type == "text/plain" and not text_body:
                payload = part.get_payload(decode=True)
                if payload:
                    charset = part.get_content_charset() or "utf-8"
                    text_body = payload.decode(charset, errors="replace")
            elif content_type == "text/html" and not text_body:
                payload = part.get_payload(decode=True)
                if payload:
                    charset = part.get_content_charset() or "utf-8"
                    text_body = _strip_html(payload.decode(charset, errors="replace"))
    else:
        content_type = msg.get_content_type()
        payload = msg.get_payload(decode=True)
        if payload:
            charset = msg.get_content_charset() or "utf-8"
            raw_text = payload.decode(charset, errors="replace")
            text_body = _strip_html(raw_text) if content_type == "text/html" else raw_text

    preview = text_body[:_MAX_PREVIEW_CHARS].strip()
    if len(text_body) > _MAX_PREVIEW_CHARS:
        preview += "..."

    return preview, has_attachments


def _parse_email_message(msg: email.message.Message, uid: str) -> dict[str, Any]:
    """Parst eine E-Mail-Nachricht in ein standardisiertes Dict."""
    preview, has_attachments = _extract_body_preview(msg)

    return {
        "uid": uid,
        "from": _decode_header(msg.get("From")),
        "to": _decode_header(msg.get("To")),
        "subject": _decode_header(msg.get("Subject")),
        "date": _decode_header(msg.get("Date")),
        "preview": preview,
        "has_attachments": has_attachments,
    }


class EmailTools:
    """E-Mail-Operationen ueber IMAP/SMTP. [B§5.3]

    Attributes:
        _imap_host: IMAP-Server Hostname.
        _smtp_host: SMTP-Server Hostname.
    """

    def __init__(self, config: CognithorConfig) -> None:
        """Initialisiert EmailTools mit Konfiguration.

        Args:
            config: Jarvis-Konfiguration mit email-Sektion.
        """
        self._config = config
        email_cfg = config.email
        self._imap_host = email_cfg.imap_host
        self._imap_port = email_cfg.imap_port
        self._smtp_host = email_cfg.smtp_host
        self._smtp_port = email_cfg.smtp_port
        self._username = email_cfg.username
        self._password_env = email_cfg.password_env

        # Workspace root for attachment path validation
        self._allowed_roots: list[Path] = [
            Path(p).expanduser().resolve() for p in config.security.allowed_paths
        ]

        # IMAP connection cache
        self._imap_conn: imaplib.IMAP4_SSL | None = None
        self._imap_last_used: float = 0.0
        self._imap_lock = asyncio.Lock()

        # Rate limiting for sends
        self._send_timestamps: list[float] = []

    def _get_password(self) -> str:
        """Liest das E-Mail-Passwort aus der Umgebungsvariable.

        Raises:
            EmailError: Wenn die Umgebungsvariable nicht gesetzt ist.
        """
        import os

        password = os.environ.get(self._password_env, "")
        if not password:
            raise EmailError(t("email.missing_password", env_var=self._password_env))
        return password

    def _validate_email(self, addr: str) -> bool:
        """Validiert eine E-Mail-Adresse."""
        return bool(_EMAIL_REGEX.match(addr.strip()))

    def _validate_attachment_path(self, path_str: str) -> Path:
        """Validiert einen Anhang-Pfad gegen die Sandbox.

        Raises:
            EmailError: Wenn der Pfad nicht erlaubt ist.
        """
        try:
            path = Path(path_str).expanduser().resolve()
        except (ValueError, OSError) as exc:
            raise EmailError(f"Ungültiger Anhang-Pfad: {path_str}") from exc

        for root in self._allowed_roots:
            try:
                path.relative_to(root)
                if not path.exists():
                    raise EmailError(t("email.attachment_not_found", path=str(path_str)))
                if not path.is_file():
                    raise EmailError(f"Anhang ist keine Datei: {path_str}")
                return path
            except ValueError:
                continue

        raise EmailError(
            f"Anhang-Pfad nicht erlaubt: {path_str} — "
            f"muss in einem der erlaubten Verzeichnisse liegen."
        )

    def _check_send_rate_limit(self) -> None:
        """Prueft das Rate-Limit fuer E-Mail-Versand.

        Raises:
            EmailError: Wenn das Limit ueberschritten ist.
        """
        now = time.monotonic()
        one_hour_ago = now - 3600
        self._send_timestamps = [ts for ts in self._send_timestamps if ts > one_hour_ago]
        if len(self._send_timestamps) >= _MAX_SEND_PER_HOUR:
            raise EmailError(
                f"Rate-Limit erreicht: Maximal {_MAX_SEND_PER_HOUR} E-Mails pro Stunde. "
                f"Bitte warte, bis das Limit zurückgesetzt wird."
            )

    async def _get_imap_connection(self) -> imaplib.IMAP4_SSL:
        """Gibt eine gecachte oder neue IMAP-Verbindung zurueck."""
        async with self._imap_lock:
            now = time.monotonic()
            # Verbindung wiederverwenden, wenn sie nicht aelter als 5 Minuten ist
            if self._imap_conn is not None and (now - self._imap_last_used) < 300:
                try:
                    self._imap_conn.noop()
                    self._imap_last_used = now
                    return self._imap_conn
                except Exception:
                    # Verbindung verloren, neue erstellen
                    with contextlib.suppress(Exception):
                        self._imap_conn.logout()
                    self._imap_conn = None

            password = self._get_password()
            loop = asyncio.get_running_loop()

            def _connect() -> imaplib.IMAP4_SSL:
                ctx = ssl.create_default_context()
                conn = imaplib.IMAP4_SSL(
                    self._imap_host,
                    self._imap_port,
                    ssl_context=ctx,
                    timeout=_IMAP_TIMEOUT,
                )
                conn.login(self._username, password)
                return conn

            conn = await loop.run_in_executor(None, _connect)
            self._imap_conn = conn
            self._imap_last_used = now
            return conn

    def _fetch_emails_sync(
        self,
        conn: imaplib.IMAP4_SSL,
        folder: str,
        search_criteria: str,
        max_count: int,
    ) -> list[dict[str, Any]]:
        """Synchrones Abrufen von E-Mails ueber IMAP.

        Args:
            conn: IMAP-Verbindung.
            folder: Ordnername.
            search_criteria: IMAP SEARCH-Kriterien.
            max_count: Maximale Anzahl E-Mails.

        Returns:
            Liste von E-Mail-Dicts.
        """
        conn.select(folder, readonly=True)
        status, data = conn.search(None, search_criteria)

        if status != "OK" or not data or not data[0]:
            return []

        uids = data[0].split()
        # Neueste zuerst
        uids = list(reversed(uids[-max_count:]))

        results: list[dict[str, Any]] = []
        for uid_bytes in uids:
            uid = uid_bytes.decode("utf-8")
            status, msg_data = conn.fetch(uid_bytes, "(RFC822)")
            if status != "OK" or not msg_data or msg_data[0] is None:
                continue

            raw_bytes = msg_data[0]
            if isinstance(raw_bytes, tuple) and len(raw_bytes) >= 2:
                raw_email = raw_bytes[1]
            else:
                continue

            if isinstance(raw_email, bytes):
                msg = email.message_from_bytes(raw_email)
            else:
                msg = email.message_from_string(str(raw_email))

            results.append(_parse_email_message(msg, uid))

        return results

    async def email_read_inbox(
        self,
        count: int = 10,
        folder: str = "INBOX",
        unread_only: bool = False,
    ) -> str:
        """Liest die neuesten E-Mails aus einem Ordner.

        Args:
            count: Anzahl E-Mails (1-50, Default: 10).
            folder: IMAP-Ordner (Default: INBOX).
            unread_only: Nur ungelesene E-Mails.

        Returns:
            Formatierte E-Mail-Liste.
        """
        count = max(1, min(count, 50))
        criteria = "UNSEEN" if unread_only else "ALL"

        conn = await self._get_imap_connection()
        loop = asyncio.get_running_loop()
        emails = await loop.run_in_executor(
            None, self._fetch_emails_sync, conn, folder, criteria, count
        )

        if not emails:
            if unread_only:
                return t("email.no_emails_unread", folder=folder)
            return t("email.no_emails", folder=folder)

        return _format_email_list(emails, folder, unread_only)

    async def email_search(
        self,
        query: str = "",
        from_addr: str = "",
        subject: str = "",
        since: str = "",
        folder: str = "INBOX",
        max_results: int = 20,
    ) -> str:
        """Durchsucht E-Mails nach verschiedenen Kriterien.

        Args:
            query: Allgemeiner Suchbegriff (BODY).
            from_addr: Absender-Filter.
            subject: Betreff-Filter.
            since: Datum-Filter (ISO-Format, z.B. 2024-01-15).
            folder: IMAP-Ordner (Default: INBOX).
            max_results: Maximale Ergebnisse (1-50, Default: 20).

        Returns:
            Formatierte Suchergebnisse.
        """
        max_results = max(1, min(max_results, 50))

        # IMAP SEARCH-Kriterien aufbauen
        criteria_parts: list[str] = []
        if query:
            criteria_parts.append(f'BODY "{query}"')
        if from_addr:
            criteria_parts.append(f'FROM "{from_addr}"')
        if subject:
            criteria_parts.append(f'SUBJECT "{subject}"')
        if since:
            # ISO-Format (2024-01-15) → IMAP-Format (15-Jan-2024)
            try:
                from datetime import datetime

                dt = datetime.fromisoformat(since)
                imap_date = dt.strftime("%d-%b-%Y")
                criteria_parts.append(f"SINCE {imap_date}")
            except ValueError as exc:
                raise EmailError(f"Ungültiges Datum: {since} (erwartet: YYYY-MM-DD)") from exc

        if not criteria_parts:
            criteria_parts.append("ALL")

        search_str = " ".join(criteria_parts)

        conn = await self._get_imap_connection()
        loop = asyncio.get_running_loop()
        emails = await loop.run_in_executor(
            None, self._fetch_emails_sync, conn, folder, search_str, max_results
        )

        if not emails:
            return f"Keine E-Mails gefunden für: {search_str}"

        header = f"Suchergebnisse ({len(emails)} Treffer)"
        lines = [header, "=" * len(header), ""]
        for em in emails:
            lines.append(f"Von: {em['from']}")
            lines.append(f"An: {em['to']}")
            lines.append(f"Betreff: {em['subject']}")
            lines.append(f"Datum: {em['date']}")
            lines.append(f"UID: {em['uid']}")
            if em["has_attachments"]:
                lines.append("Anhänge: Ja")
            lines.append(f"Vorschau: {em['preview']}")
            lines.append("-" * 40)

        return "\n".join(lines)

    async def email_send(
        self,
        to: str | list[str] = "",
        subject: str = "",
        body: str = "",
        cc: str | list[str] = "",
        bcc: str | list[str] = "",
        html: bool = False,
        attachments: list[str] | None = None,
    ) -> str:
        """Sendet eine E-Mail ueber SMTP.

        Args:
            to: Empfaenger (einzeln oder Liste).
            subject: Betreff.
            body: Nachrichtentext.
            cc: CC-Empfaenger (optional).
            bcc: BCC-Empfaenger (optional).
            html: HTML-Format (Default: False).
            attachments: Liste von Datei-Pfaden (im Workspace).

        Returns:
            Bestaetigungsnachricht.
        """
        # Rate-Limit pruefen
        self._check_send_rate_limit()

        # Empfaenger normalisieren
        if isinstance(to, str):
            to_list = [addr.strip() for addr in to.split(",") if addr.strip()]
        else:
            to_list = [addr.strip() for addr in to if addr.strip()]

        if not to_list:
            raise EmailError(t("email.missing_recipient"))

        if not subject:
            raise EmailError(t("email.missing_subject"))

        if not body:
            raise EmailError(t("email.missing_body"))

        # Alle Adressen validieren
        all_addrs: list[str] = list(to_list)

        cc_list: list[str] = []
        if cc:
            if isinstance(cc, str):
                cc_list = [addr.strip() for addr in cc.split(",") if addr.strip()]
            else:
                cc_list = [addr.strip() for addr in cc if addr.strip()]
            all_addrs.extend(cc_list)

        bcc_list: list[str] = []
        if bcc:
            if isinstance(bcc, str):
                bcc_list = [addr.strip() for addr in bcc.split(",") if addr.strip()]
            else:
                bcc_list = [addr.strip() for addr in bcc if addr.strip()]
            all_addrs.extend(bcc_list)

        for addr in all_addrs:
            if not self._validate_email(addr):
                raise EmailError(f"Ungültige E-Mail-Adresse: {addr}")

        # Nachricht aufbauen
        if attachments:
            msg = MIMEMultipart()
            content_type = "html" if html else "plain"
            msg.attach(MIMEText(body, content_type, "utf-8"))

            for att_path_str in attachments:
                att_path = self._validate_attachment_path(att_path_str)
                att_data = att_path.read_bytes()
                att_part = MIMEApplication(att_data, Name=att_path.name)
                att_part["Content-Disposition"] = f'attachment; filename="{att_path.name}"'
                msg.attach(att_part)
        else:
            content_type = "html" if html else "plain"
            msg = MIMEText(body, content_type, "utf-8")

        msg["From"] = self._username
        msg["To"] = ", ".join(to_list)
        if cc_list:
            msg["Cc"] = ", ".join(cc_list)
        msg["Subject"] = subject

        # SMTP-Versand
        password = self._get_password()
        loop = asyncio.get_running_loop()

        def _send() -> None:
            ctx = ssl.create_default_context()
            if self._smtp_port == 465:
                # SMTP_SSL (implizites TLS)
                with smtplib.SMTP_SSL(
                    self._smtp_host,
                    self._smtp_port,
                    context=ctx,
                    timeout=_SMTP_TIMEOUT,
                ) as server:
                    server.login(self._username, password)
                    server.sendmail(self._username, all_addrs, msg.as_string())
            else:
                # STARTTLS (explizites TLS, typisch Port 587)
                with smtplib.SMTP(
                    self._smtp_host,
                    self._smtp_port,
                    timeout=_SMTP_TIMEOUT,
                ) as server:
                    server.ehlo()
                    server.starttls(context=ctx)
                    server.ehlo()
                    server.login(self._username, password)
                    server.sendmail(self._username, all_addrs, msg.as_string())

        await loop.run_in_executor(None, _send)

        # Rate-Limit-Timestamp erfassen
        self._send_timestamps.append(time.monotonic())

        att_count = len(attachments) if attachments else 0
        log.info(
            "email_sent",
            to=to_list,
            subject=subject[:60],
            attachments=att_count,
        )

        recipient_str = ", ".join(to_list)
        return t(
            "email.send_success_full",
            to=recipient_str,
            subject=subject,
            count=att_count,
        )

    async def email_summarize(
        self,
        count: int = 20,
        folder: str = "INBOX",
    ) -> str:
        """Fasst den Posteingang algorithmisch zusammen.

        Gruppiert E-Mails nach Absender und Thread, gibt Statistiken zurueck.

        Args:
            count: Anzahl E-Mails zum Analysieren (1-50, Default: 20).
            folder: IMAP-Ordner (Default: INBOX).

        Returns:
            Formatierte Zusammenfassung.
        """
        count = max(1, min(count, 50))

        conn = await self._get_imap_connection()
        loop = asyncio.get_running_loop()
        emails = await loop.run_in_executor(
            None, self._fetch_emails_sync, conn, folder, "ALL", count
        )

        if not emails:
            return f"Keine E-Mails in {folder} zum Zusammenfassen."

        # Gruppierung nach Absender
        by_sender: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for em in emails:
            sender = em["from"]
            # Nur E-Mail-Adresse extrahieren (ohne Name)
            match = re.search(r"<([^>]+)>", sender)
            sender_addr = match.group(1) if match else sender
            by_sender[sender_addr].append(em)

        # Gruppierung nach Thread (basierend auf Betreff)
        by_thread: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for em in emails:
            subj = em["subject"]
            # Re:/Fwd: entfernen fuer Thread-Gruppierung
            clean_subj = re.sub(r"^(Re|Fwd|AW|WG):\s*", "", subj, flags=re.IGNORECASE).strip()
            if not clean_subj:
                clean_subj = "(kein Betreff)"
            by_thread[clean_subj].append(em)

        # Statistiken
        total = len(emails)
        with_attachments = sum(1 for em in emails if em["has_attachments"])
        unique_senders = len(by_sender)
        unique_threads = len(by_thread)

        lines: list[str] = [
            f"Posteingang-Zusammenfassung ({folder})",
            "=" * 40,
            "",
            f"Analysierte E-Mails: {total}",
            f"Eindeutige Absender: {unique_senders}",
            f"Eindeutige Threads: {unique_threads}",
            f"Mit Anhängen: {with_attachments}",
            "",
            "Top-Absender:",
            "-" * 20,
        ]

        # Top-Absender (nach Haeufigkeit)
        sorted_senders = sorted(by_sender.items(), key=lambda x: len(x[1]), reverse=True)
        for sender, sender_emails in sorted_senders[:5]:
            lines.append(f"  {sender}: {len(sender_emails)} E-Mail(s)")
            for em in sender_emails[:2]:
                lines.append(f"    - {em['subject'][:60]}")

        lines.extend(["", "Aktive Threads:", "-" * 20])

        # Top-Threads (nach Nachrichten-Anzahl)
        sorted_threads = sorted(by_thread.items(), key=lambda x: len(x[1]), reverse=True)
        for thread_subj, thread_emails in sorted_threads[:5]:
            lines.append(f"  [{len(thread_emails)}x] {thread_subj[:60]}")
            participants = {em["from"] for em in thread_emails}
            if len(participants) > 1:
                lines.append(f"    Teilnehmer: {len(participants)}")

        return "\n".join(lines)


def _format_email_list(emails: list[dict[str, Any]], folder: str, unread_only: bool) -> str:
    """Formatiert eine E-Mail-Liste fuer die Ausgabe."""
    label = f"E-Mails in {folder}"
    if unread_only:
        label += " (ungelesen)"
    header = f"{label} ({len(emails)})"
    lines = [header, "=" * len(header), ""]

    for em in emails:
        lines.append(f"Von: {em['from']}")
        lines.append(f"An: {em['to']}")
        lines.append(f"Betreff: {em['subject']}")
        lines.append(f"Datum: {em['date']}")
        lines.append(f"UID: {em['uid']}")
        if em["has_attachments"]:
            lines.append("Anhänge: Ja")
        lines.append(f"Vorschau: {em['preview']}")
        lines.append("-" * 40)

    return "\n".join(lines)


def register_email_tools(
    mcp_client: Any,
    config: CognithorConfig,
) -> EmailTools | None:
    """Registriert E-Mail-Tools beim MCP-Client.

    Registriert nur, wenn email.enabled=True und Credentials verfuegbar sind.

    Args:
        mcp_client: JarvisMCPClient-Instanz.
        config: CognithorConfig mit email-Sektion.

    Returns:
        EmailTools-Instanz oder None wenn deaktiviert.
    """
    import os

    email_cfg = getattr(config, "email", None)
    if email_cfg is None or not email_cfg.enabled:
        log.debug("email_tools_disabled")
        return None

    # Pruefe ob Passwort-Umgebungsvariable gesetzt ist
    password_env = email_cfg.password_env
    if not os.environ.get(password_env, ""):
        log.warning(
            "email_tools_no_password",
            env_var=password_env,
        )
        return None

    if not email_cfg.imap_host or not email_cfg.smtp_host:
        log.warning("email_tools_missing_host")
        return None

    if not email_cfg.username:
        log.warning("email_tools_missing_username")
        return None

    em = EmailTools(config)

    mcp_client.register_builtin_handler(
        "email_read_inbox",
        em.email_read_inbox,
        description=(
            "E-Mails aus dem Posteingang lesen. "
            "Gibt Absender, Betreff, Datum und eine Vorschau zurück."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "count": {
                    "type": "integer",
                    "description": "Anzahl E-Mails (1-50, Default: 10)",
                    "default": 10,
                },
                "folder": {
                    "type": "string",
                    "description": "IMAP-Ordner (Default: INBOX)",
                    "default": "INBOX",
                },
                "unread_only": {
                    "type": "boolean",
                    "description": "Nur ungelesene E-Mails",
                    "default": False,
                },
            },
        },
    )

    mcp_client.register_builtin_handler(
        "email_search",
        em.email_search,
        description=("E-Mails durchsuchen nach Absender, Betreff, Inhalt oder Datum."),
        input_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Suchbegriff (durchsucht den E-Mail-Body)",
                    "default": "",
                },
                "from_addr": {
                    "type": "string",
                    "description": "Absender-Filter",
                    "default": "",
                },
                "subject": {
                    "type": "string",
                    "description": "Betreff-Filter",
                    "default": "",
                },
                "since": {
                    "type": "string",
                    "description": "Datum-Filter (ISO-Format, z.B. 2024-01-15)",
                    "default": "",
                },
                "folder": {
                    "type": "string",
                    "description": "IMAP-Ordner (Default: INBOX)",
                    "default": "INBOX",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximale Ergebnisse (1-50, Default: 20)",
                    "default": 20,
                },
            },
        },
    )

    mcp_client.register_builtin_handler(
        "email_send",
        em.email_send,
        description=(
            "E-Mail senden über SMTP. Unterstützt Text und HTML, "
            "CC/BCC und Datei-Anhänge aus dem Workspace."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "to": {
                    "type": "string",
                    "description": "Empfänger (kommagetrennt oder einzeln)",
                },
                "subject": {
                    "type": "string",
                    "description": "Betreff",
                },
                "body": {
                    "type": "string",
                    "description": "Nachrichtentext",
                },
                "cc": {
                    "type": "string",
                    "description": "CC-Empfänger (kommagetrennt)",
                    "default": "",
                },
                "bcc": {
                    "type": "string",
                    "description": "BCC-Empfänger (kommagetrennt)",
                    "default": "",
                },
                "html": {
                    "type": "boolean",
                    "description": "HTML-Format (Default: False)",
                    "default": False,
                },
                "attachments": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Liste von Datei-Pfaden im Workspace",
                    "default": [],
                },
            },
            "required": ["to", "subject", "body"],
        },
    )

    mcp_client.register_builtin_handler(
        "email_summarize",
        em.email_summarize,
        description=(
            "Posteingang zusammenfassen: Gruppierung nach Absender und Thread, "
            "Statistiken und Top-Absender."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "count": {
                    "type": "integer",
                    "description": "Anzahl E-Mails zum Analysieren (1-50, Default: 20)",
                    "default": 20,
                },
                "folder": {
                    "type": "string",
                    "description": "IMAP-Ordner (Default: INBOX)",
                    "default": "INBOX",
                },
            },
        },
    )

    log.info(
        "email_tools_registered",
        tools=["email_read_inbox", "email_search", "email_send", "email_summarize"],
    )
    return em
