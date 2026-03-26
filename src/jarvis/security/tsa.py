"""RFC 3161 Timestamp Authority Client.

Creates timestamp requests via OpenSSL CLI (universally available),
sends them to a TSA server, and stores the signed responses.

The TSA response proves that a specific hash (and thus the entire
audit chain up to that point) existed at a specific time.

Usage:
    client = TSAClient(tsa_url="https://freetsa.org/tsr",
                       storage_dir=Path("~/.jarvis/tsa/"))
    tsr_path = client.request_timestamp(sha256_hex, "2026-03-25")

Verification (manual, with OpenSSL):
    openssl ts -verify -in audit_2026-03-25.tsr \\
               -digest <sha256hex> -sha256 \\
               -CAfile cacert.pem -untrusted tsa.crt
"""

from __future__ import annotations

import shutil
import subprocess
import urllib.request
from pathlib import Path
from typing import Any

from jarvis.utils.logging import get_logger

log = get_logger(__name__)

__all__ = ["TSAClient"]

# Default TSA servers (free, no registration)
DEFAULT_TSA_URL = "https://freetsa.org/tsr"
FALLBACK_TSA_URLS = [
    "http://timestamp.digicert.com",
    "http://timestamp.apple.com/ts01",
]


class TSAClient:
    """RFC 3161 Timestamp Authority client using OpenSSL CLI.

    Workflow:
      1. build_request(digest) -> creates .tsq file via `openssl ts -query`
      2. send_request(tsq_path) -> HTTP POST to TSA, returns .tsr bytes
      3. store_response(date, tsr_bytes) -> saves .tsr alongside audit logs

    If OpenSSL is not available, falls back to raw urllib POST with
    manually constructed minimal ASN.1 DER request (SHA-256 only).
    """

    def __init__(
        self,
        tsa_url: str = DEFAULT_TSA_URL,
        storage_dir: Path | str = "",
    ) -> None:
        self._tsa_url = tsa_url
        self._storage_dir = Path(storage_dir) if storage_dir else Path.home() / ".jarvis" / "tsa"
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        self._openssl = shutil.which("openssl")

    def has_openssl(self) -> bool:
        """Check if OpenSSL CLI is available."""
        return self._openssl is not None

    def build_request(
        self,
        sha256_hex: str,
        output_path: Path | None = None,
    ) -> Path | None:
        """Build a TimeStampReq (.tsq) file for a SHA-256 digest.

        Args:
            sha256_hex: SHA-256 hex digest of the data to timestamp.
            output_path: Where to write the .tsq file.

        Returns:
            Path to the .tsq file, or None if OpenSSL unavailable.
        """
        if not self._openssl:
            log.debug("tsa_openssl_not_available")
            return None

        if output_path is None:
            output_path = self._storage_dir / f"request_{sha256_hex[:16]}.tsq"

        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            result = subprocess.run(
                [
                    self._openssl,
                    "ts",
                    "-query",
                    "-digest",
                    sha256_hex,
                    "-sha256",
                    "-cert",
                    "-no_nonce",
                    "-out",
                    str(output_path),
                ],
                capture_output=True,
                timeout=10,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            if result.returncode != 0:
                log.warning(
                    "tsa_build_request_failed",
                    stderr=result.stderr.decode(errors="replace")[:200],
                )
                return None
            return output_path
        except Exception as exc:
            log.warning("tsa_build_request_error", error=str(exc))
            return None

    def send_request(self, tsq_path: Path) -> bytes | None:
        """Send a .tsq file to the TSA server via HTTP POST.

        Args:
            tsq_path: Path to the TimeStampReq file.

        Returns:
            Raw TSA response bytes (.tsr), or None on failure.
        """
        if not tsq_path.exists():
            return None

        tsq_data = tsq_path.read_bytes()
        urls = [self._tsa_url] + [u for u in FALLBACK_TSA_URLS if u != self._tsa_url]

        for url in urls:
            try:
                req = urllib.request.Request(
                    url,
                    data=tsq_data,
                    headers={"Content-Type": "application/timestamp-query"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=15) as resp:
                    if resp.status == 200:
                        tsr_data = resp.read()
                        log.info("tsa_response_received", url=url, size=len(tsr_data))
                        return tsr_data
            except Exception as exc:
                log.debug("tsa_send_failed", url=url, error=str(exc))
                continue

        log.warning("tsa_all_servers_failed")
        return None

    def request_timestamp(
        self,
        sha256_hex: str,
        date_str: str,
    ) -> Path | None:
        """Full flow: build request -> send to TSA -> store response.

        Args:
            sha256_hex: SHA-256 hex digest of the audit anchor.
            date_str: Date string for the filename (e.g., "2026-03-25").

        Returns:
            Path to the stored .tsr file, or None on failure.
        """
        # Step 1: Build .tsq
        tsq_path = self._storage_dir / f"audit_{date_str}.tsq"
        built = self.build_request(sha256_hex, tsq_path)
        if built is None:
            # Fallback: try raw HTTP POST without openssl
            return self._request_timestamp_raw(sha256_hex, date_str)

        # Step 2: Send to TSA
        tsr_data = self.send_request(tsq_path)
        if tsr_data is None:
            return None

        # Step 3: Store response
        return self.store_response(date_str, tsr_data)

    def _request_timestamp_raw(
        self,
        sha256_hex: str,
        date_str: str,
    ) -> Path | None:
        """Fallback: Build minimal ASN.1 DER request without OpenSSL.

        This builds a bare-minimum TimeStampReq for SHA-256 digests.
        Not as robust as OpenSSL, but works when it's not available.
        """
        digest_bytes = bytes.fromhex(sha256_hex)

        # Minimal ASN.1 DER TimeStampReq for SHA-256
        # SEQUENCE {
        #   INTEGER 1 (version)
        #   SEQUENCE { (messageImprint)
        #     SEQUENCE { (hashAlgorithm - SHA-256 OID: 2.16.840.1.101.3.4.2.1)
        #       OID 2.16.840.1.101.3.4.2.1
        #       NULL
        #     }
        #     OCTET STRING (32 bytes hash)
        #   }
        #   BOOLEAN TRUE (certReq)
        # }
        sha256_oid = bytes(
            [
                0x30,
                0x0D,  # SEQUENCE (13 bytes)
                0x06,
                0x09,  # OID (9 bytes)
                0x60,
                0x86,
                0x48,
                0x01,
                0x65,
                0x03,
                0x04,
                0x02,
                0x01,  # SHA-256
                0x05,
                0x00,  # NULL
            ]
        )
        message_imprint = (
            bytes([0x30, len(sha256_oid) + 2 + len(digest_bytes)])
            + sha256_oid
            + bytes([0x04, len(digest_bytes)])
            + digest_bytes
        )
        version = bytes([0x02, 0x01, 0x01])  # INTEGER 1
        cert_req = bytes([0x01, 0x01, 0xFF])  # BOOLEAN TRUE
        inner = version + message_imprint + cert_req
        tsq_data = bytes([0x30, len(inner)]) + inner

        # Send directly
        try:
            req = urllib.request.Request(
                self._tsa_url,
                data=tsq_data,
                headers={"Content-Type": "application/timestamp-query"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                if resp.status == 200:
                    tsr_data = resp.read()
                    return self.store_response(date_str, tsr_data)
        except Exception as exc:
            log.debug("tsa_raw_request_failed", error=str(exc))

        return None

    def store_response(self, date_str: str, tsr_data: bytes) -> Path:
        """Store a TSA response as .tsr file.

        Args:
            date_str: Date string for filename.
            tsr_data: Raw TSA response bytes.

        Returns:
            Path to the stored .tsr file.
        """
        tsr_path = self._storage_dir / f"audit_{date_str}.tsr"
        tsr_path.write_bytes(tsr_data)
        log.info("tsa_response_stored", path=str(tsr_path), size=len(tsr_data))
        return tsr_path

    def list_timestamps(self) -> list[dict[str, Any]]:
        """List all stored TSA timestamps.

        Returns:
            List of dicts with date, path, size for each .tsr file.
        """
        results = []
        for tsr_file in sorted(self._storage_dir.glob("audit_*.tsr")):
            # Extract date from filename: audit_2026-03-25.tsr -> 2026-03-25
            stem = tsr_file.stem  # audit_2026-03-25
            date_str = stem.replace("audit_", "")
            results.append(
                {
                    "date": date_str,
                    "path": str(tsr_file),
                    "size_bytes": tsr_file.stat().st_size,
                }
            )
        return results

    def get_timestamp(self, date_str: str) -> Path | None:
        """Get a stored TSA response for a specific date.

        Returns:
            Path to .tsr file, or None if not found.
        """
        tsr_path = self._storage_dir / f"audit_{date_str}.tsr"
        return tsr_path if tsr_path.exists() else None

    def verify_timestamp(
        self,
        date_str: str,
        sha256_hex: str,
        ca_cert: Path | None = None,
        tsa_cert: Path | None = None,
    ) -> dict[str, Any]:
        """Verify a stored TSA response using OpenSSL.

        Args:
            date_str: Date of the timestamp.
            sha256_hex: Expected SHA-256 digest.
            ca_cert: Path to CA certificate (optional).
            tsa_cert: Path to TSA certificate (optional).

        Returns:
            Dict with verified (bool), output (str), error (str).
        """
        if not self._openssl:
            return {"verified": False, "error": "OpenSSL not available"}

        tsr_path = self.get_timestamp(date_str)
        if tsr_path is None:
            return {"verified": False, "error": f"No timestamp for {date_str}"}

        cmd = [
            self._openssl,
            "ts",
            "-verify",
            "-in",
            str(tsr_path),
            "-digest",
            sha256_hex,
            "-sha256",
        ]
        if ca_cert and ca_cert.exists():
            cmd.extend(["-CAfile", str(ca_cert)])
        if tsa_cert and tsa_cert.exists():
            cmd.extend(["-untrusted", str(tsa_cert)])

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=10,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            output = result.stdout.decode(errors="replace").strip()
            error = result.stderr.decode(errors="replace").strip()
            verified = result.returncode == 0 and "verification: ok" in output.lower()
            return {
                "verified": verified,
                "output": output,
                "error": error if not verified else "",
            }
        except Exception as exc:
            return {"verified": False, "error": str(exc)}
