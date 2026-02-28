# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.22.x  | Yes       |
| < 0.22  | No        |

## Reporting a Vulnerability

If you discover a security vulnerability in Cognithor, please report it responsibly:

1. **Do NOT open a public issue.** Security vulnerabilities must be reported privately.
2. **Email:** Send a detailed report to the repository owner via GitHub's private vulnerability reporting feature (Security tab → "Report a vulnerability").
3. **Include:**
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

We aim to acknowledge reports within 48 hours and provide a fix within 7 days for critical issues.

## Security Architecture

Cognithor implements defense-in-depth with multiple security layers:

- **Gatekeeper** — Deterministic policy engine (no LLM). Every tool call is validated against security policies with 4 risk levels: GREEN (auto-approve) → YELLOW (inform) → ORANGE (require approval) → RED (block).
- **Sandbox** — Multi-level execution isolation: Process-level → Linux Namespaces (nsjail) → Docker containers → Windows Job Objects.
- **Audit Trail** — Append-only JSONL log with SHA-256 hash chain. Tamper-evident. Credentials are masked before logging.
- **Credential Vault** — Fernet-encrypted (AES-256) per-agent secret storage. Keys never appear in logs or API responses.
- **Input Sanitization** — Protection against shell injection, path traversal, and prompt injection attacks.
- **Path Sandbox** — File operations restricted to explicitly allowed directories.
- **Red-Teaming** — Automated offensive security test suite (1,425 LOC).

## Credential Handling

- API keys in configuration are masked (`***`) in all API responses by default.
- The `.env` file (`~/.jarvis/.env`) is excluded from version control via `.gitignore`.
- The Control Center API never writes masked placeholder values (`***`) back to configuration files.

## Dependencies

We regularly review dependencies for known vulnerabilities. If you find a vulnerable dependency, please report it using the process above.
