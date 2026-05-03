"""storage — persistent storage layer for the identity subsystem.

Currently ships ``local_store`` (SQLite-backed) and
``merkle_batcher`` (audit-trail anchoring helper). The Arweave / IPFS /
Blockchain providers were scaffolded but never wired into a runtime
path; they were removed in Sprint-23 PR#G to keep the surface
professional. If a remote-anchor provider is needed in future, it
should be added behind the existing ``LocalStore`` interface so the
fallback chain composes cleanly.
"""
