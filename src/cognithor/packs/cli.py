"""cognithor pack CLI — thin argparse wrapper over PackInstaller.

Entry point: ``cognithor pack <subcommand> [args]``

Subcommands
-----------
install <path-or-url>   Install a pack from a local zip or HTTP(S) URL.
list                    List all installed packs.
remove <namespace/id>   Remove an installed pack.
update <namespace/id>   MVP stub: prints re-install instructions.
accept-eula <namespace/id>  Re-prompt and record EULA acceptance.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, cast

from cognithor.packs.errors import PackInstallError
from cognithor.packs.installer import PackInstaller

if TYPE_CHECKING:
    from collections.abc import Sequence

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_packs_root() -> Path:
    """Return the packs root directory from environment / defaults.

    Resolution order:
    1. ``$COGNITHOR_PACKS_DIR``
    2. ``$COGNITHOR_HOME/packs``
    3. ``~/.cognithor/packs``
    """
    env_dir = os.environ.get("COGNITHOR_PACKS_DIR")
    if env_dir:
        return Path(env_dir)

    cognithor_home = os.environ.get("COGNITHOR_HOME")
    if cognithor_home:
        return Path(cognithor_home) / "packs"

    return Path.home() / ".cognithor" / "packs"


def _make_installer() -> PackInstaller:
    return PackInstaller(packs_root=_resolve_packs_root())


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------


def _cmd_install(args: argparse.Namespace) -> int:
    installer = _make_installer()
    source: str = args.source
    try:
        if source.startswith("http://") or source.startswith("https://"):
            manifest = installer.install_from_url(source)
        else:
            manifest = installer.install_from_path(Path(source))
    except PackInstallError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"[OK] Installed {manifest.qualified_id} v{manifest.version}")
    return 0


def _cmd_list(args: argparse.Namespace) -> int:
    installer = _make_installer()
    try:
        packs = installer.list_installed()
    except PackInstallError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if not packs:
        print("(no packs installed)")
        return 0

    # Simple tabular output
    col_id = max(len(p.qualified_id) for p in packs)
    col_ver = max(len(p.version) for p in packs)
    header = f"{'PACK ID':<{col_id}}  {'VERSION':<{col_ver}}  DISPLAY NAME"
    print(header)
    print("-" * len(header))
    for p in packs:
        print(f"{p.qualified_id:<{col_id}}  {p.version:<{col_ver}}  {p.display_name}")
    return 0


def _cmd_remove(args: argparse.Namespace) -> int:
    installer = _make_installer()
    try:
        installer.remove(args.qualified_id)
    except PackInstallError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"[OK] Removed {args.qualified_id}")
    return 0


def _cmd_update(args: argparse.Namespace) -> int:
    print(
        "Update is not yet automated.\n"
        "To update a pack, re-install it with the URL from your purchase email:\n"
        "  cognithor pack install https://example.com/updated-pack.zip"
    )
    return 0


def _cmd_rollback(args: argparse.Namespace) -> int:
    """``cognithor pack rollback <qualified_id> [--to-version X]``.

    TRUST-4 (operational-trust audit, 2026-05-04). Restore a pack from
    its pre-upgrade snapshot taken automatically during the last
    ``install``. With ``--list`` shows available versions instead of
    rolling back.
    """
    installer = _make_installer()
    try:
        if args.list:
            versions = installer.list_backups(args.qualified_id)
            if not versions:
                print(f"No backups available for {args.qualified_id}.")
                return 0
            print(f"Available rollback versions for {args.qualified_id}:")
            for v in versions:
                print(f"  - {v}")
            return 0

        manifest = installer.rollback(
            args.qualified_id,
            to_version=args.to_version,
        )
    except PackInstallError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"[OK] Rolled back {args.qualified_id} to version {manifest.version}")
    return 0


def _cmd_accept_eula(args: argparse.Namespace) -> int:
    installer = _make_installer()
    try:
        installer.accept_eula(args.qualified_id)
    except PackInstallError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"[OK] EULA accepted for {args.qualified_id}")
    return 0


def _cmd_create(args: argparse.Namespace) -> int:
    from cognithor.packs.scaffolder import scaffold_pack

    output = Path(args.output) if args.output else _resolve_packs_root()
    try:
        pack_dir = scaffold_pack(
            output_dir=output,
            name=args.name,
            namespace=args.namespace,
            description=args.description or f"{args.name} pack for Cognithor",
            with_leads=args.with_leads,
            license_type=args.license,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"\n[OK] Created pack at {pack_dir}/\n")
    print("  pack_manifest.json    [OK]")
    print("  pack.py               [OK]")
    print("  eula.md               [OK]")
    print("  src/__init__.py       [OK]")
    print("  tests/test_pack.py    [OK]")
    print("  catalog/catalog.mdx   [OK]")
    print("\nNext steps:")
    print("  1. Edit src/ to add your tools")
    print("  2. Wire them in pack.py register()")
    print(f"  3. Test: cognithor pack install {pack_dir}")
    return 0


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Return the top-level ``cognithor pack`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="cognithor pack",
        description="Manage agent packs for Cognithor.",
    )
    sub = parser.add_subparsers(dest="command", metavar="<command>")
    sub.required = True

    # install
    p_install = sub.add_parser("install", help="Install a pack from a local zip or URL.")
    p_install.add_argument(
        "source",
        metavar="PATH_OR_URL",
        help="Local .zip path or HTTP(S) URL.",
    )
    p_install.set_defaults(func=_cmd_install)

    # list
    p_list = sub.add_parser("list", help="List all installed packs.")
    p_list.set_defaults(func=_cmd_list)

    # remove
    p_remove = sub.add_parser("remove", help="Remove an installed pack.")
    p_remove.add_argument(
        "qualified_id",
        metavar="NAMESPACE/PACK_ID",
        help="Qualified pack identifier, e.g. cognithor-official/my-pack.",
    )
    p_remove.set_defaults(func=_cmd_remove)

    # update
    p_update = sub.add_parser("update", help="Update an installed pack (MVP stub).")
    p_update.add_argument(
        "qualified_id",
        metavar="NAMESPACE/PACK_ID",
        nargs="?",
        help="Qualified pack identifier (optional for stub).",
    )
    p_update.set_defaults(func=_cmd_update)

    # accept-eula
    p_eula = sub.add_parser("accept-eula", help="Re-accept the EULA for an installed pack.")
    p_eula.add_argument(
        "qualified_id",
        metavar="NAMESPACE/PACK_ID",
        help="Qualified pack identifier.",
    )
    p_eula.set_defaults(func=_cmd_accept_eula)

    # rollback (TRUST-4)
    p_rollback = sub.add_parser(
        "rollback",
        help="Roll an installed pack back to a previously-snapshotted version.",
    )
    p_rollback.add_argument(
        "qualified_id",
        metavar="NAMESPACE/PACK_ID",
        help="Qualified pack identifier.",
    )
    p_rollback.add_argument(
        "--to-version",
        dest="to_version",
        metavar="VERSION",
        default=None,
        help="Specific version to restore. Default: most recent snapshot.",
    )
    p_rollback.add_argument(
        "--list",
        action="store_true",
        help="List available rollback versions instead of rolling back.",
    )
    p_rollback.set_defaults(func=_cmd_rollback)

    # create
    p_create = sub.add_parser("create", help="Scaffold a new pack from template.")
    p_create.add_argument("--name", required=True, help="Pack identifier (lowercase, e.g. my-pack)")
    p_create.add_argument("--namespace", default="cognithor-community", help="Publisher namespace")
    p_create.add_argument("--description", default="", help="Pack description")
    p_create.add_argument("--with-leads", action="store_true", help="Include LeadSource stub")
    p_create.add_argument(
        "--license",
        default="apache-2.0",
        choices=["apache-2.0", "proprietary"],
        help="License type",
    )
    p_create.add_argument("--output", default="", help="Output directory (default: packs root)")
    p_create.set_defaults(func=_cmd_create)

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    """Parse *argv* and dispatch to the appropriate subcommand handler.

    Parameters
    ----------
    argv:
        Argument list (defaults to ``sys.argv[1:]`` when ``None``).

    Returns
    -------
    int
        Exit code (0 = success, non-zero = failure).
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    return cast("int", args.func(args))


if __name__ == "__main__":
    sys.exit(main())
