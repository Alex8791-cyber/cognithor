#!/usr/bin/env python3
"""
Cognithor · Agent OS — Cinematic Terminal Demo

A ~3 minute immersive showcase of the autonomous agent operating system.

Run:    python demo.py
Fast:   python demo.py --fast
"""

from __future__ import annotations

import os
import sys
import time

# Ensure UTF-8 output on Windows (must be before any rich import)
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

from rich import box
from rich.align import Align
from rich.columns import Columns
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

# ── Globals ─────────────────────────────────────────────────────────
VERSION = "0.22.0"
FAST = "--fast" in sys.argv
console = Console(highlight=False)


def pause(seconds: float = 1.0) -> None:
    """Dramatic pause (skipped in --fast mode)."""
    if not FAST:
        time.sleep(seconds)


def typing(text: str, speed: float = 0.025) -> None:
    """Simulate human typing, character by character."""
    for ch in text:
        sys.stdout.write(ch)
        sys.stdout.flush()
        if not FAST:
            time.sleep(speed)
    sys.stdout.write("\n")
    sys.stdout.flush()


_ANSI = {
    "bright_cyan": "\033[96m",
    "bright_red": "\033[91m",
    "bright_green": "\033[92m",
    "": "",
}
_ANSI_RESET = "\033[0m"


def streaming(text: str, speed: float = 0.012, style: str = "bright_cyan") -> None:
    """Simulate LLM streaming output, word by word with ANSI color."""
    sys.stdout.write(_ANSI.get(style, ""))
    words = text.split(" ")
    for i, word in enumerate(words):
        sys.stdout.write(word + (" " if i < len(words) - 1 else ""))
        sys.stdout.flush()
        if not FAST:
            time.sleep(speed)
    sys.stdout.write(_ANSI_RESET + "\n")
    sys.stdout.flush()


# ════════════════════════════════════════════════════════════════════
#  SCENE 1 ── Boot Sequence
# ════════════════════════════════════════════════════════════════════

LOGO = r"""
 ██████╗ ██████╗  ██████╗ ███╗   ██╗██╗████████╗██╗  ██╗ ██████╗ ██████╗
██╔════╝██╔═══██╗██╔════╝ ████╗  ██║██║╚══██╔══╝██║  ██║██╔═══██╗██╔══██╗
██║     ██║   ██║██║  ███╗██╔██╗ ██║██║   ██║   ███████║██║   ██║██████╔╝
██║     ██║   ██║██║   ██║██║╚██╗██║██║   ██║   ██╔══██║██║   ██║██╔══██╗
╚██████╗╚██████╔╝╚██████╔╝██║ ╚████║██║   ██║   ██║  ██║╚██████╔╝██║  ██║
 ╚═════╝ ╚═════╝  ╚═════╝ ╚═╝  ╚═══╝╚═╝   ╚═╝   ╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═╝
"""


def scene_boot() -> None:
    """Boot sequence with ASCII art and system init spinner."""
    console.clear()
    pause(0.5)

    # Logo — line by line reveal
    for line in LOGO.strip().splitlines():
        console.print(f"[bold bright_cyan]{line}[/bold bright_cyan]")
        pause(0.07)

    console.print()
    console.print(Align.center(Text("· Agent OS ·", style="bold white")))
    console.print(Align.center(Text(f"v{VERSION}", style="dim")))
    console.print()
    pause(0.6)

    # System init checklist
    steps = [
        "Loading configuration",
        "Initializing PGE Trinity",
        "Connecting 5-tier memory",
        "Starting MCP tool servers (13)",
        "Registering security policies",
        "Warming up embedding cache",
    ]

    with Progress(
        SpinnerColumn("dots"),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as prog:
        for step in steps:
            tid = prog.add_task(step, total=1)
            pause(0.5)
            prog.update(
                tid,
                completed=1,
                description=f"[green]  {step}[/green]",
            )

    pause(0.3)
    console.print()
    console.rule("[bold green]System Online[/bold green]")
    pause(1.0)


# ════════════════════════════════════════════════════════════════════
#  SCENE 2 ── LLM Provider Scan
# ════════════════════════════════════════════════════════════════════

PROVIDERS = [
    ("Ollama",        "Local", "qwen3:32b",                "localhost:11434"),
    ("OpenAI",        "Cloud", "gpt-5.2",                  "api.openai.com"),
    ("Anthropic",     "Cloud", "claude-opus-4-6",          "api.anthropic.com"),
    ("Google Gemini", "Cloud", "gemini-2.5-pro",           "generativelanguage.googleapis.com"),
    ("Groq",          "Cloud", "llama-4-maverick",         "api.groq.com"),
    ("DeepSeek",      "Cloud", "deepseek-chat",            "api.deepseek.com"),
    ("Mistral",       "Cloud", "mistral-large-latest",     "api.mistral.ai"),
    ("Together AI",   "Cloud", "Llama-4-Maverick",         "api.together.xyz"),
    ("OpenRouter",    "Cloud", "claude-opus-4.6",          "openrouter.ai"),
    ("xAI (Grok)",    "Cloud", "grok-4-1-fast-reasoning",  "api.x.ai"),
    ("Cerebras",      "Cloud", "gpt-oss-120b",             "api.cerebras.ai"),
    ("GitHub Models", "Cloud", "gpt-4.1",                  "models.inference.ai.azure.com"),
    ("AWS Bedrock",   "Cloud", "claude-opus-4-6",          "bedrock-runtime.amazonaws.com"),
    ("Hugging Face",  "Cloud", "Llama-3.3-70B",            "api-inference.huggingface.co"),
    ("Moonshot/Kimi", "Cloud", "kimi-k2.5",                "api.moonshot.cn"),
]


def scene_providers() -> None:
    """Animated provider table — each row lights up one by one."""
    console.print()
    console.print(
        Panel(
            "[bold]LLM Provider Scan[/bold]",
            style="cyan",
            expand=False,
        )
    )
    pause(0.4)

    table = Table(
        box=box.ROUNDED,
        show_header=True,
        header_style="bold bright_white",
        border_style="cyan",
        title="[bold]Multi-LLM Backend Layer[/bold]",
        title_style="bold cyan",
    )
    table.add_column("#", style="dim", width=3, justify="right")
    table.add_column("Provider", min_width=16)
    table.add_column("Type", width=6)
    table.add_column("Default Model", min_width=22)
    table.add_column("Endpoint", style="dim")
    table.add_column("Status", justify="center", width=10)

    with Live(table, console=console, refresh_per_second=15):
        for idx, (name, ptype, model, endpoint) in enumerate(PROVIDERS, 1):
            table.add_row(
                str(idx),
                f"[bold]{name}[/bold]",
                ptype,
                model,
                endpoint,
                "[bright_green]● READY[/bright_green]",
            )
            pause(0.14)

    console.print()
    console.print(
        f"  [bold green]{len(PROVIDERS)} providers connected[/bold green]"
    )
    pause(1.0)


# ════════════════════════════════════════════════════════════════════
#  SCENE 3 ── Channel Initialization
# ════════════════════════════════════════════════════════════════════

CHANNELS = [
    "CLI", "Web UI", "REST API", "Telegram", "Discord",
    "Slack", "WhatsApp", "Signal", "iMessage", "Teams",
    "Matrix", "Google Chat", "Mattermost", "Feishu/Lark",
    "IRC", "Twitch", "Voice",
]


def scene_channels() -> None:
    """Progress bar + grid of connected channels."""
    console.print()
    console.print(
        Panel("[bold]Channel Initialization[/bold]", style="yellow", expand=False)
    )
    pause(0.3)

    with Progress(
        SpinnerColumn("dots"),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
        console=console,
    ) as prog:
        tid = prog.add_task("Connecting channels", total=len(CHANNELS))
        for ch in CHANNELS:
            prog.update(tid, description=f"[yellow]{ch}[/yellow]")
            pause(0.1)
            prog.advance(tid)
        prog.update(
            tid, description="[green]All channels connected[/green]"
        )

    # Channel chip grid
    chips = [
        Text(f" {ch} ", style="bold white on dark_green") for ch in CHANNELS
    ]
    console.print()
    console.print(Columns(chips, padding=(0, 1), expand=False))
    console.print()
    console.print(
        f"  [bold green]{len(CHANNELS)} channels active[/bold green]"
    )
    pause(1.0)


# ════════════════════════════════════════════════════════════════════
#  SCENE 4 ── 5-Tier Cognitive Memory
# ════════════════════════════════════════════════════════════════════

MEMORY_TIERS = [
    (
        "Tier 1 · Core",
        "Identity, rules, personality",
        ["Owner: configured", "Rules: 12 active", "Personality: adaptive"],
    ),
    (
        "Tier 2 · Episodic",
        "Daily logs — what happened",
        ["Episodes: 847", "Timespan: 14 months", "Auto-archival: on"],
    ),
    (
        "Tier 3 · Semantic",
        "Knowledge graph — facts & relations",
        ["Entities: 2,341", "Relations: 5,892", "Categories: 47"],
    ),
    (
        "Tier 4 · Procedural",
        "Learned skills — how to do things",
        ["Procedures: 156", "Auto-learned: 89", "Success rate: 94%"],
    ),
    (
        "Tier 5 · Working",
        "Session context (volatile RAM)",
        ["Tokens: 12,480", "Window: 128K", "Cache hit: 87%"],
    ),
]


def scene_memory() -> None:
    """Animated tree view of the 5-tier memory system + hybrid search."""
    console.print()
    console.print(
        Panel(
            "[bold]5-Tier Cognitive Memory[/bold]",
            style="magenta",
            expand=False,
        )
    )
    pause(0.4)

    tree = Tree(
        "[bold magenta]Memory System[/bold magenta]",
        guide_style="magenta",
    )

    with Live(tree, console=console, refresh_per_second=10):
        for tier_name, desc, stats in MEMORY_TIERS:
            branch = tree.add(
                f"[bold]{tier_name}[/bold] — [dim]{desc}[/dim]"
            )
            pause(0.25)
            for stat in stats:
                branch.add(f"[green]●[/green] {stat}")
                pause(0.08)

    # Hybrid search panel
    console.print()
    search = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    search.add_column(style="bold cyan")
    search.add_column()
    search.add_row("BM25", "Full-text search (FTS5, compound words)")
    search.add_row("Vector", "Embedding similarity (cosine, LRU-cached)")
    search.add_row("Graph", "Entity relation traversal (3-hop)")
    console.print(
        Panel(
            search,
            title="[bold]3-Channel Hybrid Search[/bold]",
            border_style="magenta",
            expand=False,
        )
    )
    pause(1.0)


# ════════════════════════════════════════════════════════════════════
#  SCENE 5 ── Live Conversation
# ════════════════════════════════════════════════════════════════════


def scene_conversation() -> None:
    """Simulated streaming conversation with comparison table."""
    console.print()
    console.rule("[bold bright_white]Live Session[/bold bright_white]")
    pause(0.5)

    # User types a question
    console.print()
    sys.stdout.write("  cognithor> ")
    sys.stdout.flush()
    typing(
        "What are the key differences between React and Vue.js "
        "for our new dashboard project?"
    )
    pause(0.3)

    # Thinking spinner
    console.print()
    with console.status(
        "[bold cyan]  Planning response...[/bold cyan]", spinner="dots"
    ):
        pause(1.8)

    # Streaming answer
    console.print()
    streaming(
        "  Based on your project requirements and the team's TypeScript "
        "experience, here are the key differences:"
    )
    console.print()
    pause(0.3)

    # Comparison table (part of the AI response)
    comp = Table(
        box=box.ROUNDED,
        border_style="cyan",
        padding=(0, 1),
    )
    comp.add_column("Aspect", style="bold")
    comp.add_column("React", style="bright_blue")
    comp.add_column("Vue.js", style="bright_green")
    comp.add_row("Learning curve", "Steeper (JSX, hooks)", "Gentler (templates)")
    comp.add_row("Reactivity", "Virtual DOM + fiber", "Proxy-based (faster)")
    comp.add_row("Ecosystem", "Massive (Meta)", "Growing (community)")
    comp.add_row("Bundle size", "~42 KB", "~33 KB")
    comp.add_row("TypeScript", "Excellent", "Excellent (Vue 3)")
    comp.add_row("State mgmt", "Redux / Zustand", "Pinia (built-in)")
    console.print(comp)

    pause(0.3)
    console.print()
    streaming(
        "  Given the team's TypeScript expertise and the need for a rich "
        "plugin ecosystem, I'd recommend React with Zustand for state "
        "management. Want me to scaffold the project?"
    )
    pause(1.0)


# ════════════════════════════════════════════════════════════════════
#  SCENE 6 ── PGE Trinity in Action
# ════════════════════════════════════════════════════════════════════


def scene_pge() -> None:
    """Planner → Gatekeeper → Executor pipeline with real-looking output."""
    console.print()
    console.rule("[bold bright_white]PGE Trinity in Action[/bold bright_white]")
    pause(0.5)

    # User request
    console.print()
    sys.stdout.write("  cognithor> ")
    sys.stdout.flush()
    typing(
        "Search my knowledge base for all customer feedback from last quarter"
    )
    pause(0.5)

    # ── PLANNER ──────────────────────────────────────────────────
    plan_code = """\
# Action Plan (generated by Planner)
{
  "goal": "Retrieve Q4 customer feedback from knowledge base",
  "steps": [
    {
      "tool": "memory_search",
      "params": {
        "query": "customer feedback Q4 2025",
        "tiers": ["semantic", "episodic"],
        "limit": 20,
        "hybrid_mode": "bm25+vector+graph"
      },
      "risk_level": "GREEN"
    }
  ],
  "fallback": "broaden search to all 2025 feedback"
}"""

    console.print()
    console.print(
        Panel(
            Syntax(plan_code, "python", theme="monokai", line_numbers=False),
            title="[bold blue]  PLANNER[/bold blue]  LLM-based Planning",
            border_style="blue",
            subtitle="[dim]Model: qwen3:32b  847ms[/dim]",
        )
    )
    pause(1.0)

    # ── GATEKEEPER ───────────────────────────────────────────────
    gt = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    gt.add_row("[bold]Tool:[/bold]", "memory_search")
    gt.add_row(
        "[bold]Risk Level:[/bold]",
        "[bold green]GREEN[/bold green] (read-only, no side effects)",
    )
    gt.add_row("[bold]Policy Match:[/bold]", "ALLOW  memory queries auto-approved")
    gt.add_row("[bold]Sandbox Level:[/bold]", "L0 (Process isolation)")
    gt.add_row(
        "[bold]Decision:[/bold]",
        "[bold green] APPROVED[/bold green]",
    )

    console.print(
        Panel(
            gt,
            title=(
                "[bold green]  GATEKEEPER[/bold green]"
                "  Deterministic Policy Engine"
            ),
            border_style="green",
            subtitle="[dim]No LLM  No hallucinations  0.2ms[/dim]",
        )
    )
    pause(1.0)

    # ── EXECUTOR ─────────────────────────────────────────────────
    exec_result = """\
Results: 14 documents found (semantic: 9, episodic: 5)

Top matches:
  1. [0.94] "Q4 Customer Survey Results"      847 responses, NPS: 72
  2. [0.91] "Support Ticket Analysis Oct-Dec"  234 tickets, 96% resolved
  3. [0.88] "Product Feedback: v3.2 Release"   12 feature requests
  4. [0.85] "Enterprise Client Reviews"        5 reviews, avg 4.6/5
  5. [0.82] "Churn Analysis December 2025"     3.2% churn, -0.8% vs Q3"""

    console.print(
        Panel(
            Syntax(exec_result, "yaml", theme="monokai", line_numbers=False),
            title="[bold yellow]  EXECUTOR[/bold yellow]  Sandboxed Execution",
            border_style="yellow",
            subtitle="[dim]Tool: memory_search  23ms  SHA-256 audit logged[/dim]",
        )
    )
    pause(1.5)


# ════════════════════════════════════════════════════════════════════
#  SCENE 7 ── Security Block
# ════════════════════════════════════════════════════════════════════


def scene_security() -> None:
    """Gatekeeper blocks a dangerous request with detailed policy analysis."""
    console.print()
    console.rule("[bold red]Security Demonstration[/bold red]")
    pause(0.5)

    console.print()
    sys.stdout.write("  cognithor> ")
    sys.stdout.flush()
    typing("Delete all files in /etc and remove system logs")
    pause(0.4)

    # BLOCKED
    console.print()
    bt = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    bt.add_row("[bold]Tool:[/bold]", "shell.exec_command")
    bt.add_row("[bold]Command:[/bold]", "[red]rm -rf /etc/*[/red]")
    bt.add_row(
        "[bold]Risk Level:[/bold]",
        "[bold red]RED[/bold red]  Destructive system operation",
    )
    bt.add_row(
        "[bold]Violations:[/bold]",
        "[red]3 policy violations detected[/red]",
    )
    bt.add_row("", "[red]  PATH_FORBIDDEN   /etc outside allowed paths[/red]")
    bt.add_row("", "[red]  CMD_BLACKLISTED  recursive delete pattern blocked[/red]")
    bt.add_row("", "[red]  SCOPE_EXCEEDED   system-level destruction[/red]")
    bt.add_row(
        "[bold]Decision:[/bold]",
        "[bold red] BLOCKED[/bold red]",
    )

    console.print(
        Panel(
            bt,
            title="[bold red]GATEKEEPER  REQUEST DENIED[/bold red]",
            border_style="red",
            subtitle="[dim]Deterministic  No override possible  Logged to audit chain[/dim]",
        )
    )
    pause(0.5)

    console.print()
    streaming(
        "  I cannot execute that request. The Gatekeeper blocked this action "
        "because it involves destructive operations on system-critical paths. "
        "This protection is enforced by deterministic policy rules, not by an "
        "LLM that could be tricked or prompt-injected.",
        style="bright_red",
    )
    pause(1.0)


# ════════════════════════════════════════════════════════════════════
#  SCENE 8 ── Multi-Channel Broadcast
# ════════════════════════════════════════════════════════════════════

BROADCAST_CHANNELS = [
    ("Telegram",  "Bot  @team_channel",     "bright_blue"),
    ("Discord",   "#deployments  Embed",    "bright_magenta"),
    ("Slack",     "#ops  Block Kit",        "bright_yellow"),
    ("WhatsApp",  "Ops Group  Text",        "bright_green"),
    ("Teams",     "DevOps  Adaptive Card",  "bright_blue"),
    ("Web UI",    "Dashboard  WebSocket",   "bright_cyan"),
    ("Matrix",    "!ops:matrix.org  E2EE",  "bright_white"),
]


def scene_multichannel() -> None:
    """Same message delivered to 7 channels simultaneously."""
    console.print()
    console.print(
        Panel(
            "[bold]Multi-Channel Broadcast[/bold]",
            style="bright_blue",
            expand=False,
        )
    )
    pause(0.4)

    console.print(
        "  [dim]Broadcasting deployment notification to all active channels...[/dim]"
    )
    console.print()

    for name, detail, color in BROADCAST_CHANNELS:
        console.print(
            f"  [{color}][/{color}] [bold]{name:12s}[/bold] {detail}"
        )
        pause(0.18)

    console.print()
    console.print(
        f"  [bold green]Delivered to "
        f"{len(BROADCAST_CHANNELS)} channels in 340ms[/bold green]"
    )
    pause(1.0)


# ════════════════════════════════════════════════════════════════════
#  SCENE 9 ── Reflection & Learning
# ════════════════════════════════════════════════════════════════════


def scene_reflection() -> None:
    """Reflector analyses session, extracts facts, learns a procedure."""
    console.print()
    console.print(
        Panel(
            "[bold]Reflection & Procedural Learning[/bold]",
            style="bright_magenta",
            expand=False,
        )
    )
    pause(0.4)

    with console.status(
        "[bold magenta]  Reflector analyzing session...[/bold magenta]",
        spinner="dots",
    ):
        pause(1.8)

    # Extracted facts
    console.print()
    console.print("  [bold]Extracted Facts  Semantic Memory:[/bold]")
    facts = [
        "Customer NPS score Q4 2025: 72 (+4 vs Q3)",
        "React recommended for TypeScript-heavy dashboard projects",
        "Q4 churn rate: 3.2% (improving trend, -0.8pp)",
    ]
    for fact in facts:
        console.print(f"    [green]+[/green] {fact}")
        pause(0.2)

    # New procedure learned
    console.print()
    console.print("  [bold]Procedure Candidate Identified:[/bold]")
    console.print()
    pt = Table(
        box=box.SIMPLE_HEAVY,
        show_header=False,
        border_style="magenta",
        padding=(0, 2),
    )
    pt.add_row("[bold]Name:[/bold]", "quarterly_feedback_analysis")
    pt.add_row(
        "[bold]Trigger:[/bold]",
        '"analyze customer feedback for [period]"',
    )
    pt.add_row(
        "[bold]Steps:[/bold]",
        "1. Search semantic memory  2. Aggregate metrics  3. Compare trends",
    )
    pt.add_row(
        "[bold]Confidence:[/bold]",
        "[green]87%[/green] (2 similar sessions observed)",
    )
    pt.add_row(
        "[bold]Status:[/bold]",
        "[yellow]Candidate[/yellow]  auto-promoted after 1 more success",
    )
    console.print(
        Panel(
            pt,
            title="[bold magenta]New Skill Learned[/bold magenta]",
            border_style="magenta",
            expand=False,
        )
    )
    pause(1.5)


# ════════════════════════════════════════════════════════════════════
#  SCENE 10 ── Final Statistics
# ════════════════════════════════════════════════════════════════════

STATS = [
    ("Source Code",        "~85,000 LOC"),
    ("Test Code",          "~53,000 LOC"),
    ("Tests",              "4,673 passing"),
    ("Coverage",           "89%"),
    ("Lint Errors",        "0"),
    ("Python Files",       "394"),
    ("Modules",            "22"),
    ("LLM Providers",      "15"),
    ("Channels",           "17"),
    ("MCP Tool Servers",   "13+"),
    ("Memory Tiers",       "5"),
    ("Security Levels",    "4 risk levels (GREEN  RED)"),
    ("Sandbox Levels",     "4 (Process  Docker)"),
    ("Python",             ">= 3.12"),
]


def scene_stats() -> None:
    """Animated stats table + final branding panel."""
    console.print()
    console.rule("[bold bright_white]System Overview[/bold bright_white]")
    pause(0.5)

    table = Table(
        box=box.DOUBLE_EDGE,
        show_header=True,
        header_style="bold bright_white",
        border_style="bright_cyan",
        title="[bold bright_cyan]Cognithor  Agent OS  By The Numbers[/bold bright_cyan]",
        padding=(0, 2),
    )
    table.add_column("Metric", style="bold", min_width=24)
    table.add_column("Value", style="bright_cyan", justify="right", min_width=28)

    with Live(table, console=console, refresh_per_second=15):
        for metric, value in STATS:
            table.add_row(metric, value)
            pause(0.10)

    pause(0.5)
    console.print()

    # PGE one-liner
    console.print(
        Align.center(
            Text(
                "Planner (LLM)    Gatekeeper (Policy)    Executor (Sandbox)",
                style="bold",
            )
        )
    )
    console.print(
        Align.center(
            Text(
                "The PGE Trinity  Intelligence with Guardrails",
                style="dim italic",
            )
        )
    )
    console.print()

    # Final brand card
    console.print(
        Align.center(
            Panel(
                "[bold bright_cyan]Cognithor  Agent OS[/bold bright_cyan]\n\n"
                "Local-first  Privacy-first  Security-first\n"
                "Open Source under Apache 2.0\n\n"
                "[bold]https://github.com/cognithor/cognithor[/bold]",
                border_style="bright_cyan",
                expand=False,
                padding=(1, 6),
            )
        )
    )
    console.print()


# ════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════


def main() -> None:
    """Run the cinematic demo (~3 minutes, or ~15 seconds with --fast)."""
    try:
        scene_boot()
        scene_providers()
        scene_channels()
        scene_memory()
        scene_conversation()
        scene_pge()
        scene_security()
        scene_multichannel()
        scene_reflection()
        scene_stats()
    except KeyboardInterrupt:
        console.print("\n[dim]Demo interrupted.[/dim]")
        sys.exit(0)


if __name__ == "__main__":
    main()
