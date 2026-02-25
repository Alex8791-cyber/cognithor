#!/usr/bin/env python3
"""
Cognithor - Agent OS -- Cinematic Terminal Demo

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
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

# -- Globals ---------------------------------------------------------------
VERSION = "0.22.0"
FAST = "--fast" in sys.argv
_FORCE = bool(os.environ.get("FORCE_COLOR"))
console = Console(
    highlight=False,
    force_terminal=_FORCE or None,
    color_system="truecolor" if _FORCE else "auto",
    width=int(os.environ.get("COLUMNS", 0)) or None,
    legacy_windows=False,
)

# ASCII-safe box styles for recording compatibility
TABLE_BOX = box.ASCII2
PANEL_BOX = box.ASCII2


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
    "bright_yellow": "\033[93m",
    "bright_blue": "\033[94m",
    "bright_magenta": "\033[95m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "": "",
}
_ANSI_RESET = "\033[0m"


def ansi_print(text: str, style: str = "") -> None:
    """Print a line with ANSI color (bypasses rich markup entirely)."""
    sys.stdout.write(_ANSI.get(style, "") + text + _ANSI_RESET + "\n")
    sys.stdout.flush()


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


# ====================================================================
#  SCENE 1 -- Boot Sequence
# ====================================================================

def scene_boot() -> None:
    """Boot sequence with title and system init steps."""
    console.clear()
    pause(0.5)

    # Clean title block (no ASCII art - renders perfectly in svg-term)
    sys.stdout.write("\n\n")
    ansi_print("    ========================================================", "bright_cyan")
    ansi_print("    ||                                                    ||", "bright_cyan")
    ansi_print("    ||             C O G N I T H O R                      ||", "bright_cyan")
    ansi_print("    ||               Agent OS v" + VERSION + "                       ||", "bright_cyan")
    ansi_print("    ||                                                    ||", "bright_cyan")
    ansi_print("    ||      Cognition + Thor -- Intelligence with Power   ||", "bright_cyan")
    ansi_print("    ||                                                    ||", "bright_cyan")
    ansi_print("    ========================================================", "bright_cyan")
    sys.stdout.write("\n")
    pause(0.6)

    # System init checklist (plain text, no Unicode spinners)
    steps = [
        "Loading configuration",
        "Initializing PGE Trinity",
        "Connecting 5-tier memory",
        "Starting MCP tool servers (13)",
        "Registering security policies",
        "Warming up embedding cache",
    ]

    for step in steps:
        ansi_print("  [ok] " + step, "bright_green")
        pause(0.35)

    sys.stdout.write("\n")
    ansi_print("  ============== System Online ==============", "bright_green")
    pause(1.0)


# ====================================================================
#  SCENE 2 -- LLM Provider Scan
# ====================================================================

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
    """Animated provider table -- each row lights up one by one."""
    sys.stdout.write("\n")
    console.print(
        Panel(
            Text("LLM Provider Scan", style="bold"),
            style="cyan",
            expand=False,
            box=PANEL_BOX,
        )
    )
    pause(0.4)

    table = Table(
        box=TABLE_BOX,
        show_header=True,
        header_style="bold bright_white",
        border_style="cyan",
        title="Multi-LLM Backend Layer",
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
                Text(name, style="bold"),
                ptype,
                model,
                endpoint,
                Text("* READY", style="bright_green"),
            )
            pause(0.14)

    sys.stdout.write("\n")
    ansi_print(f"  {len(PROVIDERS)} providers connected", "bright_green")
    pause(1.0)


# ====================================================================
#  SCENE 3 -- Channel Initialization
# ====================================================================

CHANNELS = [
    "CLI", "Web UI", "REST API", "Telegram", "Discord",
    "Slack", "WhatsApp", "Signal", "iMessage", "Teams",
    "Matrix", "Google Chat", "Mattermost", "Feishu/Lark",
    "IRC", "Twitch", "Voice",
]


def scene_channels() -> None:
    """Progress bar + grid of connected channels."""
    sys.stdout.write("\n")
    console.print(
        Panel(
            Text("Channel Initialization", style="bold"),
            style="yellow",
            expand=False,
            box=PANEL_BOX,
        )
    )
    pause(0.3)

    with Progress(
        SpinnerColumn("line"),
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
    sys.stdout.write("\n")
    console.print(Columns(chips, padding=(0, 1), expand=False))
    sys.stdout.write("\n")
    ansi_print(f"  {len(CHANNELS)} channels active", "bright_green")
    pause(1.0)


# ====================================================================
#  SCENE 4 -- 5-Tier Cognitive Memory
# ====================================================================

MEMORY_TIERS = [
    (
        "Tier 1 - Core",
        "Identity, rules, personality",
        ["Owner: configured", "Rules: 12 active", "Personality: adaptive"],
    ),
    (
        "Tier 2 - Episodic",
        "Daily logs -- what happened",
        ["Episodes: 847", "Timespan: 14 months", "Auto-archival: on"],
    ),
    (
        "Tier 3 - Semantic",
        "Knowledge graph -- facts & relations",
        ["Entities: 2,341", "Relations: 5,892", "Categories: 47"],
    ),
    (
        "Tier 4 - Procedural",
        "Learned skills -- how to do things",
        ["Procedures: 156", "Auto-learned: 89", "Success rate: 94%"],
    ),
    (
        "Tier 5 - Working",
        "Session context (volatile RAM)",
        ["Tokens: 12,480", "Window: 128K", "Cache hit: 87%"],
    ),
]


def scene_memory() -> None:
    """Animated tree view of the 5-tier memory system + hybrid search."""
    sys.stdout.write("\n")
    console.print(
        Panel(
            Text("5-Tier Cognitive Memory", style="bold"),
            style="magenta",
            expand=False,
            box=PANEL_BOX,
        )
    )
    pause(0.4)

    tree = Tree(
        Text("Memory System", style="bold magenta"),
        guide_style="magenta",
    )

    with Live(tree, console=console, refresh_per_second=10):
        for tier_name, desc, stats in MEMORY_TIERS:
            branch = tree.add(
                Text.assemble(
                    (tier_name, "bold"),
                    " -- ",
                    (desc, "dim"),
                )
            )
            pause(0.25)
            for stat in stats:
                branch.add(Text.assemble(("* ", "green"), stat))
                pause(0.08)

    # Hybrid search panel
    sys.stdout.write("\n")
    search = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    search.add_column(style="bold cyan")
    search.add_column()
    search.add_row("BM25", "Full-text search (FTS5, compound words)")
    search.add_row("Vector", "Embedding similarity (cosine, LRU-cached)")
    search.add_row("Graph", "Entity relation traversal (3-hop)")
    console.print(
        Panel(
            search,
            title="3-Channel Hybrid Search",
            title_style="bold",
            border_style="magenta",
            expand=False,
            box=PANEL_BOX,
        )
    )
    pause(1.0)


# ====================================================================
#  SCENE 5 -- Live Conversation
# ====================================================================


def scene_conversation() -> None:
    """Simulated streaming conversation with comparison table."""
    sys.stdout.write("\n")
    ansi_print("  ============== Live Session ==============", "bold")
    pause(0.5)

    # User types a question
    sys.stdout.write("\n")
    sys.stdout.write("  cognithor> ")
    sys.stdout.flush()
    typing(
        "What are the key differences between React and Vue.js "
        "for our new dashboard project?"
    )
    pause(0.3)

    # Thinking
    sys.stdout.write("\n")
    ansi_print("  [thinking] Planning response...", "bright_cyan")
    pause(1.8)

    # Streaming answer
    sys.stdout.write("\n")
    streaming(
        "  Based on your project requirements and the team's TypeScript "
        "experience, here are the key differences:"
    )
    sys.stdout.write("\n")
    pause(0.3)

    # Comparison table (part of the AI response)
    comp = Table(
        box=TABLE_BOX,
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
    sys.stdout.write("\n")
    streaming(
        "  Given the team's TypeScript expertise and the need for a rich "
        "plugin ecosystem, I'd recommend React with Zustand for state "
        "management. Want me to scaffold the project?"
    )
    pause(1.0)


# ====================================================================
#  SCENE 6 -- PGE Trinity in Action
# ====================================================================


def scene_pge() -> None:
    """Planner -> Gatekeeper -> Executor pipeline with real-looking output."""
    sys.stdout.write("\n")
    ansi_print("  ========== PGE Trinity in Action ==========", "bold")
    pause(0.5)

    # User request
    sys.stdout.write("\n")
    sys.stdout.write("  cognithor> ")
    sys.stdout.flush()
    typing(
        "Search my knowledge base for all customer feedback from last quarter"
    )
    pause(0.5)

    # -- PLANNER --------------------------------------------------------
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

    sys.stdout.write("\n")
    console.print(
        Panel(
            Syntax(plan_code, "python", theme="monokai", line_numbers=False),
            title=">> PLANNER << LLM-based Planning",
            title_style="bold blue",
            border_style="blue",
            subtitle="Model: qwen3:32b  847ms",
            subtitle_style="dim",
            box=PANEL_BOX,
        )
    )
    pause(1.0)

    # -- GATEKEEPER -----------------------------------------------------
    gt = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    gt.add_row(Text("Tool:", style="bold"), "memory_search")
    gt.add_row(
        Text("Risk Level:", style="bold"),
        Text("GREEN (read-only, no side effects)", style="bold green"),
    )
    gt.add_row(Text("Policy Match:", style="bold"), "ALLOW  memory queries auto-approved")
    gt.add_row(Text("Sandbox Level:", style="bold"), "L0 (Process isolation)")
    gt.add_row(
        Text("Decision:", style="bold"),
        Text(">> APPROVED <<", style="bold green"),
    )

    console.print(
        Panel(
            gt,
            title=">> GATEKEEPER << Deterministic Policy Engine",
            title_style="bold green",
            border_style="green",
            subtitle="No LLM | No hallucinations | 0.2ms",
            subtitle_style="dim",
            box=PANEL_BOX,
        )
    )
    pause(1.0)

    # -- EXECUTOR -------------------------------------------------------
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
            title=">> EXECUTOR << Sandboxed Execution",
            title_style="bold yellow",
            border_style="yellow",
            subtitle="Tool: memory_search | 23ms | SHA-256 audit logged",
            subtitle_style="dim",
            box=PANEL_BOX,
        )
    )
    pause(1.5)


# ====================================================================
#  SCENE 7 -- Security Block
# ====================================================================


def scene_security() -> None:
    """Gatekeeper blocks a dangerous request with detailed policy analysis."""
    sys.stdout.write("\n")
    ansi_print("  ========= Security Demonstration =========", "bright_red")
    pause(0.5)

    sys.stdout.write("\n")
    sys.stdout.write("  cognithor> ")
    sys.stdout.flush()
    typing("Delete all files in /etc and remove system logs")
    pause(0.4)

    # BLOCKED
    sys.stdout.write("\n")
    bt = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    bt.add_row(Text("Tool:", style="bold"), "shell.exec_command")
    bt.add_row(Text("Command:", style="bold"), Text("rm -rf /etc/*", style="red"))
    bt.add_row(
        Text("Risk Level:", style="bold"),
        Text("RED -- Destructive system operation", style="bold red"),
    )
    bt.add_row(
        Text("Violations:", style="bold"),
        Text("3 policy violations detected", style="red"),
    )
    bt.add_row("", Text("  PATH_FORBIDDEN   /etc outside allowed paths", style="red"))
    bt.add_row("", Text("  CMD_BLACKLISTED  recursive delete blocked", style="red"))
    bt.add_row("", Text("  SCOPE_EXCEEDED   system-level destruction", style="red"))
    bt.add_row(
        Text("Decision:", style="bold"),
        Text(">>> BLOCKED <<<", style="bold red"),
    )

    console.print(
        Panel(
            bt,
            title="GATEKEEPER -- REQUEST DENIED",
            title_style="bold red",
            border_style="red",
            subtitle="Deterministic | No override possible | Logged to audit chain",
            subtitle_style="dim",
            box=PANEL_BOX,
        )
    )
    pause(0.5)

    sys.stdout.write("\n")
    streaming(
        "  I cannot execute that request. The Gatekeeper blocked this action "
        "because it involves destructive operations on system-critical paths. "
        "This protection is enforced by deterministic policy rules, not by an "
        "LLM that could be tricked or prompt-injected.",
        style="bright_red",
    )
    pause(1.0)


# ====================================================================
#  SCENE 8 -- Multi-Channel Broadcast
# ====================================================================

BROADCAST_CHANNELS = [
    ("Telegram",  "Bot >> @team_channel",     "bright_blue"),
    ("Discord",   "#deployments >> Embed",    "bright_magenta"),
    ("Slack",     "#ops >> Block Kit",        "bright_yellow"),
    ("WhatsApp",  "Ops Group >> Text",        "bright_green"),
    ("Teams",     "DevOps >> Adaptive Card",  "bright_blue"),
    ("Web UI",    "Dashboard >> WebSocket",   "bright_cyan"),
    ("Matrix",    "!ops:matrix.org >> E2EE",  ""),
]


def scene_multichannel() -> None:
    """Same message delivered to 7 channels simultaneously."""
    sys.stdout.write("\n")
    console.print(
        Panel(
            Text("Multi-Channel Broadcast", style="bold"),
            style="bright_blue",
            expand=False,
            box=PANEL_BOX,
        )
    )
    pause(0.4)

    ansi_print("  Broadcasting deployment notification to all active channels...", "dim")
    sys.stdout.write("\n")

    for name, detail, color in BROADCAST_CHANNELS:
        ansi_print(f"  -> {name:12s} {detail}", color or "bold")
        pause(0.18)

    sys.stdout.write("\n")
    ansi_print(
        f"  Delivered to {len(BROADCAST_CHANNELS)} channels in 340ms",
        "bright_green",
    )
    pause(1.0)


# ====================================================================
#  SCENE 9 -- Reflection & Learning
# ====================================================================


def scene_reflection() -> None:
    """Reflector analyses session, extracts facts, learns a procedure."""
    sys.stdout.write("\n")
    console.print(
        Panel(
            Text("Reflection & Procedural Learning", style="bold"),
            style="bright_magenta",
            expand=False,
            box=PANEL_BOX,
        )
    )
    pause(0.4)

    ansi_print("  [analyzing] Reflector analyzing session...", "bright_magenta")
    pause(1.8)

    # Extracted facts
    sys.stdout.write("\n")
    ansi_print("  Extracted Facts >> Semantic Memory:", "bold")
    facts = [
        "Customer NPS score Q4 2025: 72 (+4 vs Q3)",
        "React recommended for TypeScript-heavy dashboard projects",
        "Q4 churn rate: 3.2% (improving trend, -0.8pp)",
    ]
    for fact in facts:
        ansi_print("    + " + fact, "bright_green")
        pause(0.2)

    # New procedure learned
    sys.stdout.write("\n")
    ansi_print("  Procedure Candidate Identified:", "bold")
    sys.stdout.write("\n")
    pt = Table(
        box=TABLE_BOX,
        show_header=False,
        border_style="magenta",
        padding=(0, 2),
    )
    pt.add_row(Text("Name:", style="bold"), "quarterly_feedback_analysis")
    pt.add_row(
        Text("Trigger:", style="bold"),
        '"analyze customer feedback for [period]"',
    )
    pt.add_row(
        Text("Steps:", style="bold"),
        "1. Search semantic memory  2. Aggregate metrics  3. Compare trends",
    )
    pt.add_row(
        Text("Confidence:", style="bold"),
        Text("87% (2 similar sessions observed)", style="green"),
    )
    pt.add_row(
        Text("Status:", style="bold"),
        Text("Candidate -- auto-promoted after 1 more success", style="yellow"),
    )
    console.print(
        Panel(
            pt,
            title="New Skill Learned",
            title_style="bold magenta",
            border_style="magenta",
            expand=False,
            box=PANEL_BOX,
        )
    )
    pause(1.5)


# ====================================================================
#  SCENE 10 -- Final Statistics
# ====================================================================

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
    ("Security Levels",    "4 risk levels (GREEN >> RED)"),
    ("Sandbox Levels",     "4 (Process >> Docker)"),
    ("Python",             ">= 3.12"),
]


def scene_stats() -> None:
    """Animated stats table + final branding panel."""
    sys.stdout.write("\n")
    ansi_print("  ============= System Overview =============", "bold")
    pause(0.5)

    table = Table(
        box=TABLE_BOX,
        show_header=True,
        header_style="bold bright_white",
        border_style="bright_cyan",
        title="Cognithor - Agent OS - By The Numbers",
        title_style="bold bright_cyan",
        padding=(0, 2),
    )
    table.add_column("Metric", style="bold", min_width=24)
    table.add_column("Value", style="bright_cyan", justify="right", min_width=28)

    with Live(table, console=console, refresh_per_second=15):
        for metric, value in STATS:
            table.add_row(metric, value)
            pause(0.10)

    pause(0.5)
    sys.stdout.write("\n")

    # PGE one-liner
    console.print(
        Align.center(
            Text(
                "Planner (LLM)  >>  Gatekeeper (Policy)  >>  Executor (Sandbox)",
                style="bold",
            )
        )
    )
    console.print(
        Align.center(
            Text(
                "The PGE Trinity -- Intelligence with Guardrails",
                style="dim italic",
            )
        )
    )
    sys.stdout.write("\n")

    # Final brand card
    console.print(
        Align.center(
            Panel(
                Text.assemble(
                    ("Cognithor - Agent OS", "bold bright_cyan"),
                    "\n\n",
                    "Local-first | Privacy-first | Security-first\n",
                    "Open Source under Apache 2.0\n\n",
                    ("https://github.com/Alex8791-cyber/cognithor", "bold"),
                ),
                border_style="bright_cyan",
                expand=False,
                padding=(1, 6),
                box=PANEL_BOX,
            )
        )
    )
    sys.stdout.write("\n")


# ====================================================================
#  MAIN
# ====================================================================


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
