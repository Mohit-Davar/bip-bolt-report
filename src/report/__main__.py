"""CLI entry point for the Report Generator.
Usage
    python -m spec_report generate           # full pipeline
    python -m spec_report generate --help    # option reference
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

from .utils.logging_config import setup_logging

console = Console()
logger = logging.getLogger(__name__)

# Paths resolved relative to the installed package root 
# src/report/__main__.py  →  ../../..  == project root
_PKG_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _PKG_DIR.parents[1]  # Resolve to bip-bolt/
_DATA_DIR = _PROJECT_ROOT / "data"
_REPORTS_DIR = _PROJECT_ROOT / "reports"
_CONFIG_PATH = _PROJECT_ROOT / "config" / "config.json"



@click.group()
def cli() -> None:
    """Report Generator.
    Fetches the official BIP and BOLT repositories, analyses the complete
    specification corpus, and produces a publishable Markdown research report
    with cost estimates and charts.
    """

@cli.command()
@click.option(
    "--log-level",
    default="INFO",
    show_default=True,
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
    help="Console logging verbosity.",
)
def generate(log_level: str) -> None:
    """Run the full analysis pipeline and produce the report.
    \b
    Steps:
      1. Clone / pull bitcoin/bips and lightning/bolts
      2. Parse every .mediawiki (BIP) and .md (BOLT) file
      3. Compute token counts and structural metadata
      4. Model LLM processing costs across all configured pricing tiers
      5. Generate matplotlib charts
      6. Assemble the Markdown research report
    """
    setup_logging(log_level)

    console.print()
    console.print(
        Panel.fit(
            "[bold cyan]BIPs and BOLTs[/bold cyan]\n",
            title="[bold blue] v0.1.0 [/bold blue]",
            title_align="right",
            border_style="cyan",
            padding=(1, 2),
        )
    )
    console.print()

    # Validate config
    if not _CONFIG_PATH.exists():
        logger.error(
            f"config not found: {_CONFIG_PATH}\n"
            "Expected at config/config.json relative to project root."
        )
        sys.exit(1)

    # Step 1: Sync repositories
    console.print(Rule("[bold cyan]Step 1 / 5  — Repository Sync[/bold cyan]"))
    from .ingestion.repo_sync import sync_all  # noqa: PLC0415

    sync_results = sync_all(_DATA_DIR)
    commit_shas = {name: sha for name, (_, sha) in sync_results.items()}
    repo_paths = {name: path for name, (path, _) in sync_results.items()}
    console.print()

    # Step 2: Parse specifications
    console.print(Rule("[bold cyan]Step 2 / 5  — Parsing[/bold cyan]"))
    from .parsing.parser import parse_all_files  # noqa: PLC0415

    records = parse_all_files(repo_paths["bips"], repo_paths["bolts"])
    if not records:
        logger.error(
            "No records were parsed.  "
            "Check that the cloned repositories contain .mediawiki and .md files."
        )
        sys.exit(1)

    logger.info(f"  Total specs parsed: [bold]{len(records)}[/]")
    console.print()

    # Step 3: Dataset analysis
    console.print(Rule("[bold cyan]Step 3 / 5  — Dataset Analysis[/bold cyan]"))
    from .analysis.dataset_analyser import analyse  # noqa: PLC0415

    stats, df = analyse(records, _DATA_DIR)
    console.print()

    # Step 4: LLM cost modelling
    console.print(Rule("[bold cyan]Step 4 / 5  — LLM Cost Modelling[/bold cyan]"))
    from .cost_model.llm_cost import compute_costs  # noqa: PLC0415

    cost_report = compute_costs(records, _CONFIG_PATH, _DATA_DIR)
    console.print()

    # Step 5: Charts + report
    console.print(
        Rule("[bold cyan]Step 5 / 5  — Visualisation & Report[/bold cyan]")
    )
    from .reporting.chart_builder import build_all_charts  # noqa: PLC0415
    from .reporting.report_writer import generate_report  # noqa: PLC0415

    charts = build_all_charts(df, stats, cost_report, _REPORTS_DIR / "assets")
    report_path = generate_report(
        stats, cost_report, df, charts, _REPORTS_DIR, commit_shas
    )

    # ── Summary ────────────────────────────────────────────────────────────
    console.print()
    console.print(
        Panel.fit(
            f"[bold green]SUCCESS: Analysis complete![/bold green]\n\n"
            f"[cyan]Report    :[/cyan]  {report_path}\n"
            f"[cyan]Charts    :[/cyan]  {_REPORTS_DIR / 'assets'}\n"
            f"[cyan]Spec CSV  :[/cyan]  {_DATA_DIR / 'dataset_stats.csv'}\n"
            f"[cyan]Cost CSV  :[/cyan]  {_DATA_DIR / 'per_spec_costs.csv'}\n\n"
            f"[dim]BIPs: {stats.total_bips}  BOLTs: {stats.total_bolts}  "
            f"Total tokens: {stats.total_tokens:,}[/dim]",
            border_style="green",
            padding=(1, 4),
        )
    )


# ---------------------------------------------------------------------------
# Module entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cli()
