"""assembles the final Markdown research report.

The report integrates analytical stats, cost models, and charts into a 
single, self-contained document for research deliverables.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ..analysis.dataset_analyser import DatasetStats
from ..cost_model.llm_cost import CostReport

logger = logging.getLogger(__name__)

# --- Formatting Helpers ---
def _usd(value: float, decimals: int = 4) -> str:
    """Format numeric value as USD string."""
    return f"${value:,.{decimals}f}"


def _num(value: int) -> str:
    """Format integer with thousands separator."""
    return f"{value:,}"


def _pct(part: int, whole: int) -> str:
    """Calculate percentage of whole."""
    if whole == 0:
        return "0%"
    return f"{part / whole * 100:.1f}%"


# --- Section Builders ---
def _section_intro(commit_shas: dict[str, str], now: str) -> str:
    """Generate Title and Introduction."""
    bip_sha = commit_shas.get("bips", "unknown")[:12]
    bolt_sha = commit_shas.get("bolts", "unknown")[:12]
    return f"""# Bitcoin Spec Intelligence Report

> **Generated:** {now}
> **BIP Repository HEAD:** `{bip_sha}`
> **BOLT Repository HEAD:** `{bolt_sha}`

---

## 1. Introduction

This report provides a **technical feasibility analysis** for building an LLM-powered
explainer system over the complete corpus of Bitcoin Improvement Proposals (BIPs) and
Lightning Network specification documents (BOLTs).

The analysis is fully automated and reproducible: it fetches live data from the official
upstream repositories, parses every specification document, computes accurate GPT-family
token counts, and models the economic cost of processing the entire corpus.

---
"""


def _section_data_sources(commit_shas: dict[str, str]) -> str:
    """Generate data history and parsing methodology."""
    bip_sha = commit_shas.get("bips", "unknown")
    bolt_sha = commit_shas.get("bolts", "unknown")
    return f"""## 2. Data Sources

| Source | Repository | Commit |
|---|---|---|
| Bitcoin BIPs | https://github.com/bitcoin/bips | `{bip_sha[:12]}` |
| Lightning BOLTs | https://github.com/lightning/bolts | `{bolt_sha[:12]}` |

All data was fetched live from upstream repositories and parsed locally.

- **BIP files** use MediaWiki markup (`.mediawiki`) and are parsed with `mwparserfromhell`.
- **BOLT files** use GitHub Markdown (`.md`) stripped of non-semantic elements.

---
"""


def _section_dataset_overview(stats: DatasetStats) -> str:
    """Generate high-level corpus metrics and distributions."""
    return f"""## 3. Dataset Overview

| Metric | Value |
|---|---|
| **Total Specifications** | {_num(stats.total_specs)} |
| BIPs | {_num(stats.total_bips)} |
| BOLTs | {_num(stats.total_bolts)} |
| **Total Corpus Tokens** | {_num(stats.total_tokens)} |
| Average Tokens / Spec | {stats.avg_tokens:,.1f} |
| Median Tokens / Spec | {stats.median_tokens:,.1f} |
| Specs With Embedded Images | {_num(stats.specs_with_images)} ({_pct(stats.specs_with_images, stats.total_specs)}) |
| Total Embedded Images | {_num(stats.total_images)} |
| Total Code Blocks | {_num(stats.total_code_blocks)} |

### Token Distribution

![Token Distribution](assets/token_distribution.png)

### BIPs vs BOLTs Comparison

![BIPs vs BOLTs](assets/bips_vs_bolts.png)

---
"""


def _section_spec_statistics(stats: DatasetStats) -> str:
    """Generate Top-10 tables and volume charts."""
    top10_rows = "\n".join(
        f"| {s['identifier']} | `{s['filename']}` "
        f"| {_num(int(s['token_count']))} | {s['complexity']} |"
        for s in stats.largest_specs
    )
    return f"""## 4. Specification Statistics

### Top 10 Largest Specifications

| Identifier | Filename | Tokens | Complexity |
|---|---|---|---|
{top10_rows}

### Token Count — Top 50 Specs

![Token Count per Spec](assets/token_per_spec.png)

---
"""


def _section_multimodal(stats: DatasetStats) -> str:
    """Generate complexity tier analysis."""
    return f"""## 5. Multimodal Complexity Analysis

| Tier | Detection Criteria | Count | Share |
|---|---|---|---|
| `text-only` | No images · fewer than 3 code blocks | {_num(stats.text_only_count)} | {_pct(stats.text_only_count, stats.total_specs)} |
| `diagram-assisted` | Has images · fewer than 5 code blocks | {_num(stats.diagram_assisted_count)} | {_pct(stats.diagram_assisted_count, stats.total_specs)} |
| `complex-technical` | Many images and/or numerous code blocks | {_num(stats.complex_technical_count)} | {_pct(stats.complex_technical_count, stats.total_specs)} |

---
"""


def _section_strategy(cost_report: CostReport) -> str:
    """Generate LLM processing assumptions and targets."""
    a = cost_report.assumptions
    return f"""## 6. LLM Processing Strategy

Each specification will be processed to produce **{a['explanations_per_spec']} distinct
explanation levels** (Beginner, Developer, Researcher).

### Cost Model Assumptions

| Parameter | Value |
|---|---|
| Explanations generated per spec | {a['explanations_per_spec']} |
| Context tokens per explanation call | {_num(a['context_tokens'])} |
| Output tokens per explanation | {_num(a['output_tokens_per_explanation'])} |

Token counts use **`cl100k_base`** encoding (OpenAI tiktoken).

---
"""


def _section_cost(cost_report: CostReport) -> str:
    """Generate cost tables and recommendations."""
    emb_rows = "\n".join(
        f"| `{m}` | {_usd(cost_report.total_embedding_cost[m])} |"
        for m in cost_report.total_embedding_cost
    )
    gen_rows = "\n".join(
        f"| `{m}` "
        f"| {_usd(cost_report.total_one_time_cost.get(m, 0))} |"
        for m in cost_report.total_generation_cost
    )

    return f"""## 7. Cost Estimation

### 7.1  Embedding Cost (One-Time)
| Embedding Model | Total Cost (USD) |
|---|---|
{emb_rows}

### 7.2  Generation Cost
| Generation Model | One-Time Processing |
|---|---|
{gen_rows}

### 7.3  Model Cost Comparison
![Cost Comparison](assets/cost_comparison.png)

### 7.4  Recommendation
> **Recommended model for MVP:** `{cost_report.recommended_model}`
>
> At **{_usd(cost_report.total_one_time_cost.get(cost_report.recommended_model, 0))}** one-time
> processing cost, this model provides the best cost-to-capability ratio.

---
"""


def _section_per_spec_table(cost_report: CostReport) -> str:
    """Generate the exhaustive per-file cost breakdown CSV-equivalent table."""
    emb_models = list(cost_report.total_embedding_cost)
    gen_models = list(cost_report.total_generation_cost)

    # Header
    emb_headers = " | ".join(f"Embed · {m}" for m in emb_models)
    gen_headers = " | ".join(f"Gen · {m}" for m in gen_models)
    header = f"| Identifier | Filename | Tokens | {emb_headers} | {gen_headers} |"
    
    num_cols = 3 + len(emb_models) + len(gen_models)
    sep_cols = "| " + " | ".join(["---"] * num_cols) + " |"

    # Rows — sort by token count desc
    sorted_specs = sorted(
        cost_report.per_spec, key=lambda s: s.token_count, reverse=True
    )
    rows: list[str] = []
    for sc in sorted_specs:
        emb_cells = " | ".join(_usd(sc.embedding_cost.get(m, 0.0), 6) for m in emb_models)
        gen_cells = " | ".join(_usd(sc.generation_cost.get(m, 0.0), 6) for m in gen_models)
        rows.append(f"| {sc.identifier} | `{sc.filename}` | {_num(sc.token_count)} | {emb_cells} | {gen_cells} |")

    table = "\n".join([header, sep_cols] + rows)
    return f"""## 8. Per-Specification Cost Breakdown\n\n{table}\n\n---\n"""


def _section_implementation_scope(stats: DatasetStats, cost_report: CostReport) -> str:
    """Generate implementation phase roadmap."""
    rec = cost_report.recommended_model
    return f"""## 9. Recommended Implementation Scope

### Phase 1 — Text-Only MVP
- Target the **{_num(stats.text_only_count)} `text-only`** specifications.
- Use `{rec}` for generation.

### Phase 2 — Full Corpus Coverage
- Extend coverage to all **{_num(stats.total_specs)}** specifications.

---
"""


def _section_reproducibility() -> str:
    """Generate setup and run instructions."""
    return """## 10. Reproducibility Instructions

```bash
# Install and run analysis
pip install -e .
python -m report generate
```

**Outputs produced:**
| File | Description |
|---|---|
| `reports/spec_intelligence_report.md` | This report |
| `reports/assets/*.png` | Analytical charts |
| `data/*.csv` | Raw metadata and cost exports |

---
*Report generated by the BIPs & BOLTs Explanation Cost Report Generator.*
"""


# --- Orchestrator ---
def generate_report(
    stats: DatasetStats,
    cost_report: CostReport,
    df: pd.DataFrame,
    charts: dict[str, Path],
    reports_dir: Path,
    commit_shas: dict[str, str],
) -> Path:
    """Assemble and write the complete research report."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / "spec_intelligence_report.md"

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    sections = [
        _section_intro(commit_shas, now),
        _section_data_sources(commit_shas),
        _section_dataset_overview(stats),
        _section_spec_statistics(stats),
        _section_multimodal(stats),
        _section_strategy(cost_report),
        _section_cost(cost_report),
        _section_per_spec_table(cost_report),
        _section_implementation_scope(stats, cost_report),
        _section_reproducibility(),
    ]

    out_path.write_text("\n".join(sections), encoding="utf-8")
    logger.info(f"Report written -> {out_path.name}")
    return out_path
