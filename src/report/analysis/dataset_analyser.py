"""aggregates metadata into summary statistics.

Exports:
- ``data/dataset_stats.csv``  — one row per record, all fields.
- ``data/dataset_stats.json`` — aggregate summary as a JSON document.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from ..parsing.parser import SpecRecord

logger = logging.getLogger(__name__)


# Summary dataclass

@dataclass
class DatasetStats:
    """Aggregate statistics over the complete BIP + BOLT corpus."""

    total_specs: int
    total_bips: int
    total_bolts: int

    # Token distribution
    total_tokens: int
    avg_tokens: float
    median_tokens: float
    max_tokens: int
    min_tokens: int

    # Structural content
    total_images: int
    total_code_blocks: int
    specs_with_images: int

    # Multimodal complexity breakdown
    text_only_count: int
    diagram_assisted_count: int
    complex_technical_count: int

    # Top-10 largest specs (for report table)
    largest_specs: list[dict]

    # Full complexity distribution map
    complexity_distribution: dict[str, int]


# Core analysis function
def analyse(
    records: list[SpecRecord],
    data_dir: Path,
) -> tuple[DatasetStats, pd.DataFrame]:
    """Compute aggregate statistics and export raw data files."""
    if not records:
        raise ValueError("No records to analyse")

    # use asdict() for cleaner DataFrame construction
    df = pd.DataFrame([asdict(r) for r in records])

    # Boolean indexing for subset counts
    spec_counts = df["spec_type"].value_counts()
    
    complexity_dist: dict[str, int] = df["complexity"].value_counts().to_dict()

    # Streamlined top-10 extraction
    top10 = (
        df.nlargest(10, "token_count")[
            ["identifier", "filename", "token_count", "complexity"]
        ]
        .to_dict("records")
    )

    # Using .agg() for bulk descriptive stats
    token_stats = df["token_count"].agg(["sum", "mean", "median", "max", "min"])

    stats = DatasetStats(
        total_specs=len(df),
        total_bips=int(spec_counts.get("BIP", 0)),
        total_bolts=int(spec_counts.get("BOLT", 0)),
        total_tokens=int(token_stats["sum"]),
        avg_tokens=round(float(token_stats["mean"]), 1),
        median_tokens=round(float(token_stats["median"]), 1),
        max_tokens=int(token_stats["max"]),
        min_tokens=int(token_stats["min"]),
        total_images=int(df["image_count"].sum()),
        total_code_blocks=int(df["code_block_count"].sum()),
        specs_with_images=int((df["image_count"] > 0).sum()),
        text_only_count=complexity_dist.get("text-only", 0),
        diagram_assisted_count=complexity_dist.get("diagram-assisted", 0),
        complex_technical_count=complexity_dist.get("complex-technical", 0),
        largest_specs=top10,
        complexity_distribution=complexity_dist,
    )

    # Export
    data_dir.mkdir(parents=True, exist_ok=True)

    csv_path = data_dir / "dataset_stats.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"  Exported spec table → {csv_path}")

    json_path = data_dir / "dataset_stats.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(asdict(stats), fh, indent=2)
    logger.info(f"  Exported aggregate stats → {json_path}")

    return stats, df