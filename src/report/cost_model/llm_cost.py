"""computes one-time processing and monthly serving costs."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..parsing.parser import SpecRecord

logger = logging.getLogger(__name__)


# Data models
@dataclass
class ModelPricing:
    """Per-model token pricing (USD per 1 million tokens)."""
    name: str
    input_per_1m: float
    output_per_1m: float  # 0.0 for embedding-only models


@dataclass
class SpecCost:
    """Cost breakdown for a single file across all models."""
    identifier: str
    filename: str
    token_count: int
    embedding_cost: dict[str, float]   # model -> USD
    generation_cost: dict[str, float]  # model -> USD


@dataclass
class CostReport:
    """Corpus-wide cost aggregates and per-file breakdowns."""
    assumptions: dict
    per_spec: list[SpecCost]
    total_embedding_cost: dict[str, float]
    total_generation_cost: dict[str, float]
    total_one_time_cost: dict[str, float]
    cheapest_model: str
    recommended_model: str


# Helpers
def _load_pricing(config_path: Path) -> tuple[dict[str, ModelPricing], dict]:
    """Read config.json and return (models_dict, assumptions_dict).

    Filters out non-pricing keys like 'provider' so they
    don't break the ModelPricing constructor.
    """
    with open(config_path, encoding="utf-8") as fh:
        cfg = json.load(fh)

    models: dict[str, ModelPricing] = {}
    for name, vals in cfg["models"].items():
        models[name] = ModelPricing(
            name=name,
            input_per_1m=float(vals["input_per_1m"]),
            output_per_1m=float(vals["output_per_1m"]),
        )

    return models, cfg["assumptions"]


# Main cost-computation function
def compute_costs(
    records: list[SpecRecord],
    config_path: Path,
    data_dir: Path,
) -> CostReport:
    """Compute one-time LLM processing costs for the full corpus.

    Cost semantics
    --------------
    Embedding cost  — Embed every spec once to build a vector index.
                      Cost = sum(spec_tokens) × price_per_input_token.

    Generation cost — For each spec, generate N explanations.
                      Each call sends (spec_tokens + ctx_tokens) as input
                      and produces out_tokens as output.
                      Cost = sum over specs of:
                        [(spec_tokens + ctx_tokens) × explanations × input_price
                         + out_tokens × explanations × output_price]
    """
    models, assumptions = _load_pricing(config_path)

    explanations: int = assumptions["explanations_per_spec"]
    ctx_tokens: int = assumptions["context_tokens"]
    out_tokens: int = assumptions["output_tokens_per_explanation"]

    embedding_models = {k: v for k, v in models.items() if v.output_per_1m == 0.0}
    generation_models = {k: v for k, v in models.items() if v.output_per_1m > 0.0}

    # 1. One-time embedding: embed each spec's plain-text tokens once
    total_corpus_tokens = sum(r.token_count for r in records)
    total_embedding = {
        name: round((total_corpus_tokens * m.input_per_1m) / 1_000_000, 6)
        for name, m in embedding_models.items()
    }

    # 2. One-time generation: for each spec produce N explanations.
    #    Input = spec tokens + ctx_tokens overhead.  Output = fixed out_tokens.
    total_generation: dict[str, float] = {name: 0.0 for name in generation_models}
    for rec in records:
        input_per_call = rec.token_count + ctx_tokens  # tokens sent per explanation
        for name, m in generation_models.items():
            cost = (
                input_per_call * explanations * m.input_per_1m
                + out_tokens * explanations * m.output_per_1m
            ) / 1_000_000
            total_generation[name] += cost
    total_generation = {k: round(v, 6) for k, v in total_generation.items()}

    # 3. Per-spec breakdown (for CSV export)
    per_spec_costs: list[SpecCost] = []
    for rec in records:
        emb_map = {
            n: round((rec.token_count * m.input_per_1m) / 1_000_000, 8)
            for n, m in embedding_models.items()
        }
        gen_map = {
            n: round((
                (rec.token_count + ctx_tokens) * explanations * m.input_per_1m
                + out_tokens * explanations * m.output_per_1m
            ) / 1_000_000, 8)
            for n, m in generation_models.items()
        }
        per_spec_costs.append(
            SpecCost(rec.identifier, rec.filename, rec.token_count, emb_map, gen_map)
        )

    # 4. Combined one-time cost = cheapest embedding + each generation model
    cheapest_emb_val = min(total_embedding.values()) if total_embedding else 0.0
    total_one_time = {
        gen_m: round(cheapest_emb_val + gen_cost, 4)
        for gen_m, gen_cost in total_generation.items()
    }

    cheapest_model = min(total_one_time, key=total_one_time.get)
    candidates = [m for m in generation_models if any(x in m.lower() for x in ["mini", "flash", "haiku"])]
    recommended_model = min(candidates, key=total_one_time.get) if candidates else cheapest_model

    # 5. Export per-spec CSV
    rows = []
    for sc in per_spec_costs:
        row = {"identifier": sc.identifier, "filename": sc.filename, "token_count": sc.token_count}
        row.update({f"embed__{k}": v for k, v in sc.embedding_cost.items()})
        row.update({f"gen__{k}": v for k, v in sc.generation_cost.items()})
        rows.append(row)

    data_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(data_dir / "per_spec_costs.csv", index=False)
    logger.info(f"  Exported per-spec cost table -> {data_dir / 'per_spec_costs.csv'}")

    return CostReport(
        assumptions=assumptions,
        per_spec=per_spec_costs,
        total_embedding_cost=total_embedding,
        total_generation_cost=total_generation,
        total_one_time_cost=total_one_time,
        cheapest_model=cheapest_model,
        recommended_model=recommended_model,
    )