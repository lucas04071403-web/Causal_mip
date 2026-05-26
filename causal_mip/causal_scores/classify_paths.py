from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable


CATEGORY_FORGET = "P_forget"
CATEGORY_SHARED = "P_shared"
CATEGORY_RETAIN = "P_retain"
CATEGORY_IRRELEVANT = "P_irrelevant"


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"Quantile must be in [0, 1], got {q}")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def load_score_records_jsonl(paths: list[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def _aggregate(values: list[float], mode: str) -> float:
    if not values:
        return 0.0
    if mode == "mean":
        return float(mean(values))
    if mode == "median":
        return float(median(values))
    raise ValueError(f"Unsupported aggregation mode: {mode}")


def _positive_or_signed(value: float, clip_negative_effects: bool) -> float:
    return max(0.0, value) if clip_negative_effects else value


def aggregate_path_score_records(
    records: list[dict[str, Any]],
    alpha: float = 1.0,
    aggregation: str = "mean",
    clip_negative_effects: bool = True,
    include_non_ok: bool = False,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    skipped = 0

    for record in records:
        if not include_non_ok and record.get("status") != "ok":
            skipped += 1
            continue
        path_id = record.get("path_id")
        if path_id is None:
            skipped += 1
            continue
        if _as_float(record.get("Nec")) is None or _as_float(record.get("Suf")) is None or _as_float(record.get("Ret")) is None:
            skipped += 1
            continue
        grouped[path_id].append(record)

    aggregated: list[dict[str, Any]] = []
    for path_id, path_records in sorted(grouped.items()):
        nec_values = [_as_float(record.get("Nec")) for record in path_records]
        suf_values = [_as_float(record.get("Suf")) for record in path_records]
        ret_values = [_as_float(record.get("Ret")) for record in path_records]
        nec_values = [value for value in nec_values if value is not None]
        suf_values = [value for value in suf_values if value is not None]
        ret_values = [value for value in ret_values if value is not None]

        nec = _aggregate(nec_values, aggregation)
        suf = _aggregate(suf_values, aggregation)
        ret = _aggregate(ret_values, aggregation)
        forget_effect = _positive_or_signed(nec, clip_negative_effects) + alpha * _positive_or_signed(suf, clip_negative_effects)
        retain_impact = _positive_or_signed(ret, clip_negative_effects)
        first = path_records[0]

        aggregated.append(
            {
                "path_id": path_id,
                "path_source": first.get("path_source"),
                "path_modality": first.get("path_modality"),
                "mip_score": first.get("mip_score"),
                "num_nodes": first.get("num_nodes"),
                "num_patchable_nodes": first.get("num_patchable_nodes"),
                "num_score_records": len(path_records),
                "pair_ids": sorted({record.get("pair_id") for record in path_records if record.get("pair_id") is not None}),
                "Nec": nec,
                "Suf": suf,
                "Ret": ret,
                "forget_effect": forget_effect,
                "retain_impact": retain_impact,
                "aggregation": aggregation,
                "alpha": alpha,
                "clip_negative_effects": clip_negative_effects,
            }
        )

    for item in aggregated:
        item["_skipped_score_records"] = skipped
    return aggregated


def compute_classification_thresholds(
    aggregated_paths: list[dict[str, Any]],
    forget_threshold: float | None = None,
    retain_threshold: float | None = None,
    forget_quantile: float = 0.75,
    retain_quantile: float = 0.75,
    min_forget_effect: float = 0.0,
    min_retain_impact: float = 0.0,
) -> dict[str, float]:
    forget_values = [float(path["forget_effect"]) for path in aggregated_paths]
    retain_values = [float(path["retain_impact"]) for path in aggregated_paths]
    resolved_forget = (
        float(forget_threshold)
        if forget_threshold is not None
        else max(float(min_forget_effect), _quantile(forget_values, forget_quantile))
    )
    resolved_retain = (
        float(retain_threshold)
        if retain_threshold is not None
        else max(float(min_retain_impact), _quantile(retain_values, retain_quantile))
    )
    return {
        "forget_threshold": resolved_forget,
        "retain_threshold": resolved_retain,
        "forget_quantile": forget_quantile,
        "retain_quantile": retain_quantile,
        "min_forget_effect": min_forget_effect,
        "min_retain_impact": min_retain_impact,
    }


def classify_aggregated_path(path: dict[str, Any], thresholds: dict[str, float]) -> str:
    forget_high = float(path["forget_effect"]) > thresholds["forget_threshold"]
    retain_high = float(path["retain_impact"]) > thresholds["retain_threshold"]
    if forget_high and not retain_high:
        return CATEGORY_FORGET
    if forget_high and retain_high:
        return CATEGORY_SHARED
    if not forget_high and retain_high:
        return CATEGORY_RETAIN
    return CATEGORY_IRRELEVANT


def classify_path_scores(
    records: list[dict[str, Any]],
    alpha: float = 1.0,
    aggregation: str = "mean",
    clip_negative_effects: bool = True,
    forget_threshold: float | None = None,
    retain_threshold: float | None = None,
    forget_quantile: float = 0.75,
    retain_quantile: float = 0.75,
    min_forget_effect: float = 0.0,
    min_retain_impact: float = 0.0,
    include_non_ok: bool = False,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    aggregated_paths = aggregate_path_score_records(
        records=records,
        alpha=alpha,
        aggregation=aggregation,
        clip_negative_effects=clip_negative_effects,
        include_non_ok=include_non_ok,
    )
    thresholds = compute_classification_thresholds(
        aggregated_paths=aggregated_paths,
        forget_threshold=forget_threshold,
        retain_threshold=retain_threshold,
        forget_quantile=forget_quantile,
        retain_quantile=retain_quantile,
        min_forget_effect=min_forget_effect,
        min_retain_impact=min_retain_impact,
    )
    categories = {
        CATEGORY_FORGET: [],
        CATEGORY_SHARED: [],
        CATEGORY_RETAIN: [],
        CATEGORY_IRRELEVANT: [],
    }
    for path in aggregated_paths:
        category = classify_aggregated_path(path, thresholds)
        path_record = dict(path)
        path_record["category"] = category
        path_record["thresholds"] = thresholds
        path_record.pop("_skipped_score_records", None)
        categories[category].append(path_record)

    skipped = aggregated_paths[0].get("_skipped_score_records", 0) if aggregated_paths else 0
    summary = build_classification_summary(categories, thresholds, len(records), skipped)
    return categories, summary


def build_classification_summary(
    categories: dict[str, list[dict[str, Any]]],
    thresholds: dict[str, float],
    num_input_records: int,
    num_skipped_records: int,
) -> dict[str, Any]:
    all_paths = [path for paths in categories.values() for path in paths]
    return {
        "num_input_score_records": num_input_records,
        "num_skipped_score_records": num_skipped_records,
        "num_classified_paths": len(all_paths),
        "category_counts": {category: len(paths) for category, paths in categories.items()},
        "thresholds": thresholds,
        "modalities": {
            modality: sum(1 for path in all_paths if path.get("path_modality") == modality)
            for modality in sorted({path.get("path_modality") for path in all_paths})
        },
    }


def write_classified_paths(
    categories: dict[str, list[dict[str, Any]]],
    summary: dict[str, Any],
    output_dir: str,
    write_combined: bool = True,
) -> dict[str, str]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}
    for category, paths in categories.items():
        path = output / f"{category}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for record in paths:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        written[category] = str(path)

    if write_combined:
        combined_path = output / "P_classified.jsonl"
        with combined_path.open("w", encoding="utf-8") as handle:
            for category in [CATEGORY_FORGET, CATEGORY_SHARED, CATEGORY_RETAIN, CATEGORY_IRRELEVANT]:
                for record in categories[category]:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        written["P_classified"] = str(combined_path)

    summary_path = output / "classification_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    written["summary"] = str(summary_path)
    return written


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Classify Step 5 causal path scores into Step 6 path sets.")
    parser.add_argument("--scores_path", nargs="+", required=True, help="One or more Step 5 path score JSONL files.")
    parser.add_argument("--output_dir", required=True, help="Directory for P_forget/P_shared/P_retain/P_irrelevant JSONL files.")
    parser.add_argument("--alpha", type=float, default=1.0, help="Weight for Suf in forget_effect = Nec + alpha * Suf.")
    parser.add_argument("--aggregation", choices=["mean", "median"], default="mean")
    parser.add_argument("--forget_threshold", type=float, default=None)
    parser.add_argument("--retain_threshold", type=float, default=None)
    parser.add_argument("--forget_quantile", type=float, default=0.75)
    parser.add_argument("--retain_quantile", type=float, default=0.75)
    parser.add_argument("--min_forget_effect", type=float, default=0.0)
    parser.add_argument("--min_retain_impact", type=float, default=0.0)
    parser.add_argument(
        "--use_signed_effects",
        action="store_true",
        default=False,
        help="Use signed Nec/Suf/Ret values instead of clipping negative effects to zero.",
    )
    parser.add_argument("--include_non_ok", action="store_true", default=False)
    parser.add_argument("--no_combined_output", action="store_true", default=False)
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    records = load_score_records_jsonl(args.scores_path)
    categories, summary = classify_path_scores(
        records=records,
        alpha=args.alpha,
        aggregation=args.aggregation,
        clip_negative_effects=not args.use_signed_effects,
        forget_threshold=args.forget_threshold,
        retain_threshold=args.retain_threshold,
        forget_quantile=args.forget_quantile,
        retain_quantile=args.retain_quantile,
        min_forget_effect=args.min_forget_effect,
        min_retain_impact=args.min_retain_impact,
        include_non_ok=args.include_non_ok,
    )
    written = write_classified_paths(
        categories=categories,
        summary=summary,
        output_dir=args.output_dir,
        write_combined=not args.no_combined_output,
    )
    print(json.dumps({"summary": summary, "outputs": written}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
