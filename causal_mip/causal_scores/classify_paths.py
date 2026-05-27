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


def _as_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


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


def _aggregate_optional_float(
    records: list[dict[str, Any]],
    key: str,
    aggregation: str,
) -> float | None:
    values = [_as_float(record.get(key)) for record in records]
    values = [value for value in values if value is not None]
    if not values:
        return None
    return _aggregate(values, aggregation)


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
        full_patchable_records = []
        for record in path_records:
            num_nodes = _as_int(record.get("num_nodes"), -1)
            num_patchable_nodes = _as_int(record.get("num_patchable_nodes"), 0)
            if num_nodes >= 0 and num_patchable_nodes == num_nodes:
                full_patchable_records.append(record)

        aggregated.append(
            {
                "path_id": path_id,
                "path_source": first.get("path_source"),
                "path_modality": first.get("path_modality"),
                "mip_score": first.get("mip_score"),
                "num_nodes": first.get("num_nodes"),
                "num_patchable_nodes": first.get("num_patchable_nodes"),
                "min_num_patchable_nodes": min(_as_int(record.get("num_patchable_nodes"), 0) for record in path_records),
                "max_num_patchable_nodes": max(_as_int(record.get("num_patchable_nodes"), 0) for record in path_records),
                "num_fully_patchable_score_records": len(full_patchable_records),
                "all_score_records_fully_patchable": len(full_patchable_records) == len(path_records),
                "num_positive_suf_records": sum(1 for value in suf_values if value > 0.0),
                "contains_projector": any(bool(record.get("contains_projector", False)) for record in path_records),
                "projector_patchable": any(bool(record.get("projector_patchable", False)) for record in path_records),
                "num_score_records": len(path_records),
                "pair_ids": sorted({record.get("pair_id") for record in path_records if record.get("pair_id") is not None}),
                "Nec": nec,
                "Suf": suf,
                "Ret": ret,
                "forget_effect": forget_effect,
                "retain_impact": retain_impact,
                "forget_saliency": _aggregate_optional_float(path_records, "forget_saliency", aggregation),
                "retain_anchor_saliency": _aggregate_optional_float(path_records, "retain_anchor_saliency", aggregation),
                "saliency_specificity_margin": _aggregate_optional_float(
                    path_records,
                    "saliency_specificity_margin",
                    aggregation,
                ),
                "saliency_specificity_ratio": _aggregate_optional_float(
                    path_records,
                    "saliency_specificity_ratio",
                    aggregation,
                ),
                "forget_fisher_saliency": _aggregate_optional_float(path_records, "forget_fisher_saliency", aggregation),
                "retain_anchor_fisher_saliency": _aggregate_optional_float(
                    path_records,
                    "retain_anchor_fisher_saliency",
                    aggregation,
                ),
                "fisher_specificity_margin": _aggregate_optional_float(
                    path_records,
                    "fisher_specificity_margin",
                    aggregation,
                ),
                "fisher_specificity_ratio": _aggregate_optional_float(
                    path_records,
                    "fisher_specificity_ratio",
                    aggregation,
                ),
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


def evaluate_forget_eligibility(
    path: dict[str, Any],
    min_forget_sufficiency: float = 0.0,
    require_positive_forget_sufficiency: bool = True,
    require_full_patchable_forget: bool = True,
    require_saliency_specificity: bool = False,
    saliency_specificity_key: str = "saliency_specificity_margin",
    min_saliency_specificity: float = 0.0,
    max_retain_anchor_saliency: float | None = None,
) -> dict[str, Any]:
    reasons = []
    sufficiency = float(path.get("Suf", 0.0))
    if require_positive_forget_sufficiency and sufficiency <= min_forget_sufficiency:
        reasons.append("sufficiency_not_positive")

    if require_full_patchable_forget:
        if not bool(path.get("all_score_records_fully_patchable", False)):
            reasons.append("not_all_score_records_fully_patchable")

    specificity_value = _as_float(path.get(saliency_specificity_key))
    if require_saliency_specificity:
        if specificity_value is None:
            reasons.append("missing_saliency_specificity")
        elif specificity_value <= min_saliency_specificity:
            reasons.append("saliency_specificity_too_low")

    retain_anchor_saliency = _as_float(path.get("retain_anchor_saliency"))
    if max_retain_anchor_saliency is not None:
        if retain_anchor_saliency is None:
            reasons.append("missing_retain_anchor_saliency")
        elif retain_anchor_saliency > max_retain_anchor_saliency:
            reasons.append("retain_anchor_saliency_too_high")

    return {
        "eligible": not reasons,
        "reasons": reasons,
        "min_forget_sufficiency": min_forget_sufficiency,
        "require_positive_forget_sufficiency": require_positive_forget_sufficiency,
        "require_full_patchable_forget": require_full_patchable_forget,
        "require_saliency_specificity": require_saliency_specificity,
        "saliency_specificity_key": saliency_specificity_key,
        "min_saliency_specificity": min_saliency_specificity,
        "max_retain_anchor_saliency": max_retain_anchor_saliency,
    }


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
    min_forget_sufficiency: float = 0.0,
    require_positive_forget_sufficiency: bool = True,
    require_full_patchable_forget: bool = True,
    require_saliency_specificity: bool = False,
    saliency_specificity_key: str = "saliency_specificity_margin",
    min_saliency_specificity: float = 0.0,
    max_retain_anchor_saliency: float | None = None,
    shared_on_retain_anchor_saliency: bool = True,
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
        path_record["pre_eligibility_category"] = category
        if category == CATEGORY_FORGET:
            eligibility = evaluate_forget_eligibility(
                path_record,
                min_forget_sufficiency=min_forget_sufficiency,
                require_positive_forget_sufficiency=require_positive_forget_sufficiency,
                require_full_patchable_forget=require_full_patchable_forget,
                require_saliency_specificity=require_saliency_specificity,
                saliency_specificity_key=saliency_specificity_key,
                min_saliency_specificity=min_saliency_specificity,
                max_retain_anchor_saliency=max_retain_anchor_saliency,
            )
            path_record["forget_eligibility"] = eligibility
            if not eligibility["eligible"]:
                if shared_on_retain_anchor_saliency and "retain_anchor_saliency_too_high" in eligibility["reasons"]:
                    category = CATEGORY_SHARED
                    path_record["demoted_from"] = CATEGORY_FORGET
                    path_record["promoted_shared_by"] = "retain_anchor_saliency"
                else:
                    category = CATEGORY_IRRELEVANT
                    path_record["demoted_from"] = CATEGORY_FORGET
        path_record["category"] = category
        path_record["thresholds"] = thresholds
        path_record.pop("_skipped_score_records", None)
        categories[category].append(path_record)

    skipped = aggregated_paths[0].get("_skipped_score_records", 0) if aggregated_paths else 0
    summary = build_classification_summary(
        categories,
        thresholds,
        len(records),
        skipped,
        eligibility={
            "min_forget_sufficiency": min_forget_sufficiency,
            "require_positive_forget_sufficiency": require_positive_forget_sufficiency,
            "require_full_patchable_forget": require_full_patchable_forget,
            "require_saliency_specificity": require_saliency_specificity,
            "saliency_specificity_key": saliency_specificity_key,
            "min_saliency_specificity": min_saliency_specificity,
            "max_retain_anchor_saliency": max_retain_anchor_saliency,
            "shared_on_retain_anchor_saliency": shared_on_retain_anchor_saliency,
            "num_demoted_from_forget": sum(
                1
                for paths in categories.values()
                for path in paths
                if path.get("demoted_from") == CATEGORY_FORGET
            ),
            "num_forget_to_shared_by_retain_anchor": sum(
                1
                for path in categories[CATEGORY_SHARED]
                if path.get("promoted_shared_by") == "retain_anchor_saliency"
            ),
        },
    )
    return categories, summary


def build_classification_summary(
    categories: dict[str, list[dict[str, Any]]],
    thresholds: dict[str, float],
    num_input_records: int,
    num_skipped_records: int,
    eligibility: dict[str, Any] | None = None,
) -> dict[str, Any]:
    all_paths = [path for paths in categories.values() for path in paths]
    return {
        "num_input_score_records": num_input_records,
        "num_skipped_score_records": num_skipped_records,
        "num_classified_paths": len(all_paths),
        "category_counts": {category: len(paths) for category, paths in categories.items()},
        "thresholds": thresholds,
        "eligibility": eligibility or {},
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
        "--min_forget_sufficiency",
        type=float,
        default=0.0,
        help="P_forget requires aggregated Suf greater than this value.",
    )
    parser.add_argument(
        "--allow_zero_sufficiency_forget",
        action="store_true",
        default=False,
        help="Compatibility mode: allow P_forget even when Suf is not positive.",
    )
    parser.add_argument(
        "--allow_partial_patchable_forget",
        action="store_true",
        default=False,
        help="Compatibility mode: allow P_forget even when not every score record is fully patchable.",
    )
    parser.add_argument(
        "--require_saliency_specificity",
        action="store_true",
        default=False,
        help="Require a SalUn/SSD-style forget-vs-retain saliency specificity field before assigning P_forget.",
    )
    parser.add_argument(
        "--saliency_specificity_key",
        choices=[
            "saliency_specificity_margin",
            "saliency_specificity_ratio",
            "fisher_specificity_margin",
            "fisher_specificity_ratio",
        ],
        default="saliency_specificity_margin",
    )
    parser.add_argument("--min_saliency_specificity", type=float, default=0.0)
    parser.add_argument(
        "--max_retain_anchor_saliency",
        type=float,
        default=None,
        help="If set, P_forget candidates above this retain-anchor saliency are routed to P_shared by default.",
    )
    parser.add_argument(
        "--demote_high_retain_anchor_to_irrelevant",
        action="store_true",
        default=False,
        help="Compatibility/debug mode: do not route high retain-anchor saliency P_forget candidates to P_shared.",
    )
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
        min_forget_sufficiency=args.min_forget_sufficiency,
        require_positive_forget_sufficiency=not args.allow_zero_sufficiency_forget,
        require_full_patchable_forget=not args.allow_partial_patchable_forget,
        require_saliency_specificity=args.require_saliency_specificity,
        saliency_specificity_key=args.saliency_specificity_key,
        min_saliency_specificity=args.min_saliency_specificity,
        max_retain_anchor_saliency=args.max_retain_anchor_saliency,
        shared_on_retain_anchor_saliency=not args.demote_high_retain_anchor_to_irrelevant,
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
