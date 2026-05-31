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
    alpha_name: float = 1.0,
    aggregation: str = "mean",
    clip_negative_effects: bool = True,
    include_non_ok: bool = False,
    aggregation_key: str = "path",
) -> list[dict[str, Any]]:
    if aggregation_key not in {"path", "path_pair"}:
        raise ValueError(f"Unsupported aggregation_key: {aggregation_key}")

    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    skipped = 0

    for record in records:
        if not include_non_ok and record.get("status") != "ok":
            skipped += 1
            continue
        path_id = record.get("path_id")
        if path_id is None:
            skipped += 1
            continue
        pair_id = record.get("pair_id")
        if aggregation_key == "path_pair" and pair_id is None:
            skipped += 1
            continue
        if _as_float(record.get("Nec")) is None or _as_float(record.get("Suf")) is None or _as_float(record.get("Ret")) is None:
            skipped += 1
            continue
        if aggregation_key == "path_pair":
            grouped[(str(pair_id), str(path_id))].append(record)
        else:
            grouped[(str(path_id),)].append(record)

    aggregated: list[dict[str, Any]] = []
    for group_key, path_records in sorted(grouped.items()):
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
        name_nec = _aggregate_optional_float(path_records, "NameNec", aggregation)
        name_suf = _aggregate_optional_float(path_records, "NameSuf", aggregation)
        name_ret = _aggregate_optional_float(path_records, "NameRet", aggregation)
        name_forget_effect = None
        if name_nec is not None or name_suf is not None:
            name_forget_effect = _positive_or_signed(name_nec or 0.0, clip_negative_effects) + alpha_name * _positive_or_signed(name_suf or 0.0, clip_negative_effects)
        name_retain_impact = None
        if name_ret is not None:
            name_retain_impact = _positive_or_signed(name_ret, clip_negative_effects)
        name_editable_score = None
        if name_forget_effect is not None:
            name_editable_score = float(name_forget_effect) / (1e-6 + float(name_retain_impact or 0.0))
        first = path_records[0]
        path_id = str(first.get("path_id"))
        pair_ids = sorted({record.get("pair_id") for record in path_records if record.get("pair_id") is not None})
        full_patchable_records = []
        for record in path_records:
            num_nodes = _as_int(record.get("num_nodes"), -1)
            num_patchable_nodes = _as_int(record.get("num_patchable_nodes"), 0)
            if num_nodes >= 0 and num_patchable_nodes == num_nodes:
                full_patchable_records.append(record)

        aggregated_record = {
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
            "num_positive_name_suf_records": sum(
                1
                for value in (_as_float(record.get("NameSuf")) for record in path_records)
                if value is not None and value > 0.0
            ),
            "contains_projector": any(bool(record.get("contains_projector", False)) for record in path_records),
            "projector_patchable": any(bool(record.get("projector_patchable", False)) for record in path_records),
            "num_score_records": len(path_records),
            "pair_ids": pair_ids,
            "Nec": nec,
            "Suf": suf,
            "Ret": ret,
            "forget_effect": forget_effect,
            "retain_impact": retain_impact,
            "NameNec": name_nec,
            "NameSuf": name_suf,
            "NameRet": name_ret,
            "name_forget_effect": name_forget_effect,
            "name_retain_impact": name_retain_impact,
            "name_editable_score": name_editable_score,
            "target_names": sorted(
                {
                    str(record.get("target_name"))
                    for record in path_records
                    if record.get("target_name")
                }
            ),
            "name_match_statuses": sorted(
                {
                    str(record.get("name_match_status"))
                    for record in path_records
                    if record.get("name_match_status")
                }
            ),
            "forget_saliency": _aggregate_optional_float(path_records, "forget_saliency", aggregation),
            "retain_anchor_saliency": _aggregate_optional_float(path_records, "retain_anchor_saliency", aggregation),
            "max_anchor_retain_saliency": _aggregate_optional_float(
                path_records,
                "max_anchor_retain_saliency",
                aggregation,
            ),
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
            "min_anchor_margin": _aggregate_optional_float(
                path_records,
                "min_anchor_margin",
                aggregation,
            ),
            "min_anchor_ratio": _aggregate_optional_float(
                path_records,
                "min_anchor_ratio",
                aggregation,
            ),
            "forget_fisher_saliency": _aggregate_optional_float(path_records, "forget_fisher_saliency", aggregation),
            "retain_anchor_fisher_saliency": _aggregate_optional_float(
                path_records,
                "retain_anchor_fisher_saliency",
                aggregation,
            ),
            "max_anchor_retain_fisher_saliency": _aggregate_optional_float(
                path_records,
                "max_anchor_retain_fisher_saliency",
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
            "min_anchor_fisher_margin": _aggregate_optional_float(
                path_records,
                "min_anchor_fisher_margin",
                aggregation,
            ),
            "min_anchor_fisher_ratio": _aggregate_optional_float(
                path_records,
                "min_anchor_fisher_ratio",
                aggregation,
            ),
            "aggregation": aggregation,
            "aggregation_key": aggregation_key,
            "alpha": alpha,
            "alpha_name": alpha_name,
            "clip_negative_effects": clip_negative_effects,
        }
        if aggregation_key == "path_pair":
            pair_id = str(first.get("pair_id"))
            aggregated_record["pair_id"] = pair_id
            aggregated_record["pair_path_id"] = f"{pair_id}::{path_id}"
        aggregated.append(aggregated_record)

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


def compute_name_classification_thresholds(
    aggregated_paths: list[dict[str, Any]],
    name_forget_threshold: float | None = None,
    name_retain_threshold: float | None = None,
    name_forget_quantile: float = 0.75,
    name_retain_quantile: float = 0.75,
    min_name_forget_effect: float = 0.0,
    min_name_retain_impact: float = 0.0,
) -> dict[str, float]:
    name_forget_values = [
        float(path["name_forget_effect"])
        for path in aggregated_paths
        if path.get("name_forget_effect") is not None
    ]
    name_retain_values = [
        float(path["name_retain_impact"])
        for path in aggregated_paths
        if path.get("name_retain_impact") is not None
    ]
    resolved_forget = (
        float(name_forget_threshold)
        if name_forget_threshold is not None
        else max(float(min_name_forget_effect), _quantile(name_forget_values, name_forget_quantile))
    )
    resolved_retain = (
        float(name_retain_threshold)
        if name_retain_threshold is not None
        else max(float(min_name_retain_impact), _quantile(name_retain_values, name_retain_quantile))
    )
    return {
        "name_forget_threshold": resolved_forget,
        "name_retain_threshold": resolved_retain,
        "name_forget_quantile": name_forget_quantile,
        "name_retain_quantile": name_retain_quantile,
        "min_name_forget_effect": min_name_forget_effect,
        "min_name_retain_impact": min_name_retain_impact,
        "num_name_forget_values": len(name_forget_values),
        "num_name_retain_values": len(name_retain_values),
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


def classify_name_aware_path(path: dict[str, Any], name_thresholds: dict[str, float]) -> str:
    name_forget = _as_float(path.get("name_forget_effect"))
    name_retain = _as_float(path.get("name_retain_impact"))
    forget_high = name_forget is not None and name_forget > name_thresholds["name_forget_threshold"]
    retain_high = name_retain is not None and name_retain > name_thresholds["name_retain_threshold"]
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
    retain_anchor_saliency_key: str = "retain_anchor_saliency",
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

    retain_anchor_saliency = _as_float(path.get(retain_anchor_saliency_key))
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
        "retain_anchor_saliency_key": retain_anchor_saliency_key,
    }


def evaluate_name_forget_eligibility(
    path: dict[str, Any],
    min_name_sufficiency: float = 0.0,
    require_positive_name_sufficiency: bool = True,
    require_full_patchable_forget: bool = True,
) -> dict[str, Any]:
    reasons = []
    name_suf = _as_float(path.get("NameSuf"))
    if require_positive_name_sufficiency:
        if name_suf is None:
            reasons.append("missing_name_sufficiency")
        elif name_suf <= min_name_sufficiency:
            reasons.append("name_sufficiency_not_positive")

    if require_full_patchable_forget:
        if not bool(path.get("all_score_records_fully_patchable", False)):
            reasons.append("not_all_score_records_fully_patchable")

    return {
        "eligible": not reasons,
        "reasons": reasons,
        "min_name_sufficiency": min_name_sufficiency,
        "require_positive_name_sufficiency": require_positive_name_sufficiency,
        "require_full_patchable_forget": require_full_patchable_forget,
    }


def classify_path_scores(
    records: list[dict[str, Any]],
    alpha: float = 1.0,
    alpha_name: float = 1.0,
    aggregation: str = "mean",
    clip_negative_effects: bool = True,
    aggregation_key: str = "path",
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
    retain_anchor_saliency_key: str = "retain_anchor_saliency",
    shared_on_retain_anchor_saliency: bool = True,
    name_aware_forget: bool = False,
    name_forget_threshold: float | None = None,
    name_retain_threshold: float | None = None,
    name_forget_quantile: float = 0.75,
    name_retain_quantile: float = 0.75,
    min_name_forget_effect: float = 0.0,
    min_name_retain_impact: float = 0.0,
    min_name_sufficiency: float = 0.0,
    require_positive_name_sufficiency: bool = True,
    max_forget_projector_paths: int = 0,
    projector_name_effect_ratio_threshold: float = 1.2,
    projector_topk_metric: str = "name_forget_effect",
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    aggregated_paths = aggregate_path_score_records(
        records=records,
        alpha=alpha,
        alpha_name=alpha_name,
        aggregation=aggregation,
        clip_negative_effects=clip_negative_effects,
        include_non_ok=include_non_ok,
        aggregation_key=aggregation_key,
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
    name_thresholds = compute_name_classification_thresholds(
        aggregated_paths=aggregated_paths,
        name_forget_threshold=name_forget_threshold,
        name_retain_threshold=name_retain_threshold,
        name_forget_quantile=name_forget_quantile,
        name_retain_quantile=name_retain_quantile,
        min_name_forget_effect=min_name_forget_effect,
        min_name_retain_impact=min_name_retain_impact,
    ) if name_aware_forget else None
    categories = {
        CATEGORY_FORGET: [],
        CATEGORY_SHARED: [],
        CATEGORY_RETAIN: [],
        CATEGORY_IRRELEVANT: [],
    }
    projector_candidates: list[dict[str, Any]] = []
    for path in aggregated_paths:
        category = (
            classify_name_aware_path(path, name_thresholds)
            if name_aware_forget and name_thresholds is not None
            else classify_aggregated_path(path, thresholds)
        )
        path_record = dict(path)
        path_record["pre_eligibility_category"] = category
        if name_aware_forget:
            path_record["answer_level_category"] = classify_aggregated_path(path, thresholds)
        if category == CATEGORY_FORGET:
            if name_aware_forget:
                eligibility = evaluate_name_forget_eligibility(
                    path_record,
                    min_name_sufficiency=min_name_sufficiency,
                    require_positive_name_sufficiency=require_positive_name_sufficiency,
                    require_full_patchable_forget=require_full_patchable_forget,
                )
            else:
                eligibility = evaluate_forget_eligibility(
                    path_record,
                    min_forget_sufficiency=min_forget_sufficiency,
                    require_positive_forget_sufficiency=require_positive_forget_sufficiency,
                    require_full_patchable_forget=require_full_patchable_forget,
                    require_saliency_specificity=require_saliency_specificity,
                    saliency_specificity_key=saliency_specificity_key,
                    min_saliency_specificity=min_saliency_specificity,
                    max_retain_anchor_saliency=max_retain_anchor_saliency,
                    retain_anchor_saliency_key=retain_anchor_saliency_key,
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
        elif (
            name_aware_forget
            and max_forget_projector_paths > 0
            and bool(path_record.get("contains_projector", False))
            and path_record.get("name_forget_effect") is not None
        ):
            name_effect = float(path_record.get("name_forget_effect") or 0.0)
            retain_effect = float(path_record.get("name_retain_impact") or 0.0)
            min_effect = name_thresholds["name_forget_threshold"] if name_thresholds is not None else 0.0
            ratio = float("inf") if retain_effect <= 0.0 else name_effect / retain_effect
            name_suf = _as_float(path_record.get("NameSuf"))
            if (
                name_effect > min_effect
                and ratio >= projector_name_effect_ratio_threshold
                and (not require_positive_name_sufficiency or (name_suf is not None and name_suf > min_name_sufficiency))
                and (not require_full_patchable_forget or bool(path_record.get("all_score_records_fully_patchable", False)))
            ):
                projector_candidates.append(path_record)
        path_record["category"] = category
        path_record["thresholds"] = thresholds
        if name_thresholds is not None:
            path_record["name_thresholds"] = name_thresholds
        path_record.pop("_skipped_score_records", None)
        categories[category].append(path_record)

    promoted_projectors = []
    if name_aware_forget and max_forget_projector_paths > 0 and projector_candidates:
        existing_forget_ids = {path["path_id"] for path in categories[CATEGORY_FORGET]}
        projector_candidates = [
            path
            for path in projector_candidates
            if path["path_id"] not in existing_forget_ids
        ]
        projector_candidates.sort(
            key=lambda path: (
                float(path.get(projector_topk_metric) or 0.0),
                float(path.get("name_forget_effect") or 0.0),
                float(path.get("NameSuf") or 0.0),
            ),
            reverse=True,
        )
        promoted_ids = {path["path_id"] for path in projector_candidates[:max_forget_projector_paths]}
        if promoted_ids:
            for category_name in [CATEGORY_SHARED, CATEGORY_RETAIN, CATEGORY_IRRELEVANT]:
                kept = []
                for path in categories[category_name]:
                    if path["path_id"] in promoted_ids:
                        promoted = dict(path)
                        promoted["category"] = CATEGORY_FORGET
                        promoted["promoted_to_forget_by"] = "projector_name_effect_topk"
                        promoted["previous_category"] = category_name
                        categories[CATEGORY_FORGET].append(promoted)
                        promoted_projectors.append(promoted)
                    else:
                        kept.append(path)
                categories[category_name] = kept

    skipped = aggregated_paths[0].get("_skipped_score_records", 0) if aggregated_paths else 0
    summary = build_classification_summary(
        categories,
        thresholds,
        len(records),
        skipped,
        aggregation_key=aggregation_key,
        eligibility={
            "aggregation_key": aggregation_key,
            "min_forget_sufficiency": min_forget_sufficiency,
            "require_positive_forget_sufficiency": require_positive_forget_sufficiency,
            "require_full_patchable_forget": require_full_patchable_forget,
            "require_saliency_specificity": require_saliency_specificity,
            "saliency_specificity_key": saliency_specificity_key,
            "min_saliency_specificity": min_saliency_specificity,
            "max_retain_anchor_saliency": max_retain_anchor_saliency,
            "retain_anchor_saliency_key": retain_anchor_saliency_key,
            "shared_on_retain_anchor_saliency": shared_on_retain_anchor_saliency,
            "name_aware_forget": name_aware_forget,
            "alpha_name": alpha_name,
            "name_thresholds": name_thresholds,
            "min_name_sufficiency": min_name_sufficiency,
            "require_positive_name_sufficiency": require_positive_name_sufficiency,
            "max_forget_projector_paths": max_forget_projector_paths,
            "projector_name_effect_ratio_threshold": projector_name_effect_ratio_threshold,
            "projector_topk_metric": projector_topk_metric,
            "num_projector_promoted_to_forget": len(promoted_projectors),
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
    aggregation_key: str = "path",
    eligibility: dict[str, Any] | None = None,
) -> dict[str, Any]:
    all_paths = [path for paths in categories.values() for path in paths]
    return {
        "num_input_score_records": num_input_records,
        "num_skipped_score_records": num_skipped_records,
        "num_classified_paths": len(all_paths),
        "aggregation_key": aggregation_key,
        "category_counts": {category: len(paths) for category, paths in categories.items()},
        "thresholds": thresholds,
        "eligibility": eligibility or {},
        "modalities": {
            modality: sum(1 for path in all_paths if path.get("path_modality") == modality)
            for modality in sorted({path.get("path_modality") for path in all_paths})
        },
        "category_modalities": {
            category: {
                modality: sum(1 for path in paths if path.get("path_modality") == modality)
                for modality in sorted({path.get("path_modality") for path in paths})
            }
            for category, paths in categories.items()
        },
        "category_projector_counts": {
            category: {
                "contains_projector_true": sum(1 for path in paths if bool(path.get("contains_projector", False))),
                "contains_projector_false": sum(1 for path in paths if not bool(path.get("contains_projector", False))),
            }
            for category, paths in categories.items()
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
    parser.add_argument("--alpha_name", type=float, default=1.0, help="Weight for NameSuf in name_forget_effect.")
    parser.add_argument("--aggregation", choices=["mean", "median"], default="mean")
    parser.add_argument(
        "--aggregation_key",
        choices=["path", "path_pair"],
        default="path",
        help="Aggregate globally by path_id or pair-specifically by (pair_id, path_id).",
    )
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
            "min_anchor_margin",
            "min_anchor_ratio",
            "fisher_specificity_margin",
            "fisher_specificity_ratio",
            "min_anchor_fisher_margin",
            "min_anchor_fisher_ratio",
        ],
        default="saliency_specificity_margin",
    )
    parser.add_argument("--min_saliency_specificity", type=float, default=0.0)
    parser.add_argument(
        "--retain_anchor_saliency_key",
        choices=[
            "retain_anchor_saliency",
            "max_anchor_retain_saliency",
            "retain_anchor_fisher_saliency",
            "max_anchor_retain_fisher_saliency",
        ],
        default="retain_anchor_saliency",
        help="Retain saliency field used by --max_retain_anchor_saliency.",
    )
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
    parser.add_argument(
        "--name_aware_forget",
        action="store_true",
        default=False,
        help="Classify paths using Step10 NameNec/NameSuf/NameRet fields instead of answer-level effects.",
    )
    parser.add_argument("--name_forget_threshold", type=float, default=None)
    parser.add_argument("--name_retain_threshold", type=float, default=None)
    parser.add_argument("--name_forget_quantile", type=float, default=0.75)
    parser.add_argument("--name_retain_quantile", type=float, default=0.75)
    parser.add_argument("--min_name_forget_effect", type=float, default=0.0)
    parser.add_argument("--min_name_retain_impact", type=float, default=0.0)
    parser.add_argument("--min_name_sufficiency", type=float, default=0.0)
    parser.add_argument(
        "--allow_zero_name_sufficiency_forget",
        action="store_true",
        default=False,
        help="Compatibility/debug mode: allow name-aware P_forget when NameSuf is not positive.",
    )
    parser.add_argument("--max_forget_projector_paths", type=int, default=0)
    parser.add_argument("--projector_name_effect_ratio_threshold", type=float, default=1.2)
    parser.add_argument(
        "--projector_topk_metric",
        choices=["name_forget_effect", "name_editable_score", "NameSuf"],
        default="name_forget_effect",
        help="Metric used to rank high name-sensitive projector paths promoted into P_forget.",
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
        alpha_name=args.alpha_name,
        aggregation=args.aggregation,
        clip_negative_effects=not args.use_signed_effects,
        aggregation_key=args.aggregation_key,
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
        retain_anchor_saliency_key=args.retain_anchor_saliency_key,
        shared_on_retain_anchor_saliency=not args.demote_high_retain_anchor_to_irrelevant,
        name_aware_forget=args.name_aware_forget,
        name_forget_threshold=args.name_forget_threshold,
        name_retain_threshold=args.name_retain_threshold,
        name_forget_quantile=args.name_forget_quantile,
        name_retain_quantile=args.name_retain_quantile,
        min_name_forget_effect=args.min_name_forget_effect,
        min_name_retain_impact=args.min_name_retain_impact,
        min_name_sufficiency=args.min_name_sufficiency,
        require_positive_name_sufficiency=not args.allow_zero_name_sufficiency_forget,
        max_forget_projector_paths=args.max_forget_projector_paths,
        projector_name_effect_ratio_threshold=args.projector_name_effect_ratio_threshold,
        projector_topk_metric=args.projector_topk_metric,
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
