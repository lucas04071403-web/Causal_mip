from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from causal_mip.path_localization.path_schema import CandidatePath


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_jsonl(path: str) -> list[dict[str, Any]]:
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _load_candidate_paths_jsonl(path: str) -> list[CandidatePath]:
    return [CandidatePath.from_dict(record) for record in _load_jsonl(path)]


def _write_jsonl(path: str | Path, records: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _aggregate_score_records(records: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        pair_id = record.get("pair_id")
        path_id = record.get("path_id")
        if pair_id is None or path_id is None:
            continue
        if record.get("status") != "ok":
            continue
        grouped[(str(pair_id), str(path_id))].append(record)

    aggregated = {}
    for key, group in grouped.items():
        first = group[0]
        margins = [_as_float(record.get("saliency_specificity_margin")) for record in group]
        ratios = [_as_float(record.get("saliency_specificity_ratio")) for record in group]
        forget = [_as_float(record.get("forget_saliency")) for record in group]
        retain = [_as_float(record.get("retain_anchor_saliency")) for record in group]
        fisher_margins = [_as_float(record.get("fisher_specificity_margin")) for record in group]
        margins = [value for value in margins if value is not None]
        ratios = [value for value in ratios if value is not None]
        forget = [value for value in forget if value is not None]
        retain = [value for value in retain if value is not None]
        fisher_margins = [value for value in fisher_margins if value is not None]
        if not margins:
            continue
        aggregated[key] = {
            "pair_id": first.get("pair_id"),
            "path_id": first.get("path_id"),
            "path_modality": first.get("path_modality"),
            "path_source": first.get("path_source"),
            "specificity_score": float(mean(margins)),
            "saliency_specificity_margin": float(mean(margins)),
            "saliency_specificity_ratio": float(mean(ratios)) if ratios else None,
            "forget_saliency": float(mean(forget)) if forget else None,
            "retain_anchor_saliency": float(mean(retain)) if retain else None,
            "fisher_specificity_margin": float(mean(fisher_margins)) if fisher_margins else None,
            "num_score_records": len(group),
            "contains_projector": any(bool(record.get("contains_projector", False)) for record in group),
            "projector_patchable": any(bool(record.get("projector_patchable", False)) for record in group),
            "all_nodes_patchable": all(bool(record.get("all_nodes_patchable", False)) for record in group),
        }
    return aggregated


def _filter_records(
    records: list[dict[str, Any]],
    modalities: set[str] | None,
    min_specificity_margin: float | None,
    require_positive_specificity: bool,
    require_full_patchable: bool,
    require_projector_for_vision_text: bool,
) -> list[dict[str, Any]]:
    filtered = []
    for record in records:
        if modalities and record.get("path_modality") not in modalities:
            continue
        specificity = _as_float(record.get("saliency_specificity_margin"))
        if specificity is None:
            continue
        if require_positive_specificity and specificity <= 0.0:
            continue
        if min_specificity_margin is not None and specificity < min_specificity_margin:
            continue
        if require_full_patchable and not bool(record.get("all_nodes_patchable", False)):
            continue
        if (
            require_projector_for_vision_text
            and record.get("path_modality") == "vision_text"
            and not bool(record.get("projector_patchable", False))
        ):
            continue
        filtered.append(record)
    return filtered


def _rank_key(record: dict[str, Any]) -> tuple[float, float, float]:
    specificity = _as_float(record.get("saliency_specificity_margin")) or 0.0
    ratio = _as_float(record.get("saliency_specificity_ratio")) or 0.0
    forget = _as_float(record.get("forget_saliency")) or 0.0
    return specificity, ratio, forget


def _select_top_records(
    records: list[dict[str, Any]],
    top_k_per_pair: int | None,
    top_k_per_pair_modality: int | None,
    top_k_total: int | None,
) -> list[dict[str, Any]]:
    selected_by_key: dict[tuple[str, str], dict[str, Any]] = {}

    if top_k_per_pair_modality is not None:
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            grouped[(str(record["pair_id"]), str(record.get("path_modality")))].append(record)
        for group in grouped.values():
            for record in sorted(group, key=_rank_key, reverse=True)[:top_k_per_pair_modality]:
                selected_by_key[(str(record["pair_id"]), str(record["path_id"]))] = record

    if top_k_per_pair is not None:
        grouped_by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            grouped_by_pair[str(record["pair_id"])].append(record)
        for group in grouped_by_pair.values():
            for record in sorted(group, key=_rank_key, reverse=True)[:top_k_per_pair]:
                selected_by_key[(str(record["pair_id"]), str(record["path_id"]))] = record

    if top_k_per_pair is None and top_k_per_pair_modality is None:
        for record in records:
            selected_by_key[(str(record["pair_id"]), str(record["path_id"]))] = record

    selected = sorted(selected_by_key.values(), key=_rank_key, reverse=True)
    if top_k_total is not None:
        selected = selected[:top_k_total]
    return selected


def _clone_selected_candidate_paths(
    candidate_by_id: dict[str, CandidatePath],
    selected_records: list[dict[str, Any]],
    path_id_prefix: str,
) -> tuple[list[CandidatePath], dict[tuple[str, str], str]]:
    selected_path_ids = []
    for record in selected_records:
        path_id = str(record["path_id"])
        if path_id not in selected_path_ids:
            selected_path_ids.append(path_id)

    old_to_new = {path_id: f"{path_id_prefix}_p{idx:06d}" for idx, path_id in enumerate(selected_path_ids)}
    selected_by_path: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in selected_records:
        selected_by_path[str(record["path_id"])].append(record)

    candidates = []
    for old_path_id in selected_path_ids:
        original = candidate_by_id.get(old_path_id)
        if original is None:
            continue
        path_records = selected_by_path[old_path_id]
        best_record = max(path_records, key=_rank_key)
        candidate = CandidatePath.from_dict(original.to_dict())
        candidate.path_id = old_to_new[old_path_id]
        candidate.source = f"{original.source}_saliency_specific"
        candidate.mip_score = float(best_record["specificity_score"])
        candidate.metadata = {
            **(candidate.metadata or {}),
            "original_path_id": old_path_id,
            "saliency_specificity_margin": best_record.get("saliency_specificity_margin"),
            "saliency_specificity_ratio": best_record.get("saliency_specificity_ratio"),
            "forget_saliency": best_record.get("forget_saliency"),
            "retain_anchor_saliency": best_record.get("retain_anchor_saliency"),
            "fisher_specificity_margin": best_record.get("fisher_specificity_margin"),
            "saliency_selected_pair_ids": sorted({str(record["pair_id"]) for record in path_records}),
            "candidate_generation": "saliency_specificity",
        }
        candidates.append(candidate)

    pair_path_to_new = {
        (str(record["pair_id"]), str(record["path_id"])): old_to_new[str(record["path_id"])]
        for record in selected_records
        if str(record["path_id"]) in old_to_new
    }
    return candidates, pair_path_to_new


def export_saliency_specific_candidates(
    score_paths: list[str],
    candidate_paths_path: str,
    output_candidates_path: str,
    output_bindings_path: str,
    summary_path: str | None = None,
    modalities: set[str] | None = None,
    min_specificity_margin: float | None = None,
    require_positive_specificity: bool = True,
    require_full_patchable: bool = True,
    require_projector_for_vision_text: bool = True,
    top_k_per_pair: int | None = None,
    top_k_per_pair_modality: int | None = 2,
    top_k_total: int | None = None,
    path_id_prefix: str = "saliency_specific",
) -> dict[str, Any]:
    raw_scores = []
    for score_path in score_paths:
        raw_scores.extend(_load_jsonl(score_path))

    candidate_by_id = {path.path_id: path for path in _load_candidate_paths_jsonl(candidate_paths_path)}
    aggregated = list(_aggregate_score_records(raw_scores).values())
    filtered = _filter_records(
        aggregated,
        modalities=modalities,
        min_specificity_margin=min_specificity_margin,
        require_positive_specificity=require_positive_specificity,
        require_full_patchable=require_full_patchable,
        require_projector_for_vision_text=require_projector_for_vision_text,
    )
    selected = _select_top_records(
        filtered,
        top_k_per_pair=top_k_per_pair,
        top_k_per_pair_modality=top_k_per_pair_modality,
        top_k_total=top_k_total,
    )
    candidates, pair_path_to_new = _clone_selected_candidate_paths(
        candidate_by_id=candidate_by_id,
        selected_records=selected,
        path_id_prefix=path_id_prefix,
    )
    candidate_dicts = [candidate.to_dict() for candidate in candidates]
    _write_jsonl(output_candidates_path, candidate_dicts)

    binding_records = []
    for record in selected:
        new_path_id = pair_path_to_new.get((str(record["pair_id"]), str(record["path_id"])))
        if new_path_id is None:
            continue
        binding_records.append(
            {
                "pair_id": record["pair_id"],
                "path_id": new_path_id,
                "original_path_id": record["path_id"],
                "path_modality": record.get("path_modality"),
                "path_source": record.get("path_source"),
                "binding_strategy": "saliency_specificity_score",
                "specificity_score": record.get("specificity_score"),
                "saliency_specificity_margin": record.get("saliency_specificity_margin"),
                "saliency_specificity_ratio": record.get("saliency_specificity_ratio"),
                "forget_saliency": record.get("forget_saliency"),
                "retain_anchor_saliency": record.get("retain_anchor_saliency"),
            }
        )
    _write_jsonl(output_bindings_path, binding_records)

    summary = {
        "score_paths": score_paths,
        "candidate_paths_path": candidate_paths_path,
        "output_candidates_path": output_candidates_path,
        "output_bindings_path": output_bindings_path,
        "num_score_records": len(raw_scores),
        "num_pair_path_scores": len(aggregated),
        "num_filtered_pair_path_scores": len(filtered),
        "num_selected_bindings": len(binding_records),
        "num_selected_candidate_paths": len(candidates),
        "modalities": {
            modality: sum(1 for record in binding_records if record.get("path_modality") == modality)
            for modality in sorted({record.get("path_modality") for record in binding_records})
        },
        "filters": {
            "modalities": sorted(modalities) if modalities else None,
            "min_specificity_margin": min_specificity_margin,
            "require_positive_specificity": require_positive_specificity,
            "require_full_patchable": require_full_patchable,
            "require_projector_for_vision_text": require_projector_for_vision_text,
            "top_k_per_pair": top_k_per_pair,
            "top_k_per_pair_modality": top_k_per_pair_modality,
            "top_k_total": top_k_total,
        },
    }
    if summary_path is not None:
        Path(summary_path).parent.mkdir(parents=True, exist_ok=True)
        with Path(summary_path).open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    return summary


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export Step2 saliency-specific candidate paths from Step5 saliency scores.")
    parser.add_argument("--scores_path", nargs="+", required=True)
    parser.add_argument("--candidate_paths_path", required=True)
    parser.add_argument("--output_candidates", required=True)
    parser.add_argument("--output_bindings", required=True)
    parser.add_argument("--summary", default=None)
    parser.add_argument("--modalities", nargs="*", default=None)
    parser.add_argument("--min_specificity_margin", type=float, default=None)
    parser.add_argument("--allow_nonpositive_specificity", action="store_true", default=False)
    parser.add_argument("--allow_partial_patchable", action="store_true", default=False)
    parser.add_argument("--allow_unpatchable_vision_text_projector", action="store_true", default=False)
    parser.add_argument("--top_k_per_pair", type=int, default=None)
    parser.add_argument("--top_k_per_pair_modality", type=int, default=2)
    parser.add_argument("--top_k_total", type=int, default=None)
    parser.add_argument("--path_id_prefix", default="saliency_specific")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    summary = export_saliency_specific_candidates(
        score_paths=args.scores_path,
        candidate_paths_path=args.candidate_paths_path,
        output_candidates_path=args.output_candidates,
        output_bindings_path=args.output_bindings,
        summary_path=args.summary,
        modalities=set(args.modalities) if args.modalities else None,
        min_specificity_margin=args.min_specificity_margin,
        require_positive_specificity=not args.allow_nonpositive_specificity,
        require_full_patchable=not args.allow_partial_patchable,
        require_projector_for_vision_text=not args.allow_unpatchable_vision_text_projector,
        top_k_per_pair=args.top_k_per_pair,
        top_k_per_pair_modality=args.top_k_per_pair_modality,
        top_k_total=args.top_k_total,
        path_id_prefix=args.path_id_prefix,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
