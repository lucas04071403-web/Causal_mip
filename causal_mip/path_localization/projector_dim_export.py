from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from causal_mip.path_localization.path_schema import CandidatePath, PathNode


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


def _write_jsonl(path: str | Path, records: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _load_candidate_paths_jsonl(path: str) -> dict[str, CandidatePath]:
    return {record["path_id"]: CandidatePath.from_dict(record) for record in _load_jsonl(path)}


def _dim_scores_by_index(node_score: dict[str, Any]) -> dict[int, dict[str, Any]]:
    by_dim = {}
    for dim_score in node_score.get("dim_scores", []):
        try:
            by_dim[int(dim_score["dim_index"])] = dim_score
        except (KeyError, TypeError, ValueError):
            continue
    return by_dim


def _summarize_values(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "mean": None, "min": None, "max": None}
    return {
        "n": len(values),
        "mean": float(mean(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _mean_retain_dim_scores(retain_anchors: dict[str, Any], node_index: int) -> dict[int, dict[str, float]]:
    saliencies: dict[int, list[float]] = defaultdict(list)
    fishers: dict[int, list[float]] = defaultdict(list)
    for retain_result in retain_anchors.values():
        for node_score in retain_result.get("node_scores", []):
            if int(node_score.get("node_index", -1)) != node_index:
                continue
            for dim_index, dim_score in _dim_scores_by_index(node_score).items():
                saliency = _as_float(dim_score.get("saliency"))
                fisher = _as_float(dim_score.get("fisher_saliency"))
                if saliency is not None:
                    saliencies[dim_index].append(saliency)
                if fisher is not None:
                    fishers[dim_index].append(fisher)
    return {
        dim_index: {
            "retain_anchor_saliency": float(mean(values)) if values else 0.0,
            "retain_anchor_fisher_saliency": float(mean(fishers.get(dim_index, [])))
            if fishers.get(dim_index)
            else 0.0,
        }
        for dim_index, values in saliencies.items()
    }


def _retain_dim_anchor_scores(
    retain_anchors: dict[str, Any],
    node_index: int,
) -> dict[int, dict[str, dict[str, float]]]:
    anchor_scores: dict[int, dict[str, dict[str, float]]] = defaultdict(dict)
    for anchor_name, retain_result in retain_anchors.items():
        for node_score in retain_result.get("node_scores", []):
            try:
                if int(node_score.get("node_index", -1)) != node_index:
                    continue
            except (TypeError, ValueError):
                continue
            for dim_index, dim_score in _dim_scores_by_index(node_score).items():
                saliency = _as_float(dim_score.get("saliency"))
                fisher = _as_float(dim_score.get("fisher_saliency"))
                anchor_scores[dim_index][str(anchor_name)] = {
                    "saliency": float(saliency) if saliency is not None else 0.0,
                    "fisher_saliency": float(fisher) if fisher is not None else 0.0,
                }
    return anchor_scores


def score_projector_dims(record: dict[str, Any], gamma: float = 1.0, eps: float = 1e-6) -> list[dict[str, Any]]:
    saliency = record.get("saliency_specificity") or {}
    forget = saliency.get("forget") or {}
    retain_anchors = saliency.get("retain_anchors") or {}
    dim_records = []
    for node_score in forget.get("node_scores", []):
        if node_score.get("module_kind") != "projector":
            continue
        try:
            node_index = int(node_score["node_index"])
        except (KeyError, TypeError, ValueError):
            continue
        retain_scores = _mean_retain_dim_scores(retain_anchors, node_index)
        retain_anchor_scores = _retain_dim_anchor_scores(retain_anchors, node_index)
        for dim_index, forget_dim_score in _dim_scores_by_index(node_score).items():
            forget_saliency = _as_float(forget_dim_score.get("saliency"))
            forget_fisher = _as_float(forget_dim_score.get("fisher_saliency"))
            if forget_saliency is None:
                continue
            retain_score = retain_scores.get(dim_index, {})
            retain_saliency = float(retain_score.get("retain_anchor_saliency", 0.0))
            retain_fisher = float(retain_score.get("retain_anchor_fisher_saliency", 0.0))
            per_anchor = retain_anchor_scores.get(dim_index, {})
            per_anchor_saliency = {
                anchor_name: float(anchor_score.get("saliency", 0.0))
                for anchor_name, anchor_score in per_anchor.items()
            }
            per_anchor_margin = {
                anchor_name: forget_saliency - gamma * anchor_saliency
                for anchor_name, anchor_saliency in per_anchor_saliency.items()
            }
            per_anchor_ratio = {
                anchor_name: forget_saliency / (anchor_saliency + eps)
                for anchor_name, anchor_saliency in per_anchor_saliency.items()
            }
            max_anchor_retain_saliency = max(per_anchor_saliency.values()) if per_anchor_saliency else retain_saliency
            min_anchor_margin = min(per_anchor_margin.values()) if per_anchor_margin else forget_saliency - gamma * retain_saliency
            min_anchor_ratio = min(per_anchor_ratio.values()) if per_anchor_ratio else forget_saliency / (retain_saliency + eps)
            dim_records.append(
                {
                    "node_index": node_index,
                    "module": node_score.get("module"),
                    "layer": node_score.get("layer"),
                    "token_selector": node_score.get("token_selector"),
                    "module_kind": "projector",
                    "dim_index": dim_index,
                    "forget_saliency": forget_saliency,
                    "retain_anchor_saliency": retain_saliency,
                    "saliency_specificity_margin": forget_saliency - gamma * retain_saliency,
                    "saliency_specificity_ratio": forget_saliency / (retain_saliency + eps),
                    "max_anchor_retain_saliency": max_anchor_retain_saliency,
                    "min_anchor_margin": min_anchor_margin,
                    "min_anchor_ratio": min_anchor_ratio,
                    "retain_anchor_saliency_summary": _summarize_values(list(per_anchor_saliency.values())),
                    "retain_anchor_margins": per_anchor_margin,
                    "retain_anchor_ratios": per_anchor_ratio,
                    "forget_fisher_saliency": float(forget_fisher) if forget_fisher is not None else None,
                    "retain_anchor_fisher_saliency": retain_fisher,
                    "fisher_specificity_margin": (float(forget_fisher) if forget_fisher is not None else 0.0)
                    - gamma * retain_fisher,
                }
            )
    return dim_records


def _rank_key(record: dict[str, Any]) -> tuple[float, ...]:
    return (
        float(record.get("min_anchor_margin") or 0.0),
        float(record.get("min_anchor_ratio") or 0.0),
        float(record.get("saliency_specificity_margin") or 0.0),
        float(record.get("saliency_specificity_ratio") or 0.0),
        float(record.get("forget_saliency") or 0.0),
    )


def _select_dims(
    dim_records: list[dict[str, Any]],
    top_k_dims: int,
    min_specificity_margin: float | None,
    min_anchor_margin: float | None = None,
    min_anchor_ratio: float | None = None,
    max_anchor_retain_saliency: float | None = None,
) -> list[dict[str, Any]]:
    filtered = []
    for dim_record in dim_records:
        margin = _as_float(dim_record.get("saliency_specificity_margin"))
        if margin is None:
            continue
        if min_specificity_margin is None:
            if margin <= 0.0:
                continue
        elif margin < min_specificity_margin:
            continue
        worst_margin = _as_float(dim_record.get("min_anchor_margin"))
        if min_anchor_margin is not None and (worst_margin is None or worst_margin < min_anchor_margin):
            continue
        worst_ratio = _as_float(dim_record.get("min_anchor_ratio"))
        if min_anchor_ratio is not None and (worst_ratio is None or worst_ratio < min_anchor_ratio):
            continue
        anchor_retain = _as_float(dim_record.get("max_anchor_retain_saliency"))
        if max_anchor_retain_saliency is not None and (
            anchor_retain is None or anchor_retain > max_anchor_retain_saliency
        ):
            continue
        filtered.append(dim_record)
    return sorted(filtered, key=_rank_key, reverse=True)[:top_k_dims]


def _selected_dim_signature(item: dict[str, Any]) -> tuple[str, tuple[int, ...]]:
    record = item["record"]
    dims = tuple(sorted(int(dim["dim_index"]) for dim in item["selected_dims"]))
    return str(record.get("pair_id")), dims


def _dedupe_sorted_items(
    items: list[dict[str, Any]],
    limit: int | None = None,
) -> list[dict[str, Any]]:
    deduped = []
    seen = set()
    for item in items:
        signature = _selected_dim_signature(item)
        if signature in seen:
            continue
        seen.add(signature)
        deduped.append(item)
        if limit is not None and len(deduped) >= limit:
            break
    return deduped


def _clone_dim_path(
    original: CandidatePath,
    selected_dims: list[dict[str, Any]],
    new_path_id: str,
    pair_id: str,
) -> CandidatePath:
    nodes = []
    for dim_record in selected_dims:
        original_node = original.nodes[int(dim_record["node_index"])]
        nodes.append(
            PathNode(
                module=original_node.module,
                layer=original_node.layer,
                neuron=int(dim_record["dim_index"]),
                token_selector=original_node.token_selector,
            )
        )
    candidate = CandidatePath(
        path_id=new_path_id,
        source=f"{original.source}_projector_dim_specific",
        modality=original.modality,
        mip_score=float(mean([float(dim["saliency_specificity_margin"]) for dim in selected_dims])),
        nodes=nodes,
        source_sample_idx=original.source_sample_idx,
        metadata={
            **(original.metadata or {}),
            "original_path_id": original.path_id,
            "candidate_generation": "projector_dim_specificity",
            "projector_dim_level": True,
            "projector_dim_pair_id": pair_id,
            "selected_projector_dims": selected_dims,
        },
    )
    return candidate


def export_projector_dim_candidates(
    score_paths: list[str],
    candidate_paths_path: str,
    output_candidates_path: str,
    output_bindings_path: str,
    summary_path: str | None = None,
    top_k_dims: int = 8,
    top_k_per_pair: int | None = 2,
    min_specificity_margin: float | None = None,
    require_suf_positive: bool = True,
    require_high_ret: bool = False,
    retain_threshold: float = 0.005208333333333333,
    path_id_prefix: str = "projector_dim",
    gamma: float = 1.0,
    eps: float = 1e-6,
    min_anchor_margin: float | None = None,
    min_anchor_ratio: float | None = None,
    max_anchor_retain_saliency: float | None = None,
    dedupe_selected: bool = False,
) -> dict[str, Any]:
    raw_scores = []
    for score_path in score_paths:
        raw_scores.extend(_load_jsonl(score_path))
    candidate_by_id = _load_candidate_paths_jsonl(candidate_paths_path)

    scored = []
    for record in raw_scores:
        if record.get("status") != "ok" or not bool(record.get("contains_projector")):
            continue
        if require_suf_positive and (_as_float(record.get("Suf")) or 0.0) <= 0.0:
            continue
        if require_high_ret and (_as_float(record.get("Ret")) or 0.0) <= retain_threshold:
            continue
        original = candidate_by_id.get(str(record.get("path_id")))
        if original is None:
            continue
        dim_records = score_projector_dims(record, gamma=gamma, eps=eps)
        selected_dims = _select_dims(
            dim_records,
            top_k_dims=top_k_dims,
            min_specificity_margin=min_specificity_margin,
            min_anchor_margin=min_anchor_margin,
            min_anchor_ratio=min_anchor_ratio,
            max_anchor_retain_saliency=max_anchor_retain_saliency,
        )
        if not selected_dims:
            continue
        scored.append(
            {
                "record": record,
                "selected_dims": selected_dims,
                "best_margin": float(max(float(dim["saliency_specificity_margin"]) for dim in selected_dims)),
                "mean_margin": float(mean([float(dim["saliency_specificity_margin"]) for dim in selected_dims])),
                "best_min_anchor_margin": float(max(float(dim["min_anchor_margin"]) for dim in selected_dims)),
                "mean_min_anchor_margin": float(mean([float(dim["min_anchor_margin"]) for dim in selected_dims])),
            }
        )

    if top_k_per_pair is None:
        ordered = sorted(
            scored,
            key=lambda item: (
                item["best_min_anchor_margin"],
                item["mean_min_anchor_margin"],
                item["best_margin"],
                item["mean_margin"],
            ),
            reverse=True,
        )
        selected = _dedupe_sorted_items(ordered) if dedupe_selected else ordered
    else:
        selected = []
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in scored:
            grouped[str(item["record"].get("pair_id"))].append(item)
        for group in grouped.values():
            ordered = sorted(
                group,
                key=lambda item: (
                    item["best_min_anchor_margin"],
                    item["mean_min_anchor_margin"],
                    item["best_margin"],
                    item["mean_margin"],
                ),
                reverse=True,
            )
            if dedupe_selected:
                selected.extend(_dedupe_sorted_items(ordered, limit=top_k_per_pair))
            else:
                selected.extend(ordered[:top_k_per_pair])

    candidates = []
    bindings = []
    for index, item in enumerate(selected):
        record = item["record"]
        old_path_id = str(record["path_id"])
        pair_id = str(record["pair_id"])
        new_path_id = f"{path_id_prefix}_p{index:06d}"
        candidate = _clone_dim_path(
            original=candidate_by_id[old_path_id],
            selected_dims=item["selected_dims"],
            new_path_id=new_path_id,
            pair_id=pair_id,
        )
        candidates.append(candidate.to_dict())
        bindings.append(
            {
                "pair_id": pair_id,
                "path_id": new_path_id,
                "original_path_id": old_path_id,
                "path_modality": record.get("path_modality"),
                "path_source": record.get("path_source"),
                "binding_strategy": "projector_dim_specificity_score",
                "num_selected_dims": len(item["selected_dims"]),
                "selected_dim_indices": [int(dim["dim_index"]) for dim in item["selected_dims"]],
                "best_dim_specificity_margin": item["best_margin"],
                "mean_dim_specificity_margin": item["mean_margin"],
                "best_dim_min_anchor_margin": item["best_min_anchor_margin"],
                "mean_dim_min_anchor_margin": item["mean_min_anchor_margin"],
            }
        )

    _write_jsonl(output_candidates_path, candidates)
    _write_jsonl(output_bindings_path, bindings)
    summary = {
        "score_paths": score_paths,
        "candidate_paths_path": candidate_paths_path,
        "output_candidates_path": output_candidates_path,
        "output_bindings_path": output_bindings_path,
        "num_score_records": len(raw_scores),
        "num_scored_projector_records": len(scored),
        "num_selected_bindings": len(bindings),
        "num_selected_candidate_paths": len(candidates),
        "filters": {
            "top_k_dims": top_k_dims,
            "top_k_per_pair": top_k_per_pair,
            "min_specificity_margin": min_specificity_margin,
            "require_suf_positive": require_suf_positive,
            "require_high_ret": require_high_ret,
            "retain_threshold": retain_threshold,
            "path_id_prefix": path_id_prefix,
            "gamma": gamma,
            "eps": eps,
            "min_anchor_margin": min_anchor_margin,
            "min_anchor_ratio": min_anchor_ratio,
            "max_anchor_retain_saliency": max_anchor_retain_saliency,
            "dedupe_selected": dedupe_selected,
        },
    }
    if summary_path:
        summary_output = Path(summary_path)
        summary_output.parent.mkdir(parents=True, exist_ok=True)
        with summary_output.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    return summary


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export projector dim-specific candidates from Step5 dim saliency scores.")
    parser.add_argument("--scores_path", nargs="+", required=True)
    parser.add_argument("--candidate_paths_path", required=True)
    parser.add_argument("--output_candidates", required=True)
    parser.add_argument("--output_bindings", required=True)
    parser.add_argument("--summary", default=None)
    parser.add_argument("--top_k_dims", type=int, default=8)
    parser.add_argument("--top_k_per_pair", type=int, default=2)
    parser.add_argument("--no_top_k_per_pair", action="store_true", default=False)
    parser.add_argument("--min_specificity_margin", type=float, default=None)
    parser.add_argument("--allow_nonpositive_suf", action="store_true", default=False)
    parser.add_argument("--require_high_ret", action="store_true", default=False)
    parser.add_argument("--retain_threshold", type=float, default=0.005208333333333333)
    parser.add_argument("--path_id_prefix", default="projector_dim")
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--min_anchor_margin", type=float, default=None)
    parser.add_argument("--min_anchor_ratio", type=float, default=None)
    parser.add_argument("--max_anchor_retain_saliency", type=float, default=None)
    parser.add_argument("--dedupe_selected", action="store_true", default=False)
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    summary = export_projector_dim_candidates(
        score_paths=args.scores_path,
        candidate_paths_path=args.candidate_paths_path,
        output_candidates_path=args.output_candidates,
        output_bindings_path=args.output_bindings,
        summary_path=args.summary,
        top_k_dims=args.top_k_dims,
        top_k_per_pair=None if args.no_top_k_per_pair else args.top_k_per_pair,
        min_specificity_margin=args.min_specificity_margin,
        require_suf_positive=not args.allow_nonpositive_suf,
        require_high_ret=args.require_high_ret,
        retain_threshold=args.retain_threshold,
        path_id_prefix=args.path_id_prefix,
        gamma=args.gamma,
        eps=args.eps,
        min_anchor_margin=args.min_anchor_margin,
        min_anchor_ratio=args.min_anchor_ratio,
        max_anchor_retain_saliency=args.max_anchor_retain_saliency,
        dedupe_selected=args.dedupe_selected,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
