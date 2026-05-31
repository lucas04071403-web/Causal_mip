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


def _write_jsonl(path: str | Path, records: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _load_candidate_paths_jsonl(path: str) -> dict[str, CandidatePath]:
    return {record["path_id"]: CandidatePath.from_dict(record) for record in _load_jsonl(path)}


def _node_key(node_score: dict[str, Any]) -> int | None:
    try:
        return int(node_score["node_index"])
    except (KeyError, TypeError, ValueError):
        return None


def _index_node_scores(node_scores: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    indexed = {}
    for node_score in node_scores:
        node_index = _node_key(node_score)
        if node_index is not None:
            indexed[node_index] = node_score
    return indexed


def _summarize_values(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "mean": None, "min": None, "max": None}
    return {
        "n": len(values),
        "mean": float(mean(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _mean_retain_node_scores(retain_anchors: dict[str, Any]) -> dict[int, dict[str, float]]:
    saliencies: dict[int, list[float]] = defaultdict(list)
    fishers: dict[int, list[float]] = defaultdict(list)
    for retain_result in retain_anchors.values():
        for node_score in retain_result.get("node_scores", []):
            node_index = _node_key(node_score)
            if node_index is None or node_score.get("status") != "ok":
                continue
            saliency = _as_float(node_score.get("saliency"))
            fisher = _as_float(node_score.get("fisher_saliency"))
            if saliency is not None:
                saliencies[node_index].append(saliency)
            if fisher is not None:
                fishers[node_index].append(fisher)
    return {
        node_index: {
            "retain_anchor_saliency": float(mean(values)) if values else 0.0,
            "retain_anchor_fisher_saliency": float(mean(fishers.get(node_index, [])))
            if fishers.get(node_index)
            else 0.0,
        }
        for node_index, values in saliencies.items()
    }


def _retain_node_anchor_scores(retain_anchors: dict[str, Any]) -> dict[int, dict[str, dict[str, float]]]:
    anchor_scores: dict[int, dict[str, dict[str, float]]] = defaultdict(dict)
    for anchor_name, retain_result in retain_anchors.items():
        for node_score in retain_result.get("node_scores", []):
            node_index = _node_key(node_score)
            if node_index is None or node_score.get("status") != "ok":
                continue
            saliency = _as_float(node_score.get("saliency"))
            fisher = _as_float(node_score.get("fisher_saliency"))
            anchor_scores[node_index][str(anchor_name)] = {
                "saliency": float(saliency) if saliency is not None else 0.0,
                "fisher_saliency": float(fisher) if fisher is not None else 0.0,
            }
    return anchor_scores


def score_record_nodes(record: dict[str, Any], gamma: float = 1.0, eps: float = 1e-6) -> list[dict[str, Any]]:
    saliency = record.get("saliency_specificity") or {}
    forget = saliency.get("forget") or {}
    forget_scores = _index_node_scores(forget.get("node_scores", []))
    retain_scores = _mean_retain_node_scores(saliency.get("retain_anchors") or {})
    retain_anchor_scores = _retain_node_anchor_scores(saliency.get("retain_anchors") or {})
    node_records = []
    for node_index, forget_score in sorted(forget_scores.items()):
        if forget_score.get("status") != "ok":
            continue
        forget_saliency = _as_float(forget_score.get("saliency"))
        forget_fisher = _as_float(forget_score.get("fisher_saliency"))
        if forget_saliency is None:
            continue
        retain_score = retain_scores.get(node_index, {})
        retain_saliency = float(retain_score.get("retain_anchor_saliency", 0.0))
        retain_fisher = float(retain_score.get("retain_anchor_fisher_saliency", 0.0))
        per_anchor = retain_anchor_scores.get(node_index, {})
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
        node_records.append(
            {
                "node_index": node_index,
                "module": forget_score.get("module"),
                "layer": forget_score.get("layer"),
                "neuron": forget_score.get("neuron"),
                "token_selector": forget_score.get("token_selector"),
                "module_kind": forget_score.get("module_kind"),
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
                "num_token_positions": forget_score.get("num_token_positions"),
            }
        )
    return node_records


def _rank_key(record: dict[str, Any]) -> tuple[float, ...]:
    return (
        float(record.get("min_anchor_margin") or 0.0),
        float(record.get("min_anchor_ratio") or 0.0),
        float(record.get("saliency_specificity_margin") or 0.0),
        float(record.get("saliency_specificity_ratio") or 0.0),
        float(record.get("forget_saliency") or 0.0),
    )


def _select_nodes(
    node_records: list[dict[str, Any]],
    top_k_nodes: int,
    min_specificity_margin: float | None,
    allow_projector_nodes: bool,
    min_anchor_margin: float | None = None,
    min_anchor_ratio: float | None = None,
    max_anchor_retain_saliency: float | None = None,
) -> list[dict[str, Any]]:
    filtered = []
    for node_record in node_records:
        if node_record.get("module_kind") == "projector" and not allow_projector_nodes:
            continue
        margin = _as_float(node_record.get("saliency_specificity_margin"))
        if margin is None:
            continue
        if min_specificity_margin is not None and margin < min_specificity_margin:
            continue
        if min_specificity_margin is None and margin <= 0.0:
            continue
        worst_margin = _as_float(node_record.get("min_anchor_margin"))
        if min_anchor_margin is not None and (worst_margin is None or worst_margin < min_anchor_margin):
            continue
        worst_ratio = _as_float(node_record.get("min_anchor_ratio"))
        if min_anchor_ratio is not None and (worst_ratio is None or worst_ratio < min_anchor_ratio):
            continue
        anchor_retain = _as_float(node_record.get("max_anchor_retain_saliency"))
        if max_anchor_retain_saliency is not None and (
            anchor_retain is None or anchor_retain > max_anchor_retain_saliency
        ):
            continue
        filtered.append(node_record)
    return sorted(filtered, key=_rank_key, reverse=True)[:top_k_nodes]


def _selected_node_signature(item: dict[str, Any]) -> tuple[str, str, tuple[tuple[str, str, str, str], ...]]:
    record = item["record"]
    nodes = tuple(
        sorted(
            (
                str(node.get("module")),
                str(node.get("layer")),
                str(node.get("neuron")),
                str(node.get("token_selector")),
            )
            for node in item["selected_nodes"]
        )
    )
    return str(record.get("pair_id")), str(record.get("path_modality")), nodes


def _dedupe_sorted_items(
    items: list[dict[str, Any]],
    limit: int | None = None,
) -> list[dict[str, Any]]:
    deduped = []
    seen = set()
    for item in items:
        signature = _selected_node_signature(item)
        if signature in seen:
            continue
        seen.add(signature)
        deduped.append(item)
        if limit is not None and len(deduped) >= limit:
            break
    return deduped


def _clone_compact_path(
    original: CandidatePath,
    selected_nodes: list[dict[str, Any]],
    new_path_id: str,
    pair_id: str,
) -> CandidatePath:
    selected_indices = sorted(int(node["node_index"]) for node in selected_nodes)
    candidate = CandidatePath.from_dict(original.to_dict())
    candidate.path_id = new_path_id
    candidate.source = f"{original.source}_node_specific"
    candidate.nodes = [original.nodes[index] for index in selected_indices]
    candidate.mip_score = float(mean([float(node["saliency_specificity_margin"]) for node in selected_nodes]))
    candidate.metadata = {
        **(candidate.metadata or {}),
        "original_path_id": original.path_id,
        "candidate_generation": "node_specificity",
        "node_specific_pair_id": pair_id,
        "selected_original_node_indices": selected_indices,
        "selected_node_scores": selected_nodes,
    }
    return candidate


def export_node_specific_candidates(
    score_paths: list[str],
    candidate_paths_path: str,
    output_candidates_path: str,
    output_bindings_path: str,
    summary_path: str | None = None,
    modalities: set[str] | None = None,
    top_k_nodes: int = 4,
    top_k_per_pair_modality: int | None = 2,
    min_specificity_margin: float | None = None,
    require_suf_positive: bool = True,
    allow_projector_nodes: bool = False,
    path_id_prefix: str = "node_specific",
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

    scored_candidates = []
    for record in raw_scores:
        if record.get("status") != "ok":
            continue
        if modalities and record.get("path_modality") not in modalities:
            continue
        if require_suf_positive and (_as_float(record.get("Suf")) or 0.0) <= 0.0:
            continue
        original = candidate_by_id.get(str(record.get("path_id")))
        if original is None:
            continue
        node_records = score_record_nodes(record, gamma=gamma, eps=eps)
        selected_nodes = _select_nodes(
            node_records,
            top_k_nodes=top_k_nodes,
            min_specificity_margin=min_specificity_margin,
            allow_projector_nodes=allow_projector_nodes,
            min_anchor_margin=min_anchor_margin,
            min_anchor_ratio=min_anchor_ratio,
            max_anchor_retain_saliency=max_anchor_retain_saliency,
        )
        if not selected_nodes:
            continue
        best_margin = float(max(float(node["saliency_specificity_margin"]) for node in selected_nodes))
        best_min_anchor_margin = float(max(float(node["min_anchor_margin"]) for node in selected_nodes))
        mean_min_anchor_margin = float(mean([float(node["min_anchor_margin"]) for node in selected_nodes]))
        scored_candidates.append(
            {
                "record": record,
                "selected_nodes": selected_nodes,
                "best_margin": best_margin,
                "mean_margin": float(mean([float(node["saliency_specificity_margin"]) for node in selected_nodes])),
                "best_min_anchor_margin": best_min_anchor_margin,
                "mean_min_anchor_margin": mean_min_anchor_margin,
            }
        )

    selected = []
    if top_k_per_pair_modality is None:
        ordered = sorted(
            scored_candidates,
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
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for item in scored_candidates:
            record = item["record"]
            grouped[(str(record.get("pair_id")), str(record.get("path_modality")))].append(item)
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
                selected.extend(_dedupe_sorted_items(ordered, limit=top_k_per_pair_modality))
            else:
                selected.extend(ordered[:top_k_per_pair_modality])

    candidates = []
    bindings = []
    for index, item in enumerate(selected):
        record = item["record"]
        old_path_id = str(record["path_id"])
        pair_id = str(record["pair_id"])
        original = candidate_by_id[old_path_id]
        new_path_id = f"{path_id_prefix}_p{index:06d}"
        compact = _clone_compact_path(
            original=original,
            selected_nodes=item["selected_nodes"],
            new_path_id=new_path_id,
            pair_id=pair_id,
        )
        candidates.append(compact.to_dict())
        bindings.append(
            {
                "pair_id": pair_id,
                "path_id": new_path_id,
                "original_path_id": old_path_id,
                "path_modality": record.get("path_modality"),
                "path_source": record.get("path_source"),
                "binding_strategy": "node_specificity_score",
                "num_selected_nodes": len(item["selected_nodes"]),
                "selected_original_node_indices": compact.metadata["selected_original_node_indices"],
                "best_node_specificity_margin": item["best_margin"],
                "mean_node_specificity_margin": item["mean_margin"],
                "best_node_min_anchor_margin": item["best_min_anchor_margin"],
                "mean_node_min_anchor_margin": item["mean_min_anchor_margin"],
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
        "num_scored_candidates": len(scored_candidates),
        "num_selected_bindings": len(bindings),
        "num_selected_candidate_paths": len(candidates),
        "modalities": {
            modality: sum(1 for record in bindings if record.get("path_modality") == modality)
            for modality in sorted({record.get("path_modality") for record in bindings})
        },
        "filters": {
            "modalities": sorted(modalities) if modalities else None,
            "top_k_nodes": top_k_nodes,
            "top_k_per_pair_modality": top_k_per_pair_modality,
            "min_specificity_margin": min_specificity_margin,
            "require_suf_positive": require_suf_positive,
            "allow_projector_nodes": allow_projector_nodes,
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
    parser = argparse.ArgumentParser(description="Export compact node-specific candidate paths from Step5 saliency scores.")
    parser.add_argument("--scores_path", nargs="+", required=True)
    parser.add_argument("--candidate_paths_path", required=True)
    parser.add_argument("--output_candidates", required=True)
    parser.add_argument("--output_bindings", required=True)
    parser.add_argument("--summary", default=None)
    parser.add_argument("--modalities", nargs="*", default=None)
    parser.add_argument("--top_k_nodes", type=int, default=4)
    parser.add_argument("--top_k_per_pair_modality", type=int, default=2)
    parser.add_argument("--no_top_k_per_pair_modality", action="store_true", default=False)
    parser.add_argument("--min_specificity_margin", type=float, default=None)
    parser.add_argument("--allow_nonpositive_suf", action="store_true", default=False)
    parser.add_argument("--allow_projector_nodes", action="store_true", default=False)
    parser.add_argument("--path_id_prefix", default="node_specific")
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--min_anchor_margin", type=float, default=None)
    parser.add_argument("--min_anchor_ratio", type=float, default=None)
    parser.add_argument("--max_anchor_retain_saliency", type=float, default=None)
    parser.add_argument("--dedupe_selected", action="store_true", default=False)
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    summary = export_node_specific_candidates(
        score_paths=args.scores_path,
        candidate_paths_path=args.candidate_paths_path,
        output_candidates_path=args.output_candidates,
        output_bindings_path=args.output_bindings,
        summary_path=args.summary,
        modalities=set(args.modalities) if args.modalities else None,
        top_k_nodes=args.top_k_nodes,
        top_k_per_pair_modality=None if args.no_top_k_per_pair_modality else args.top_k_per_pair_modality,
        min_specificity_margin=args.min_specificity_margin,
        require_suf_positive=not args.allow_nonpositive_suf,
        allow_projector_nodes=args.allow_projector_nodes,
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
