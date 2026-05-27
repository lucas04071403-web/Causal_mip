from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from causal_mip.causal_scores.classify_paths import aggregate_path_score_records, load_score_records_jsonl


def _rank_paths(paths: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    return sorted(paths, key=lambda item: (float(item[key]), float(item["forget_effect"])), reverse=True)


def _with_specificity_scores(paths: list[dict[str, Any]], gamma: float, eps: float) -> list[dict[str, Any]]:
    scored = []
    for path in paths:
        forget_effect = float(path["forget_effect"])
        retain_impact = float(path["retain_impact"])
        record = dict(path)
        record["specificity_ratio"] = forget_effect / (retain_impact + eps)
        record["specificity_margin"] = forget_effect - gamma * retain_impact
        record["specificity_gamma"] = gamma
        record["specificity_eps"] = eps
        scored.append(record)
    return scored


def _attach_ranks(paths: list[dict[str, Any]], keys: list[str]) -> None:
    for key in keys:
        for rank, path in enumerate(_rank_paths(paths, key), start=1):
            path[f"rank_by_{key}"] = rank


def _summarize_modality(paths: list[dict[str, Any]], modality: str) -> dict[str, Any]:
    items = [path for path in paths if path.get("path_modality") == modality]
    if not items:
        return {
            "num_paths": 0,
            "path_ids": [],
        }
    return {
        "num_paths": len(items),
        "path_ids": [path["path_id"] for path in items],
        "rank_by_forget_effect": {path["path_id"]: path["rank_by_forget_effect"] for path in items},
        "rank_by_retain_impact": {path["path_id"]: path["rank_by_retain_impact"] for path in items},
        "rank_by_specificity_ratio": {path["path_id"]: path["rank_by_specificity_ratio"] for path in items},
        "rank_by_specificity_margin": {path["path_id"]: path["rank_by_specificity_margin"] for path in items},
        "forget_effect": {path["path_id"]: path["forget_effect"] for path in items},
        "retain_impact": {path["path_id"]: path["retain_impact"] for path in items},
        "specificity_ratio": {path["path_id"]: path["specificity_ratio"] for path in items},
        "specificity_margin": {path["path_id"]: path["specificity_margin"] for path in items},
    }


def _top_paths(paths: list[dict[str, Any]], key: str, top_k: int) -> list[dict[str, Any]]:
    fields = [
        "path_id",
        "path_modality",
        "forget_effect",
        "retain_impact",
        "specificity_ratio",
        "specificity_margin",
        "rank_by_forget_effect",
        "rank_by_retain_impact",
        "rank_by_specificity_ratio",
        "rank_by_specificity_margin",
    ]
    return [{field: path.get(field) for field in fields} for path in _rank_paths(paths, key)[:top_k]]


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def build_probe_records(
    paths: list[dict[str, Any]],
    modality: str,
    max_retain_impact: float | None,
    min_forget_effect: float,
    top_k: int | None,
    ranking_key: str,
) -> list[dict[str, Any]]:
    candidates = [
        path
        for path in paths
        if path.get("path_modality") == modality
        and float(path["forget_effect"]) >= min_forget_effect
        and (max_retain_impact is None or float(path["retain_impact"]) <= max_retain_impact)
    ]
    candidates = _rank_paths(candidates, ranking_key)
    if top_k is not None:
        candidates = candidates[:top_k]
    return [
        {
            **path,
            "category": "P_projector_probe",
            "probe_reason": (
                "high_forget_effect_with_nonzero_retain_impact; use weak editing with retain anchors, not hard P_forget"
            ),
            "probe_ranking_key": ranking_key,
        }
        for path in candidates
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank Step 6 paths by forget/retain specificity and optionally export P_projector_probe.")
    parser.add_argument("--scores_path", nargs="+", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--aggregation", choices=["mean", "median"], default="mean")
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--probe_modality", default="vision_text")
    parser.add_argument("--probe_top_k", type=int, default=None)
    parser.add_argument("--probe_min_forget_effect", type=float, default=0.0)
    parser.add_argument("--probe_max_retain_impact", type=float, default=None)
    parser.add_argument(
        "--probe_ranking_key",
        choices=["forget_effect", "specificity_margin", "specificity_ratio"],
        default="specificity_margin",
    )
    parser.add_argument("--no_probe_output", action="store_true", default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_score_records_jsonl(args.scores_path)
    aggregated = aggregate_path_score_records(
        records=records,
        alpha=args.alpha,
        aggregation=args.aggregation,
        clip_negative_effects=True,
        include_non_ok=False,
    )
    paths = _with_specificity_scores(aggregated, gamma=args.gamma, eps=args.eps)
    _attach_ranks(paths, ["forget_effect", "retain_impact", "specificity_ratio", "specificity_margin"])

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ranked_path = output_dir / "P_specificity_ranked.jsonl"
    _write_jsonl(ranked_path, sorted(paths, key=lambda item: item["rank_by_specificity_margin"]))

    probe_records: list[dict[str, Any]] = []
    probe_path = output_dir / "P_projector_probe.jsonl"
    if not args.no_probe_output:
        probe_records = build_probe_records(
            paths=paths,
            modality=args.probe_modality,
            max_retain_impact=args.probe_max_retain_impact,
            min_forget_effect=args.probe_min_forget_effect,
            top_k=args.probe_top_k,
            ranking_key=args.probe_ranking_key,
        )
        _write_jsonl(probe_path, probe_records)

    summary = {
        "num_input_score_records": len(records),
        "num_aggregated_paths": len(paths),
        "modalities": dict(sorted(Counter(path.get("path_modality") for path in paths).items())),
        "alpha": args.alpha,
        "aggregation": args.aggregation,
        "gamma": args.gamma,
        "eps": args.eps,
        "top_by_forget_effect": _top_paths(paths, "forget_effect", args.top_k),
        "top_by_specificity_margin": _top_paths(paths, "specificity_margin", args.top_k),
        "top_by_specificity_ratio": _top_paths(paths, "specificity_ratio", args.top_k),
        "probe_modality_summary": _summarize_modality(paths, args.probe_modality),
        "probe_output": None
        if args.no_probe_output
        else {
            "path": str(probe_path),
            "num_records": len(probe_records),
            "ranking_key": args.probe_ranking_key,
            "min_forget_effect": args.probe_min_forget_effect,
            "max_retain_impact": args.probe_max_retain_impact,
            "top_k": args.probe_top_k,
        },
        "ranked_output": str(ranked_path),
    }
    summary_path = output_dir / "specificity_summary.json"
    _write_json(summary_path, summary)

    print(json.dumps({"summary": summary, "outputs": {"ranked": str(ranked_path), "summary": str(summary_path), "probe": str(probe_path) if not args.no_probe_output else None}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
