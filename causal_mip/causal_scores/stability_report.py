from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any


DEFAULT_FORGET_THRESHOLD = 0.015625
DEFAULT_RETAIN_THRESHOLD = 0.005208333333333333


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


def _summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "max": None,
            "ci95_low": None,
            "ci95_high": None,
        }
    avg = float(mean(values))
    if len(values) > 1:
        std = float(stdev(values))
        radius = 1.96 * std / math.sqrt(len(values))
    else:
        std = 0.0
        radius = 0.0
    return {
        "n": len(values),
        "mean": avg,
        "median": float(median(values)),
        "std": std,
        "min": float(min(values)),
        "max": float(max(values)),
        "ci95_low": avg - radius,
        "ci95_high": avg + radius,
    }


def _record_passes(
    record: dict[str, Any],
    forget_threshold: float,
    retain_threshold: float,
    min_saliency_specificity: float,
    min_suf: float,
) -> bool:
    suf = _as_float(record.get("Suf"))
    nec = _as_float(record.get("Nec"))
    ret = _as_float(record.get("Ret"))
    margin = _as_float(record.get("saliency_specificity_margin"))
    num_nodes = int(record.get("num_nodes") or 0)
    num_patchable_nodes = int(record.get("num_patchable_nodes") or 0)
    forget_effect = max(0.0, nec or 0.0) + max(0.0, suf or 0.0)
    return (
        record.get("status") == "ok"
        and suf is not None
        and suf > min_suf
        and forget_effect > forget_threshold
        and ret is not None
        and ret <= retain_threshold
        and margin is not None
        and margin > min_saliency_specificity
        and num_nodes > 0
        and num_patchable_nodes == num_nodes
    )


def build_stability_report(
    score_paths: list[str],
    focus_path_ids: set[str] | None = None,
    focus_pair_ids: set[str] | None = None,
    forget_threshold: float = DEFAULT_FORGET_THRESHOLD,
    retain_threshold: float = DEFAULT_RETAIN_THRESHOLD,
    min_saliency_specificity: float = 0.0,
    min_suf: float = 0.0,
    min_pass_rate: float = 0.8,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    num_loaded = 0
    for run_index, path in enumerate(score_paths):
        for record in _load_jsonl(path):
            num_loaded += 1
            pair_id = str(record.get("pair_id"))
            path_id = str(record.get("path_id"))
            if focus_path_ids and path_id not in focus_path_ids:
                continue
            if focus_pair_ids and pair_id not in focus_pair_ids:
                continue
            grouped[(pair_id, path_id)].append({**record, "_run_index": run_index, "_score_path": path})

    reports = []
    for (pair_id, path_id), records in sorted(grouped.items()):
        passes = [
            _record_passes(
                record,
                forget_threshold=forget_threshold,
                retain_threshold=retain_threshold,
                min_saliency_specificity=min_saliency_specificity,
                min_suf=min_suf,
            )
            for record in records
        ]
        pass_count = sum(1 for passed in passes if passed)
        pass_rate = pass_count / len(records) if records else 0.0
        reports.append(
            {
                "pair_id": pair_id,
                "path_id": path_id,
                "path_modality": records[0].get("path_modality") if records else None,
                "num_runs": len(records),
                "pass_count": pass_count,
                "pass_rate": pass_rate,
                "stable": pass_rate >= min_pass_rate,
                "thresholds": {
                    "forget_threshold": forget_threshold,
                    "retain_threshold": retain_threshold,
                    "min_saliency_specificity": min_saliency_specificity,
                    "min_suf": min_suf,
                    "min_pass_rate": min_pass_rate,
                },
                "Suf": _summary([value for value in (_as_float(record.get("Suf")) for record in records) if value is not None]),
                "Ret": _summary([value for value in (_as_float(record.get("Ret")) for record in records) if value is not None]),
                "saliency_specificity_margin": _summary(
                    [
                        value
                        for value in (_as_float(record.get("saliency_specificity_margin")) for record in records)
                        if value is not None
                    ]
                ),
                "forget_saliency": _summary(
                    [value for value in (_as_float(record.get("forget_saliency")) for record in records) if value is not None]
                ),
                "retain_anchor_saliency": _summary(
                    [
                        value
                        for value in (_as_float(record.get("retain_anchor_saliency")) for record in records)
                        if value is not None
                    ]
                ),
                "max_anchor_retain_saliency": _summary(
                    [
                        value
                        for value in (_as_float(record.get("max_anchor_retain_saliency")) for record in records)
                        if value is not None
                    ]
                ),
                "min_anchor_margin": _summary(
                    [
                        value
                        for value in (_as_float(record.get("min_anchor_margin")) for record in records)
                        if value is not None
                    ]
                ),
                "min_anchor_ratio": _summary(
                    [
                        value
                        for value in (_as_float(record.get("min_anchor_ratio")) for record in records)
                        if value is not None
                    ]
                ),
                "score_paths": sorted({str(record["_score_path"]) for record in records}),
            }
        )

    summary = {
        "score_paths": score_paths,
        "num_loaded_score_records": num_loaded,
        "num_report_records": len(reports),
        "num_stable": sum(1 for report in reports if report["stable"]),
        "thresholds": {
            "forget_threshold": forget_threshold,
            "retain_threshold": retain_threshold,
            "min_saliency_specificity": min_saliency_specificity,
            "min_suf": min_suf,
            "min_pass_rate": min_pass_rate,
        },
        "focus_path_ids": sorted(focus_path_ids) if focus_path_ids else None,
        "focus_pair_ids": sorted(focus_pair_ids) if focus_pair_ids else None,
    }
    return reports, summary


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize repeated Step5 causal-score stability.")
    parser.add_argument("--scores_path", nargs="+", required=True)
    parser.add_argument("--output", required=True, help="JSONL output with one row per pair/path.")
    parser.add_argument("--summary", default=None)
    parser.add_argument("--focus_path_ids", nargs="*", default=None)
    parser.add_argument("--focus_pair_ids", nargs="*", default=None)
    parser.add_argument("--forget_threshold", type=float, default=DEFAULT_FORGET_THRESHOLD)
    parser.add_argument("--retain_threshold", type=float, default=DEFAULT_RETAIN_THRESHOLD)
    parser.add_argument("--min_saliency_specificity", type=float, default=0.0)
    parser.add_argument("--min_suf", type=float, default=0.0)
    parser.add_argument("--min_pass_rate", type=float, default=0.8)
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    reports, summary = build_stability_report(
        score_paths=args.scores_path,
        focus_path_ids=set(args.focus_path_ids) if args.focus_path_ids else None,
        focus_pair_ids=set(args.focus_pair_ids) if args.focus_pair_ids else None,
        forget_threshold=args.forget_threshold,
        retain_threshold=args.retain_threshold,
        min_saliency_specificity=args.min_saliency_specificity,
        min_suf=args.min_suf,
        min_pass_rate=args.min_pass_rate,
    )
    _write_jsonl(args.output, reports)
    if args.summary:
        summary_path = Path(args.summary)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
