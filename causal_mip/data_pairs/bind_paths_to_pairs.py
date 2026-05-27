from __future__ import annotations

import argparse
import json
import re
from math import gcd
from pathlib import Path
from typing import Any

from causal_mip.interventions.activation_cache import load_candidate_paths_jsonl, load_pairs_jsonl


def _sample_key(sample: dict[str, Any]) -> tuple[Any, Any, Any]:
    image_ref = sample.get("image_ref") or {}
    dataset_path = image_ref.get("dataset_path") or sample.get("dataset_path")
    row_idx = image_ref.get("row_idx", sample.get("row_idx"))
    sample_id = sample.get("id", image_ref.get("item_id"))
    return dataset_path, row_idx, sample_id


def infer_pair_key(pair: dict[str, Any]) -> tuple[Any, Any, Any]:
    return _sample_key(pair["forget_clean"])


def build_path_pair_bindings(
    candidate_paths_path: str,
    pairs_path: str,
    output_path: str,
    fallback: str = "sample_idx",
    global_fallback_modalities: set[str] | None = None,
    paths_per_sample: dict[str, int] | None = None,
) -> dict[str, Any]:
    if fallback not in {"sample_idx", "all"}:
        raise ValueError(f"Unsupported fallback mode: {fallback}")

    candidate_paths = load_candidate_paths_jsonl(candidate_paths_path)
    pairs = load_pairs_jsonl(pairs_path)
    global_fallback_modalities = set(global_fallback_modalities or set())
    pair_by_row_idx = {
        int(pair["forget_clean"]["image_ref"]["row_idx"]): pair
        for pair in pairs
        if pair.get("forget_clean", {}).get("image_ref", {}).get("row_idx") is not None
    }
    pair_by_ordinal = {idx: pair for idx, pair in enumerate(pairs)}
    inferred_paths_per_sample = _infer_paths_per_sample(candidate_paths, pairs)
    if paths_per_sample:
        inferred_paths_per_sample.update(
            {modality: int(value) for modality, value in paths_per_sample.items() if value}
        )

    records = []
    unbound_path_ids = []
    for path in candidate_paths:
        source_sample_idx = path.source_sample_idx
        inferred_source_sample_idx = False
        if source_sample_idx is None:
            source_sample_idx = path.metadata.get("source_sample_idx") if path.metadata else None
        if source_sample_idx is None and inferred_paths_per_sample.get(path.modality):
            path_ordinal = _path_ordinal(path.path_id)
            if path_ordinal is not None:
                source_sample_idx = path_ordinal // inferred_paths_per_sample[path.modality]
                inferred_source_sample_idx = True

        pair = None
        binding_strategy = None
        if source_sample_idx is not None:
            source_sample_idx = int(source_sample_idx)
            pair = pair_by_row_idx.get(source_sample_idx)
            binding_strategy = "source_sample_idx_to_row_idx"
            if pair is None and not inferred_source_sample_idx:
                pair = pair_by_ordinal.get(source_sample_idx)
                binding_strategy = "source_sample_idx_to_pair_ordinal"

        if pair is None and (
            fallback == "all" or path.modality in global_fallback_modalities
        ):
            binding_strategy = (
                "global_fallback_all_pairs"
                if fallback == "all"
                else f"global_fallback_{path.modality}"
            )
            for fallback_pair in pairs:
                records.append(_binding_record(path, fallback_pair, source_sample_idx, binding_strategy))
            continue

        if pair is None:
            unbound_path_ids.append(path.path_id)
            continue

        records.append(_binding_record(path, pair, source_sample_idx, binding_strategy))

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    return {
        "candidate_paths_path": candidate_paths_path,
        "pairs_path": pairs_path,
        "output_path": output_path,
        "num_candidate_paths": len(candidate_paths),
        "num_pairs": len(pairs),
        "num_bindings": len(records),
        "num_unbound_paths": len(unbound_path_ids),
        "fallback": fallback,
        "global_fallback_modalities": sorted(global_fallback_modalities),
        "inferred_paths_per_sample": inferred_paths_per_sample,
        "unbound_path_ids_preview": unbound_path_ids[:20],
    }


def _path_ordinal(path_id: str) -> int | None:
    match = re.search(r"(\d+)$", path_id)
    return int(match.group(1)) if match else None


def _infer_paths_per_sample(candidate_paths, pairs: list[dict[str, Any]]) -> dict[str, int]:
    result: dict[str, int] = {}
    row_indices = [
        int(pair["forget_clean"]["image_ref"]["row_idx"])
        for pair in pairs
        if pair.get("forget_clean", {}).get("image_ref", {}).get("row_idx") is not None
    ]
    inferred_num_samples = max(row_indices) + 1 if row_indices else None
    if inferred_num_samples is None:
        return result
    for modality in sorted({path.modality for path in candidate_paths}):
        paths = [path for path in candidate_paths if path.modality == modality]
        if not paths:
            continue
        explicit_indices = [
            path.source_sample_idx
            for path in paths
            if path.source_sample_idx is not None
        ]
        if explicit_indices:
            continue
        ordinals = [_path_ordinal(path.path_id) for path in paths]
        if any(ordinal is None for ordinal in ordinals):
            continue
        ordinals = sorted(int(ordinal) for ordinal in ordinals if ordinal is not None)
        if ordinals != list(range(len(ordinals))):
            continue
        if modality == "vision_text":
            # Cross-modal paths are currently built from top global unimodal paths,
            # so their path_id ordinal is not a reliable sample index.
            continue
        if len(paths) % inferred_num_samples == 0:
            result[modality] = len(paths) // inferred_num_samples
            continue
        common = gcd(len(paths), inferred_num_samples)
        if common > 0:
            paths_per_sample = len(paths) // (len(paths) // common)
            if paths_per_sample > 0:
                result[modality] = paths_per_sample
    return result


def _binding_record(path, pair: dict[str, Any], source_sample_idx: int | None, strategy: str | None) -> dict[str, Any]:
    forget_clean = pair["forget_clean"]
    return {
        "pair_id": pair["pair_id"],
        "path_id": path.path_id,
        "sample_id": forget_clean.get("id"),
        "source_row_idx": forget_clean.get("image_ref", {}).get("row_idx", forget_clean.get("row_idx")),
        "source_sample_idx": source_sample_idx,
        "binding_strategy": strategy,
        "path_modality": path.modality,
        "path_source": path.source,
        "sample_key": list(infer_pair_key(pair)),
    }


def load_path_pair_bindings(path: str) -> dict[str, set[str]]:
    bindings: dict[str, set[str]] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            pair_id = record.get("pair_id")
            path_id = record.get("path_id")
            if pair_id is None or path_id is None:
                continue
            bindings.setdefault(pair_id, set()).add(path_id)
    return bindings


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bind Step2 candidate paths to Step3 pair ids.")
    parser.add_argument("--candidate_paths_path", required=True)
    parser.add_argument("--pairs_path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fallback", choices=["sample_idx", "all"], default="sample_idx")
    parser.add_argument("--global_fallback_modalities", nargs="*", default=[])
    parser.add_argument("--text_paths_per_sample", type=int, default=None)
    parser.add_argument("--vision_paths_per_sample", type=int, default=None)
    parser.add_argument("--vision_text_paths_per_sample", type=int, default=None)
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    summary = build_path_pair_bindings(
        candidate_paths_path=args.candidate_paths_path,
        pairs_path=args.pairs_path,
        output_path=args.output,
        fallback=args.fallback,
        global_fallback_modalities=set(args.global_fallback_modalities),
        paths_per_sample={
            "text": args.text_paths_per_sample,
            "vision": args.vision_paths_per_sample,
            "vision_text": args.vision_text_paths_per_sample,
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
