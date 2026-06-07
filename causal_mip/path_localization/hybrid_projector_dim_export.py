from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

from causal_mip.path_localization.path_schema import CandidatePath, PathNode


def _load_jsonl(path: str | Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    records: list[dict[str, Any]] = []
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


def _candidate_by_id(path: str | Path) -> dict[str, CandidatePath]:
    return {record["path_id"]: CandidatePath.from_dict(record) for record in _load_jsonl(path)}


def _records_by_id(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(record["path_id"]): record for record in records if record.get("path_id") is not None}


def _pair_key(record: dict[str, Any]) -> tuple[str, ...]:
    pair_ids = record.get("pair_ids")
    if isinstance(pair_ids, list):
        return tuple(str(pair_id) for pair_id in pair_ids)
    pair_id = record.get("pair_id")
    if pair_id is not None:
        return (str(pair_id),)
    return ()


def _records_by_pair(records: list[dict[str, Any]]) -> dict[tuple[str, ...], dict[str, Any]]:
    by_pair: dict[tuple[str, ...], dict[str, Any]] = {}
    for record in records:
        key = _pair_key(record)
        if key and key not in by_pair:
            by_pair[key] = record
    return by_pair


def _node_template(candidate: CandidatePath) -> PathNode:
    if candidate.nodes:
        return candidate.nodes[0]
    return PathNode(module="mm_projector", layer=None, neuron=0, token_selector="image_tokens")


def _dim_record_from_node(candidate: CandidatePath, node: PathNode, index: int) -> dict[str, Any]:
    return {
        "node_index": index,
        "module": "visual.merger",
        "layer": node.layer,
        "token_selector": node.token_selector,
        "module_kind": "projector",
        "dim_index": int(node.neuron),
        "hybrid_dim_source": "node",
    }


def _candidate_dim_records(candidate: CandidatePath) -> list[dict[str, Any]]:
    metadata = candidate.metadata or {}
    selected = metadata.get("selected_projector_dims")
    if isinstance(selected, list) and selected:
        dim_records = []
        for item in selected:
            if not isinstance(item, dict) or item.get("dim_index") is None:
                continue
            dim_records.append({**item, "dim_index": int(item["dim_index"])})
        if dim_records:
            return dim_records
    return [_dim_record_from_node(candidate, node, index) for index, node in enumerate(candidate.nodes)]


def _dim_indices(dim_records: list[dict[str, Any]]) -> list[int]:
    return [int(record["dim_index"]) for record in dim_records]


def _dedupe_dim_records(dim_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped = []
    seen = set()
    for record in dim_records:
        dim_index = int(record["dim_index"])
        if dim_index in seen:
            continue
        seen.add(dim_index)
        deduped.append({**record, "dim_index": dim_index})
    return deduped


def _mean_optional(dim_records: list[dict[str, Any]], key: str) -> float | None:
    values = []
    for record in dim_records:
        value = record.get(key)
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    if not values:
        return None
    return float(mean(values))


def _path_mip_score(template: CandidatePath, dim_records: list[dict[str, Any]]) -> float:
    for key in ("saliency_specificity_margin", "min_anchor_margin", "forget_saliency"):
        value = _mean_optional(dim_records, key)
        if value is not None:
            return value
    return float(template.mip_score)


def _clone_hybrid_candidate(
    *,
    new_path_id: str,
    template: CandidatePath,
    dim_records: list[dict[str, Any]],
    base_record: dict[str, Any],
    source_record: dict[str, Any],
    hybrid_metadata: dict[str, Any],
) -> CandidatePath:
    node_template = _node_template(template)
    nodes = [
        PathNode(
            module=node_template.module,
            layer=node_template.layer,
            neuron=int(dim_record["dim_index"]),
            token_selector=node_template.token_selector,
        )
        for dim_record in dim_records
    ]
    metadata = {
        **(template.metadata or {}),
        "candidate_generation": "hybrid_projector_dim",
        "projector_dim_level": True,
        "projector_dim_pair_id": (_pair_key(source_record) or _pair_key(base_record) or [None])[0],
        "selected_projector_dims": dim_records,
        "hybrid_projector_dim": hybrid_metadata,
    }
    salun_mean = _mean_optional(dim_records, "salun_ssd_score")
    fisher_mean = _mean_optional(dim_records, "fisher_specificity_margin")
    if salun_mean is not None:
        metadata["mean_selected_salun_ssd_score"] = salun_mean
    if fisher_mean is not None:
        metadata["mean_selected_fisher_specificity_margin"] = fisher_mean
    return CandidatePath(
        path_id=new_path_id,
        source=f"{template.source}_hybrid_projector_dim",
        modality=template.modality,
        mip_score=_path_mip_score(template, dim_records),
        nodes=nodes,
        source_sample_idx=template.source_sample_idx,
        metadata=metadata,
    )


def _augment_dims_for_pair(
    *,
    pair_key: tuple[str, ...],
    augment_records_by_pair: dict[tuple[str, ...], dict[str, Any]],
    augment_candidates: dict[str, CandidatePath],
    existing_dims: set[int],
    augment_top_k_dims: int,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    if augment_top_k_dims <= 0:
        return [], None
    augment_record = augment_records_by_pair.get(pair_key)
    if augment_record is None:
        return [], None
    augment_candidate = augment_candidates.get(str(augment_record.get("path_id")))
    if augment_candidate is None:
        return [], None
    appended = []
    for dim_record in _candidate_dim_records(augment_candidate):
        dim_index = int(dim_record["dim_index"])
        if dim_index in existing_dims:
            continue
        appended.append({**dim_record, "hybrid_dim_source": "augment", "augment_path_id": augment_candidate.path_id})
        existing_dims.add(dim_index)
        if len(appended) >= augment_top_k_dims:
            break
    return appended, augment_record


def export_hybrid_projector_dim_candidates(
    *,
    base_candidates_path: str,
    base_p_forget_path: str,
    output_candidates_path: str,
    output_p_forget_path: str,
    output_bindings_path: str,
    output_p_shared_path: str | None = None,
    summary_path: str | None = None,
    path_id_prefix: str = "hybrid_projdim",
    drop_path_ids: list[str] | None = None,
    replace_path_ids: list[str] | None = None,
    replace_candidates_path: str | None = None,
    replace_p_forget_path: str | None = None,
    augment_candidates_path: str | None = None,
    augment_p_forget_path: str | None = None,
    augment_pair_ids: list[str] | None = None,
    augment_top_k_dims: int = 0,
) -> dict[str, Any]:
    base_candidates = _candidate_by_id(base_candidates_path)
    base_records = _load_jsonl(base_p_forget_path)
    replace_candidates = _candidate_by_id(replace_candidates_path) if replace_candidates_path else {}
    replace_records = _load_jsonl(replace_p_forget_path)
    replace_records_by_pair = _records_by_pair(replace_records)
    augment_candidates = _candidate_by_id(augment_candidates_path) if augment_candidates_path else {}
    augment_records = _load_jsonl(augment_p_forget_path)
    augment_records_by_pair = _records_by_pair(augment_records)

    drop_set = {str(path_id) for path_id in (drop_path_ids or [])}
    replace_set = {str(path_id) for path_id in (replace_path_ids or [])}
    augment_pair_set = {str(pair_id) for pair_id in (augment_pair_ids or [])}

    output_candidates = []
    output_p_forget = []
    output_bindings = []
    components = []

    for index, base_record in enumerate(record for record in base_records if str(record.get("path_id")) not in drop_set):
        base_path_id = str(base_record["path_id"])
        pair_key = _pair_key(base_record)
        base_candidate = base_candidates.get(base_path_id)
        if base_candidate is None:
            raise KeyError(f"Base P_forget path_id not found in candidates: {base_path_id}")

        template = base_candidate
        source_record = base_record
        source_candidate = base_candidate
        replaced_by = None

        if base_path_id in replace_set:
            replacement_record = replace_records_by_pair.get(pair_key)
            if replacement_record is None:
                raise KeyError(f"No replacement P_forget record found for pair {pair_key} and path {base_path_id}")
            replacement_path_id = str(replacement_record["path_id"])
            replacement_candidate = replace_candidates.get(replacement_path_id)
            if replacement_candidate is None:
                raise KeyError(f"Replacement path_id not found in replacement candidates: {replacement_path_id}")
            source_record = replacement_record
            source_candidate = replacement_candidate
            template = replacement_candidate
            replaced_by = replacement_path_id

        dim_records = [
            {**record, "hybrid_dim_source": "core", "core_path_id": source_candidate.path_id}
            for record in _candidate_dim_records(source_candidate)
        ]
        dim_records = _dedupe_dim_records(dim_records)
        augmented_from = None
        if not augment_pair_set or any(pair_id in augment_pair_set for pair_id in pair_key):
            existing_dims = set(_dim_indices(dim_records))
            appended, augment_record = _augment_dims_for_pair(
                pair_key=pair_key,
                augment_records_by_pair=augment_records_by_pair,
                augment_candidates=augment_candidates,
                existing_dims=existing_dims,
                augment_top_k_dims=augment_top_k_dims,
            )
            if appended:
                dim_records = _dedupe_dim_records(dim_records + appended)
                augmented_from = str(augment_record["path_id"]) if augment_record is not None else None

        new_path_id = f"{path_id_prefix}_p{len(output_candidates):06d}"
        hybrid_metadata = {
            "base_path_id": base_path_id,
            "base_pair_ids": list(pair_key),
            "source_path_id": source_candidate.path_id,
            "replaced_by_path_id": replaced_by,
            "augmented_by_path_id": augmented_from,
            "augment_top_k_dims": augment_top_k_dims,
            "num_core_dims": len(_candidate_dim_records(source_candidate)),
            "num_total_dims": len(dim_records),
            "selected_dim_indices": _dim_indices(dim_records),
        }
        candidate = _clone_hybrid_candidate(
            new_path_id=new_path_id,
            template=template,
            dim_records=dim_records,
            base_record=base_record,
            source_record=source_record,
            hybrid_metadata=hybrid_metadata,
        )
        output_candidates.append(candidate.to_dict())

        p_forget_record = {
            **source_record,
            "path_id": new_path_id,
            "category": "P_forget",
            "hybrid_base_path_id": base_path_id,
            "hybrid_source_path_id": source_candidate.path_id,
            "hybrid_replaced_by_path_id": replaced_by,
            "hybrid_augmented_by_path_id": augmented_from,
            "hybrid_num_dims": len(dim_records),
            "hybrid_selected_dim_indices": _dim_indices(dim_records),
        }
        output_p_forget.append(p_forget_record)
        binding_pair_id = pair_key[0] if pair_key else None
        output_bindings.append(
            {
                "pair_id": binding_pair_id,
                "path_id": new_path_id,
                "base_path_id": base_path_id,
                "source_path_id": source_candidate.path_id,
                "replaced_by_path_id": replaced_by,
                "augmented_by_path_id": augmented_from,
                "binding_strategy": "hybrid_projector_dim",
                "num_selected_dims": len(dim_records),
                "selected_dim_indices": _dim_indices(dim_records),
            }
        )
        components.append(hybrid_metadata)

    _write_jsonl(output_candidates_path, output_candidates)
    _write_jsonl(output_p_forget_path, output_p_forget)
    _write_jsonl(output_bindings_path, output_bindings)
    if output_p_shared_path:
        _write_jsonl(output_p_shared_path, [])

    summary = {
        "base_candidates_path": base_candidates_path,
        "base_p_forget_path": base_p_forget_path,
        "replace_candidates_path": replace_candidates_path,
        "replace_p_forget_path": replace_p_forget_path,
        "augment_candidates_path": augment_candidates_path,
        "augment_p_forget_path": augment_p_forget_path,
        "output_candidates_path": output_candidates_path,
        "output_p_forget_path": output_p_forget_path,
        "output_p_shared_path": output_p_shared_path,
        "output_bindings_path": output_bindings_path,
        "path_id_prefix": path_id_prefix,
        "drop_path_ids": sorted(drop_set),
        "replace_path_ids": sorted(replace_set),
        "augment_pair_ids": sorted(augment_pair_set),
        "augment_top_k_dims": augment_top_k_dims,
        "num_output_candidates": len(output_candidates),
        "num_output_p_forget": len(output_p_forget),
        "num_unique_dims": len({dim for record in output_bindings for dim in record["selected_dim_indices"]}),
        "components": components,
    }
    if summary_path:
        summary_output = Path(summary_path)
        summary_output.parent.mkdir(parents=True, exist_ok=True)
        with summary_output.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    return summary


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build hybrid projector-dim candidates from CE-driving core and safe augment dims.")
    parser.add_argument("--base_candidates", required=True)
    parser.add_argument("--base_p_forget", required=True)
    parser.add_argument("--output_candidates", required=True)
    parser.add_argument("--output_p_forget", required=True)
    parser.add_argument("--output_bindings", required=True)
    parser.add_argument("--output_p_shared", default=None)
    parser.add_argument("--summary", default=None)
    parser.add_argument("--path_id_prefix", default="hybrid_projdim")
    parser.add_argument("--drop_path_ids", nargs="*", default=[])
    parser.add_argument("--replace_path_ids", nargs="*", default=[])
    parser.add_argument("--replace_candidates", default=None)
    parser.add_argument("--replace_p_forget", default=None)
    parser.add_argument("--augment_candidates", default=None)
    parser.add_argument("--augment_p_forget", default=None)
    parser.add_argument("--augment_pair_ids", nargs="*", default=[])
    parser.add_argument("--augment_top_k_dims", type=int, default=0)
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    summary = export_hybrid_projector_dim_candidates(
        base_candidates_path=args.base_candidates,
        base_p_forget_path=args.base_p_forget,
        output_candidates_path=args.output_candidates,
        output_p_forget_path=args.output_p_forget,
        output_p_shared_path=args.output_p_shared,
        output_bindings_path=args.output_bindings,
        summary_path=args.summary,
        path_id_prefix=args.path_id_prefix,
        drop_path_ids=args.drop_path_ids,
        replace_path_ids=args.replace_path_ids,
        replace_candidates_path=args.replace_candidates,
        replace_p_forget_path=args.replace_p_forget,
        augment_candidates_path=args.augment_candidates,
        augment_p_forget_path=args.augment_p_forget,
        augment_pair_ids=args.augment_pair_ids,
        augment_top_k_dims=args.augment_top_k_dims,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
