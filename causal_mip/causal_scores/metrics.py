from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from causal_mip.causal_scores.necessity import compute_necessity
from causal_mip.causal_scores.retain_impact import compute_retain_impact
from causal_mip.causal_scores.sufficiency import compute_sufficiency
from causal_mip.interventions.activation_cache import (
    PreparedSampleBatch,
    SampleReferenceResolver,
    extract_pair_sample,
    prepare_sample_batch,
    resolve_candidate_path_targets,
)
from causal_mip.path_localization.path_schema import CandidatePath


def filter_candidate_paths_for_step5(
    candidate_paths: list[CandidatePath],
    modality_filter: str = "all",
) -> list[CandidatePath]:
    allowed_modalities = {"all", "text", "vision_text", "vision"}
    if modality_filter not in allowed_modalities:
        raise ValueError(f"Unsupported modality_filter: {modality_filter}")
    if modality_filter == "all":
        return list(candidate_paths)
    return [path for path in candidate_paths if path.modality == modality_filter]


def build_pair_prepared_batches(
    pair: dict[str, Any],
    processor,
    model,
    image_resize: int,
    resolver: SampleReferenceResolver | None = None,
) -> dict[str, PreparedSampleBatch]:
    resolver = resolver or SampleReferenceResolver()
    forget_clean = extract_pair_sample(pair, "forget_clean")
    forget_corrupt = extract_pair_sample(pair, "forget_corrupt")
    counterfactual = extract_pair_sample(pair, "counterfactual_retain")
    same_topic = extract_pair_sample(pair, "hard_retain", hard_retain_type="same_topic")

    forget_clean_batch = prepare_sample_batch(
        forget_clean,
        processor=processor,
        model=model,
        image_resize=image_resize,
        resolver=resolver,
    )
    corrupt_batch = prepare_sample_batch(
        forget_corrupt,
        processor=processor,
        model=model,
        image_resize=image_resize,
        resolver=resolver,
        target_answer_text=forget_clean.get("answer"),
    )
    corrupt_source = "forget_corrupt"
    if _same_model_inputs_for_sufficiency(
        forget_clean_batch,
        corrupt_batch,
    ):
        corrupt_batch = prepare_sample_batch(
            counterfactual,
            processor=processor,
            model=model,
            image_resize=image_resize,
            resolver=resolver,
            target_answer_text=forget_clean.get("answer"),
        )
        corrupt_source = "counterfactual_retain_fallback"

    forget_clean_batch.sample = dict(forget_clean_batch.sample)
    forget_clean_batch.sample["step5_role"] = "forget_clean"
    corrupt_batch.sample = dict(corrupt_batch.sample)
    corrupt_batch.sample["step5_role"] = "forget_corrupt_target_clean_answer"
    corrupt_batch.sample["step5_corrupt_source"] = corrupt_source

    batches = {
        "forget_clean": forget_clean_batch,
        "forget_corrupt_target_clean_answer": corrupt_batch,
        "same_topic": prepare_sample_batch(
            same_topic,
            processor=processor,
            model=model,
            image_resize=image_resize,
            resolver=resolver,
        ),
        "counterfactual_retain": prepare_sample_batch(
            counterfactual,
            processor=processor,
            model=model,
            image_resize=image_resize,
            resolver=resolver,
        ),
    }

    hard_retain = pair.get("hard_retain", [])
    if any(sample.get("type") == "same_reasoning" for sample in hard_retain):
        same_reasoning = extract_pair_sample(pair, "hard_retain", hard_retain_type="same_reasoning")
        batches["same_reasoning"] = prepare_sample_batch(
            same_reasoning,
            processor=processor,
            model=model,
            image_resize=image_resize,
            resolver=resolver,
        )

    return batches


def _tensor_equal(left: torch.Tensor | None, right: torch.Tensor | None) -> bool:
    if left is None or right is None:
        return left is None and right is None
    if left.shape != right.shape:
        return False
    return bool(torch.equal(left.detach().cpu(), right.detach().cpu()))


def _same_model_inputs_for_sufficiency(clean_batch: PreparedSampleBatch, corrupt_batch: PreparedSampleBatch) -> bool:
    keys = sorted(set(clean_batch.model_inputs) | set(corrupt_batch.model_inputs))
    for key in keys:
        if key == "labels":
            continue
        left = clean_batch.model_inputs.get(key)
        right = corrupt_batch.model_inputs.get(key)
        if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
            if not _tensor_equal(left, right):
                return False
        elif left != right:
            return False
    return True


def compute_path_causal_score_record(
    model,
    candidate_path: CandidatePath,
    pair: dict[str, Any],
    prepared_batches: dict[str, PreparedSampleBatch],
    strict: bool = False,
) -> dict[str, Any]:
    clean_batch = prepared_batches["forget_clean"]
    resolved_nodes = resolve_candidate_path_targets(candidate_path, clean_batch, strict=False, model=model)

    record = {
        "pair_id": pair.get("pair_id"),
        "path_id": candidate_path.path_id,
        "path_source": candidate_path.source,
        "path_modality": candidate_path.modality,
        "mip_score": candidate_path.mip_score,
        "num_nodes": len(candidate_path.nodes),
        "num_patchable_nodes": len(resolved_nodes),
        "sufficiency_corrupt_source": prepared_batches["forget_corrupt_target_clean_answer"].sample.get(
            "step5_corrupt_source",
            "forget_corrupt",
        ),
    }

    if not resolved_nodes:
        record.update(
            {
                "status": "no_patchable_nodes",
                "Nec": None,
                "Suf": None,
                "Ret": None,
                "necessity": None,
                "sufficiency": None,
                "retain_impact": None,
                "retain_details": {},
            }
        )
        return record

    necessity = compute_necessity(
        model=model,
        clean_batch=clean_batch,
        candidate_path=candidate_path,
        strict=strict,
    )
    sufficiency = compute_sufficiency(
        model=model,
        clean_batch=clean_batch,
        corrupt_batch=prepared_batches["forget_corrupt_target_clean_answer"],
        candidate_path=candidate_path,
        strict=strict,
    )
    retain = compute_retain_impact(
        model=model,
        retain_batches={
            key: value
            for key, value in prepared_batches.items()
            if key in {"same_topic", "same_reasoning", "counterfactual_retain"}
        },
        candidate_path=candidate_path,
        strict=strict,
    )

    record.update(
        {
            "status": "ok",
            "Nec": necessity["necessity"],
            "Suf": sufficiency["sufficiency"],
            "Ret": retain["retain_impact"],
            "necessity": necessity,
            "sufficiency": sufficiency,
            "retain_impact": retain["retain_impact"],
            "retain_details": retain["retain_details"],
        }
    )
    return record


def write_path_score_records_jsonl(records: list[dict[str, Any]], output_path: str) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
