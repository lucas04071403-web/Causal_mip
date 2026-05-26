from __future__ import annotations

from typing import Any

from causal_mip.interventions.activation_cache import (
    PreparedSampleBatch,
    cache_candidate_path_activations,
    compute_target_answer_logprob,
    resolve_candidate_path_targets,
)
from causal_mip.interventions.restoration import restore_path_activations
from causal_mip.path_localization.path_schema import CandidatePath


def compute_sufficiency(
    model,
    clean_batch: PreparedSampleBatch,
    corrupt_batch: PreparedSampleBatch,
    candidate_path: CandidatePath,
    strict: bool = False,
) -> dict[str, Any]:
    resolved_nodes = resolve_candidate_path_targets(candidate_path, clean_batch, strict=strict)
    if not resolved_nodes:
        return {
            "status": "no_patchable_nodes",
            "num_patchable_nodes": 0,
            "corrupt_score": None,
            "restored_score": None,
            "sufficiency": None,
        }

    cached_path = cache_candidate_path_activations(
        model=model,
        prepared_batch=clean_batch,
        candidate_path=candidate_path,
        strict=strict,
        no_grad=True,
    )

    corrupt_outputs = model(**corrupt_batch.model_inputs)
    corrupt_score = float(
        compute_target_answer_logprob(corrupt_outputs.logits.detach(), corrupt_batch).detach().cpu().item()
    )

    restored_outputs, _ = restore_path_activations(
        model=model,
        prepared_batch=corrupt_batch,
        cached_path=cached_path,
        no_grad=True,
    )
    restored_score = float(
        compute_target_answer_logprob(restored_outputs.logits.detach(), corrupt_batch).detach().cpu().item()
    )

    return {
        "status": "ok",
        "num_patchable_nodes": len(resolved_nodes),
        "clean_score": cached_path.target_answer_logprob,
        "corrupt_score": corrupt_score,
        "restored_score": restored_score,
        "sufficiency": restored_score - corrupt_score,
    }
