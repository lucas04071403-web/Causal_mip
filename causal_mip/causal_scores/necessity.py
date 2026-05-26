from __future__ import annotations

from typing import Any

from causal_mip.interventions.ablation import ablate_candidate_path
from causal_mip.interventions.activation_cache import (
    PreparedSampleBatch,
    compute_target_answer_logprob,
    resolve_candidate_path_targets,
)
from causal_mip.path_localization.path_schema import CandidatePath


def compute_necessity(
    model,
    clean_batch: PreparedSampleBatch,
    candidate_path: CandidatePath,
    strict: bool = False,
) -> dict[str, Any]:
    resolved_nodes = resolve_candidate_path_targets(candidate_path, clean_batch, strict=strict)
    if not resolved_nodes:
        return {
            "status": "no_patchable_nodes",
            "num_patchable_nodes": 0,
            "clean_score": None,
            "ablated_score": None,
            "necessity": None,
        }

    clean_outputs = model(**clean_batch.model_inputs)
    clean_score = float(compute_target_answer_logprob(clean_outputs.logits.detach(), clean_batch).detach().cpu().item())

    ablated_outputs, _ = ablate_candidate_path(
        model=model,
        prepared_batch=clean_batch,
        candidate_path=candidate_path,
        strict=strict,
        no_grad=True,
    )
    ablated_score = float(
        compute_target_answer_logprob(ablated_outputs.logits.detach(), clean_batch).detach().cpu().item()
    )

    return {
        "status": "ok",
        "num_patchable_nodes": len(resolved_nodes),
        "clean_score": clean_score,
        "ablated_score": ablated_score,
        "necessity": clean_score - ablated_score,
    }
