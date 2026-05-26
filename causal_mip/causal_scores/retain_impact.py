from __future__ import annotations

from typing import Any

from causal_mip.interventions.ablation import ablate_candidate_path
from causal_mip.interventions.activation_cache import (
    PreparedSampleBatch,
    compute_target_answer_logprob,
    resolve_candidate_path_targets,
)
from causal_mip.path_localization.path_schema import CandidatePath


def compute_retain_impact(
    model,
    retain_batches: dict[str, PreparedSampleBatch],
    candidate_path: CandidatePath,
    strict: bool = False,
) -> dict[str, Any]:
    if not retain_batches:
        return {
            "status": "no_retain_batches",
            "num_patchable_nodes": 0,
            "retain_impact": None,
            "retain_details": {},
        }

    first_batch = next(iter(retain_batches.values()))
    resolved_nodes = resolve_candidate_path_targets(candidate_path, first_batch, strict=strict)
    if not resolved_nodes:
        return {
            "status": "no_patchable_nodes",
            "num_patchable_nodes": 0,
            "retain_impact": None,
            "retain_details": {},
        }

    retain_details: dict[str, dict[str, float]] = {}
    impacts: list[float] = []
    for retain_name, retain_batch in retain_batches.items():
        baseline_outputs = model(**retain_batch.model_inputs)
        baseline_score = float(
            compute_target_answer_logprob(baseline_outputs.logits.detach(), retain_batch).detach().cpu().item()
        )

        ablated_outputs, _ = ablate_candidate_path(
            model=model,
            prepared_batch=retain_batch,
            candidate_path=candidate_path,
            strict=strict,
            no_grad=True,
        )
        ablated_score = float(
            compute_target_answer_logprob(ablated_outputs.logits.detach(), retain_batch).detach().cpu().item()
        )
        impact = baseline_score - ablated_score
        retain_details[retain_name] = {
            "baseline_score": baseline_score,
            "ablated_score": ablated_score,
            "impact": impact,
        }
        impacts.append(impact)

    return {
        "status": "ok",
        "num_patchable_nodes": len(resolved_nodes),
        "retain_impact": sum(impacts) / len(impacts),
        "retain_details": retain_details,
    }
