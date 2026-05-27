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


def _target_token_diagnostics(batch: PreparedSampleBatch) -> dict[str, Any]:
    answer_positions = [position for position in batch.answer_token_positions if position > 0]
    token_ids = []
    score_positions = []
    for answer_position in answer_positions:
        score_position = answer_position - 1
        if score_position < 0 or score_position >= batch.input_ids.shape[1] - 1:
            continue
        score_positions.append(score_position)
        token_ids.append(int(batch.input_ids[0, answer_position].detach().cpu().item()))
    return {
        "target_answer_text": batch.target_answer_text,
        "answer_token_positions": list(batch.answer_token_positions),
        "score_positions": score_positions,
        "answer_token_ids": token_ids,
        "prompt_length": batch.prompt_length,
    }


def compute_sufficiency(
    model,
    clean_batch: PreparedSampleBatch,
    corrupt_batch: PreparedSampleBatch,
    candidate_path: CandidatePath,
    strict: bool = False,
) -> dict[str, Any]:
    resolved_nodes = resolve_candidate_path_targets(candidate_path, clean_batch, strict=strict, model=model)
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
    restored_nodes = [
        {
            "module": node.module,
            "layer": node.layer,
            "neuron": node.neuron,
            "token_selector": node.token_selector,
            "module_kind": node.module_kind,
            "clean_token_positions": list(node.token_positions),
            "num_restored_tokens": int(node.values.numel()),
        }
        for node in cached_path.nodes
    ]

    return {
        "status": "ok",
        "num_patchable_nodes": len(resolved_nodes),
        "clean_score": cached_path.target_answer_logprob,
        "corrupt_score": corrupt_score,
        "restored_score": restored_score,
        "sufficiency": restored_score - corrupt_score,
        "clean_target": _target_token_diagnostics(clean_batch),
        "corrupt_target": _target_token_diagnostics(corrupt_batch),
        "restored_nodes": restored_nodes,
    }
