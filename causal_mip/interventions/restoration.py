from __future__ import annotations

from causal_mip.interventions.activation_cache import (
    ALL_VISUAL_TOKEN_POSITIONS,
    CachedPathActivations,
    PreparedSampleBatch,
    resolve_token_positions,
)
from causal_mip.interventions.patching import NodeIntervention, run_patched_forward


def build_restoration_interventions(
    cached_path: CachedPathActivations,
    prepared_batch: PreparedSampleBatch | None = None,
) -> list[NodeIntervention]:
    interventions = []
    for node in cached_path.nodes:
        token_positions = list(node.token_positions)
        if prepared_batch is not None and node.module_kind != "vision":
            candidate_positions = resolve_token_positions(prepared_batch, node.token_selector)
            if len(candidate_positions) == node.values.numel():
                token_positions = candidate_positions
            elif token_positions == list(ALL_VISUAL_TOKEN_POSITIONS):
                token_positions = candidate_positions
        interventions.append(
            NodeIntervention(
                module=node.module,
                token_positions=token_positions,
                neuron=node.neuron,
                mode="restore",
                values=node.values,
            )
        )
    return interventions


def restore_path_activations(
    model,
    prepared_batch: PreparedSampleBatch,
    cached_path: CachedPathActivations,
    no_grad: bool = True,
):
    interventions = build_restoration_interventions(cached_path, prepared_batch=prepared_batch)
    return run_patched_forward(
        model=model,
        model_inputs=prepared_batch.model_inputs,
        interventions=interventions,
        trace_layers=[intervention.module for intervention in interventions],
        no_grad=no_grad,
    )
