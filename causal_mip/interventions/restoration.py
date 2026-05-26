from __future__ import annotations

from causal_mip.interventions.activation_cache import (
    CachedPathActivations,
    PreparedSampleBatch,
    resolve_token_positions,
)
from causal_mip.interventions.patching import NodeIntervention, run_patched_forward


def build_restoration_interventions(
    cached_path: CachedPathActivations,
    prepared_batch: PreparedSampleBatch | None = None,
) -> list[NodeIntervention]:
    return [
        NodeIntervention(
            module=node.module,
            token_positions=(
                resolve_token_positions(prepared_batch, node.token_selector)
                if prepared_batch is not None and node.module_kind != "vision"
                else list(node.token_positions)
            ),
            neuron=node.neuron,
            mode="restore",
            values=node.values,
        )
        for node in cached_path.nodes
    ]


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
