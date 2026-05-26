from __future__ import annotations

from causal_mip.interventions.activation_cache import PreparedSampleBatch, resolve_candidate_path_targets
from causal_mip.interventions.patching import NodeIntervention, run_patched_forward
from causal_mip.path_localization.path_schema import CandidatePath


def build_zero_ablation_interventions(
    candidate_path: CandidatePath,
    prepared_batch: PreparedSampleBatch,
    strict: bool = False,
) -> list[NodeIntervention]:
    resolved_nodes = resolve_candidate_path_targets(candidate_path, prepared_batch, strict=strict)
    return [
        NodeIntervention(
            module=node.module,
            token_positions=list(node.token_positions),
            neuron=node.neuron,
            mode="zero",
        )
        for node in resolved_nodes
    ]


def ablate_candidate_path(
    model,
    prepared_batch: PreparedSampleBatch,
    candidate_path: CandidatePath,
    strict: bool = False,
    no_grad: bool = True,
):
    interventions = build_zero_ablation_interventions(
        candidate_path=candidate_path,
        prepared_batch=prepared_batch,
        strict=strict,
    )
    return run_patched_forward(
        model=model,
        model_inputs=prepared_batch.model_inputs,
        interventions=interventions,
        trace_layers=[intervention.module for intervention in interventions],
        no_grad=no_grad,
    )
