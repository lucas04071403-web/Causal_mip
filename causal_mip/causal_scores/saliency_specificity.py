from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Any

import torch

from causal_mip.interventions.activation_cache import (
    ALL_VISUAL_TOKEN_POSITIONS,
    WHOLE_VECTOR_NEURON,
    PreparedSampleBatch,
    ResolvedPathNode,
    compute_target_answer_logprob,
    resolve_candidate_path_targets,
)
from causal_mip.interventions.hooks import get_module
from causal_mip.path_localization.path_schema import CandidatePath


def _untuple(output):
    return output[0] if isinstance(output, tuple) else output


def _repack(original_input, new_tensor):
    if isinstance(original_input, tuple):
        return (new_tensor,) + tuple(original_input[1:])
    return new_tensor


def _expand_token_positions(token_positions: list[int], tensor: torch.Tensor) -> list[int]:
    if token_positions == ALL_VISUAL_TOKEN_POSITIONS:
        if tensor.ndim < 2:
            return []
        return list(range(tensor.shape[-2]))
    return list(token_positions)


def _select_node_tensor(tensor: torch.Tensor, token_positions: list[int], neuron: int) -> torch.Tensor:
    if neuron == WHOLE_VECTOR_NEURON:
        if tensor.ndim == 3:
            return tensor[:, token_positions, :]
        if tensor.ndim == 2:
            return tensor[token_positions, :]
        raise ValueError(f"Unsupported activation rank for saliency: ndim={tensor.ndim}")
    if tensor.ndim == 3:
        return tensor[:, token_positions, neuron]
    if tensor.ndim == 2:
        return tensor[token_positions, neuron]
    raise ValueError(f"Unsupported activation rank for saliency: ndim={tensor.ndim}")


def _node_saliency_from_activation_grad(
    activation: torch.Tensor,
    gradient: torch.Tensor,
    node: ResolvedPathNode,
) -> dict[str, Any]:
    token_positions = _expand_token_positions(node.token_positions, activation)
    if not token_positions:
        return {
            "saliency": 0.0,
            "fisher_saliency": 0.0,
            "num_token_positions": 0,
            "status": "no_token_positions",
        }

    selected_activation = _select_node_tensor(activation, token_positions, node.neuron).detach()
    selected_gradient = _select_node_tensor(gradient, token_positions, node.neuron).detach()
    grad_act = torch.abs(selected_activation * selected_gradient)
    fisher = selected_gradient.pow(2)
    return {
        "saliency": float(grad_act.mean().detach().cpu().item()),
        "fisher_saliency": float(fisher.mean().detach().cpu().item()),
        "num_token_positions": len(token_positions),
        "status": "ok",
    }


def _summarize_node_scores(
    candidate_path: CandidatePath,
    resolved_nodes: list[ResolvedPathNode],
    node_scores: list[dict[str, Any]],
    status: str,
) -> dict[str, Any]:
    valid_scores = [score for score in node_scores if score.get("status") == "ok"]
    saliencies = [float(score["saliency"]) for score in valid_scores]
    fisher_saliencies = [float(score["fisher_saliency"]) for score in valid_scores]
    return {
        "status": status,
        "path_id": candidate_path.path_id,
        "num_patchable_nodes": len(resolved_nodes),
        "num_scored_nodes": len(valid_scores),
        "saliency": float(mean(saliencies)) if saliencies else 0.0,
        "saliency_sum": float(sum(saliencies)),
        "fisher_saliency": float(mean(fisher_saliencies)) if fisher_saliencies else 0.0,
        "fisher_saliency_sum": float(sum(fisher_saliencies)),
        "node_scores": node_scores,
    }


def _compute_saliency_with_existing_graph(
    model,
    prepared_batch: PreparedSampleBatch,
    candidate_path: CandidatePath,
    resolved_nodes: list[ResolvedPathNode],
) -> tuple[dict[str, Any], set[tuple[str, int, int]]]:
    captured: dict[str, torch.Tensor] = {}
    handles = []

    def make_hook(module_name: str):
        def hook(_module, inputs):
            tensor = _untuple(inputs)
            if isinstance(tensor, torch.Tensor):
                if tensor.requires_grad:
                    tensor.retain_grad()
                captured[module_name] = tensor
            return None

        return hook

    for module_name in sorted({node.module for node in resolved_nodes}):
        handles.append(get_module(model, module_name).register_forward_pre_hook(make_hook(module_name)))

    try:
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            outputs = model(**prepared_batch.model_inputs)
            score = compute_target_answer_logprob(outputs.logits, prepared_batch)
            if not bool(getattr(score, "requires_grad", False)):
                node_scores = [
                    {
                        "node_index": node_index,
                        "module": node.module,
                        "layer": node.layer,
                        "neuron": node.neuron,
                        "token_selector": node.token_selector,
                        "module_kind": node.module_kind,
                        "saliency": 0.0,
                        "fisher_saliency": 0.0,
                        "num_token_positions": 0,
                        "status": "missing_gradient_graph",
                    }
                    for node_index, node in enumerate(resolved_nodes)
                ]
                return _summarize_node_scores(candidate_path, resolved_nodes, node_scores, "missing_gradient_graph"), {
                    (node.module, idx, node.neuron) for idx, node in enumerate(resolved_nodes)
                }
            score.backward()

        missing: set[tuple[str, int, int]] = set()
        node_scores = []
        for node_index, node in enumerate(resolved_nodes):
            activation = captured.get(node.module)
            gradient = activation.grad if isinstance(activation, torch.Tensor) else None
            base = {
                "node_index": node_index,
                "module": node.module,
                "layer": node.layer,
                "neuron": node.neuron,
                "token_selector": node.token_selector,
                "module_kind": node.module_kind,
            }
            if activation is None or gradient is None:
                missing.add((node.module, node_index, node.neuron))
                node_scores.append(
                    {
                        **base,
                        "saliency": 0.0,
                        "fisher_saliency": 0.0,
                        "num_token_positions": 0,
                        "status": "missing_activation_gradient",
                    }
                )
                continue
            node_scores.append({**base, **_node_saliency_from_activation_grad(activation, gradient, node)})

        status = "ok" if not missing else "partial_missing_gradients"
        return _summarize_node_scores(candidate_path, resolved_nodes, node_scores, status), missing
    finally:
        for handle in reversed(handles):
            handle.remove()
        model.zero_grad(set_to_none=True)


def _compute_leaf_saliency_for_module_nodes(
    model,
    prepared_batch: PreparedSampleBatch,
    candidate_path: CandidatePath,
    nodes: list[tuple[int, ResolvedPathNode]],
) -> list[dict[str, Any]]:
    module_name = nodes[0][1].module
    captured: dict[str, torch.Tensor] = {}

    def hook(_module, inputs):
        tensor = _untuple(inputs)
        if not isinstance(tensor, torch.Tensor):
            return None
        leaf = tensor.detach().clone().requires_grad_(True)
        captured["activation"] = leaf
        return _repack(inputs, leaf)

    handle = get_module(model, module_name).register_forward_pre_hook(hook)
    try:
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            outputs = model(**prepared_batch.model_inputs)
            score = compute_target_answer_logprob(outputs.logits, prepared_batch)
            if not bool(getattr(score, "requires_grad", False)):
                return [
                    {
                        "node_index": node_index,
                        "module": node.module,
                        "layer": node.layer,
                        "neuron": node.neuron,
                        "token_selector": node.token_selector,
                        "module_kind": node.module_kind,
                        "saliency": 0.0,
                        "fisher_saliency": 0.0,
                        "num_token_positions": 0,
                        "status": "missing_leaf_gradient_graph",
                    }
                    for node_index, node in nodes
                ]
            score.backward()

        activation = captured.get("activation")
        gradient = activation.grad if isinstance(activation, torch.Tensor) else None
        node_scores = []
        for node_index, node in nodes:
            base = {
                "node_index": node_index,
                "module": node.module,
                "layer": node.layer,
                "neuron": node.neuron,
                "token_selector": node.token_selector,
                "module_kind": node.module_kind,
            }
            if activation is None or gradient is None:
                node_scores.append(
                    {
                        **base,
                        "saliency": 0.0,
                        "fisher_saliency": 0.0,
                        "num_token_positions": 0,
                        "status": "missing_leaf_activation_gradient",
                    }
                )
                continue
            node_scores.append({**base, **_node_saliency_from_activation_grad(activation, gradient, node)})
        return node_scores
    finally:
        handle.remove()
        model.zero_grad(set_to_none=True)


def compute_batch_path_saliency(
    model,
    prepared_batch: PreparedSampleBatch,
    candidate_path: CandidatePath,
    strict: bool = False,
) -> dict[str, Any]:
    resolved_nodes = resolve_candidate_path_targets(
        candidate_path=candidate_path,
        prepared_batch=prepared_batch,
        strict=strict,
        model=model,
    )
    if not resolved_nodes:
        return {
            "status": "no_patchable_nodes",
            "path_id": candidate_path.path_id,
            "num_patchable_nodes": 0,
            "num_scored_nodes": 0,
            "saliency": None,
            "saliency_sum": None,
            "fisher_saliency": None,
            "fisher_saliency_sum": None,
            "node_scores": [],
        }

    summary, missing = _compute_saliency_with_existing_graph(
        model=model,
        prepared_batch=prepared_batch,
        candidate_path=candidate_path,
        resolved_nodes=resolved_nodes,
    )
    if not missing:
        return summary

    missing_by_module: dict[str, list[tuple[int, ResolvedPathNode]]] = defaultdict(list)
    for module, node_index, neuron in missing:
        node = resolved_nodes[node_index]
        if node.module == module and node.neuron == neuron:
            missing_by_module[module].append((node_index, node))

    replacement_scores: dict[int, dict[str, Any]] = {}
    for nodes in missing_by_module.values():
        for node_score in _compute_leaf_saliency_for_module_nodes(
            model=model,
            prepared_batch=prepared_batch,
            candidate_path=candidate_path,
            nodes=nodes,
        ):
            replacement_scores[int(node_score["node_index"])] = node_score

    merged_scores = []
    for node_score in summary["node_scores"]:
        node_index = int(node_score["node_index"])
        merged_scores.append(replacement_scores.get(node_index, node_score))

    status = "ok" if all(score.get("status") == "ok" for score in merged_scores) else "partial_missing_gradients"
    return _summarize_node_scores(candidate_path, resolved_nodes, merged_scores, status)


def compute_path_saliency_specificity(
    model,
    forget_batch: PreparedSampleBatch,
    retain_batches: dict[str, PreparedSampleBatch],
    candidate_path: CandidatePath,
    strict: bool = False,
    gamma: float = 1.0,
    eps: float = 1e-6,
) -> dict[str, Any]:
    forget = compute_batch_path_saliency(
        model=model,
        prepared_batch=forget_batch,
        candidate_path=candidate_path,
        strict=strict,
    )
    if forget.get("saliency") is None:
        return {
            "status": forget.get("status"),
            "forget": forget,
            "retain_anchors": {},
            "forget_saliency": None,
            "retain_anchor_saliency": None,
            "saliency_specificity_margin": None,
            "saliency_specificity_ratio": None,
            "forget_fisher_saliency": None,
            "retain_anchor_fisher_saliency": None,
            "fisher_specificity_margin": None,
            "fisher_specificity_ratio": None,
            "gamma": gamma,
            "eps": eps,
        }

    retain_results = {
        retain_name: compute_batch_path_saliency(
            model=model,
            prepared_batch=retain_batch,
            candidate_path=candidate_path,
            strict=strict,
        )
        for retain_name, retain_batch in retain_batches.items()
    }
    retain_saliencies = [
        float(result["saliency"])
        for result in retain_results.values()
        if result.get("saliency") is not None
    ]
    retain_fisher_saliencies = [
        float(result["fisher_saliency"])
        for result in retain_results.values()
        if result.get("fisher_saliency") is not None
    ]

    forget_saliency = float(forget["saliency"])
    forget_fisher_saliency = float(forget["fisher_saliency"])
    retain_anchor_saliency = float(mean(retain_saliencies)) if retain_saliencies else 0.0
    retain_anchor_fisher_saliency = float(mean(retain_fisher_saliencies)) if retain_fisher_saliencies else 0.0
    return {
        "status": "ok" if retain_results else "ok_no_retain_anchors",
        "forget": forget,
        "retain_anchors": retain_results,
        "forget_saliency": forget_saliency,
        "retain_anchor_saliency": retain_anchor_saliency,
        "saliency_specificity_margin": forget_saliency - gamma * retain_anchor_saliency,
        "saliency_specificity_ratio": forget_saliency / (retain_anchor_saliency + eps),
        "forget_fisher_saliency": forget_fisher_saliency,
        "retain_anchor_fisher_saliency": retain_anchor_fisher_saliency,
        "fisher_specificity_margin": forget_fisher_saliency - gamma * retain_anchor_fisher_saliency,
        "fisher_specificity_ratio": forget_fisher_saliency / (retain_anchor_fisher_saliency + eps),
        "gamma": gamma,
        "eps": eps,
    }
