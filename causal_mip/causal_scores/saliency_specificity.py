from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Any

import torch

from causal_mip.causal_scores.name_token_metrics import find_name_token_positions
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
    max_dim_scores: int = 0,
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
    result = {
        "saliency": float(grad_act.mean().detach().cpu().item()),
        "fisher_saliency": float(fisher.mean().detach().cpu().item()),
        "num_token_positions": len(token_positions),
        "status": "ok",
    }
    if node.neuron == WHOLE_VECTOR_NEURON and max_dim_scores > 0 and grad_act.ndim >= 2:
        reduce_dims = tuple(range(grad_act.ndim - 1))
        dim_saliency = grad_act.mean(dim=reduce_dims)
        dim_fisher = fisher.mean(dim=reduce_dims)
        k = min(int(max_dim_scores), int(dim_saliency.numel()))
        if k > 0:
            top_values, top_indices = torch.topk(dim_saliency, k=k)
            result["dim_scores"] = [
                {
                    "dim_index": int(dim_index.detach().cpu().item()),
                    "saliency": float(value.detach().cpu().item()),
                    "fisher_saliency": float(dim_fisher[int(dim_index)].detach().cpu().item()),
                    "status": "ok",
                }
                for value, dim_index in zip(top_values, top_indices)
            ]
            result["num_dim_scores"] = k
    return result


def compute_target_name_logprob(
    logits: torch.Tensor,
    prepared_batch: PreparedSampleBatch,
    processor_or_tokenizer,
    reduction: str = "mean",
) -> torch.Tensor:
    target_name = prepared_batch.sample.get("name", "")
    location = find_name_token_positions(
        processor_or_tokenizer=processor_or_tokenizer,
        input_ids=prepared_batch.input_ids,
        answer_positions=list(prepared_batch.answer_token_positions),
        target_text=prepared_batch.target_answer_text,
        name_text=target_name,
    )
    positions = [
        position
        for position in location["name_token_positions"]
        if position > 0 and position < prepared_batch.input_ids.shape[1]
    ]
    if not positions:
        return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

    log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
    contributions = []
    for position in positions:
        score_position = position - 1
        token_id = prepared_batch.input_ids[:, position]
        contributions.append(log_probs[:, score_position, :].gather(-1, token_id.unsqueeze(-1)).squeeze(-1))
    stacked = torch.stack(contributions, dim=0)
    if reduction == "sum":
        return stacked.sum()
    return stacked.mean()


def _compute_saliency_score(
    logits: torch.Tensor,
    prepared_batch: PreparedSampleBatch,
    target: str,
    processor_or_tokenizer=None,
) -> torch.Tensor:
    if target == "answer":
        return compute_target_answer_logprob(logits, prepared_batch)
    if target == "target_name":
        if processor_or_tokenizer is None:
            raise ValueError("target_name saliency requires processor_or_tokenizer.")
        return compute_target_name_logprob(
            logits=logits,
            prepared_batch=prepared_batch,
            processor_or_tokenizer=processor_or_tokenizer,
        )
    raise ValueError(f"Unsupported saliency target: {target}")


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
    max_dim_scores: int = 0,
    target: str = "answer",
    processor_or_tokenizer=None,
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
            score = _compute_saliency_score(
                logits=outputs.logits,
                prepared_batch=prepared_batch,
                target=target,
                processor_or_tokenizer=processor_or_tokenizer,
            )
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
            node_scores.append({**base, **_node_saliency_from_activation_grad(activation, gradient, node, max_dim_scores)})

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
    max_dim_scores: int = 0,
    target: str = "answer",
    processor_or_tokenizer=None,
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
            score = _compute_saliency_score(
                logits=outputs.logits,
                prepared_batch=prepared_batch,
                target=target,
                processor_or_tokenizer=processor_or_tokenizer,
            )
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
            node_scores.append({**base, **_node_saliency_from_activation_grad(activation, gradient, node, max_dim_scores)})
        return node_scores
    finally:
        handle.remove()
        model.zero_grad(set_to_none=True)


def compute_batch_path_saliency(
    model,
    prepared_batch: PreparedSampleBatch,
    candidate_path: CandidatePath,
    strict: bool = False,
    max_dim_scores: int = 0,
    target: str = "answer",
    processor_or_tokenizer=None,
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
        max_dim_scores=max_dim_scores,
        target=target,
        processor_or_tokenizer=processor_or_tokenizer,
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
            max_dim_scores=max_dim_scores,
            target=target,
            processor_or_tokenizer=processor_or_tokenizer,
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
    max_dim_scores: int = 0,
    target: str = "answer",
    processor_or_tokenizer=None,
) -> dict[str, Any]:
    forget = compute_batch_path_saliency(
        model=model,
        prepared_batch=forget_batch,
        candidate_path=candidate_path,
        strict=strict,
        max_dim_scores=max_dim_scores,
        target=target,
        processor_or_tokenizer=processor_or_tokenizer,
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
            max_dim_scores=max_dim_scores,
            target=target,
            processor_or_tokenizer=processor_or_tokenizer,
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
    max_anchor_retain_saliency = max(retain_saliencies) if retain_saliencies else retain_anchor_saliency
    max_anchor_retain_fisher_saliency = (
        max(retain_fisher_saliencies) if retain_fisher_saliencies else retain_anchor_fisher_saliency
    )
    per_anchor_saliency = {
        retain_name: float(result["saliency"])
        for retain_name, result in retain_results.items()
        if result.get("saliency") is not None
    }
    per_anchor_fisher_saliency = {
        retain_name: float(result["fisher_saliency"])
        for retain_name, result in retain_results.items()
        if result.get("fisher_saliency") is not None
    }
    per_anchor_margin = {
        retain_name: forget_saliency - gamma * retain_saliency
        for retain_name, retain_saliency in per_anchor_saliency.items()
    }
    per_anchor_ratio = {
        retain_name: forget_saliency / (retain_saliency + eps)
        for retain_name, retain_saliency in per_anchor_saliency.items()
    }
    per_anchor_fisher_margin = {
        retain_name: forget_fisher_saliency - gamma * retain_fisher
        for retain_name, retain_fisher in per_anchor_fisher_saliency.items()
    }
    per_anchor_fisher_ratio = {
        retain_name: forget_fisher_saliency / (retain_fisher + eps)
        for retain_name, retain_fisher in per_anchor_fisher_saliency.items()
    }
    min_anchor_margin = min(per_anchor_margin.values()) if per_anchor_margin else forget_saliency - gamma * retain_anchor_saliency
    min_anchor_ratio = min(per_anchor_ratio.values()) if per_anchor_ratio else forget_saliency / (retain_anchor_saliency + eps)
    min_anchor_fisher_margin = (
        min(per_anchor_fisher_margin.values())
        if per_anchor_fisher_margin
        else forget_fisher_saliency - gamma * retain_anchor_fisher_saliency
    )
    min_anchor_fisher_ratio = (
        min(per_anchor_fisher_ratio.values())
        if per_anchor_fisher_ratio
        else forget_fisher_saliency / (retain_anchor_fisher_saliency + eps)
    )
    return {
        "status": "ok" if retain_results else "ok_no_retain_anchors",
        "forget": forget,
        "retain_anchors": retain_results,
        "forget_saliency": forget_saliency,
        "retain_anchor_saliency": retain_anchor_saliency,
        "saliency_specificity_margin": forget_saliency - gamma * retain_anchor_saliency,
        "saliency_specificity_ratio": forget_saliency / (retain_anchor_saliency + eps),
        "max_anchor_retain_saliency": max_anchor_retain_saliency,
        "min_anchor_margin": min_anchor_margin,
        "min_anchor_ratio": min_anchor_ratio,
        "retain_anchor_margins": per_anchor_margin,
        "retain_anchor_ratios": per_anchor_ratio,
        "forget_fisher_saliency": forget_fisher_saliency,
        "retain_anchor_fisher_saliency": retain_anchor_fisher_saliency,
        "fisher_specificity_margin": forget_fisher_saliency - gamma * retain_anchor_fisher_saliency,
        "fisher_specificity_ratio": forget_fisher_saliency / (retain_anchor_fisher_saliency + eps),
        "max_anchor_retain_fisher_saliency": max_anchor_retain_fisher_saliency,
        "min_anchor_fisher_margin": min_anchor_fisher_margin,
        "min_anchor_fisher_ratio": min_anchor_fisher_ratio,
        "retain_anchor_fisher_margins": per_anchor_fisher_margin,
        "retain_anchor_fisher_ratios": per_anchor_fisher_ratio,
        "gamma": gamma,
        "eps": eps,
        "target": target,
    }
