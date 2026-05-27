"""
Top-k IGI Path Extraction for Text Modality

This module wraps the original IGI computation to extract top-k candidate paths
instead of just the greedy best path.
"""

import torch
from tqdm import tqdm
from .path_schema import CandidatePath, PathNode, build_module_name_llm, select_token_selector


def _generate_candidate_paths_from_sample_scores(
    sample_scores: torch.Tensor,
    path_prefix: str,
    source: str,
    modality: str,
    model_name: str,
    num_candidates: int,
    per_layer_topk: int,
    path_counter_start: int = 0,
    source_sample_idx: int | None = None,
) -> tuple[list[CandidatePath], int]:
    """Create one-neuron-per-layer candidate paths from a (layers, neurons) score tensor."""
    if sample_scores.dim() != 2:
        raise ValueError(f"Expected (layers, neurons) scores, got {tuple(sample_scores.shape)}")

    sample_scores = sample_scores.to(torch.float32)
    num_layers = sample_scores.shape[0]
    per_layer_topk = max(1, min(per_layer_topk, sample_scores.shape[1]))
    num_candidates = max(1, num_candidates)

    layer_top_values = []
    layer_top_indices = []
    for layer_idx in range(num_layers):
        values, indices = torch.topk(sample_scores[layer_idx], k=per_layer_topk, dim=-1)
        layer_top_values.append(values)
        layer_top_indices.append(indices)

    def build_nodes(rank_choices: list[int]) -> tuple[list[PathNode], float]:
        nodes = []
        selected_scores = []
        for layer_idx, rank in enumerate(rank_choices):
            rank = min(rank, layer_top_indices[layer_idx].numel() - 1)
            neuron_idx = int(layer_top_indices[layer_idx][rank].item())
            selected_scores.append(float(layer_top_values[layer_idx][rank].item()))
            nodes.append(PathNode(
                module=build_module_name_llm(layer_idx, model_name),
                layer=layer_idx,
                neuron=neuron_idx,
                token_selector=select_token_selector(layer_idx, num_layers),
            ))
        return nodes, sum(selected_scores) / max(1, len(selected_scores))

    candidate_paths = []
    seen_combinations = set()
    rank_plans = [[0] * num_layers]

    for beam_rank in range(1, min(per_layer_topk, 4)):
        rank_plans.append([beam_rank] * num_layers)

    for layer_idx in range(num_layers):
        for beam_rank in range(1, min(per_layer_topk, 4)):
            if len(rank_plans) >= num_candidates:
                break
            varied_plan = [0] * num_layers
            varied_plan[layer_idx] = beam_rank
            rank_plans.append(varied_plan)
        if len(rank_plans) >= num_candidates:
            break

    generator = torch.Generator().manual_seed(42)
    while len(rank_plans) < num_candidates:
        rank_plans.append([
            int(torch.randint(per_layer_topk, (1,), generator=generator).item())
            for _ in range(num_layers)
        ])

    path_counter = path_counter_start
    for plan in rank_plans:
        nodes, mip_score = build_nodes(plan)
        combination = tuple(node.neuron for node in nodes)
        if combination in seen_combinations:
            continue
        seen_combinations.add(combination)
        candidate_paths.append(CandidatePath(
            path_id=f"{path_prefix}{path_counter:06d}",
            source=source,
            modality=modality,
            mip_score=mip_score,
            nodes=nodes,
            source_sample_idx=source_sample_idx,
            metadata={"source_sample_idx": source_sample_idx} if source_sample_idx is not None else {},
        ))
        path_counter += 1
        if len(candidate_paths) >= num_candidates:
            break

    return candidate_paths, path_counter


def compute_ig_topk_paths(
    inputs: dict,
    model,
    ig_total_step: int,
    top_k: int,
    batch_size: int,
    model_name: str,
    num_candidates: int = 10,
    per_layer_topk: int = 5
) -> list[CandidatePath]:
    """
    Compute IGI scores and extract top-k candidate paths per layer.

    Unlike the original compute_ig_path which only returns top-k neurons per layer,
    this function:
    1. Computes the full IGI attribution scores for all neurons
    2. Extracts top-k neurons per layer
    3. Generates multiple candidate paths by combining different neurons across layers

    Args:
        inputs: Model inputs dict
        model: The model
        ig_total_step: Number of steps for integrated gradients
        top_k: Number of top neurons to select per layer (for MIP-Editor compatibility)
        batch_size: Batch size
        model_name: Model name string
        num_candidates: Number of candidate paths to generate
        per_layer_topk: Number of top neurons to consider per layer when building paths

    Returns:
        List of CandidatePath objects
    """
    from ig import (
        collect_activations, scaled_input, path_scaled_input,
        forward_with_scaled_inputs, calc_per_layer_grad
    )

    # Collect activations from original forward pass
    original_ffn_activations, model_output = collect_activations(model, inputs)

    if "llava" in model_name or "gemma" in model_name:
        original_ffn_activations = [
            original_ffn_activations[layer].detach()[:, 0, :]
            for layer in original_ffn_activations
        ]
    if "Qwen" in model_name:
        original_ffn_activations = [
            original_ffn_activations[layer].detach()[:, 0]
            for layer in original_ffn_activations
        ]

    num_layers = len(original_ffn_activations)

    # Compute IGI scores for all neurons
    original_embedding = model_output.hidden_states[0]
    repeated_embedding = original_embedding.repeat(ig_total_step, 1, 1)
    new_inputs = {
        **inputs,
        'inputs_embeds': repeated_embedding,
        'labels': inputs['labels'].repeat(ig_total_step, 1).detach()
    }
    if "llava" in model.config._name_or_path or "gemma" in model.config._name_or_path:
        new_inputs["attention_mask"] = inputs['attention_mask'].repeat(ig_total_step, 1).detach()
        if 'input_ids' in new_inputs:
            del new_inputs['input_ids']

    ja_list = []  # Store IGI scores per layer: list[(batch, neurons)]
    path = []     # Greedy path used by the original IGI recursion

    for layer in range(num_layers):
        cur_layer_act = original_ffn_activations[layer]
        prev_layer_acts = original_ffn_activations[:layer]

        scaled_weights, weights_step = scaled_input(cur_layer_act, ig_total_step)
        path_scaled_weights, path_weights_steps = path_scaled_input(prev_layer_acts, path, ig_total_step)

        path_scaled_weights.append(scaled_weights)
        path_weights_steps.append(weights_step)
        for p in path_scaled_weights:
            p.requires_grad_(True)

        loss, new_ffn_acts = forward_with_scaled_inputs(model, new_inputs, path_scaled_weights)
        new_ffn_acts = new_ffn_acts[:layer+1]
        per_layer_grads = calc_per_layer_grad(loss, new_ffn_acts)
        per_layer_grads = [g[:, 0] for g in per_layer_grads]
        model.zero_grad()

        ig_pred = []
        for grad in per_layer_grads:
            ig_pred.append(grad.reshape(ig_total_step, batch_size, -1))

        ja = ig_pred[-1]
        s = path_weights_steps[-1]
        for l, p in enumerate(path):
            temp = torch.zeros_like(ig_pred[l][:, :, [0]])
            for i, b in enumerate(p):
                temp[:, i] = ig_pred[l][:, i, [b]]
            ja = ja + temp
        ja = ja.sum(dim=0)
        for l, p in enumerate(path):
            temp_s = torch.zeros_like(path_weights_steps[l][:, [0]])
            for i, b in enumerate(p):
                temp_s[i] = path_weights_steps[l][i, [b]]
            s = s + temp_s
        ja = ja.cpu() * s.cpu()
        if ja.dim() == 1:
            ja = ja.unsqueeze(0)
        ja_list.append(ja)

        # Get top-k indices for greedy path (MIP-Editor compatibility)
        _, _indices = torch.topk(ja, top_k, dim=-1)
        path.append(_indices.tolist())

    score_tensor = torch.stack(ja_list, dim=0).permute(1, 0, 2)
    candidate_paths = []
    path_counter = 0
    for sample_idx in range(score_tensor.shape[0]):
        sample_paths, path_counter = _generate_candidate_paths_from_sample_scores(
            sample_scores=score_tensor[sample_idx],
            path_prefix="text_igi_p",
            source="mip_editor_igi",
            modality="text",
            model_name=model_name,
            num_candidates=num_candidates,
            per_layer_topk=per_layer_topk,
            path_counter_start=path_counter,
            source_sample_idx=sample_idx,
        )
        candidate_paths.extend(sample_paths)
    return candidate_paths


def calculate_ig_topk_batch(
    model,
    data_loader,
    topk: int,
    ig_total_step: int,
    num_candidates_per_sample: int = 10,
    per_layer_topk: int = 5
) -> list[CandidatePath]:
    """
    Calculate top-k candidate paths for a batch of text samples.

    Args:
        model: The model
        data_loader: DataLoader for text samples
        topk: Number of top neurons per layer (MIP-Editor compatibility)
        ig_total_step: Number of steps for IGI
        num_candidates_per_sample: Number of candidate paths to generate per sample
        per_layer_topk: Number of top neurons to consider per layer

    Returns:
        List of all candidate paths from all samples
    """
    device = next(model.parameters()).device
    all_candidate_paths = []

    for batch in tqdm(data_loader, desc="Computing IGI top-k paths"):
        input_ids, attention_mask, labels, _ = batch
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        text_inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
        batch_size = input_ids.shape[0]

        candidate_paths = compute_ig_topk_paths(
            text_inputs,
            model,
            ig_total_step,
            topk,
            batch_size,
            model.config._name_or_path,
            num_candidates=num_candidates_per_sample,
            per_layer_topk=per_layer_topk
        )
        all_candidate_paths.extend(candidate_paths)

        torch.cuda.empty_cache()

    return all_candidate_paths


def extract_text_paths_from_mip_scores(
    mip_scores: torch.Tensor,
    num_candidates: int = 10,
    per_layer_topk: int = 5,
    model_name: str = "Qwen2.5-VL-3B-Instruct",
) -> list[CandidatePath]:
    """
    Extract candidate paths from pre-computed MIP scores.

    This is useful when you have pre-computed IGI scores from MIP-Editor
    and want to generate candidate paths without recomputing.

    Args:
        mip_scores: Tensor of shape (num_layers, num_samples, num_neurons)
        num_candidates: Number of candidate paths to generate
        per_layer_topk: Number of top neurons to consider per layer

    Returns:
        List of CandidatePath objects
    """
    if mip_scores.dim() != 3:
        raise ValueError(f"Expected (samples, layers, neurons) scores, got {tuple(mip_scores.shape)}")

    candidate_paths = []
    path_counter = 0

    for sample_idx in range(mip_scores.shape[0]):
        sample_paths, path_counter = _generate_candidate_paths_from_sample_scores(
            sample_scores=mip_scores[sample_idx],
            path_prefix="text_igi_p",
            source="mip_editor_igi",
            modality="text",
            model_name=model_name,
            num_candidates=num_candidates,
            per_layer_topk=per_layer_topk,
            path_counter_start=path_counter,
            source_sample_idx=sample_idx,
        )
        candidate_paths.extend(sample_paths)

    return candidate_paths
