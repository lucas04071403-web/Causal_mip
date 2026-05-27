"""
Cross-Modal Path Builder for Causal-MIP-Editor MVP

This module builds cross-modal bridge paths that connect vision encoder
neurons to LLM neurons through the multimodal projector.
"""

import os
import torch
from .path_schema import CandidatePath, PathNode, build_module_name_vision, build_module_name_llm, build_module_name_mm_projector


def build_cross_modal_paths(
    vision_neurons_per_layer: list[list[int]],
    llm_neurons_per_layer: list[list[int]],
    mm_projector_neurons: list[int],
    vision_layer_scores: list[float],
    llm_layer_scores: list[float],
    mm_projector_score: float,
    path_id_prefix: str = "cross_modal"
) -> list[CandidatePath]:
    """
    Build cross-modal bridge paths by connecting vision neurons to LLM neurons.

    The cross-modal path follows this structure:
        vision_encoder selected neuron
        → mm_projector selected neuron
        → LLM image token hidden state
        → LLM answer token MLP neuron

    Args:
        vision_neurons_per_layer: List of neuron indices per vision layer
        llm_neurons_per_layer: List of neuron indices per LLM layer
        mm_projector_neurons: List of mm_projector neuron indices
        vision_layer_scores: IGI/Fisher scores for vision neurons
        llm_layer_scores: IGI scores for LLM neurons
        mm_projector_score: Score for mm_projector neurons
        path_id_prefix: Prefix for path IDs

    Returns:
        List of CandidatePath objects representing cross-modal paths
    """
    candidate_paths = []
    path_counter = 0

    # Number of paths to generate from combinations
    num_cross_modal_paths = min(len(vision_neurons_per_layer) * len(llm_neurons_per_layer), 20)

    # Generate cross-modal paths by pairing top vision and LLM neurons
    torch.manual_seed(42)
    for _ in range(num_cross_modal_paths):
        nodes = []

        # 1. Vision encoder layer (select one neuron from top-k)
        vision_layer_idx = torch.randint(len(vision_neurons_per_layer), (1,)).item()
        vision_neurons = vision_neurons_per_layer[vision_layer_idx]
        vision_neuron = vision_neurons[torch.randint(len(vision_neurons), (1,)).item()]
        vision_score = vision_layer_scores[vision_layer_idx]

        nodes.append(PathNode(
            module=build_module_name_vision(vision_layer_idx),
            layer=vision_layer_idx,
            neuron=int(vision_neuron),
            token_selector="image_tokens"
        ))

        # 2. MM projector (single layer, select one neuron)
        if mm_projector_neurons:
            mm_neuron = mm_projector_neurons[torch.randint(len(mm_projector_neurons), (1,)).item()]
            nodes.append(PathNode(
                module=build_module_name_mm_projector(),
                layer=None,  # Projector is not layered like encoder
                neuron=int(mm_neuron),
                token_selector="image_tokens"
            ))

        # 3. LLM image token hidden state layer (early layer)
        llm_image_layer_idx = torch.randint(min(3, len(llm_neurons_per_layer)), (1,)).item()
        llm_image_neurons = llm_neurons_per_layer[llm_image_layer_idx]
        llm_image_neuron = llm_image_neurons[torch.randint(len(llm_image_neurons), (1,)).item()]
        llm_image_score = llm_layer_scores[llm_image_layer_idx]

        nodes.append(PathNode(
            module=build_module_name_llm(llm_image_layer_idx),
            layer=llm_image_layer_idx,
            neuron=int(llm_image_neuron),
            token_selector="image_tokens"
        ))

        # 4. LLM answer token MLP layer (later layer)
        llm_answer_layer_idx = len(llm_neurons_per_layer) - 1 - torch.randint(min(3, len(llm_neurons_per_layer)), (1,)).item()
        llm_answer_layer_idx = max(llm_answer_layer_idx, llm_image_layer_idx + 1)
        if llm_answer_layer_idx < len(llm_neurons_per_layer):
            llm_answer_neurons = llm_neurons_per_layer[llm_answer_layer_idx]
            llm_answer_neuron = llm_answer_neurons[torch.randint(len(llm_answer_neurons), (1,)).item()]
            llm_answer_score = llm_layer_scores[llm_answer_layer_idx]
        else:
            llm_answer_neuron = llm_neurons_per_layer[-1][0]
            llm_answer_score = llm_layer_scores[-1]

        nodes.append(PathNode(
            module=build_module_name_llm(llm_answer_layer_idx),
            layer=llm_answer_layer_idx,
            neuron=int(llm_answer_neuron),
            token_selector="answer_tokens"
        ))

        # Calculate combined cross-modal score
        combined_score = (vision_score + llm_image_score + llm_answer_score) / 3

        cross_modal_path = CandidatePath(
            path_id=f"{path_id_prefix}_p{path_counter:06d}",
            source="cross_modal",
            modality="vision_text",
            mip_score=combined_score,
            nodes=nodes
        )
        candidate_paths.append(cross_modal_path)
        path_counter += 1

    return candidate_paths


def build_simple_cross_modal_paths(
    vision_top_neurons: list[int],
    llm_top_neurons: list[int],
    num_paths: int = 10
) -> list[CandidatePath]:
    """
    Build simple cross-modal paths with fixed structure.

    This is a simplified version that assumes we already have the top neurons
    identified and just need to create path combinations.

    Args:
        vision_top_neurons: Top vision encoder neurons (flattened: layer_idx * neurons_per_layer + neuron_idx)
        llm_top_neurons: Top LLM neurons (flattened: layer_idx * neurons_per_layer + neuron_idx)
        num_paths: Number of cross-modal paths to generate

    Returns:
        List of CandidatePath objects
    """
    candidate_paths = []

    for i in range(num_paths):
        torch.manual_seed(i)
        nodes = []

        # Vision encoder: pick one neuron
        vision_neuron = vision_top_neurons[i % len(vision_top_neurons)]
        vision_layer = vision_neuron // 1000 if vision_neuron >= 1000 else 0  # Rough approximation
        vision_actual_neuron = vision_neuron % 1000

        nodes.append(PathNode(
            module=build_module_name_vision(vision_layer),
            layer=vision_layer,
            neuron=int(vision_actual_neuron),
            token_selector="image_tokens"
        ))

        # MM projector (placeholder - actual neurons are hard to access)
        nodes.append(PathNode(
            module=build_module_name_mm_projector(),
            layer=None,
            neuron=int(i * 100),  # Placeholder
            token_selector="image_tokens"
        ))

        # LLM early layer (image token processing)
        llm_early_neuron = llm_top_neurons[i % len(llm_top_neurons)]
        llm_early_layer = llm_early_neuron // 1000 if llm_early_neuron >= 1000 else 0
        llm_early_actual_neuron = llm_early_neuron % 1000

        nodes.append(PathNode(
            module=build_module_name_llm(llm_early_layer),
            layer=llm_early_layer,
            neuron=int(llm_early_actual_neuron),
            token_selector="image_tokens"
        ))

        # LLM late layer (answer token processing)
        llm_late_neuron = llm_top_neurons[(i + 5) % len(llm_top_neurons)]
        llm_late_layer = llm_late_neuron // 1000 if llm_late_neuron >= 1000 else len(range(28)) - 1
        llm_late_actual_neuron = llm_late_neuron % 1000

        nodes.append(PathNode(
            module=build_module_name_llm(llm_late_layer),
            layer=llm_late_layer,
            neuron=int(llm_late_actual_neuron),
            token_selector="answer_tokens"
        ))

        path = CandidatePath(
            path_id=f"cross_modal_p{i:06d}",
            source="cross_modal",
            modality="vision_text",
            mip_score=1.0 - (i * 0.05),  # Decreasing score for generated paths
            nodes=nodes
        )
        candidate_paths.append(path)

    return candidate_paths


def build_cross_modal_paths_from_unimodal_paths(
    vision_paths: list[CandidatePath],
    text_paths: list[CandidatePath],
    model_name: str,
    num_paths: int = 20,
) -> list[CandidatePath]:
    """Build bridge paths by pairing high-scoring vision and text candidate paths."""
    if not vision_paths or not text_paths:
        return []

    sorted_vision = sorted(vision_paths, key=lambda p: p.mip_score, reverse=True)
    sorted_text = sorted(text_paths, key=lambda p: p.mip_score, reverse=True)
    num_pairs = min(num_paths, len(sorted_vision), len(sorted_text))

    candidate_paths = []
    for idx in range(num_pairs):
        vision_path = sorted_vision[idx]
        text_path = sorted_text[idx]
        nodes = [
            PathNode(
                module=vision_path.nodes[0].module,
                layer=vision_path.nodes[0].layer,
                neuron=vision_path.nodes[0].neuron,
                token_selector="image_tokens",
            ),
            PathNode(
                module=build_module_name_mm_projector(),
                layer=None,
                neuron=0,
                token_selector="image_tokens",
            ),
            PathNode(
                module=text_path.nodes[0].module,
                layer=text_path.nodes[0].layer,
                neuron=text_path.nodes[0].neuron,
                token_selector="image_tokens",
            ),
            PathNode(
                module=text_path.nodes[-1].module,
                layer=text_path.nodes[-1].layer,
                neuron=text_path.nodes[-1].neuron,
                token_selector="answer_tokens",
            ),
        ]
        candidate_paths.append(CandidatePath(
            path_id=f"cross_modal_p{idx:06d}",
            source="cross_modal",
            modality="vision_text",
            mip_score=(vision_path.mip_score + text_path.mip_score) / 2.0,
            nodes=nodes,
            source_sample_idx=vision_path.source_sample_idx,
            metadata={
                "vision_path_id": vision_path.path_id,
                "text_path_id": text_path.path_id,
                "vision_source_sample_idx": vision_path.source_sample_idx,
                "text_source_sample_idx": text_path.source_sample_idx,
                "source_sample_idx": vision_path.source_sample_idx,
            },
        ))

    return candidate_paths


def merge_paths_from_modalities(
    text_paths: list[CandidatePath],
    vision_paths: list[CandidatePath],
    cross_modal_paths: list[CandidatePath],
    output_path: str
) -> list[CandidatePath]:
    """
    Merge paths from different modalities and save to file.

    Args:
        text_paths: Text modality candidate paths
        vision_paths: Vision modality candidate paths
        cross_modal_paths: Cross-modal candidate paths
        output_path: Path to save merged paths

    Returns:
        Merged list of all candidate paths
    """
    import json

    all_paths = text_paths + vision_paths + cross_modal_paths

    counters = {
        "mip_editor_igi": 0,
        "mip_editor_ifi": 0,
        "cross_modal": 0,
    }
    prefixes = {
        "mip_editor_igi": "text_igi_p",
        "mip_editor_ifi": "vision_fisher_p",
        "cross_modal": "cross_modal_p",
    }
    for path in all_paths:
        prefix = prefixes[path.source]
        path.path_id = f"{prefix}{counters[path.source]:06d}"
        counters[path.source] += 1

    # Save to JSONL format
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for path in all_paths:
            f.write(json.dumps(path.to_dict(), ensure_ascii=False) + '\n')

    print(f"Saved {len(all_paths)} candidate paths to {output_path}")
    return all_paths


def load_candidate_paths(path: str) -> list[CandidatePath]:
    """
    Load candidate paths from JSONL file.

    Args:
        path: Path to the JSONL file

    Returns:
        List of CandidatePath objects
    """
    import json

    paths = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            paths.append(CandidatePath.from_dict(data))

    print(f"Loaded {len(paths)} candidate paths from {path}")
    return paths
