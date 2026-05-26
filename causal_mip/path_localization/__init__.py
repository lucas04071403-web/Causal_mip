"""
Causal-MIP-Editor: Path Localization Module

This module provides functionality for extracting top-k candidate paths
from MIP-Editor instead of just the greedy best path.
"""

from .path_schema import (
    CandidatePath,
    PathNode,
    build_module_name_vision,
    build_module_name_llm,
    build_module_name_mm_projector,
    select_token_selector,
)

from .mip_topk_wrapper import (
    compute_ig_topk_paths,
    calculate_ig_topk_batch,
    extract_text_paths_from_mip_scores,
)

from .beam_search_paths import (
    compute_fisher_topk_paths,
    calculate_fisher_topk_batch,
    extract_vision_paths_from_mip_scores,
)

from .cross_modal_path_builder import (
    build_cross_modal_paths,
    build_simple_cross_modal_paths,
    build_cross_modal_paths_from_unimodal_paths,
    merge_paths_from_modalities,
    load_candidate_paths,
)

from .cached_path_export import export_candidate_paths_from_cached_scores

__all__ = [
    # Schema
    "CandidatePath",
    "PathNode",
    "build_module_name_vision",
    "build_module_name_llm",
    "build_module_name_mm_projector",
    "select_token_selector",
    # IGI
    "compute_ig_topk_paths",
    "calculate_ig_topk_batch",
    "extract_text_paths_from_mip_scores",
    # Fisher
    "compute_fisher_topk_paths",
    "calculate_fisher_topk_batch",
    "extract_vision_paths_from_mip_scores",
    # Cross-modal
    "build_cross_modal_paths",
    "build_simple_cross_modal_paths",
    "build_cross_modal_paths_from_unimodal_paths",
    "merge_paths_from_modalities",
    "load_candidate_paths",
    "export_candidate_paths_from_cached_scores",
]
