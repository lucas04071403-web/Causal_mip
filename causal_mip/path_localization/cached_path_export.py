"""
Export Step 2 candidate paths directly from cached IG/Fisher score tensors.
"""

from pathlib import Path
import torch

from .beam_search_paths import extract_vision_paths_from_mip_scores
from .cross_modal_path_builder import (
    build_cross_modal_paths_from_unimodal_paths,
    merge_paths_from_modalities,
)
from .mip_topk_wrapper import extract_text_paths_from_mip_scores


def _resolve_score_cache_paths(args) -> tuple[Path, Path]:
    base = Path(args.path_path)
    if "mllmu" in args.dataset and "Qwen" in args.model:
        text_path = base / f"text_ja_all_{args.dataset}_{args.model}.pt"
        vision_path = base / f"multi_fisher_all_{args.dataset}_{args.model}.pt"
    else:
        text_path = base / f"text_ja_forget{args.forget_ratio}_{args.dataset}_{args.model}.pt"
        vision_path = base / f"multi_fisher_forget{args.forget_ratio}_{args.dataset}_{args.model}.pt"
    return text_path, vision_path


def export_candidate_paths_from_cached_scores(args) -> dict:
    """Export text, vision, and cross-modal candidate paths from cached score tensors."""
    text_score_path, vision_score_path = _resolve_score_cache_paths(args)
    if not text_score_path.exists():
        raise FileNotFoundError(f"Missing text score cache: {text_score_path}")
    if not vision_score_path.exists():
        raise FileNotFoundError(f"Missing vision score cache: {vision_score_path}")

    text_scores = torch.load(text_score_path, weights_only=True).to(torch.float32)
    vision_scores = torch.load(vision_score_path, weights_only=True).to(torch.float32)

    text_paths = extract_text_paths_from_mip_scores(
        text_scores,
        num_candidates=args.candidate_num_paths,
        per_layer_topk=args.candidate_per_layer_topk,
        model_name=args.model,
    )
    vision_paths = extract_vision_paths_from_mip_scores(
        vision_scores,
        num_candidates=args.candidate_num_paths,
        per_layer_topk=args.candidate_per_layer_topk,
        model_name=args.model,
    )
    cross_modal_paths = build_cross_modal_paths_from_unimodal_paths(
        vision_paths=vision_paths,
        text_paths=text_paths,
        model_name=args.model,
        num_paths=args.candidate_cross_modal_paths,
    )

    output_path = Path(args.candidate_paths_output)
    all_paths = merge_paths_from_modalities(
        text_paths=text_paths,
        vision_paths=vision_paths,
        cross_modal_paths=cross_modal_paths,
        output_path=str(output_path),
    )
    return {
        "text_paths": len(text_paths),
        "vision_paths": len(vision_paths),
        "cross_modal_paths": len(cross_modal_paths),
        "all_paths": len(all_paths),
        "output_path": str(output_path),
    }
