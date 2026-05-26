"""
Path Schema Definition for Causal-MIP-Editor MVP

This module defines the standardized path schema for representing
candidate influential paths from MIP-Editor.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional
import json


@dataclass
class PathNode:
    """A single node in a path, representing a neuron in the model."""
    module: str           # e.g., "vision_encoder.blocks.12.mlp"
    layer: Optional[int]  # Layer index, None for non-layered modules
    neuron: int           # Neuron index within the layer
    token_selector: str   # "image_tokens" | "answer_tokens" | "all_tokens"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> 'PathNode':
        return cls(**data)


@dataclass
class CandidatePath:
    """A candidate influential path from MIP-Editor."""
    path_id: str
    source: str           # "mip_editor_igi" | "mip_editor_ifi" | "cross_modal"
    modality: str          # "vision" | "text" | "vision_text"
    mip_score: float      # Original MIP score (IGI or Fisher)
    nodes: list[PathNode] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "path_id": self.path_id,
            "source": self.source,
            "modality": self.modality,
            "mip_score": self.mip_score,
            "nodes": [n.to_dict() for n in self.nodes]
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'CandidatePath':
        data = data.copy()
        data['nodes'] = [PathNode.from_dict(n) for n in data['nodes']]
        return cls(**data)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> 'CandidatePath':
        return cls.from_dict(json.loads(json_str))


def build_module_name_vision(layer: int, model_name: Optional[str] = None) -> str:
    """Build the concrete vision hook module name used for intervention."""
    if model_name and "Qwen2.5-" in model_name:
        return f"model.visual.blocks.{layer}.mlp.down_proj"
    if model_name and "Qwen2-" in model_name:
        return f"visual.blocks.{layer}.mlp.fc2"
    if model_name and ("llava" in model_name or "gemma" in model_name):
        return f"model.vision_tower.vision_model.encoder.layers.{layer}.mlp.fc2"
    return f"model.visual.blocks.{layer}.mlp.down_proj"


def build_module_name_llm(layer: int, model_name: Optional[str] = None) -> str:
    """Build the concrete language-model hook module name used for intervention."""
    if model_name and "Qwen2.5-" in model_name:
        return f"model.language_model.layers.{layer}.mlp.down_proj"
    if model_name and "Qwen2-" in model_name:
        return f"model.layers.{layer}.mlp.down_proj"
    if model_name and "llava" in model_name:
        return f"language_model.model.layers.{layer}.mlp.down_proj"
    if model_name and "gemma" in model_name:
        return f"language_model.layers.{layer}.mlp.down_proj"
    return f"model.language_model.layers.{layer}.mlp.down_proj"


def build_module_name_mm_projector() -> str:
    """Build module name for multimodal projector."""
    return "mm_projector"


def select_token_selector(layer: int, num_layers: int, is_answer_token: bool = False) -> str:
    """
    Determine token selector based on layer position.

    Args:
        layer: Current layer index
        num_layers: Total number of layers
        is_answer_token: Whether this layer operates on answer tokens

    Returns:
        "image_tokens" for early layers (image processing)
        "answer_tokens" for later layers (answer generation)
    """
    if is_answer_token:
        return "answer_tokens"
    # Early layers focus on image tokens, later layers on answer
    mid_point = num_layers // 2
    if layer < mid_point:
        return "image_tokens"
    return "answer_tokens"
