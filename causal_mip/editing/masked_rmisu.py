from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal
import itertools

import torch
from torch import nn
from tqdm import tqdm
from transformers import get_scheduler

from causal_mip.interventions.hooks import get_module
from causal_mip.path_localization.path_schema import CandidatePath
from partial_linear import PartialLinear

ProjectorEditMode = Literal["skip", "qwen_merger_mlp"]
WHOLE_VECTOR_NEURON = -1


@dataclass
class PathNeuronMask:
    module: str
    layer: int | None = None
    module_kind: str = "llm"
    forget_neurons: set[int] = field(default_factory=set)
    shared_neurons: set[int] = field(default_factory=set)
    probe_neurons: set[int] = field(default_factory=set)
    forget_path_ids: set[str] = field(default_factory=set)
    shared_path_ids: set[str] = field(default_factory=set)
    probe_path_ids: set[str] = field(default_factory=set)
    trace_module: str | None = None
    trace_kind: Literal["input", "output"] = "input"
    active: bool = False

    @property
    def editable_neurons(self) -> set[int]:
        return set(self.forget_neurons) - set(self.shared_neurons)

    @property
    def trainable_neurons(self) -> set[int]:
        return set(self.editable_neurons) | set(self.probe_neurons)

    @property
    def preserve_neurons(self) -> set[int]:
        return set(self.forget_neurons) | set(self.shared_neurons) | set(self.probe_neurons)

    def to_summary(self) -> dict[str, Any]:
        return {
            "module": self.module,
            "layer": self.layer,
            "module_kind": self.module_kind,
            "num_forget_neurons": len(self.forget_neurons),
            "num_shared_neurons": len(self.shared_neurons),
            "num_probe_neurons": len(self.probe_neurons),
            "num_forget_editable_neurons": len(self.editable_neurons),
            "num_editable_neurons": len(self.trainable_neurons),
            "num_preserve_neurons": len(self.preserve_neurons),
            "forget_path_ids": sorted(self.forget_path_ids),
            "shared_path_ids": sorted(self.shared_path_ids),
            "probe_path_ids": sorted(self.probe_path_ids),
            "trace_module": self.trace_module,
            "trace_kind": self.trace_kind,
            "active": self.active,
        }


@dataclass
class MaskedRMisUConfig:
    candidate_paths_path: str
    p_forget_path: str
    p_shared_path: str | None = None
    p_probe_path: str | None = None
    alpha: float = 1.0
    beta: float = 1.0
    probe_beta: float = 0.05
    shared_alpha: float = 1.0
    forget_objective: str = "activation_random"
    forget_ce_alpha: float = 0.0
    preference_positive_alpha: float = 0.1
    bounded_delta_l2_alpha: float = 0.0
    bounded_delta_max_norm: float | None = None
    pii_noise_alpha: float = 0.0
    target_ce_scope: str = "all"
    counterfactual_anchor_alpha: float = 0.0
    counterfactual_anchor_scope: str = "name"
    projector_edit_mode: ProjectorEditMode = "qwen_merger_mlp"
    steering_coeff: float = 6.5
    coeffs: float = 1.0
    probe_steering_coeff: float = 1.0
    probe_coeffs: float = 1.0
    learning_rate: float = 1e-5
    epochs: int = 1
    use_shared_preserve: bool = True
    use_retain_preserve: bool = True
    output_path: str | None = None
    checkpoint_dir: str | None = None
    save: bool = True
    max_steps: int | None = None


def _read_jsonl(path: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_step6_path_ids(path: str | None) -> set[str]:
    if path is None:
        return set()
    return {record["path_id"] for record in _read_jsonl(path) if record.get("path_id") is not None}


def load_candidate_paths_by_id(path: str) -> dict[str, CandidatePath]:
    candidates: dict[str, CandidatePath] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                candidate = CandidatePath.from_dict(json.loads(line))
                candidates[candidate.path_id] = candidate
    return candidates


def _is_patchable_llm_down_proj(module_name: str) -> bool:
    allowed_prefixes = (
        "model.language_model.layers.",
        "base_model.model.language_model.layers.",
        "language_model.layers.",
        "model.layers.",
    )
    return module_name.startswith(allowed_prefixes) and module_name.endswith(".mlp.down_proj")


def _is_patchable_vision_down_proj(module_name: str) -> bool:
    allowed_prefixes = (
        "model.visual.blocks.",
        "base_model.model.visual.blocks.",
        "visual.blocks.",
    )
    return module_name.startswith(allowed_prefixes) and module_name.endswith(".mlp.down_proj")


def _patchable_down_proj_kind(module_name: str) -> str | None:
    if _is_patchable_llm_down_proj(module_name):
        return "llm"
    if _is_patchable_vision_down_proj(module_name):
        return "vision"
    if module_name in {
        "mm_projector",
        "model.mm_projector",
        "base_model.model.mm_projector",
        "visual.merger",
        "model.visual.merger",
        "base_model.model.visual.merger",
        "base_model.model.model.visual.merger",
    }:
        return "projector"
    return None


def _candidate_projector_uses_dim_level(candidate: CandidatePath) -> bool:
    return bool((candidate.metadata or {}).get("projector_dim_level", False))


def _mask_neuron_for_node(candidate: CandidatePath, node_module_kind: str, neuron: int) -> int:
    if node_module_kind == "projector" and not _candidate_projector_uses_dim_level(candidate):
        return WHOLE_VECTOR_NEURON
    return int(neuron)


def _expand_neurons_for_width(neurons: set[int] | Iterable[int], width: int) -> set[int]:
    values = set(int(neuron) for neuron in neurons)
    if WHOLE_VECTOR_NEURON in values:
        return set(range(width))
    return {neuron for neuron in values if neuron >= 0}


def _dimensioned_editable_neurons(mask: PathNeuronMask, width: int) -> set[int]:
    forget = _expand_neurons_for_width(mask.forget_neurons, width)
    shared = _expand_neurons_for_width(mask.shared_neurons, width)
    return forget - shared


def _dimensioned_trainable_neurons(mask: PathNeuronMask, width: int) -> set[int]:
    editable = _dimensioned_editable_neurons(mask, width)
    probe = _expand_neurons_for_width(mask.probe_neurons, width)
    return editable | probe


def _dimensioned_preserve_neurons(mask: PathNeuronMask, width: int) -> set[int]:
    return (
        _expand_neurons_for_width(mask.forget_neurons, width)
        | _expand_neurons_for_width(mask.shared_neurons, width)
        | _expand_neurons_for_width(mask.probe_neurons, width)
    )


def _summary_with_dimensioned_neurons(mask: PathNeuronMask, width: int) -> dict[str, Any]:
    editable = sorted(_dimensioned_editable_neurons(mask, width))
    trainable = sorted(_dimensioned_trainable_neurons(mask, width))
    preserve = sorted(_dimensioned_preserve_neurons(mask, width))
    summary = mask.to_summary()
    summary.update(
        {
            "num_forget_editable_neurons": len(editable),
            "num_editable_neurons": len(trainable),
            "num_preserve_neurons": len(preserve),
            "uses_whole_vector_neuron": any(
                WHOLE_VECTOR_NEURON in neurons
                for neurons in (mask.forget_neurons, mask.shared_neurons, mask.probe_neurons)
            ),
        }
    )
    return summary


def _resolve_projector_edit_module(
    model: nn.Module,
    module_name: str,
    mode: ProjectorEditMode,
) -> str | None:
    if mode == "skip":
        return None
    if mode != "qwen_merger_mlp":
        raise ValueError(f"Unsupported projector_edit_mode: {mode}")

    candidates = [module_name]
    if module_name == "mm_projector":
        candidates.extend(
            [
                "model.mm_projector",
                "base_model.model.mm_projector",
                "visual.merger",
                "model.visual.merger",
                "base_model.model.visual.merger",
                "base_model.model.model.visual.merger",
            ]
        )

    for candidate in candidates:
        try:
            module = get_module(model, candidate)
        except KeyError:
            continue
        mlp = getattr(module, "mlp", None)
        if isinstance(mlp, nn.Sequential) and len(mlp) >= 3:
            if isinstance(mlp[0], nn.Linear) and isinstance(mlp[2], nn.Linear):
                return f"{candidate}.mlp.0"
        if isinstance(module, nn.Linear):
            return candidate
    return None


def build_path_neuron_masks(
    candidate_paths_path: str,
    p_forget_path: str,
    p_shared_path: str | None = None,
    p_probe_path: str | None = None,
    strict: bool = False,
) -> dict[str, PathNeuronMask]:
    candidates = load_candidate_paths_by_id(candidate_paths_path)
    forget_ids = load_step6_path_ids(p_forget_path)
    shared_ids = load_step6_path_ids(p_shared_path)
    probe_ids = load_step6_path_ids(p_probe_path)
    masks: dict[str, PathNeuronMask] = {}
    missing_ids = sorted((forget_ids | shared_ids | probe_ids) - set(candidates))
    if missing_ids and strict:
        raise KeyError(f"Step 6 path ids not found in candidate paths: {missing_ids[:10]}")

    def ensure_mask(module: str, layer: int | None, module_kind: str) -> PathNeuronMask:
        if module not in masks:
            masks[module] = PathNeuronMask(module=module, layer=layer, module_kind=module_kind)
        return masks[module]

    for path_id in sorted(forget_ids):
        candidate = candidates.get(path_id)
        if candidate is None:
            continue
        for node in candidate.nodes:
            module_kind = _patchable_down_proj_kind(node.module)
            if module_kind is None:
                continue
            mask = ensure_mask(node.module, node.layer, module_kind)
            mask.forget_neurons.add(_mask_neuron_for_node(candidate, module_kind, node.neuron))
            mask.forget_path_ids.add(path_id)

    for path_id in sorted(shared_ids):
        candidate = candidates.get(path_id)
        if candidate is None:
            continue
        for node in candidate.nodes:
            module_kind = _patchable_down_proj_kind(node.module)
            if module_kind is None:
                continue
            mask = ensure_mask(node.module, node.layer, module_kind)
            mask.shared_neurons.add(_mask_neuron_for_node(candidate, module_kind, node.neuron))
            mask.shared_path_ids.add(path_id)

    for path_id in sorted(probe_ids):
        candidate = candidates.get(path_id)
        if candidate is None:
            continue
        for node in candidate.nodes:
            module_kind = _patchable_down_proj_kind(node.module)
            if module_kind is None:
                continue
            mask = ensure_mask(node.module, node.layer, module_kind)
            mask.probe_neurons.add(_mask_neuron_for_node(candidate, module_kind, node.neuron))
            mask.probe_path_ids.add(path_id)

    return masks


def _parent_module_name(module_name: str) -> str:
    suffix = ".down_proj"
    if not module_name.endswith(suffix):
        raise ValueError(f"Expected down_proj module, got: {module_name}")
    return module_name[: -len(suffix)]


def _replace_child_module(root: nn.Module, dotted_name: str, new_module: nn.Module) -> None:
    parent_name, child_name = dotted_name.rsplit(".", 1)
    parent = get_module(root, parent_name)
    setattr(parent, child_name, new_module)


def _wrap_linear_with_partial(linear: nn.Module, trainable_cols: list[int]) -> PartialLinear:
    if isinstance(linear, PartialLinear):
        linear = linear.merge_to_linear()
    if not isinstance(linear, nn.Linear):
        raise TypeError(f"Expected nn.Linear or PartialLinear, got {type(linear)}")
    if not trainable_cols:
        raise ValueError("trainable_cols must not be empty")
    max_col = max(trainable_cols)
    if max_col >= linear.out_features:
        raise IndexError(
            f"Trainable neuron index {max_col} is out of range for linear out_features={linear.out_features}"
        )
    return PartialLinear(linear, trainable_cols=trainable_cols)


def apply_masked_rmisu_parameter_mask(
    model: nn.Module,
    masks: dict[str, PathNeuronMask],
    strict: bool = False,
    projector_edit_mode: ProjectorEditMode = "qwen_merger_mlp",
) -> dict[str, Any]:
    model.requires_grad_(False)
    summary = {
        "num_modules": 0,
        "num_editable_neurons": 0,
        "num_probe_neurons": 0,
        "modules": [],
        "skipped_modules": [],
    }

    for down_proj_name, mask in sorted(masks.items()):
        if not mask.trainable_neurons:
            summary["skipped_modules"].append(
                {"module": down_proj_name, "reason": "no_editable_neurons"}
            )
            continue
        if mask.module_kind == "projector":
            edit_module_name = _resolve_projector_edit_module(
                model,
                down_proj_name,
                mode=projector_edit_mode,
            )
            if edit_module_name is None:
                reason = (
                    "projector_editing_disabled"
                    if projector_edit_mode == "skip"
                    else "projector_edit_module_not_found"
                )
                if strict and projector_edit_mode != "skip":
                    raise KeyError(f"Projector edit module not found for {down_proj_name}")
                summary["skipped_modules"].append(
                    {"module": down_proj_name, "reason": reason}
                )
                continue
            try:
                linear = get_module(model, edit_module_name)
                if isinstance(linear, PartialLinear):
                    width = linear.num_columns
                elif isinstance(linear, nn.Linear):
                    width = linear.out_features
                else:
                    raise TypeError(f"Expected nn.Linear or PartialLinear, got {type(linear)}")
                trainable = sorted(_dimensioned_trainable_neurons(mask, width))
                if not trainable:
                    summary["skipped_modules"].append(
                        {"module": down_proj_name, "reason": "no_editable_neurons"}
                    )
                    continue
                wrapped = _wrap_linear_with_partial(linear, trainable)
                _replace_child_module(model, edit_module_name, wrapped)
            except (KeyError, TypeError, IndexError):
                if strict:
                    raise
                summary["skipped_modules"].append(
                    {"module": down_proj_name, "reason": "projector_edit_module_not_patchable"}
                )
                continue
            mask.trace_module = edit_module_name
            mask.trace_kind = "output"
            mask.active = True
            summary["num_modules"] += 1
            summary["num_editable_neurons"] += len(trainable)
            summary["num_probe_neurons"] += len(mask.probe_neurons)
            summary["modules"].append(
                {
                    **_summary_with_dimensioned_neurons(mask, width),
                    "editable_neurons": trainable,
                    "forget_editable_neurons": sorted(_dimensioned_editable_neurons(mask, width)),
                    "probe_neurons": sorted(_expand_neurons_for_width(mask.probe_neurons, width)),
                    "edit_module": edit_module_name,
                }
            )
            continue
        mlp_name = _parent_module_name(down_proj_name)
        try:
            mlp = get_module(model, mlp_name)
        except KeyError:
            if strict:
                raise
            summary["skipped_modules"].append(
                {"module": down_proj_name, "reason": "module_not_found"}
            )
            continue

        for proj_name in ("up_proj", "gate_proj"):
            if not hasattr(mlp, proj_name):
                if strict:
                    raise KeyError(f"{mlp_name}.{proj_name} not found")
                continue
            target_linear = getattr(mlp, proj_name)
            if isinstance(target_linear, PartialLinear):
                width = target_linear.num_columns
            else:
                width = target_linear.out_features
            trainable = sorted(_dimensioned_trainable_neurons(mask, width))
            if not trainable:
                break
            wrapped = _wrap_linear_with_partial(target_linear, trainable)
            _replace_child_module(model, f"{mlp_name}.{proj_name}", wrapped)
        if not trainable:
            summary["skipped_modules"].append(
                {"module": down_proj_name, "reason": "no_editable_neurons"}
            )
            continue

        summary["num_modules"] += 1
        summary["num_editable_neurons"] += len(trainable)
        summary["num_probe_neurons"] += len(_expand_neurons_for_width(mask.probe_neurons, width))
        mask.active = True
        summary["modules"].append(
            {
                **_summary_with_dimensioned_neurons(mask, width),
                "editable_neurons": trainable,
                "forget_editable_neurons": sorted(_dimensioned_editable_neurons(mask, width)),
                "probe_neurons": sorted(_expand_neurons_for_width(mask.probe_neurons, width)),
                "edit_module": down_proj_name,
            }
        )

    return summary


def merge_partial_linear_modules(model: nn.Module) -> int:
    merged = 0

    def visit(module: nn.Module) -> None:
        nonlocal merged
        for child_name, child in list(module.named_children()):
            if isinstance(child, PartialLinear):
                setattr(module, child_name, child.merge_to_linear())
                merged += 1
            else:
                visit(child)

    visit(model)
    return merged


class DownProjInputTracer:
    def __init__(
        self,
        model: nn.Module,
        masks: dict[str, PathNeuronMask],
        neuron_kind: str = "editable",
        detach: bool = False,
    ):
        self.model = model
        self.masks = masks
        self.neuron_kind = neuron_kind
        self.detach = detach
        self.handles = []
        self.activations: dict[str, torch.Tensor] = {}

    def _raw_neurons_for_mask(self, mask: PathNeuronMask) -> set[int]:
        if self.neuron_kind == "editable":
            return mask.editable_neurons
        if self.neuron_kind == "probe":
            return mask.probe_neurons
        if self.neuron_kind == "trainable":
            return mask.trainable_neurons
        if self.neuron_kind == "shared":
            return mask.shared_neurons
        if self.neuron_kind == "preserve":
            return mask.preserve_neurons
        raise ValueError(f"Unsupported neuron_kind: {self.neuron_kind}")

    def __enter__(self):
        for module_name, mask in sorted(self.masks.items()):
            if not mask.active:
                continue
            raw_neurons = self._raw_neurons_for_mask(mask)
            if not raw_neurons:
                continue
            trace_module_name = mask.trace_module or module_name
            module = get_module(self.model, trace_module_name)

            def pre_hook(current_module, inputs, module_name=module_name, raw_neurons=raw_neurons):
                tensor = inputs[0] if isinstance(inputs, tuple) else inputs
                neurons = sorted(_expand_neurons_for_width(raw_neurons, tensor.shape[-1]))
                if not neurons:
                    return None
                selected = tensor[..., neurons]
                if self.detach:
                    selected = selected.detach()
                self.activations[module_name] = selected
                return None

            def forward_hook(current_module, inputs, output, module_name=module_name, raw_neurons=raw_neurons):
                tensor = output[0] if isinstance(output, tuple) else output
                neurons = sorted(_expand_neurons_for_width(raw_neurons, tensor.shape[-1]))
                if not neurons:
                    return None
                selected = tensor[..., neurons]
                if self.detach:
                    selected = selected.detach()
                self.activations[module_name] = selected
                return None

            if mask.trace_kind == "output":
                self.handles.append(module.register_forward_hook(forward_hook))
            else:
                self.handles.append(module.register_forward_pre_hook(pre_hook))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        return None


def _model_device(model: nn.Module) -> torch.device:
    return next(model.parameters()).device


def _forward_batch(model: nn.Module, batch, compute_loss: bool = False):
    device = _model_device(model)
    arch = getattr(model.config, "architectures", [""])[0]
    if "Qwen" in arch:
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        pixel_values = batch[2].to(device) if batch[2] is not None else None
        image_grid_thw = batch[3].to(device) if batch[3] is not None else None
        labels = batch[4].to(device) if compute_loss else None
        return model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            labels=labels,
        )
    if "Llava" in arch:
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        pixel_values = batch[2].to(device)
        labels = batch[3].to(device) if compute_loss else None
        return model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values, labels=labels)
    if "Gemma" in arch:
        inputs = batch[0]
        labels = inputs["labels"].to(device) if compute_loss else None
        return model(
            input_ids=inputs["input_ids"].to(device),
            attention_mask=inputs["attention_mask"].to(device),
            pixel_values=inputs["pixel_values"].to(device),
            labels=labels,
        )
    return model(*batch)


def _batch_labels(batch) -> torch.Tensor | None:
    arch_batch_len = len(batch) if isinstance(batch, tuple) else 0
    if arch_batch_len >= 5 and torch.is_tensor(batch[4]):
        return batch[4]
    if arch_batch_len >= 4 and torch.is_tensor(batch[3]):
        return batch[3]
    if arch_batch_len >= 1 and isinstance(batch[0], dict):
        labels = batch[0].get("labels")
        if torch.is_tensor(labels):
            return labels
    return None


def _batch_item_list(batch) -> list[Any]:
    if isinstance(batch, tuple) and batch:
        maybe_items = batch[-1]
        if isinstance(maybe_items, list):
            return maybe_items
    return []


def _metadata_positions(item: Any, scope: str) -> list[int]:
    if not isinstance(item, dict):
        return []
    if scope == "answer":
        positions = item.get("answer_token_positions")
    elif scope == "name":
        positions = item.get("name_token_positions")
    else:
        positions = None
    if positions is None:
        return []
    return [int(position) for position in positions]


def _labels_for_target_ce(batch, scope: str, device: torch.device) -> tuple[torch.Tensor | None, int]:
    labels = _batch_labels(batch)
    if labels is None:
        return None, 0
    labels = labels.to(device)
    if scope == "all":
        token_count = int((labels != -100).detach().sum().cpu().item())
        return labels, token_count

    item_list = _batch_item_list(batch)
    masked_labels = torch.full_like(labels, -100)
    for row_idx, item in enumerate(item_list[: labels.shape[0]]):
        positions = _metadata_positions(item, scope)
        for position in positions:
            if 0 <= position < labels.shape[1] and labels[row_idx, position] != -100:
                masked_labels[row_idx, position] = labels[row_idx, position]

    token_count = int((masked_labels != -100).detach().sum().cpu().item())
    if token_count == 0:
        if scope == "name":
            return None, 0
        token_count = int((labels != -100).detach().sum().cpu().item())
        return labels, token_count
    return masked_labels, token_count


def _labels_for_answer_without_name_ce(batch, device: torch.device) -> tuple[torch.Tensor | None, int]:
    labels = _batch_labels(batch)
    if labels is None:
        return None, 0
    labels = labels.to(device)
    item_list = _batch_item_list(batch)
    masked_labels = torch.full_like(labels, -100)
    for row_idx, item in enumerate(item_list[: labels.shape[0]]):
        answer_positions = set(_metadata_positions(item, "answer"))
        name_positions = set(_metadata_positions(item, "name"))
        for position in sorted(answer_positions - name_positions):
            if 0 <= position < labels.shape[1] and labels[row_idx, position] != -100:
                masked_labels[row_idx, position] = labels[row_idx, position]

    token_count = int((masked_labels != -100).detach().sum().cpu().item())
    if token_count == 0:
        return None, 0
    return masked_labels, token_count


def _labels_for_redacted_positive_ce(batch, device: torch.device) -> tuple[torch.Tensor | None, int]:
    labels = _batch_labels(batch)
    if labels is None:
        return None, 0
    labels = labels.to(device)
    item_list = _batch_item_list(batch)
    masked_labels = torch.full_like(labels, -100)
    for row_idx, item in enumerate(item_list[: labels.shape[0]]):
        if not isinstance(item, dict):
            continue
        name_positions = _metadata_positions(item, "name")
        positive_token_ids = item.get("redacted_name_token_ids")
        if not name_positions:
            continue
        if not isinstance(positive_token_ids, list):
            positive_token_ids = item.get("redacted_positive_token_ids")
        if not isinstance(positive_token_ids, list):
            continue
        positions = name_positions[: len(positive_token_ids)]
        for offset, token_id in enumerate(positive_token_ids):
            if offset >= len(positions):
                break
            position = positions[offset]
            if 0 <= position < labels.shape[1] and labels[row_idx, position] != -100:
                masked_labels[row_idx, position] = int(token_id)

    token_count = int((masked_labels != -100).detach().sum().cpu().item())
    if token_count == 0:
        return None, 0
    return masked_labels, token_count


def _targeted_ce_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
    return loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1).to(shift_logits.device),
    )


def _targeted_entropy_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous().to(shift_logits.device)
    active = shift_labels != -100
    if not bool(active.detach().any().cpu().item()):
        return torch.tensor(0.0, device=shift_logits.device)
    log_probs = torch.nn.functional.log_softmax(shift_logits[active], dim=-1)
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=-1)
    return entropy.mean()


def _partial_linear_delta_l2_loss(model: nn.Module) -> torch.Tensor:
    losses = []
    for module in model.modules():
        if isinstance(module, PartialLinear):
            losses.append(torch.nn.functional.mse_loss(module.trainable_weight, module.original_linear.weight[module.trainable_cols, :].to(module.trainable_weight.device)))
            if module.trainable_bias is not None and module.original_linear.bias is not None:
                losses.append(torch.nn.functional.mse_loss(module.trainable_bias, module.original_linear.bias[module.trainable_cols].to(module.trainable_bias.device)))
    if not losses:
        return torch.tensor(0.0, device=_model_device(model))
    return torch.stack(losses).mean()


def _clip_partial_linear_delta_(model: nn.Module, max_norm: float | None) -> None:
    if max_norm is None or max_norm <= 0:
        return
    with torch.no_grad():
        for module in model.modules():
            if not isinstance(module, PartialLinear):
                continue
            base_weight = module.original_linear.weight[module.trainable_cols, :].to(module.trainable_weight.device)
            delta = module.trainable_weight - base_weight
            flat_delta = delta.reshape(delta.shape[0], -1)
            norms = flat_delta.norm(dim=1, keepdim=True).clamp_min(1e-12)
            scale = (float(max_norm) / norms).clamp(max=1.0).view(-1, *([1] * (delta.ndim - 1)))
            module.trainable_weight.copy_(base_weight + delta * scale)
            if module.trainable_bias is not None and module.original_linear.bias is not None:
                base_bias = module.original_linear.bias[module.trainable_cols].to(module.trainable_bias.device)
                bias_delta = module.trainable_bias - base_bias
                module.trainable_bias.copy_(base_bias + bias_delta.clamp(min=-float(max_norm), max=float(max_norm)))


def _activation_mse_to_random(
    activations: dict[str, torch.Tensor],
    steering_coeff: float,
    coeffs: float,
) -> torch.Tensor:
    losses = []
    for activation in activations.values():
        target = torch.randn_like(activation)
        target = target / target.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        target = target * steering_coeff * coeffs
        losses.append(torch.nn.functional.mse_loss(activation, target))
    if not losses:
        raise ValueError("No editable activations were captured for masked RMisU unlearn loss")
    return torch.stack(losses).mean()


def _activation_preserve_loss(
    updated: dict[str, torch.Tensor],
    frozen: dict[str, torch.Tensor],
) -> torch.Tensor:
    losses = []
    for module_name, updated_activation in updated.items():
        frozen_activation = frozen.get(module_name)
        if frozen_activation is None:
            continue
        losses.append(torch.nn.functional.mse_loss(updated_activation, frozen_activation.to(updated_activation.device)))
    if not losses:
        return torch.tensor(0.0)
    return torch.stack(losses).mean()


def _validate_forget_objective(config: MaskedRMisUConfig) -> None:
    allowed = {
        "activation_random",
        "ce_ascent",
        "answer_ce_ascent",
        "name_ce_ascent",
        "bounded_name_ce_ascent",
        "pii_name_token_noise",
        "name_preference_unlearning",
        "redacted_name_preference",
        "activation_random_ce",
        "activation_random_answer_ce",
        "activation_random_name_ce",
    }
    if config.forget_objective not in allowed:
        raise ValueError(
            f"Unsupported forget_objective={config.forget_objective}. "
            f"Expected one of {sorted(allowed)}."
        )
    if config.forget_objective in {
        "ce_ascent",
        "answer_ce_ascent",
        "name_ce_ascent",
        "bounded_name_ce_ascent",
        "pii_name_token_noise",
        "name_preference_unlearning",
        "redacted_name_preference",
        "activation_random_ce",
        "activation_random_answer_ce",
        "activation_random_name_ce",
    } and config.forget_ce_alpha <= 0.0:
        raise ValueError(
            f"forget_objective={config.forget_objective} requires forget_ce_alpha > 0"
        )
    allowed_scopes = {"all", "answer", "name"}
    if config.target_ce_scope not in allowed_scopes:
        raise ValueError(
            f"Unsupported target_ce_scope={config.target_ce_scope}. "
            f"Expected one of {sorted(allowed_scopes)}."
        )
    if config.counterfactual_anchor_scope not in allowed_scopes:
        raise ValueError(
            f"Unsupported counterfactual_anchor_scope={config.counterfactual_anchor_scope}. "
            f"Expected one of {sorted(allowed_scopes)}."
        )


def _ce_scope_for_objective(config: MaskedRMisUConfig) -> str:
    objective_to_scope = {
        "ce_ascent": "all",
        "activation_random_ce": "all",
        "answer_ce_ascent": "answer",
        "activation_random_answer_ce": "answer",
        "name_ce_ascent": "name",
        "bounded_name_ce_ascent": "name",
        "pii_name_token_noise": "name",
        "name_preference_unlearning": "name",
        "redacted_name_preference": "name",
        "activation_random_name_ce": "name",
    }
    return objective_to_scope.get(config.forget_objective, config.target_ce_scope)


def masked_rmisu_finetune(
    updated_model: nn.Module,
    frozen_model: nn.Module | None,
    retain_loader,
    forget_loader,
    config: MaskedRMisUConfig,
    counterfactual_anchor_loader=None,
) -> tuple[nn.Module, dict[str, Any]]:
    _validate_forget_objective(config)
    masks = build_path_neuron_masks(
        candidate_paths_path=config.candidate_paths_path,
        p_forget_path=config.p_forget_path,
        p_shared_path=config.p_shared_path,
        p_probe_path=config.p_probe_path,
    )
    mask_summary = apply_masked_rmisu_parameter_mask(
        updated_model,
        masks,
        projector_edit_mode=config.projector_edit_mode,
    )
    trainable_params = [param for param in updated_model.parameters() if param.requires_grad]
    if not trainable_params:
        raise ValueError("No trainable parameters after applying masked RMisU parameter mask")

    optimizer = torch.optim.AdamW(trainable_params, lr=config.learning_rate)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer,
        num_warmup_steps=0,
        num_training_steps=config.epochs * len(retain_loader),
    )
    losses = []
    updated_model.train()
    if frozen_model is not None:
        frozen_model.eval()
    counterfactual_anchor_iterator = None
    if config.counterfactual_anchor_alpha != 0.0:
        if counterfactual_anchor_loader is None:
            raise ValueError(
                "counterfactual_anchor_alpha requires counterfactual_anchor_loader"
            )
        counterfactual_anchor_iterator = itertools.cycle(counterfactual_anchor_loader)

    for epoch in range(config.epochs):
        iterator = tqdm(
            enumerate(zip(retain_loader, forget_loader)),
            total=len(retain_loader),
            desc=f"Masked RMisU {epoch}",
        )
        for step, (retain_batch, forget_batch) in iterator:
            if config.max_steps is not None and len(losses) >= config.max_steps:
                break
            optimizer.zero_grad()

            with DownProjInputTracer(
                updated_model, masks, neuron_kind="editable", detach=False
            ) as forget_trace, DownProjInputTracer(
                updated_model, masks, neuron_kind="probe", detach=False
            ) as probe_trace:
                forget_output = _forward_batch(updated_model, forget_batch)
            device = _model_device(updated_model)
            if config.forget_objective in {
                "activation_random",
                "activation_random_ce",
                "activation_random_answer_ce",
                "activation_random_name_ce",
            }:
                if forget_trace.activations:
                    unlearn_loss = _activation_mse_to_random(
                        forget_trace.activations,
                        steering_coeff=config.steering_coeff,
                        coeffs=config.coeffs,
                    )
                elif config.p_probe_path is None:
                    raise ValueError("No editable activations were captured for masked RMisU unlearn loss")
                else:
                    unlearn_loss = torch.tensor(0.0, device=device)
            else:
                unlearn_loss = torch.tensor(0.0, device=device)
            probe_loss = torch.tensor(0.0, device=device)
            if config.p_probe_path is not None and config.probe_beta != 0.0:
                if not probe_trace.activations:
                    raise ValueError("No probe activations were captured for masked RMisU probe loss")
                probe_loss = _activation_mse_to_random(
                    probe_trace.activations,
                    steering_coeff=config.probe_steering_coeff,
                    coeffs=config.probe_coeffs,
                )
            forget_ce_loss = torch.tensor(0.0, device=device)
            forget_ce_token_count = 0
            if config.forget_ce_alpha != 0.0 or config.forget_objective in {
                "ce_ascent",
                "answer_ce_ascent",
                "name_ce_ascent",
                "bounded_name_ce_ascent",
                "pii_name_token_noise",
                "name_preference_unlearning",
                "redacted_name_preference",
            }:
                target_ce_scope = _ce_scope_for_objective(config)
                target_labels, forget_ce_token_count = _labels_for_target_ce(
                    forget_batch,
                    target_ce_scope,
                    device,
                )
                if target_labels is not None:
                    forget_ce_loss = _targeted_ce_loss(forget_output.logits, target_labels)

            pii_noise_loss = torch.tensor(0.0, device=device)
            if config.forget_objective == "pii_name_token_noise" and target_labels is not None:
                pii_noise_loss = _targeted_entropy_loss(forget_output.logits, target_labels)

            bounded_delta_l2_loss = torch.tensor(0.0, device=device)
            if config.forget_objective == "bounded_name_ce_ascent" and config.bounded_delta_l2_alpha != 0.0:
                bounded_delta_l2_loss = _partial_linear_delta_l2_loss(updated_model).to(device)

            counterfactual_anchor_loss = torch.tensor(0.0, device=device)
            counterfactual_anchor_token_count = 0
            if counterfactual_anchor_iterator is not None:
                counterfactual_anchor_batch = next(counterfactual_anchor_iterator)
                counterfactual_anchor_output = _forward_batch(updated_model, counterfactual_anchor_batch)
                anchor_labels, counterfactual_anchor_token_count = _labels_for_target_ce(
                    counterfactual_anchor_batch,
                    config.counterfactual_anchor_scope,
                    device,
                )
                if anchor_labels is not None:
                    counterfactual_anchor_loss = _targeted_ce_loss(
                        counterfactual_anchor_output.logits,
                        anchor_labels,
                    )

            preference_positive_loss = torch.tensor(0.0, device=device)
            preference_positive_token_count = 0
            if config.forget_objective in {"name_preference_unlearning", "redacted_name_preference"}:
                if config.forget_objective == "redacted_name_preference":
                    positive_labels, preference_positive_token_count = _labels_for_redacted_positive_ce(
                        forget_batch,
                        device,
                    )
                else:
                    positive_labels, preference_positive_token_count = _labels_for_answer_without_name_ce(
                        forget_batch,
                        device,
                    )
                if positive_labels is not None:
                    preference_positive_loss = _targeted_ce_loss(forget_output.logits, positive_labels)

            retain_loss = torch.tensor(0.0, device=device)
            shared_loss = torch.tensor(0.0, device=device)
            if frozen_model is not None and (config.use_retain_preserve or config.use_shared_preserve):
                with torch.no_grad(), DownProjInputTracer(
                    frozen_model, masks, neuron_kind="preserve", detach=True
                ) as frozen_trace:
                    _forward_batch(frozen_model, retain_batch)

                with DownProjInputTracer(
                    updated_model, masks, neuron_kind="preserve", detach=False
                ) as retain_trace:
                    _forward_batch(updated_model, retain_batch)
                retain_loss = _activation_preserve_loss(retain_trace.activations, frozen_trace.activations).to(
                    device
                )

                if config.use_shared_preserve:
                    with torch.no_grad(), DownProjInputTracer(
                        frozen_model, masks, neuron_kind="shared", detach=True
                    ) as frozen_shared_trace:
                        _forward_batch(frozen_model, retain_batch)
                    with DownProjInputTracer(
                        updated_model, masks, neuron_kind="shared", detach=False
                    ) as shared_trace:
                        _forward_batch(updated_model, retain_batch)
                    shared_loss = _activation_preserve_loss(
                        shared_trace.activations, frozen_shared_trace.activations
                    ).to(device)

            loss = (
                config.beta * unlearn_loss
                + config.probe_beta * probe_loss
                + config.alpha * retain_loss
                + config.shared_alpha * shared_loss
                + config.preference_positive_alpha * preference_positive_loss
                + config.bounded_delta_l2_alpha * bounded_delta_l2_loss
                + config.counterfactual_anchor_alpha * counterfactual_anchor_loss
                - config.forget_ce_alpha * forget_ce_loss
                - config.pii_noise_alpha * pii_noise_loss
            )
            loss.backward()
            optimizer.step()
            if config.forget_objective == "bounded_name_ce_ascent":
                _clip_partial_linear_delta_(updated_model, config.bounded_delta_max_norm)
            lr_scheduler.step()

            record = {
                "epoch": epoch,
                "step": step,
                "loss": float(loss.detach().cpu().item()),
                "unlearn_loss": float(unlearn_loss.detach().cpu().item()),
                "probe_loss": float(probe_loss.detach().cpu().item()),
                "forget_ce_loss": float(forget_ce_loss.detach().cpu().item()),
                "forget_objective": config.forget_objective,
                "forget_ce_scope": _ce_scope_for_objective(config),
                "forget_ce_token_count": forget_ce_token_count,
                "preference_positive_loss": float(preference_positive_loss.detach().cpu().item()),
                "preference_positive_token_count": preference_positive_token_count,
                "pii_noise_loss": float(pii_noise_loss.detach().cpu().item()),
                "bounded_delta_l2_loss": float(bounded_delta_l2_loss.detach().cpu().item()),
                "counterfactual_anchor_loss": float(counterfactual_anchor_loss.detach().cpu().item()),
                "counterfactual_anchor_token_count": counterfactual_anchor_token_count,
                "retain_loss": float(retain_loss.detach().cpu().item()),
                "shared_loss": float(shared_loss.detach().cpu().item()),
            }
            losses.append(record)
            if step % 10 == 0:
                iterator.set_postfix(
                    loss=record["loss"],
                    unlearn=record["unlearn_loss"],
                    probe=record["probe_loss"],
                    forget_ce=record["forget_ce_loss"],
                    pref_pos=record["preference_positive_loss"],
                    pii_noise=record["pii_noise_loss"],
                    bound_l2=record["bounded_delta_l2_loss"],
                    cf_anchor=record["counterfactual_anchor_loss"],
                    retain=record["retain_loss"],
                    shared=record["shared_loss"],
                )

        if config.max_steps is not None and len(losses) >= config.max_steps:
            break

    summary = {
        "mask_summary": mask_summary,
        "probe_config": {
            "p_probe_path": config.p_probe_path,
            "probe_beta": config.probe_beta,
            "probe_steering_coeff": config.probe_steering_coeff,
            "probe_coeffs": config.probe_coeffs,
        },
        "forget_config": {
            "forget_objective": config.forget_objective,
            "forget_ce_alpha": config.forget_ce_alpha,
            "preference_positive_alpha": config.preference_positive_alpha,
            "bounded_delta_l2_alpha": config.bounded_delta_l2_alpha,
            "bounded_delta_max_norm": config.bounded_delta_max_norm,
            "pii_noise_alpha": config.pii_noise_alpha,
            "target_ce_scope": _ce_scope_for_objective(config),
            "counterfactual_anchor_alpha": config.counterfactual_anchor_alpha,
            "counterfactual_anchor_scope": config.counterfactual_anchor_scope,
        },
        "num_loss_records": len(losses),
        "losses": losses,
    }
    if config.output_path is not None:
        output = Path(config.output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)
    if config.save and config.checkpoint_dir is not None:
        checkpoint_dir = Path(config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        merged_partial_linear_modules = merge_partial_linear_modules(updated_model)
        updated_model.save_pretrained(str(checkpoint_dir))
        summary["checkpoint_dir"] = str(checkpoint_dir)
        summary["merged_partial_linear_modules"] = merged_partial_linear_modules
        if config.output_path is not None:
            output = Path(config.output_path)
            with output.open("w", encoding="utf-8") as handle:
                json.dump(summary, handle, ensure_ascii=False, indent=2)
    return updated_model, summary
