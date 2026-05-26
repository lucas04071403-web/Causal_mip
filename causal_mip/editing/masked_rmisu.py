from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn
from tqdm import tqdm
from transformers import get_scheduler

from causal_mip.interventions.hooks import get_module
from causal_mip.path_localization.path_schema import CandidatePath
from partial_linear import PartialLinear


@dataclass
class PathNeuronMask:
    module: str
    layer: int | None = None
    module_kind: str = "llm"
    forget_neurons: set[int] = field(default_factory=set)
    shared_neurons: set[int] = field(default_factory=set)
    forget_path_ids: set[str] = field(default_factory=set)
    shared_path_ids: set[str] = field(default_factory=set)

    @property
    def editable_neurons(self) -> set[int]:
        return set(self.forget_neurons) - set(self.shared_neurons)

    @property
    def preserve_neurons(self) -> set[int]:
        return set(self.forget_neurons) | set(self.shared_neurons)

    def to_summary(self) -> dict[str, Any]:
        return {
            "module": self.module,
            "layer": self.layer,
            "module_kind": self.module_kind,
            "num_forget_neurons": len(self.forget_neurons),
            "num_shared_neurons": len(self.shared_neurons),
            "num_editable_neurons": len(self.editable_neurons),
            "num_preserve_neurons": len(self.preserve_neurons),
            "forget_path_ids": sorted(self.forget_path_ids),
            "shared_path_ids": sorted(self.shared_path_ids),
        }


@dataclass
class MaskedRMisUConfig:
    candidate_paths_path: str
    p_forget_path: str
    p_shared_path: str | None = None
    alpha: float = 1.0
    beta: float = 1.0
    shared_alpha: float = 1.0
    forget_ce_alpha: float = 0.0
    steering_coeff: float = 6.5
    coeffs: float = 1.0
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
    return None


def build_path_neuron_masks(
    candidate_paths_path: str,
    p_forget_path: str,
    p_shared_path: str | None = None,
    strict: bool = False,
) -> dict[str, PathNeuronMask]:
    candidates = load_candidate_paths_by_id(candidate_paths_path)
    forget_ids = load_step6_path_ids(p_forget_path)
    shared_ids = load_step6_path_ids(p_shared_path)
    masks: dict[str, PathNeuronMask] = {}
    missing_ids = sorted((forget_ids | shared_ids) - set(candidates))
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
            mask.forget_neurons.add(int(node.neuron))
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
            mask.shared_neurons.add(int(node.neuron))
            mask.shared_path_ids.add(path_id)

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
) -> dict[str, Any]:
    model.requires_grad_(False)
    summary = {
        "num_modules": 0,
        "num_editable_neurons": 0,
        "modules": [],
        "skipped_modules": [],
    }

    for down_proj_name, mask in sorted(masks.items()):
        editable = sorted(mask.editable_neurons)
        if not editable:
            summary["skipped_modules"].append(
                {"module": down_proj_name, "reason": "no_editable_neurons"}
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
            wrapped = _wrap_linear_with_partial(getattr(mlp, proj_name), editable)
            _replace_child_module(model, f"{mlp_name}.{proj_name}", wrapped)

        summary["num_modules"] += 1
        summary["num_editable_neurons"] += len(editable)
        summary["modules"].append(
            {
                **mask.to_summary(),
                "editable_neurons": editable,
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

    def _neurons_for_mask(self, mask: PathNeuronMask) -> list[int]:
        if self.neuron_kind == "editable":
            return sorted(mask.editable_neurons)
        if self.neuron_kind == "shared":
            return sorted(mask.shared_neurons)
        if self.neuron_kind == "preserve":
            return sorted(mask.preserve_neurons)
        raise ValueError(f"Unsupported neuron_kind: {self.neuron_kind}")

    def __enter__(self):
        for module_name, mask in sorted(self.masks.items()):
            neurons = self._neurons_for_mask(mask)
            if not neurons:
                continue
            module = get_module(self.model, module_name)

            def hook(current_module, inputs, module_name=module_name, neurons=neurons):
                tensor = inputs[0] if isinstance(inputs, tuple) else inputs
                selected = tensor[..., neurons]
                if self.detach:
                    selected = selected.detach()
                self.activations[module_name] = selected
                return None

            self.handles.append(module.register_forward_pre_hook(hook))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        return None


def _model_device(model: nn.Module) -> torch.device:
    return next(model.parameters()).device


def _forward_batch(model: nn.Module, batch):
    device = _model_device(model)
    arch = getattr(model.config, "architectures", [""])[0]
    if "Qwen" in arch:
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        pixel_values = batch[2].to(device) if batch[2] is not None else None
        image_grid_thw = batch[3].to(device) if batch[3] is not None else None
        labels = batch[4].to(device)
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
        labels = batch[3].to(device)
        return model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values, labels=labels)
    if "Gemma" in arch:
        inputs = batch[0]
        return model(
            input_ids=inputs["input_ids"].to(device),
            attention_mask=inputs["attention_mask"].to(device),
            pixel_values=inputs["pixel_values"].to(device),
            labels=inputs["labels"].to(device),
        )
    return model(*batch)


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


def masked_rmisu_finetune(
    updated_model: nn.Module,
    frozen_model: nn.Module | None,
    retain_loader,
    forget_loader,
    config: MaskedRMisUConfig,
) -> tuple[nn.Module, dict[str, Any]]:
    masks = build_path_neuron_masks(
        candidate_paths_path=config.candidate_paths_path,
        p_forget_path=config.p_forget_path,
        p_shared_path=config.p_shared_path,
    )
    mask_summary = apply_masked_rmisu_parameter_mask(updated_model, masks)
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

            with DownProjInputTracer(updated_model, masks, neuron_kind="editable", detach=False) as forget_trace:
                forget_output = _forward_batch(updated_model, forget_batch)
            unlearn_loss = _activation_mse_to_random(
                forget_trace.activations,
                steering_coeff=config.steering_coeff,
                coeffs=config.coeffs,
            )
            forget_ce_loss = torch.tensor(0.0, device=_model_device(updated_model))
            if config.forget_ce_alpha != 0.0:
                forget_ce_loss = forget_output.loss.to(_model_device(updated_model))

            retain_loss = torch.tensor(0.0, device=_model_device(updated_model))
            shared_loss = torch.tensor(0.0, device=_model_device(updated_model))
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
                    _model_device(updated_model)
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
                    ).to(_model_device(updated_model))

            loss = (
                config.beta * unlearn_loss
                + config.alpha * retain_loss
                + config.shared_alpha * shared_loss
                - config.forget_ce_alpha * forget_ce_loss
            )
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            record = {
                "epoch": epoch,
                "step": step,
                "loss": float(loss.detach().cpu().item()),
                "unlearn_loss": float(unlearn_loss.detach().cpu().item()),
                "forget_ce_loss": float(forget_ce_loss.detach().cpu().item()),
                "retain_loss": float(retain_loss.detach().cpu().item()),
                "shared_loss": float(shared_loss.detach().cpu().item()),
            }
            losses.append(record)
            if step % 10 == 0:
                iterator.set_postfix(
                    loss=record["loss"],
                    unlearn=record["unlearn_loss"],
                    forget_ce=record["forget_ce_loss"],
                    retain=record["retain_loss"],
                    shared=record["shared_loss"],
                )

        if config.max_steps is not None and len(losses) >= config.max_steps:
            break

    summary = {
        "mask_summary": mask_summary,
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
