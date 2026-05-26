from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Literal

import torch

from causal_mip.interventions.hooks import TraceDict

ALL_VISUAL_TOKEN_POSITIONS = [-1]


@dataclass
class NodeIntervention:
    module: str
    token_positions: list[int]
    neuron: int
    mode: Literal["restore", "zero", "noise"]
    values: torch.Tensor | None = None
    noise_scale: float = 1.0


def _untuple(output):
    return output[0] if isinstance(output, tuple) else output


def _repack(original_output, new_tensor):
    if isinstance(original_output, tuple):
        return (new_tensor,) + tuple(original_output[1:])
    return new_tensor


def _group_interventions(interventions: list[NodeIntervention]) -> dict[str, list[NodeIntervention]]:
    grouped: dict[str, list[NodeIntervention]] = defaultdict(list)
    for intervention in interventions:
        grouped[intervention.module].append(intervention)
    return grouped


def _expand_token_positions(token_positions: list[int], tensor: torch.Tensor) -> list[int]:
    if token_positions == ALL_VISUAL_TOKEN_POSITIONS:
        if tensor.ndim < 2:
            return []
        return list(range(tensor.shape[-2]))
    return list(token_positions)


def _select_neuron_tokens(tensor: torch.Tensor, token_positions: list[int], neuron: int) -> torch.Tensor:
    if tensor.ndim == 3:
        return tensor[:, token_positions, neuron]
    if tensor.ndim == 2:
        return tensor[token_positions, neuron]
    raise ValueError(f"Unsupported activation rank for patching: ndim={tensor.ndim}")


def _assign_neuron_tokens(
    tensor: torch.Tensor,
    token_positions: list[int],
    neuron: int,
    values: torch.Tensor | float,
) -> None:
    if tensor.ndim == 3:
        tensor[:, token_positions, neuron] = values
        return
    if tensor.ndim == 2:
        tensor[token_positions, neuron] = values
        return
    raise ValueError(f"Unsupported activation rank for patching: ndim={tensor.ndim}")


def build_edit_input(interventions: list[NodeIntervention]):
    grouped = _group_interventions(interventions)

    def edit_input(input, layer):
        if layer not in grouped:
            return input

        tensor = _untuple(input)
        patched = tensor.clone()

        for intervention in grouped[layer]:
            if not intervention.token_positions:
                continue
            token_positions = _expand_token_positions(intervention.token_positions, patched)
            if not token_positions:
                continue
            neuron = intervention.neuron

            if intervention.mode == "zero":
                _assign_neuron_tokens(patched, token_positions, neuron, 0)
                continue

            if intervention.mode == "noise":
                noise = intervention.noise_scale * torch.randn(
                    _select_neuron_tokens(patched, token_positions, neuron).shape,
                    device=patched.device,
                    dtype=patched.dtype,
                )
                current_values = _select_neuron_tokens(patched, token_positions, neuron)
                _assign_neuron_tokens(patched, token_positions, neuron, current_values + noise)
                continue

            if intervention.mode == "restore":
                if intervention.values is None:
                    raise ValueError(f"Restore intervention for {layer} is missing values")
                restore_values = intervention.values.to(device=patched.device, dtype=patched.dtype)
                _assign_neuron_tokens(patched, token_positions, neuron, restore_values)
                continue

            raise ValueError(f"Unsupported intervention mode: {intervention.mode}")

        return _repack(input, patched)

    return edit_input


def run_patched_forward(
    model,
    model_inputs: dict[str, torch.Tensor | None],
    interventions: list[NodeIntervention],
    trace_layers: list[str] | None = None,
    no_grad: bool = True,
):
    intervention_layers = [intervention.module for intervention in interventions]
    all_layers = intervention_layers + list(trace_layers or [])
    edit_input = build_edit_input(interventions)

    if no_grad:
        context = torch.no_grad()
    else:
        from contextlib import nullcontext

        context = nullcontext()

    with context, TraceDict(model, all_layers, clone=True, detach=True, retain_input=True, edit_input=edit_input) as traces:
        outputs = model(**model_inputs)
    return outputs, traces
