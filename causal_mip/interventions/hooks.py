"""
Minimal hook utilities for Step 4 activation tracing and patching.

This file is adapted from the tracing pattern used in ROME's `util/nethook.py`,
but kept local to `causal_mip` so Step 4 does not depend on the ROME package
layout at runtime.
"""

from __future__ import annotations

import contextlib
import inspect
from collections import OrderedDict
from typing import Any

import torch


class StopForward(Exception):
    pass


def get_module(root: torch.nn.Module, dotted_name: str) -> torch.nn.Module:
    current: Any = root
    for attr in dotted_name.split("."):
        if not hasattr(current, attr):
            raise KeyError(f"Module path not found: {dotted_name}")
        current = getattr(current, attr)
    if not isinstance(current, torch.nn.Module):
        raise KeyError(f"Resolved object is not a module: {dotted_name}")
    return current


def recursive_copy(x: Any, clone: bool = False, detach: bool = False, retain_grad: bool = False):
    if not clone and not detach and not retain_grad:
        return x

    if isinstance(x, torch.Tensor):
        if retain_grad:
            if not x.requires_grad:
                x.requires_grad = True
            x.retain_grad()
        elif detach:
            x = x.detach()
        if clone:
            x = x.clone()
        return x

    if isinstance(x, dict):
        return type(x)({key: recursive_copy(value, clone=clone, detach=detach, retain_grad=retain_grad) for key, value in x.items()})
    if isinstance(x, list):
        return [recursive_copy(value, clone=clone, detach=detach, retain_grad=retain_grad) for value in x]
    if isinstance(x, tuple):
        return tuple(recursive_copy(value, clone=clone, detach=detach, retain_grad=retain_grad) for value in x)
    return x


def invoke_with_optional_args(fn, **kwargs):
    signature = inspect.signature(fn)
    accepted = {key: value for key, value in kwargs.items() if key in signature.parameters}
    return fn(**accepted)


class Trace(contextlib.AbstractContextManager):
    def __init__(
        self,
        module: torch.nn.Module,
        layer: str | None = None,
        retain_output: bool = True,
        retain_input: bool = False,
        clone: bool = False,
        detach: bool = False,
        retain_grad: bool = False,
        edit_input=None,
        edit_output=None,
        stop: bool = False,
    ):
        self.layer = layer
        self.stop = stop
        target_module = get_module(module, layer) if layer is not None else module

        def pre_hook_fn(current_module, inputs):
            if edit_input is None:
                return None
            input_value = inputs[0] if len(inputs) == 1 else inputs
            edited_input = invoke_with_optional_args(edit_input, input=input_value, layer=self.layer)
            if len(inputs) == 1:
                return (edited_input,)
            return edited_input

        def hook_fn(current_module, inputs, output):
            if retain_input:
                self.input = recursive_copy(
                    inputs[0] if len(inputs) == 1 else inputs,
                    clone=clone,
                    detach=detach,
                    retain_grad=False,
                )
            if edit_output is not None:
                output = invoke_with_optional_args(edit_output, output=output, layer=self.layer)
            if retain_output:
                self.output = recursive_copy(
                    output,
                    clone=clone,
                    detach=detach,
                    retain_grad=retain_grad,
                )
                if retain_grad:
                    output = recursive_copy(self.output, clone=True, detach=False, retain_grad=False)
            if stop:
                raise StopForward()
            return output

        self.registered_pre_hook = (
            target_module.register_forward_pre_hook(pre_hook_fn)
            if edit_input is not None
            else None
        )
        self.registered_hook = target_module.register_forward_hook(hook_fn)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        if self.stop and exc_type is not None and issubclass(exc_type, StopForward):
            return True
        return None

    def close(self):
        if self.registered_pre_hook is not None:
            self.registered_pre_hook.remove()
        self.registered_hook.remove()


class TraceDict(OrderedDict, contextlib.AbstractContextManager):
    def __init__(
        self,
        module: torch.nn.Module,
        layers: list[str] | tuple[str, ...] | None = None,
        retain_output: bool = True,
        retain_input: bool = False,
        clone: bool = False,
        detach: bool = False,
        retain_grad: bool = False,
        edit_input=None,
        edit_output=None,
        stop: bool = False,
    ):
        self.stop = stop
        layers = list(layers or [])

        seen = set()
        unique_layers = []
        for layer in layers:
            if layer not in seen:
                unique_layers.append(layer)
                seen.add(layer)

        for index, layer in enumerate(unique_layers):
            self[layer] = Trace(
                module=module,
                layer=layer,
                retain_output=retain_output,
                retain_input=retain_input,
                clone=clone,
                detach=detach,
                retain_grad=retain_grad,
                edit_input=edit_input,
                edit_output=edit_output,
                stop=stop and index == len(unique_layers) - 1,
            )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        if self.stop and exc_type is not None and issubclass(exc_type, StopForward):
            return True
        return None

    def close(self):
        for _, trace in reversed(self.items()):
            trace.close()
