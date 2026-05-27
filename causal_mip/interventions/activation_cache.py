from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import json
from pathlib import Path
from typing import Any, Literal

from PIL import Image
import torch

from causal_mip.interventions.hooks import TraceDict, get_module
from causal_mip.path_localization.path_schema import CandidatePath

ALL_VISUAL_TOKEN_POSITIONS = [-1]
WHOLE_VECTOR_NEURON = -1
ModuleKind = Literal["llm", "vision", "projector"]


def _import_pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.ipc as ipc
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError(
            "Step 4 activation caching requires pyarrow for local dataset references."
        ) from exc
    return pa, ipc, pq


def _load_arrow_rows(dataset_dir: str) -> list[dict[str, Any]]:
    pa, ipc, _ = _import_pyarrow()
    rows: list[dict[str, Any]] = []
    for shard_path in sorted(Path(dataset_dir).glob("data-*.arrow")):
        with pa.memory_map(str(shard_path), "r") as source:
            reader = ipc.open_stream(source)
            table = reader.read_all()
        rows.extend(table.to_pylist())
    return rows


def _load_parquet_rows(parquet_path: str) -> list[dict[str, Any]]:
    _, _, pq = _import_pyarrow()
    table = pq.read_table(parquet_path)
    return table.to_pylist()


def _untuple(output):
    return output[0] if isinstance(output, tuple) else output


def get_model_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def load_candidate_paths_jsonl(path: str, limit: int | None = None) -> list[CandidatePath]:
    results: list[CandidatePath] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if limit is not None and idx >= limit:
                break
            results.append(CandidatePath.from_dict(json.loads(line)))
    return results


def load_pairs_jsonl(path: str, limit: int | None = None) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if limit is not None and idx >= limit:
                break
            results.append(json.loads(line))
    return results


@dataclass
class PreparedSampleBatch:
    sample: dict[str, Any]
    model_inputs: dict[str, Any]
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    image_token_positions: list[int]
    answer_token_positions: list[int]
    all_token_positions: list[int]
    prompt_length: int
    target_answer_text: str | None = None


@dataclass
class ResolvedPathNode:
    module: str
    layer: int | None
    neuron: int
    token_selector: str
    token_positions: list[int]
    module_kind: ModuleKind = "llm"


@dataclass
class CachedNodeActivation:
    module: str
    layer: int | None
    neuron: int
    token_selector: str
    token_positions: list[int]
    values: torch.Tensor
    module_kind: ModuleKind = "llm"


@dataclass
class CachedPathActivations:
    path_id: str
    modality: str
    nodes: list[CachedNodeActivation]
    target_answer_logprob: float | None = None


class SampleReferenceResolver:
    def __init__(self):
        self._dataset_cache: dict[str, list[dict[str, Any]]] = {}

    def _load_rows(self, dataset_path: str) -> list[dict[str, Any]]:
        if dataset_path not in self._dataset_cache:
            path = Path(dataset_path)
            if path.is_dir():
                self._dataset_cache[dataset_path] = _load_arrow_rows(dataset_path)
            elif path.suffix == ".parquet":
                self._dataset_cache[dataset_path] = _load_parquet_rows(dataset_path)
            else:
                raise ValueError(f"Unsupported dataset reference path: {dataset_path}")
        return self._dataset_cache[dataset_path]

    def _decode_image(self, raw_image: Any) -> Image.Image | None:
        if raw_image is None:
            return None
        if isinstance(raw_image, Image.Image):
            return raw_image.convert("RGB")
        if isinstance(raw_image, dict):
            image_bytes = raw_image.get("bytes")
            image_path = raw_image.get("path")
            if image_bytes is not None:
                return Image.open(BytesIO(image_bytes)).convert("RGB")
            if image_path:
                return Image.open(image_path).convert("RGB")
        return None

    def resolve_image(self, sample: dict[str, Any]) -> Image.Image | None:
        direct_image = sample.get("image")
        if direct_image is not None:
            return self._decode_image(direct_image)

        image_ref = sample.get("image_ref")
        if not image_ref:
            return None
        dataset_path = image_ref.get("dataset_path")
        row_idx = image_ref.get("row_idx")
        if dataset_path is None or row_idx is None:
            return None
        rows = self._load_rows(dataset_path)
        raw_row = rows[row_idx]
        return self._decode_image(raw_row.get("image"))


def extract_pair_sample(
    pair: dict[str, Any],
    variant: Literal["forget_clean", "forget_corrupt", "counterfactual_retain", "hard_retain"],
    hard_retain_type: str | None = None,
) -> dict[str, Any]:
    if variant != "hard_retain":
        return pair[variant]

    hard_retain = pair.get("hard_retain", [])
    if not hard_retain:
        raise ValueError(f"Pair {pair.get('pair_id')} has no hard_retain samples")
    if hard_retain_type is None:
        return hard_retain[0]
    for sample in hard_retain:
        if sample.get("type") == hard_retain_type:
            return sample
    raise ValueError(
        f"hard_retain_type={hard_retain_type} not found in pair {pair.get('pair_id')}"
    )


def _build_conversation(
    sample: dict[str, Any],
    include_answer: bool,
    include_image: bool,
    answer_text: str | None = None,
) -> list[dict[str, Any]]:
    user_content = []
    if include_image:
        user_content.append({"type": "image"})
    user_content.append({"type": "text", "text": sample.get("question", "")})

    conversation = [{"role": "user", "content": user_content}]
    if include_answer:
        conversation.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": answer_text if answer_text is not None else sample.get("answer", "")}],
            }
        )
    return conversation


def _build_processor_batch(
    processor,
    text: str,
    image: Image.Image | None,
) -> dict[str, Any]:
    if image is None:
        batch = processor(text=[text], padding=True, return_tensors="pt")
        batch["pixel_values"] = None
        batch["image_grid_thw"] = None
        return batch
    return processor(text=[text], images=[[image]], padding=True, return_tensors="pt")


def prepare_sample_batch(
    sample: dict[str, Any],
    processor,
    model,
    image_resize: int,
    resolver: SampleReferenceResolver | None = None,
    include_answer: bool = True,
    target_answer_text: str | None = None,
) -> PreparedSampleBatch:
    resolver = resolver or SampleReferenceResolver()
    image = resolver.resolve_image(sample)
    if image is not None:
        image = image.resize((image_resize, image_resize))

    include_image = image is not None
    full_conversation = _build_conversation(
        sample,
        include_answer=include_answer,
        include_image=include_image,
        answer_text=target_answer_text,
    )
    full_text = processor.apply_chat_template(
        full_conversation,
        tokenize=False,
        add_generation_prompt=not include_answer,
    )
    full_batch = _build_processor_batch(processor, full_text, image)

    if include_answer:
        prompt_conversation = _build_conversation(
            sample,
            include_answer=False,
            include_image=include_image,
        )
        prompt_text = processor.apply_chat_template(
            prompt_conversation,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_batch = _build_processor_batch(processor, prompt_text, image)
        prompt_length = int(prompt_batch["input_ids"].shape[1])
    else:
        prompt_length = int(full_batch["input_ids"].shape[1])

    input_ids = full_batch["input_ids"]
    attention_mask = full_batch["attention_mask"]
    labels = input_ids.clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100

    model_inputs = {
        "input_ids": input_ids.to(get_model_device(model)),
        "attention_mask": attention_mask.to(get_model_device(model)),
        "labels": labels.to(get_model_device(model)),
    }

    pixel_values = full_batch.get("pixel_values")
    if pixel_values is not None:
        model_inputs["pixel_values"] = pixel_values.to(get_model_device(model))
    else:
        model_inputs["pixel_values"] = None

    image_grid_thw = full_batch.get("image_grid_thw")
    if image_grid_thw is not None:
        model_inputs["image_grid_thw"] = image_grid_thw.to(get_model_device(model))

    image_token_id = getattr(model.config, "image_token_id", None)
    if image_token_id is None and hasattr(processor.tokenizer, "image_token_id"):
        image_token_id = processor.tokenizer.image_token_id

    input_ids_1d = input_ids[0]
    image_positions = (
        torch.nonzero(input_ids_1d == image_token_id, as_tuple=False).squeeze(-1).tolist()
        if image_token_id is not None
        else []
    )
    answer_positions = list(range(prompt_length, int(attention_mask[0].sum().item())))
    all_positions = torch.nonzero(attention_mask[0], as_tuple=False).squeeze(-1).tolist()

    return PreparedSampleBatch(
        sample=sample,
        model_inputs=model_inputs,
        input_ids=model_inputs["input_ids"],
        attention_mask=model_inputs["attention_mask"],
        labels=model_inputs["labels"],
        image_token_positions=image_positions,
        answer_token_positions=answer_positions,
        all_token_positions=all_positions,
        prompt_length=prompt_length,
        target_answer_text=target_answer_text if target_answer_text is not None else sample.get("answer"),
    )


def resolve_token_positions(prepared_batch: PreparedSampleBatch, token_selector: str) -> list[int]:
    if token_selector == "image_tokens":
        return prepared_batch.image_token_positions
    if token_selector == "answer_tokens":
        return prepared_batch.answer_token_positions
    if token_selector == "all_tokens":
        return prepared_batch.all_token_positions
    raise ValueError(f"Unsupported token_selector: {token_selector}")


def _is_patchable_llm_module(module_name: str) -> bool:
    allowed_prefixes = (
        "model.language_model.layers.",
        "base_model.model.language_model.layers.",
        "language_model.layers.",
        "model.layers.",
    )
    return module_name.startswith(allowed_prefixes) and module_name.endswith(".mlp.down_proj")


def _is_patchable_vision_module(module_name: str) -> bool:
    allowed_prefixes = (
        "model.visual.blocks.",
        "base_model.model.visual.blocks.",
        "visual.blocks.",
    )
    return module_name.startswith(allowed_prefixes) and module_name.endswith(".mlp.down_proj")


def _is_projector_module(module_name: str) -> bool:
    return module_name in {
        "mm_projector",
        "model.mm_projector",
        "base_model.model.mm_projector",
        "visual.merger",
        "model.visual.merger",
        "base_model.model.visual.merger",
        "base_model.model.model.visual.merger",
    }


def _resolve_projector_module_name(model, module_name: str) -> str | None:
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
            get_module(model, candidate)
            return candidate
        except KeyError:
            continue
    return None


def _get_patchable_module_kind(module_name: str) -> ModuleKind | None:
    if _is_patchable_llm_module(module_name):
        return "llm"
    if _is_patchable_vision_module(module_name):
        return "vision"
    if _is_projector_module(module_name):
        return "projector"
    return None


def _expand_token_positions_for_tensor(token_positions: list[int], tensor: torch.Tensor) -> list[int]:
    if token_positions == ALL_VISUAL_TOKEN_POSITIONS:
        if tensor.ndim < 2:
            return []
        token_dim = tensor.shape[-2]
        return list(range(token_dim))
    return list(token_positions)


def resolve_candidate_path_targets(
    candidate_path: CandidatePath,
    prepared_batch: PreparedSampleBatch,
    strict: bool = False,
    model=None,
) -> list[ResolvedPathNode]:
    resolved: list[ResolvedPathNode] = []
    for node in candidate_path.nodes:
        module_kind = _get_patchable_module_kind(node.module)
        if module_kind is None:
            if strict:
                raise ValueError(f"Unsupported Step 4 module in MVP path patching: {node.module}")
            continue
        resolved_module = node.module
        neuron = node.neuron
        if module_kind == "projector":
            neuron = WHOLE_VECTOR_NEURON
            if model is not None:
                resolved_projector = _resolve_projector_module_name(model, node.module)
                if resolved_projector is None:
                    if strict:
                        raise ValueError(f"Projector module not found for path node: {node.module}")
                    continue
                resolved_module = resolved_projector
        if module_kind in {"vision", "projector"} and node.token_selector == "image_tokens":
            token_positions = list(ALL_VISUAL_TOKEN_POSITIONS)
        else:
            token_positions = resolve_token_positions(prepared_batch, node.token_selector)
        if not token_positions:
            if strict:
                raise ValueError(
                    f"No token positions resolved for selector={node.token_selector} in path {candidate_path.path_id}"
                )
            continue
        resolved.append(
            ResolvedPathNode(
                module=resolved_module,
                layer=node.layer,
                neuron=neuron,
                token_selector=node.token_selector,
                token_positions=token_positions,
                module_kind=module_kind,
            )
        )

    if strict and not resolved:
        raise ValueError(f"No patchable nodes resolved for path {candidate_path.path_id}")
    return resolved


def compute_target_answer_logprob(
    logits: torch.Tensor,
    prepared_batch: PreparedSampleBatch,
    reduction: Literal["mean", "sum"] = "mean",
) -> torch.Tensor:
    answer_positions = [position for position in prepared_batch.answer_token_positions if position > 0]
    if not answer_positions:
        return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

    log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
    token_ids = prepared_batch.input_ids[:, 1:]
    contributions = []
    for answer_position in answer_positions:
        score_position = answer_position - 1
        token_id = token_ids[:, score_position]
        contributions.append(log_probs[:, score_position, :].gather(-1, token_id.unsqueeze(-1)).squeeze(-1))
    stacked = torch.stack(contributions, dim=0)
    if reduction == "sum":
        return stacked.sum()
    return stacked.mean()


def cache_candidate_path_activations(
    model,
    prepared_batch: PreparedSampleBatch,
    candidate_path: CandidatePath,
    strict: bool = False,
    no_grad: bool = True,
) -> CachedPathActivations:
    resolved_nodes = resolve_candidate_path_targets(
        candidate_path,
        prepared_batch,
        strict=strict,
        model=model,
    )
    trace_layers = [node.module for node in resolved_nodes]

    if no_grad:
        context = torch.no_grad()
    else:
        from contextlib import nullcontext

        context = nullcontext()

    with context, TraceDict(model, trace_layers, clone=True, detach=True, retain_input=True) as traces:
        outputs = model(**prepared_batch.model_inputs)

    cached_nodes: list[CachedNodeActivation] = []
    for node in resolved_nodes:
        module_input = _untuple(traces[node.module].input)
        token_positions = _expand_token_positions_for_tensor(node.token_positions, module_input)
        if node.neuron == WHOLE_VECTOR_NEURON:
            if module_input.ndim == 3:
                activation_slice = module_input[:, token_positions, :].detach().cpu()
            elif module_input.ndim == 2:
                activation_slice = module_input[token_positions, :].detach().cpu()
            else:
                raise ValueError(
                    f"Unsupported activation rank for {node.module}: ndim={module_input.ndim}"
                )
        elif module_input.ndim == 3:
            activation_slice = module_input[:, token_positions, node.neuron].detach().cpu()
        elif module_input.ndim == 2:
            activation_slice = module_input[token_positions, node.neuron].detach().cpu()
        else:
            raise ValueError(
                f"Unsupported activation rank for {node.module}: ndim={module_input.ndim}"
            )
        cached_nodes.append(
            CachedNodeActivation(
                module=node.module,
                layer=node.layer,
                neuron=node.neuron,
                token_selector=node.token_selector,
                token_positions=token_positions,
                values=activation_slice,
                module_kind=node.module_kind,
            )
        )

    target_logprob = None
    if hasattr(outputs, "logits"):
        target_logprob = float(
            compute_target_answer_logprob(outputs.logits.detach(), prepared_batch).detach().cpu().item()
        )

    return CachedPathActivations(
        path_id=candidate_path.path_id,
        modality=candidate_path.modality,
        nodes=cached_nodes,
        target_answer_logprob=target_logprob,
    )
