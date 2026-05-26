import sys
from pathlib import Path
from types import SimpleNamespace

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn

from causal_mip.interventions.ablation import ablate_candidate_path
from causal_mip.interventions.activation_cache import (
    PreparedSampleBatch,
    cache_candidate_path_activations,
    compute_target_answer_logprob,
    resolve_candidate_path_targets,
)
from causal_mip.interventions.restoration import restore_path_activations
from causal_mip.path_localization.path_schema import CandidatePath, PathNode


class ToyMLP(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.down_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden_states):
        return self.down_proj(hidden_states)


class ToyLayer(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.mlp = ToyMLP(hidden_size)

    def forward(self, hidden_states):
        return hidden_states + torch.tanh(self.mlp(hidden_states))


class ToyLanguageModel(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([ToyLayer(hidden_size) for _ in range(num_layers)])

    def forward(self, hidden_states):
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


class ToyVisionModel(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int):
        super().__init__()
        self.blocks = nn.ModuleList([ToyLayer(hidden_size) for _ in range(num_layers)])

    def forward(self, hidden_states):
        for block in self.blocks:
            hidden_states = block(hidden_states)
        return hidden_states


class ToyWrapper(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int, vocab_size: int):
        super().__init__()
        self.language_model = ToyLanguageModel(hidden_size, num_layers)
        self.visual = ToyVisionModel(hidden_size, num_layers)

    def forward(self, hidden_states, pixel_values=None):
        if pixel_values is not None:
            visual_states = self.visual(pixel_values)
            hidden_states = hidden_states + visual_states.mean(dim=0).view(1, 1, -1)
        return self.language_model(hidden_states)


class ToyModel(nn.Module):
    def __init__(self, vocab_size: int = 32, hidden_size: int = 8, num_layers: int = 2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.model = ToyWrapper(hidden_size, num_layers, vocab_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.config = SimpleNamespace(
            image_token_id=99,
            architectures=["ToyForConditionalGeneration"],
        )

    def forward(self, input_ids, attention_mask=None, labels=None, pixel_values=None, image_grid_thw=None):
        hidden_states = self.embed(input_ids)
        hidden_states = self.model(hidden_states, pixel_values=pixel_values)
        logits = self.lm_head(hidden_states)
        return SimpleNamespace(logits=logits)


def _prepared_batch(input_ids: torch.Tensor, pixel_values: torch.Tensor | None = None) -> PreparedSampleBatch:
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    return PreparedSampleBatch(
        sample={"question": "q", "answer": "a"},
        model_inputs={
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": pixel_values,
        },
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        image_token_positions=[],
        answer_token_positions=[2, 3],
        all_token_positions=[0, 1, 2, 3],
        prompt_length=2,
    )


def _candidate_path() -> CandidatePath:
    return CandidatePath(
        path_id="toy_text_path",
        source="test",
        modality="text",
        mip_score=1.0,
        nodes=[
            PathNode("model.language_model.layers.0.mlp.down_proj", 0, 2, "answer_tokens"),
            PathNode("model.language_model.layers.1.mlp.down_proj", 1, 3, "answer_tokens"),
        ],
    )


def test_resolve_candidate_path_targets():
    batch = _prepared_batch(torch.tensor([[1, 2, 3, 4]]), pixel_values=torch.randn(5, 8))
    path = CandidatePath(
        path_id="mixed_path",
        source="test",
        modality="vision_text",
        mip_score=1.0,
        nodes=[
            PathNode("model.visual.blocks.0.mlp.down_proj", 0, 1, "image_tokens"),
            PathNode("model.language_model.layers.0.mlp.down_proj", 0, 2, "answer_tokens"),
        ],
    )
    resolved = resolve_candidate_path_targets(path, batch, strict=False)
    assert len(resolved) == 2
    assert resolved[0].module == "model.visual.blocks.0.mlp.down_proj"
    assert resolved[0].module_kind == "vision"
    assert resolved[0].token_positions == [-1]
    assert resolved[1].module == "model.language_model.layers.0.mlp.down_proj"
    assert resolved[1].module_kind == "llm"


def test_cache_ablate_restore():
    torch.manual_seed(7)
    model = ToyModel()
    clean_batch = _prepared_batch(torch.tensor([[1, 2, 3, 4]]))
    corrupt_batch = _prepared_batch(torch.tensor([[1, 2, 7, 8]]))
    path = _candidate_path()

    clean_outputs = model(**clean_batch.model_inputs)
    clean_score = compute_target_answer_logprob(clean_outputs.logits, clean_batch).item()

    cached = cache_candidate_path_activations(model, clean_batch, path, strict=True)
    assert len(cached.nodes) == 2
    assert cached.target_answer_logprob is not None

    ablated_outputs, ablated_traces = ablate_candidate_path(model, clean_batch, path, strict=True)
    ablated_score = compute_target_answer_logprob(ablated_outputs.logits, clean_batch).item()
    assert ablated_score != clean_score
    for node in cached.nodes:
        traced = ablated_traces[node.module].input
        assert torch.allclose(
            traced[:, node.token_positions, node.neuron],
            torch.zeros_like(traced[:, node.token_positions, node.neuron]),
        )

    restored_outputs, restored_traces = restore_path_activations(model, corrupt_batch, cached)
    restored_score = compute_target_answer_logprob(restored_outputs.logits, corrupt_batch).item()
    assert restored_score != ablated_score
    for node in cached.nodes:
        traced = restored_traces[node.module].input
        expected = node.values.to(device=traced.device, dtype=traced.dtype)
        assert torch.allclose(traced[:, node.token_positions, node.neuron], expected)


def test_cache_ablate_restore_vision_path():
    torch.manual_seed(11)
    model = ToyModel()
    clean_pixels = torch.randn(5, 8)
    corrupt_pixels = torch.randn(5, 8)
    clean_batch = _prepared_batch(torch.tensor([[1, 2, 3, 4]]), pixel_values=clean_pixels)
    corrupt_batch = _prepared_batch(torch.tensor([[1, 2, 3, 4]]), pixel_values=corrupt_pixels)
    path = CandidatePath(
        path_id="toy_vision_path",
        source="test",
        modality="vision",
        mip_score=1.0,
        nodes=[PathNode("model.visual.blocks.0.mlp.down_proj", 0, 2, "image_tokens")],
    )

    clean_outputs = model(**clean_batch.model_inputs)
    clean_score = compute_target_answer_logprob(clean_outputs.logits, clean_batch).item()

    cached = cache_candidate_path_activations(model, clean_batch, path, strict=True)
    assert len(cached.nodes) == 1
    assert cached.nodes[0].module_kind == "vision"
    assert cached.nodes[0].token_positions == [0, 1, 2, 3, 4]
    assert cached.nodes[0].values.shape == (5,)

    ablated_outputs, ablated_traces = ablate_candidate_path(model, clean_batch, path, strict=True)
    ablated_score = compute_target_answer_logprob(ablated_outputs.logits, clean_batch).item()
    assert ablated_score != clean_score
    for node in cached.nodes:
        traced = ablated_traces[node.module].input
        assert torch.allclose(
            traced[node.token_positions, node.neuron],
            torch.zeros_like(traced[node.token_positions, node.neuron]),
        )

    restored_outputs, restored_traces = restore_path_activations(model, corrupt_batch, cached)
    restored_score = compute_target_answer_logprob(restored_outputs.logits, corrupt_batch).item()
    assert restored_score != ablated_score
    for node in cached.nodes:
        traced = restored_traces[node.module].input
        expected = node.values.to(device=traced.device, dtype=traced.dtype)
        assert torch.allclose(traced[node.token_positions, node.neuron], expected)


def main():
    test_resolve_candidate_path_targets()
    test_cache_ablate_restore()
    test_cache_ablate_restore_vision_path()
    print("Step 4 intervention tests passed.")


if __name__ == "__main__":
    main()
