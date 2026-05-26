import copy
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn

from causal_mip.editing.masked_rmisu import (
    MaskedRMisUConfig,
    apply_masked_rmisu_parameter_mask,
    build_path_neuron_masks,
    masked_rmisu_finetune,
)
from partial_linear import PartialLinear


class ToyMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act = nn.SiLU()

    def forward(self, hidden_states):
        return self.down_proj(self.act(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class ToyLayer(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.mlp = ToyMLP(hidden_size, intermediate_size)

    def forward(self, hidden_states):
        return hidden_states + self.mlp(hidden_states)


class ToyLanguageModel(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [ToyLayer(hidden_size, intermediate_size) for _ in range(num_layers)]
        )

    def forward(self, hidden_states):
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


class ToyVisionModel(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, num_layers: int):
        super().__init__()
        self.blocks = nn.ModuleList(
            [ToyLayer(hidden_size, intermediate_size) for _ in range(num_layers)]
        )

    def forward(self, hidden_states):
        for block in self.blocks:
            hidden_states = block(hidden_states)
        return hidden_states


class ToyWrapper(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, num_layers: int):
        super().__init__()
        self.language_model = ToyLanguageModel(hidden_size, intermediate_size, num_layers)
        self.visual = ToyVisionModel(hidden_size, intermediate_size, num_layers)

    def forward(self, hidden_states, pixel_values=None):
        if pixel_values is not None and pixel_values.ndim == 2:
            visual_states = self.visual(pixel_values)
            hidden_states = hidden_states + visual_states.mean(dim=0).view(1, 1, -1)
        return self.language_model(hidden_states)


class ToyModel(nn.Module):
    def __init__(self, vocab_size=32, hidden_size=8, intermediate_size=12, num_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.model = ToyWrapper(hidden_size, intermediate_size, num_layers)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.config = SimpleNamespace(
            architectures=["Qwen2.5-VLForConditionalGeneration"],
            _name_or_path="Qwen2.5-VL-3B-Instruct",
        )

    def forward(self, input_ids, attention_mask=None, pixel_values=None, image_grid_thw=None, labels=None):
        hidden_states = self.embed(input_ids)
        hidden_states = self.model(hidden_states, pixel_values=pixel_values)
        logits = self.lm_head(hidden_states)
        loss = logits.float().mean()
        return SimpleNamespace(logits=logits, loss=loss)


def _write_jsonl(path: Path, records):
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def _candidate(path_id, nodes):
    return {
        "path_id": path_id,
        "source": "test",
        "modality": "text",
        "mip_score": 1.0,
        "nodes": nodes,
    }


def _node(layer, neuron):
    return {
        "module": f"model.language_model.layers.{layer}.mlp.down_proj",
        "layer": layer,
        "neuron": neuron,
        "token_selector": "answer_tokens",
    }


def _vision_node(layer, neuron):
    return {
        "module": f"model.visual.blocks.{layer}.mlp.down_proj",
        "layer": layer,
        "neuron": neuron,
        "token_selector": "image_tokens",
    }


def _batch(offset=0):
    input_ids = torch.tensor([[1 + offset, 2 + offset, 3 + offset, 4 + offset]]) % 20
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    pixel_values = torch.zeros(1, 3, 2, 2)
    image_grid_thw = torch.ones(1, 3, dtype=torch.long)
    return input_ids, attention_mask, pixel_values, image_grid_thw, labels, None


def _vision_batch(offset=0):
    input_ids = torch.tensor([[1 + offset, 2 + offset, 3 + offset, 4 + offset]]) % 20
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    pixel_values = torch.randn(5, 8)
    image_grid_thw = torch.ones(1, 3, dtype=torch.long)
    return input_ids, attention_mask, pixel_values, image_grid_thw, labels, None


def test_mask_build_and_parameter_wrap():
    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        candidates_path = temp / "P_cand.jsonl"
        p_forget_path = temp / "P_forget.jsonl"
        p_shared_path = temp / "P_shared.jsonl"
        _write_jsonl(
            candidates_path,
            [
                _candidate("forget_path", [_node(0, 2), _node(1, 3)]),
                _candidate("shared_path", [_node(0, 2), _node(1, 5)]),
            ],
        )
        _write_jsonl(p_forget_path, [{"path_id": "forget_path"}])
        _write_jsonl(p_shared_path, [{"path_id": "shared_path"}])

        masks = build_path_neuron_masks(str(candidates_path), str(p_forget_path), str(p_shared_path))
        assert masks["model.language_model.layers.0.mlp.down_proj"].editable_neurons == set()
        assert masks["model.language_model.layers.1.mlp.down_proj"].editable_neurons == {3}
        assert masks["model.language_model.layers.1.mlp.down_proj"].shared_neurons == {5}

        model = ToyModel()
        summary = apply_masked_rmisu_parameter_mask(model, masks)
        assert summary["num_modules"] == 1
        layer_1_mlp = model.model.language_model.layers[1].mlp
        assert isinstance(layer_1_mlp.up_proj, PartialLinear)
        assert isinstance(layer_1_mlp.gate_proj, PartialLinear)
        assert layer_1_mlp.up_proj.trainable_cols == [3]


def test_vision_mask_build_and_parameter_wrap():
    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        candidates_path = temp / "P_cand.jsonl"
        p_forget_path = temp / "P_forget.jsonl"
        p_shared_path = temp / "P_shared.jsonl"
        _write_jsonl(
            candidates_path,
            [
                _candidate("vision_forget_path", [_vision_node(0, 2), _node(1, 3)]),
                _candidate("vision_shared_path", [_vision_node(0, 4)]),
            ],
        )
        _write_jsonl(p_forget_path, [{"path_id": "vision_forget_path"}])
        _write_jsonl(p_shared_path, [{"path_id": "vision_shared_path"}])

        masks = build_path_neuron_masks(str(candidates_path), str(p_forget_path), str(p_shared_path))
        vision_mask = masks["model.visual.blocks.0.mlp.down_proj"]
        assert vision_mask.module_kind == "vision"
        assert vision_mask.editable_neurons == {2}
        assert vision_mask.shared_neurons == {4}

        model = ToyModel()
        summary = apply_masked_rmisu_parameter_mask(model, masks)
        assert summary["num_modules"] == 2
        vision_mlp = model.model.visual.blocks[0].mlp
        language_mlp = model.model.language_model.layers[1].mlp
        assert isinstance(vision_mlp.up_proj, PartialLinear)
        assert isinstance(vision_mlp.gate_proj, PartialLinear)
        assert vision_mlp.up_proj.trainable_cols == [2]
        assert isinstance(language_mlp.up_proj, PartialLinear)


def test_masked_rmisu_finetune_smoke():
    torch.manual_seed(7)
    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        candidates_path = temp / "P_cand.jsonl"
        p_forget_path = temp / "P_forget.jsonl"
        p_shared_path = temp / "P_shared.jsonl"
        output_path = temp / "masked_rmisu_summary.json"
        _write_jsonl(
            candidates_path,
            [
                _candidate("forget_path", [_node(0, 2), _node(1, 3)]),
                _candidate("shared_path", [_node(0, 4), _node(1, 5)]),
            ],
        )
        _write_jsonl(p_forget_path, [{"path_id": "forget_path"}])
        _write_jsonl(p_shared_path, [{"path_id": "shared_path"}])

        updated_model = ToyModel()
        frozen_model = copy.deepcopy(updated_model)
        retain_loader = [_batch(0), _batch(1)]
        forget_loader = [_batch(2), _batch(3)]
        config = MaskedRMisUConfig(
            candidate_paths_path=str(candidates_path),
            p_forget_path=str(p_forget_path),
            p_shared_path=str(p_shared_path),
            alpha=0.1,
            beta=0.1,
            shared_alpha=0.1,
            learning_rate=1e-4,
            epochs=1,
            output_path=str(output_path),
        )

        _, summary = masked_rmisu_finetune(
            updated_model=updated_model,
            frozen_model=frozen_model,
            retain_loader=retain_loader,
            forget_loader=forget_loader,
            config=config,
        )
        assert summary["num_loss_records"] == 2
        assert output_path.exists()


def main():
    test_mask_build_and_parameter_wrap()
    test_vision_mask_build_and_parameter_wrap()
    test_masked_rmisu_finetune_smoke()
    print("Step 7 masked RMisU tests passed.")


if __name__ == "__main__":
    main()
