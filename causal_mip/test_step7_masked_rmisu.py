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
    _labels_for_answer_without_name_ce,
    _labels_for_redacted_positive_ce,
    _labels_for_target_ce,
    _targeted_entropy_loss,
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
        self.mm_projector = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden_states, pixel_values=None):
        if pixel_values is not None and pixel_values.ndim == 2:
            visual_states = self.visual(pixel_values)
            projected_states = self.mm_projector(visual_states)
            hidden_states = hidden_states + projected_states.mean(dim=0).view(1, 1, -1)
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


def _candidate(path_id, nodes, metadata=None):
    return {
        "path_id": path_id,
        "source": "test",
        "modality": "text",
        "mip_score": 1.0,
        "nodes": nodes,
        "metadata": metadata or {},
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


def _projector_node(neuron=0):
    return {
        "module": "mm_projector",
        "layer": None,
        "neuron": neuron,
        "token_selector": "image_tokens",
    }


def _projector_dim_candidate(path_id, nodes):
    return _candidate(path_id, nodes, metadata={"projector_dim_level": True})


def _batch(offset=0):
    input_ids = torch.tensor([[1 + offset, 2 + offset, 3 + offset, 4 + offset]]) % 20
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    pixel_values = torch.zeros(1, 3, 2, 2)
    image_grid_thw = torch.ones(1, 3, dtype=torch.long)
    return input_ids, attention_mask, pixel_values, image_grid_thw, labels, None


def _batch_with_items(offset=0):
    input_ids = torch.tensor([[1 + offset, 2 + offset, 3 + offset, 4 + offset]]) % 20
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    pixel_values = torch.zeros(1, 3, 2, 2)
    image_grid_thw = torch.ones(1, 3, dtype=torch.long)
    item_list = [{"answer_token_positions": [2, 3], "name_token_positions": [2]}]
    return input_ids, attention_mask, pixel_values, image_grid_thw, labels, item_list


def _batch_with_redacted_items(offset=0):
    input_ids = torch.tensor([[1 + offset, 2 + offset, 3 + offset, 4 + offset]]) % 20
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    pixel_values = torch.zeros(1, 3, 2, 2)
    image_grid_thw = torch.ones(1, 3, dtype=torch.long)
    item_list = [
        {
            "answer_token_positions": [2, 3],
            "name_token_positions": [2],
            "redacted_positive_token_ids": [9, 10],
            "redacted_name_token_ids": [9],
        }
    ]
    return input_ids, attention_mask, pixel_values, image_grid_thw, labels, item_list


def _batch_without_name_items(offset=0):
    input_ids = torch.tensor([[1 + offset, 2 + offset, 3 + offset, 4 + offset]]) % 20
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    pixel_values = torch.zeros(1, 3, 2, 2)
    image_grid_thw = torch.ones(1, 3, dtype=torch.long)
    item_list = [{"answer_token_positions": [2, 3], "name_token_positions": []}]
    return input_ids, attention_mask, pixel_values, image_grid_thw, labels, item_list


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


def test_projector_mask_is_reported_and_skipped():
    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        candidates_path = temp / "P_cand.jsonl"
        p_forget_path = temp / "P_forget.jsonl"
        p_shared_path = temp / "P_shared.jsonl"
        _write_jsonl(
            candidates_path,
            [
                _projector_dim_candidate("projector_forget_path", [_projector_node(0), _node(1, 3)]),
            ],
        )
        _write_jsonl(p_forget_path, [{"path_id": "projector_forget_path"}])
        _write_jsonl(p_shared_path, [])

        masks = build_path_neuron_masks(str(candidates_path), str(p_forget_path), str(p_shared_path))
        assert masks["mm_projector"].module_kind == "projector"
        assert masks["mm_projector"].editable_neurons == {0}

        model = ToyModel()
        summary = apply_masked_rmisu_parameter_mask(model, masks, projector_edit_mode="skip")
        skipped = {item["module"]: item["reason"] for item in summary["skipped_modules"]}
        assert skipped["mm_projector"] == "projector_editing_disabled"
        assert summary["num_modules"] == 1


def test_projector_linear_mask_build_and_parameter_wrap():
    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        candidates_path = temp / "P_cand.jsonl"
        p_forget_path = temp / "P_forget.jsonl"
        p_shared_path = temp / "P_shared.jsonl"
        _write_jsonl(
            candidates_path,
            [
                _projector_dim_candidate("projector_forget_path", [_projector_node(3), _node(1, 4)]),
                _projector_dim_candidate("projector_shared_path", [_projector_node(5)]),
            ],
        )
        _write_jsonl(p_forget_path, [{"path_id": "projector_forget_path"}])
        _write_jsonl(p_shared_path, [{"path_id": "projector_shared_path"}])

        masks = build_path_neuron_masks(str(candidates_path), str(p_forget_path), str(p_shared_path))
        model = ToyModel()
        summary = apply_masked_rmisu_parameter_mask(model, masks)
        projector_summary = {
            item["module"]: item
            for item in summary["modules"]
            if item["module"] == "mm_projector"
        }["mm_projector"]
        assert projector_summary["edit_module"] == "model.mm_projector"
        assert projector_summary["trace_module"] == "model.mm_projector"
        assert projector_summary["trace_kind"] == "output"
        assert projector_summary["active"] is True
        assert isinstance(model.model.mm_projector, PartialLinear)
        assert model.model.mm_projector.trainable_cols == [3]


def test_projector_placeholder_uses_whole_vector_mask():
    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        candidates_path = temp / "P_cand.jsonl"
        p_forget_path = temp / "P_forget.jsonl"
        p_shared_path = temp / "P_shared.jsonl"
        _write_jsonl(
            candidates_path,
            [
                _candidate("projector_forget_path", [_projector_node(0)]),
            ],
        )
        _write_jsonl(p_forget_path, [{"path_id": "projector_forget_path"}])
        _write_jsonl(p_shared_path, [])

        masks = build_path_neuron_masks(str(candidates_path), str(p_forget_path), str(p_shared_path))
        projector_mask = masks["mm_projector"]
        assert projector_mask.forget_neurons == {-1}
        assert projector_mask.editable_neurons == {-1}

        model = ToyModel()
        projector_width = model.model.mm_projector.out_features
        summary = apply_masked_rmisu_parameter_mask(model, masks)
        projector_summary = {
            item["module"]: item
            for item in summary["modules"]
            if item["module"] == "mm_projector"
        }["mm_projector"]
        assert projector_summary["uses_whole_vector_neuron"] is True
        assert projector_summary["num_forget_editable_neurons"] == projector_width
        assert isinstance(model.model.mm_projector, PartialLinear)
        assert model.model.mm_projector.trainable_cols == list(range(projector_width))


def test_projector_probe_overrides_shared_for_weak_edit():
    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        candidates_path = temp / "P_cand.jsonl"
        p_forget_path = temp / "P_forget.jsonl"
        p_shared_path = temp / "P_shared.jsonl"
        p_probe_path = temp / "P_projector_probe.jsonl"
        _write_jsonl(
            candidates_path,
            [
                _projector_dim_candidate("projector_probe_path", [_projector_node(3)]),
                _projector_dim_candidate("projector_shared_path", [_projector_node(3)]),
            ],
        )
        _write_jsonl(p_forget_path, [])
        _write_jsonl(p_shared_path, [{"path_id": "projector_shared_path"}])
        _write_jsonl(p_probe_path, [{"path_id": "projector_probe_path"}])

        masks = build_path_neuron_masks(
            str(candidates_path),
            str(p_forget_path),
            str(p_shared_path),
            str(p_probe_path),
        )
        projector_mask = masks["mm_projector"]
        assert projector_mask.editable_neurons == set()
        assert projector_mask.shared_neurons == {3}
        assert projector_mask.probe_neurons == {3}
        assert projector_mask.trainable_neurons == {3}

        model = ToyModel()
        summary = apply_masked_rmisu_parameter_mask(model, masks)
        projector_summary = {
            item["module"]: item
            for item in summary["modules"]
            if item["module"] == "mm_projector"
        }["mm_projector"]
        assert projector_summary["num_forget_editable_neurons"] == 0
        assert projector_summary["num_probe_neurons"] == 1
        assert projector_summary["editable_neurons"] == [3]
        assert model.model.mm_projector.trainable_cols == [3]


def test_projector_activation_objective_updates_projector_weights():
    torch.manual_seed(19)
    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        candidates_path = temp / "P_cand.jsonl"
        p_forget_path = temp / "P_forget.jsonl"
        p_shared_path = temp / "P_shared.jsonl"
        _write_jsonl(candidates_path, [_projector_dim_candidate("projector_forget_path", [_projector_node(3)])])
        _write_jsonl(p_forget_path, [{"path_id": "projector_forget_path"}])
        _write_jsonl(p_shared_path, [])

        updated_model = ToyModel()
        original_weight = updated_model.model.mm_projector.weight.detach().clone()
        retain_loader = [_vision_batch(0)]
        forget_loader = [_vision_batch(2)]
        config = MaskedRMisUConfig(
            candidate_paths_path=str(candidates_path),
            p_forget_path=str(p_forget_path),
            p_shared_path=str(p_shared_path),
            alpha=0.0,
            beta=0.1,
            shared_alpha=0.0,
            learning_rate=1e-2,
            epochs=1,
            save=False,
        )

        _, summary = masked_rmisu_finetune(
            updated_model=updated_model,
            frozen_model=None,
            retain_loader=retain_loader,
            forget_loader=forget_loader,
            config=config,
        )
        merged = updated_model.model.mm_projector.merge_to_linear()
        assert summary["mask_summary"]["modules"][0]["trace_kind"] == "output"
        assert not torch.equal(merged.weight.detach()[3], original_weight[3])
        unchanged_rows = [idx for idx in range(original_weight.shape[0]) if idx != 3]
        assert torch.allclose(merged.weight.detach()[unchanged_rows], original_weight[unchanged_rows])


def test_projector_probe_objective_updates_projector_weights():
    torch.manual_seed(23)
    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        candidates_path = temp / "P_cand.jsonl"
        p_forget_path = temp / "P_forget.jsonl"
        p_shared_path = temp / "P_shared.jsonl"
        p_probe_path = temp / "P_projector_probe.jsonl"
        _write_jsonl(candidates_path, [_projector_dim_candidate("projector_probe_path", [_projector_node(3)])])
        _write_jsonl(p_forget_path, [])
        _write_jsonl(p_shared_path, [])
        _write_jsonl(p_probe_path, [{"path_id": "projector_probe_path"}])

        updated_model = ToyModel()
        original_weight = updated_model.model.mm_projector.weight.detach().clone()
        retain_loader = [_vision_batch(0)]
        forget_loader = [_vision_batch(2)]
        config = MaskedRMisUConfig(
            candidate_paths_path=str(candidates_path),
            p_forget_path=str(p_forget_path),
            p_shared_path=str(p_shared_path),
            p_probe_path=str(p_probe_path),
            alpha=0.0,
            beta=0.0,
            probe_beta=0.1,
            shared_alpha=0.0,
            probe_steering_coeff=1.0,
            learning_rate=1e-2,
            epochs=1,
            save=False,
        )

        _, summary = masked_rmisu_finetune(
            updated_model=updated_model,
            frozen_model=None,
            retain_loader=retain_loader,
            forget_loader=forget_loader,
            config=config,
        )
        merged = updated_model.model.mm_projector.merge_to_linear()
        assert summary["mask_summary"]["num_probe_neurons"] == 1
        assert summary["losses"][0]["unlearn_loss"] == 0.0
        assert summary["losses"][0]["probe_loss"] > 0.0
        assert not torch.equal(merged.weight.detach()[3], original_weight[3])
        unchanged_rows = [idx for idx in range(original_weight.shape[0]) if idx != 3]
        assert torch.allclose(merged.weight.detach()[unchanged_rows], original_weight[unchanged_rows])


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


def test_masked_rmisu_ce_ascent_objective_smoke():
    torch.manual_seed(11)
    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        candidates_path = temp / "P_cand.jsonl"
        p_forget_path = temp / "P_forget.jsonl"
        p_shared_path = temp / "P_shared.jsonl"
        _write_jsonl(candidates_path, [_candidate("forget_path", [_node(0, 2)])])
        _write_jsonl(p_forget_path, [{"path_id": "forget_path"}])
        _write_jsonl(p_shared_path, [])

        updated_model = ToyModel()
        retain_loader = [_batch(0)]
        forget_loader = [_batch(2)]
        config = MaskedRMisUConfig(
            candidate_paths_path=str(candidates_path),
            p_forget_path=str(p_forget_path),
            p_shared_path=str(p_shared_path),
            alpha=0.0,
            beta=0.0,
            shared_alpha=0.0,
            forget_objective="ce_ascent",
            forget_ce_alpha=0.1,
            learning_rate=1e-4,
            epochs=1,
            save=False,
        )

        _, summary = masked_rmisu_finetune(
            updated_model=updated_model,
            frozen_model=None,
            retain_loader=retain_loader,
            forget_loader=forget_loader,
            config=config,
        )
        assert summary["num_loss_records"] == 1
        assert summary["losses"][0]["forget_objective"] == "ce_ascent"
        assert summary["losses"][0]["forget_ce_loss"] != 0.0


def test_targeted_forget_ce_masks_answer_and_name_tokens():
    batch = _batch_with_items(2)
    answer_labels, answer_count = _labels_for_target_ce(batch, "answer", torch.device("cpu"))
    name_labels, name_count = _labels_for_target_ce(batch, "name", torch.device("cpu"))
    answer_without_name_labels, answer_without_name_count = _labels_for_answer_without_name_ce(
        batch,
        torch.device("cpu"),
    )
    redacted_labels, redacted_count = _labels_for_redacted_positive_ce(
        _batch_with_redacted_items(2),
        torch.device("cpu"),
    )
    missing_name_labels, missing_name_count = _labels_for_target_ce(
        _batch_without_name_items(2),
        "name",
        torch.device("cpu"),
    )

    assert answer_count == 2
    assert name_count == 1
    assert answer_without_name_count == 1
    assert redacted_count == 1
    assert missing_name_labels is None
    assert missing_name_count == 0
    assert answer_labels[0, 0].item() == -100
    assert answer_labels[0, 2].item() != -100
    assert answer_labels[0, 3].item() != -100
    assert name_labels[0, 2].item() != -100
    assert name_labels[0, 3].item() == -100
    assert answer_without_name_labels[0, 2].item() == -100
    assert answer_without_name_labels[0, 3].item() != -100
    assert redacted_labels[0, 2].item() == 9
    assert redacted_labels[0, 3].item() == -100


def test_masked_rmisu_name_ce_ascent_objective_smoke():
    torch.manual_seed(13)
    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        candidates_path = temp / "P_cand.jsonl"
        p_forget_path = temp / "P_forget.jsonl"
        p_shared_path = temp / "P_shared.jsonl"
        _write_jsonl(candidates_path, [_candidate("forget_path", [_node(0, 2)])])
        _write_jsonl(p_forget_path, [{"path_id": "forget_path"}])
        _write_jsonl(p_shared_path, [])

        updated_model = ToyModel()
        retain_loader = [_batch_with_items(0)]
        forget_loader = [_batch_with_items(2)]
        config = MaskedRMisUConfig(
            candidate_paths_path=str(candidates_path),
            p_forget_path=str(p_forget_path),
            p_shared_path=str(p_shared_path),
            alpha=0.0,
            beta=0.0,
            shared_alpha=0.0,
            forget_objective="name_ce_ascent",
            forget_ce_alpha=0.1,
            learning_rate=1e-4,
            epochs=1,
            save=False,
        )

        _, summary = masked_rmisu_finetune(
            updated_model=updated_model,
            frozen_model=None,
            retain_loader=retain_loader,
            forget_loader=forget_loader,
            config=config,
        )
        assert summary["forget_config"]["target_ce_scope"] == "name"
        assert summary["losses"][0]["forget_ce_token_count"] == 1
        assert summary["losses"][0]["forget_ce_loss"] != 0.0


def test_masked_rmisu_name_preference_unlearning_objective_smoke():
    torch.manual_seed(17)
    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        candidates_path = temp / "P_cand.jsonl"
        p_forget_path = temp / "P_forget.jsonl"
        p_shared_path = temp / "P_shared.jsonl"
        _write_jsonl(candidates_path, [_candidate("forget_path", [_node(0, 2)])])
        _write_jsonl(p_forget_path, [{"path_id": "forget_path"}])
        _write_jsonl(p_shared_path, [])

        updated_model = ToyModel()
        retain_loader = [_batch_with_items(0)]
        forget_loader = [_batch_with_items(2)]
        config = MaskedRMisUConfig(
            candidate_paths_path=str(candidates_path),
            p_forget_path=str(p_forget_path),
            p_shared_path=str(p_shared_path),
            alpha=0.0,
            beta=0.0,
            shared_alpha=0.0,
            forget_objective="name_preference_unlearning",
            forget_ce_alpha=0.1,
            preference_positive_alpha=0.2,
            learning_rate=1e-4,
            epochs=1,
            save=False,
        )

        _, summary = masked_rmisu_finetune(
            updated_model=updated_model,
            frozen_model=None,
            retain_loader=retain_loader,
            forget_loader=forget_loader,
            config=config,
        )
        loss = summary["losses"][0]
        assert summary["forget_config"]["target_ce_scope"] == "name"
        assert summary["forget_config"]["preference_positive_alpha"] == 0.2
        assert loss["forget_ce_token_count"] == 1
        assert loss["preference_positive_token_count"] == 1
        assert loss["forget_ce_loss"] != 0.0
        assert loss["preference_positive_loss"] != 0.0


def test_masked_rmisu_redacted_name_preference_objective_smoke():
    torch.manual_seed(19)
    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        candidates_path = temp / "P_cand.jsonl"
        p_forget_path = temp / "P_forget.jsonl"
        p_shared_path = temp / "P_shared.jsonl"
        _write_jsonl(candidates_path, [_candidate("forget_path", [_node(0, 2)])])
        _write_jsonl(p_forget_path, [{"path_id": "forget_path"}])
        _write_jsonl(p_shared_path, [])

        updated_model = ToyModel()
        retain_loader = [_batch_with_redacted_items(0)]
        forget_loader = [_batch_with_redacted_items(2)]
        config = MaskedRMisUConfig(
            candidate_paths_path=str(candidates_path),
            p_forget_path=str(p_forget_path),
            p_shared_path=str(p_shared_path),
            alpha=0.0,
            beta=0.0,
            shared_alpha=0.0,
            forget_objective="redacted_name_preference",
            forget_ce_alpha=0.1,
            preference_positive_alpha=0.2,
            learning_rate=1e-4,
            epochs=1,
            save=False,
        )

        _, summary = masked_rmisu_finetune(
            updated_model=updated_model,
            frozen_model=None,
            retain_loader=retain_loader,
            forget_loader=forget_loader,
            config=config,
        )
        loss = summary["losses"][0]
        assert summary["forget_config"]["target_ce_scope"] == "name"
        assert summary["forget_config"]["preference_positive_alpha"] == 0.2
        assert loss["forget_ce_token_count"] == 1
        assert loss["preference_positive_token_count"] == 1
        assert loss["forget_ce_loss"] != 0.0
        assert loss["preference_positive_loss"] != 0.0


def test_masked_rmisu_bounded_name_ce_ascent_objective_smoke():
    torch.manual_seed(23)
    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        candidates_path = temp / "P_cand.jsonl"
        p_forget_path = temp / "P_forget.jsonl"
        p_shared_path = temp / "P_shared.jsonl"
        _write_jsonl(candidates_path, [_candidate("forget_path", [_node(0, 2)])])
        _write_jsonl(p_forget_path, [{"path_id": "forget_path"}])
        _write_jsonl(p_shared_path, [])

        updated_model = ToyModel()
        retain_loader = [_batch_with_items(0)]
        forget_loader = [_batch_with_items(2)]
        config = MaskedRMisUConfig(
            candidate_paths_path=str(candidates_path),
            p_forget_path=str(p_forget_path),
            p_shared_path=str(p_shared_path),
            alpha=0.0,
            beta=0.0,
            shared_alpha=0.0,
            forget_objective="bounded_name_ce_ascent",
            forget_ce_alpha=0.2,
            bounded_delta_l2_alpha=0.1,
            bounded_delta_max_norm=0.05,
            learning_rate=1e-3,
            epochs=1,
            save=False,
        )

        _, summary = masked_rmisu_finetune(
            updated_model=updated_model,
            frozen_model=None,
            retain_loader=retain_loader,
            forget_loader=forget_loader,
            config=config,
        )
        loss = summary["losses"][0]
        assert summary["forget_config"]["target_ce_scope"] == "name"
        assert summary["forget_config"]["bounded_delta_l2_alpha"] == 0.1
        assert summary["forget_config"]["bounded_delta_max_norm"] == 0.05
        assert loss["forget_ce_token_count"] == 1
        assert loss["forget_ce_loss"] != 0.0
        assert "bounded_delta_l2_loss" in loss


def test_masked_rmisu_pii_name_token_noise_objective_smoke():
    torch.manual_seed(29)
    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        candidates_path = temp / "P_cand.jsonl"
        p_forget_path = temp / "P_forget.jsonl"
        p_shared_path = temp / "P_shared.jsonl"
        _write_jsonl(candidates_path, [_candidate("forget_path", [_node(0, 2)])])
        _write_jsonl(p_forget_path, [{"path_id": "forget_path"}])
        _write_jsonl(p_shared_path, [])

        updated_model = ToyModel()
        retain_loader = [_batch_with_items(0)]
        forget_loader = [_batch_with_items(2)]
        config = MaskedRMisUConfig(
            candidate_paths_path=str(candidates_path),
            p_forget_path=str(p_forget_path),
            p_shared_path=str(p_shared_path),
            alpha=0.0,
            beta=0.0,
            shared_alpha=0.0,
            forget_objective="pii_name_token_noise",
            forget_ce_alpha=0.2,
            pii_noise_alpha=0.05,
            learning_rate=1e-4,
            epochs=1,
            save=False,
        )

        _, summary = masked_rmisu_finetune(
            updated_model=updated_model,
            frozen_model=None,
            retain_loader=retain_loader,
            forget_loader=forget_loader,
            config=config,
        )
        loss = summary["losses"][0]
        assert summary["forget_config"]["target_ce_scope"] == "name"
        assert summary["forget_config"]["pii_noise_alpha"] == 0.05
        assert loss["forget_ce_token_count"] == 1
        assert loss["forget_ce_loss"] != 0.0
        assert loss["pii_noise_loss"] != 0.0

        labels, _ = _labels_for_target_ce(_batch_with_items(2), "name", torch.device("cpu"))
        assert labels is not None
        assert _targeted_entropy_loss(torch.randn(1, 4, 32), labels) > 0


def test_masked_rmisu_counterfactual_anchor_smoke():
    torch.manual_seed(31)
    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        candidates_path = temp / "P_cand.jsonl"
        p_forget_path = temp / "P_forget.jsonl"
        p_shared_path = temp / "P_shared.jsonl"
        _write_jsonl(candidates_path, [_candidate("forget_path", [_node(0, 2)])])
        _write_jsonl(p_forget_path, [{"path_id": "forget_path"}])
        _write_jsonl(p_shared_path, [])

        updated_model = ToyModel()
        retain_loader = [_batch_with_items(0)]
        forget_loader = [_batch_with_items(2)]
        counterfactual_anchor_loader = [_batch_with_items(4)]
        config = MaskedRMisUConfig(
            candidate_paths_path=str(candidates_path),
            p_forget_path=str(p_forget_path),
            p_shared_path=str(p_shared_path),
            alpha=0.0,
            beta=0.0,
            shared_alpha=0.0,
            forget_objective="pii_name_token_noise",
            forget_ce_alpha=0.2,
            pii_noise_alpha=0.05,
            counterfactual_anchor_alpha=0.03,
            counterfactual_anchor_scope="name",
            learning_rate=1e-4,
            epochs=1,
            save=False,
        )

        _, summary = masked_rmisu_finetune(
            updated_model=updated_model,
            frozen_model=None,
            retain_loader=retain_loader,
            forget_loader=forget_loader,
            counterfactual_anchor_loader=counterfactual_anchor_loader,
            config=config,
        )
        loss = summary["losses"][0]
        assert summary["forget_config"]["counterfactual_anchor_alpha"] == 0.03
        assert summary["forget_config"]["counterfactual_anchor_scope"] == "name"
        assert loss["counterfactual_anchor_token_count"] == 1
        assert loss["counterfactual_anchor_loss"] != 0.0


def main():
    test_mask_build_and_parameter_wrap()
    test_vision_mask_build_and_parameter_wrap()
    test_projector_mask_is_reported_and_skipped()
    test_projector_linear_mask_build_and_parameter_wrap()
    test_projector_placeholder_uses_whole_vector_mask()
    test_projector_probe_overrides_shared_for_weak_edit()
    test_projector_activation_objective_updates_projector_weights()
    test_projector_probe_objective_updates_projector_weights()
    test_masked_rmisu_finetune_smoke()
    test_masked_rmisu_ce_ascent_objective_smoke()
    test_targeted_forget_ce_masks_answer_and_name_tokens()
    test_masked_rmisu_name_ce_ascent_objective_smoke()
    test_masked_rmisu_name_preference_unlearning_objective_smoke()
    test_masked_rmisu_redacted_name_preference_objective_smoke()
    test_masked_rmisu_bounded_name_ce_ascent_objective_smoke()
    test_masked_rmisu_pii_name_token_noise_objective_smoke()
    test_masked_rmisu_counterfactual_anchor_smoke()
    print("Step 7 masked RMisU tests passed.")


if __name__ == "__main__":
    main()
