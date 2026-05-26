import sys
from pathlib import Path
from types import SimpleNamespace

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn

from causal_mip.causal_scores.metrics import compute_path_causal_score_record
from causal_mip.causal_scores.necessity import compute_necessity
from causal_mip.causal_scores.retain_impact import compute_retain_impact
from causal_mip.causal_scores.sufficiency import compute_sufficiency
from causal_mip.interventions.activation_cache import PreparedSampleBatch
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
    def __init__(self, hidden_size: int, num_layers: int):
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
        self.model = ToyWrapper(hidden_size, num_layers)
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


def _prepared_batch(
    input_ids: torch.Tensor,
    answer_positions=None,
    pixel_values: torch.Tensor | None = None,
) -> PreparedSampleBatch:
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
        answer_token_positions=answer_positions or [2, 3],
        all_token_positions=[0, 1, 2, 3],
        prompt_length=2,
        target_answer_text="a",
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


def _retain_batches():
    return {
        "same_topic": _prepared_batch(torch.tensor([[1, 2, 4, 5]])),
        "same_reasoning": _prepared_batch(torch.tensor([[1, 2, 6, 7]])),
        "counterfactual_retain": _prepared_batch(torch.tensor([[1, 2, 8, 9]])),
    }


def _vision_retain_batches(hidden_size: int = 8):
    return {
        "same_topic": _prepared_batch(torch.tensor([[1, 2, 4, 5]]), pixel_values=torch.randn(5, hidden_size)),
        "same_reasoning": _prepared_batch(torch.tensor([[1, 2, 6, 7]]), pixel_values=torch.randn(5, hidden_size)),
        "counterfactual_retain": _prepared_batch(torch.tensor([[1, 2, 8, 9]]), pixel_values=torch.randn(5, hidden_size)),
    }


def test_causal_scores():
    torch.manual_seed(7)
    model = ToyModel()
    clean_batch = _prepared_batch(torch.tensor([[1, 2, 3, 4]]))
    corrupt_batch = _prepared_batch(torch.tensor([[1, 2, 7, 8]]))
    candidate_path = _candidate_path()

    necessity = compute_necessity(model, clean_batch, candidate_path, strict=True)
    sufficiency = compute_sufficiency(model, clean_batch, corrupt_batch, candidate_path, strict=True)
    retain = compute_retain_impact(model, _retain_batches(), candidate_path, strict=True)

    assert necessity["status"] == "ok"
    assert sufficiency["status"] == "ok"
    assert retain["status"] == "ok"
    assert necessity["num_patchable_nodes"] == 2
    assert sufficiency["num_patchable_nodes"] == 2
    assert retain["num_patchable_nodes"] == 2
    assert set(retain["retain_details"].keys()) == {"same_topic", "same_reasoning", "counterfactual_retain"}

    record = compute_path_causal_score_record(
        model=model,
        candidate_path=candidate_path,
        pair={"pair_id": "pair_000000"},
        prepared_batches={
            "forget_clean": clean_batch,
            "forget_corrupt_target_clean_answer": corrupt_batch,
            **_retain_batches(),
        },
        strict=True,
    )
    assert record["status"] == "ok"
    assert record["Nec"] == necessity["necessity"]
    assert record["Suf"] == sufficiency["sufficiency"]
    assert record["Ret"] == retain["retain_impact"]


def test_vision_path_causal_scores():
    torch.manual_seed(13)
    model = ToyModel()
    candidate_path = CandidatePath(
        path_id="vision_only",
        source="test",
        modality="vision",
        mip_score=0.5,
        nodes=[PathNode("model.visual.blocks.0.mlp.down_proj", 0, 1, "image_tokens")],
    )
    clean_batch = _prepared_batch(torch.tensor([[1, 2, 3, 4]]), pixel_values=torch.randn(5, 8))
    corrupt_batch = _prepared_batch(torch.tensor([[1, 2, 7, 8]]), pixel_values=torch.randn(5, 8))
    record = compute_path_causal_score_record(
        model=model,
        candidate_path=candidate_path,
        pair={"pair_id": "pair_vision"},
        prepared_batches={
            "forget_clean": clean_batch,
            "forget_corrupt_target_clean_answer": corrupt_batch,
            **_vision_retain_batches(),
        },
        strict=False,
    )
    assert record["status"] == "ok"
    assert record["num_patchable_nodes"] == 1
    assert record["Nec"] is not None
    assert record["Suf"] is not None
    assert record["Ret"] is not None


def main():
    test_causal_scores()
    test_vision_path_causal_scores()
    print("Step 5 causal-score tests passed.")


if __name__ == "__main__":
    main()
