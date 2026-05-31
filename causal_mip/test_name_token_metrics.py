import math
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from causal_mip.causal_scores.name_token_metrics import (
    compute_logprob_stats,
    find_name_token_positions,
    find_subsequence_positions,
)


class DummyTokenizer:
    def __call__(self, text, add_special_tokens=False):
        del add_special_tokens
        vocab = {
            "Alice": [11],
            "Alice Smith": [11, 12],
            "Bob": [21],
        }
        return {"input_ids": vocab.get(text, [])}


class DummyProcessor:
    tokenizer = DummyTokenizer()


def test_find_name_token_positions_scopes_to_answer_tokens():
    input_ids = torch.tensor([[3, 11, 12, 4, 11, 12, 5]])
    result = find_name_token_positions(
        processor_or_tokenizer=DummyProcessor(),
        input_ids=input_ids,
        answer_positions=[4, 5, 6],
        target_text="Alice Smith is here.",
        name_text="Alice Smith",
    )

    assert result["name_match_status"] == "matched"
    assert result["name_token_positions"] == [4, 5]
    assert result["name_token_count_expected"] == 2


def test_find_name_token_positions_prefix_fallback():
    input_ids = torch.tensor([[3, 101, 102, 103]])
    result = find_name_token_positions(
        processor_or_tokenizer=DummyProcessor(),
        input_ids=input_ids,
        answer_positions=[1, 2, 3],
        target_text="Alice Smith is here.",
        name_text="Alice Smith",
    )

    assert result["name_match_status"] == "matched"
    assert result["name_token_positions"] == [1, 2]


def test_compute_logprob_stats_uses_causal_shift():
    input_ids = torch.tensor([[1, 2, 3]])
    logits = torch.zeros(1, 3, 5)
    logits[0, 0, 2] = 4.0
    logits[0, 1, 3] = 2.0

    stats = compute_logprob_stats(logits, input_ids, [1, 2])

    first = torch.log_softmax(logits[0, 0], dim=-1)[2].item()
    second = torch.log_softmax(logits[0, 1], dim=-1)[3].item()
    expected = (first + second) / 2.0
    assert stats["token_count"] == 2
    assert math.isclose(stats["mean_logprob"], expected, rel_tol=1e-6)
    assert math.isclose(stats["ce"], -expected, rel_tol=1e-6)


def main():
    assert find_subsequence_positions([4, 8, 15, 16, 15], [15, 16]) == [2, 3]
    test_find_name_token_positions_scopes_to_answer_tokens()
    test_find_name_token_positions_prefix_fallback()
    test_compute_logprob_stats_uses_causal_shift()
    print("Name-token metric tests passed.")


if __name__ == "__main__":
    main()
