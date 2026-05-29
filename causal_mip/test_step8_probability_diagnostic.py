import math
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from causal_mip.evaluation.step8_probability_diagnostic import (
    compare_model_summaries,
    compute_logprob_stats,
    find_subsequence_positions,
    summarize_records,
)


def _record(eval_set, sample_type, name_ce, name_logprob, answer_ce, generated_name_logprob=None):
    generated_name_score = None
    if generated_name_logprob is not None:
        generated_name_score = {
            "name": {
                "mean_logprob": generated_name_logprob,
            }
        }
    return {
        "eval_set": eval_set,
        "sample_type": sample_type,
        "target_score": {
            "name": {
                "ce": name_ce,
                "mean_logprob": name_logprob,
            },
            "answer": {
                "ce": answer_ce,
            },
        },
        "generated_name_score": generated_name_score,
    }


def test_find_subsequence_positions_returns_first_match():
    assert find_subsequence_positions([4, 8, 15, 16, 15], [15, 16]) == [2, 3]
    assert find_subsequence_positions([4, 8, 15], [8, 15, 16]) == []
    assert find_subsequence_positions([4, 8, 15], []) == []


def test_compute_logprob_stats_scores_target_positions_with_causal_shift():
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


def test_compare_model_summaries_reports_candidate_minus_baseline_delta():
    baseline = {
        "summary": summarize_records(
            [
                _record("forget_clean", "forget_clean", 2.0, -2.0, 1.5, -1.0),
                _record("counterfactual_retain", "counterfactual_retain", 1.0, -1.0, 1.2, -1.5),
            ]
        )
    }
    candidate = {
        "summary": summarize_records(
            [
                _record("forget_clean", "forget_clean", 3.0, -3.0, 2.0, -1.0),
                _record("counterfactual_retain", "counterfactual_retain", 0.8, -0.8, 1.0, -1.4),
            ]
        )
    }

    comparison = compare_model_summaries(baseline, candidate)
    forget_group = next(
        group
        for group in comparison["groups"]
        if group["section"] == "by_eval_set" and group["group"] == "forget_clean"
    )
    retain_group = next(
        group
        for group in comparison["groups"]
        if group["section"] == "by_eval_set" and group["group"] == "counterfactual_retain"
    )

    assert forget_group["target_name_ce"]["delta"] == 1.0
    assert forget_group["target_name_mean_logprob"]["delta"] == -1.0
    assert retain_group["target_name_ce"]["delta"] == -0.19999999999999996


def main():
    test_find_subsequence_positions_returns_first_match()
    test_compute_logprob_stats_scores_target_positions_with_causal_shift()
    test_compare_model_summaries_reports_candidate_minus_baseline_delta()
    print("Step 8 probability diagnostic tests passed.")


if __name__ == "__main__":
    main()
