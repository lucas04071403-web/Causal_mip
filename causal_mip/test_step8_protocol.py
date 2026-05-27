import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from causal_mip.evaluation.step8_protocol import build_step8_protocol_report


def _pair_eval(forget=0.5, hard=0.75, counter=0.7):
    return {
        "summary": {
            "by_eval_set": {
                "forget_clean": {"name_hit_rate": forget},
                "hard_retain": {"name_hit_rate": hard},
                "counterfactual_retain": {"name_hit_rate": counter},
            }
        }
    }


def _full_clear_summary(forget_clf=0.4, forget_gen=0.3, retain_clf=0.8, retain_gen=0.7):
    return {
        "metrics": {
            "forget_classification_remote_acc": forget_clf,
            "forget_generation_remote_acc": forget_gen,
            "retain_classification_remote_acc": retain_clf,
            "retain_generation_remote_acc": retain_gen,
        },
        "tasks": {
            "clf_forget": {"num_examples": 10, "num_scored_examples": 10},
            "clf_retain": {"num_examples": 10, "num_scored_examples": 10},
            "gen_forget": {"num_examples": 10, "num_scored_examples": 10},
            "gen_retain": {"num_examples": 10, "num_scored_examples": 10},
        },
    }


def test_step8_protocol_passes_when_pair_and_full_clear_pass():
    report = build_step8_protocol_report(
        candidate_pair_eval=_pair_eval(forget=0.4, hard=0.72, counter=0.68),
        baseline_pair_eval=_pair_eval(forget=0.5, hard=0.75, counter=0.7),
        candidate_full_clear_summary=_full_clear_summary(
            forget_clf=0.35,
            forget_gen=0.25,
            retain_clf=0.77,
            retain_gen=0.68,
        ),
        baseline_full_clear_summary=_full_clear_summary(
            forget_clf=0.45,
            forget_gen=0.35,
            retain_clf=0.8,
            retain_gen=0.7,
        ),
    )
    assert report["decision"]["can_claim_success"] is True
    assert report["decision"]["status"] == "pass"


def test_step8_protocol_fails_when_pair_forget_does_not_drop():
    report = build_step8_protocol_report(
        candidate_pair_eval=_pair_eval(forget=0.5, hard=0.75, counter=0.7),
        baseline_pair_eval=_pair_eval(forget=0.5, hard=0.75, counter=0.7),
        candidate_full_clear_summary=_full_clear_summary(forget_clf=0.35, forget_gen=0.25),
        baseline_full_clear_summary=_full_clear_summary(forget_clf=0.45, forget_gen=0.35),
    )
    assert report["decision"]["can_claim_success"] is False
    assert "pair_screen:pair_forget_clean_name_hit_rate" in report["decision"]["failed_reasons"]


def test_step8_protocol_fails_without_full_clear_summary():
    report = build_step8_protocol_report(
        candidate_pair_eval=_pair_eval(forget=0.4, hard=0.75, counter=0.7),
        baseline_pair_eval=_pair_eval(forget=0.5, hard=0.75, counter=0.7),
        candidate_full_clear_summary=None,
        baseline_full_clear_summary=_full_clear_summary(),
    )
    assert report["decision"]["can_claim_success"] is False
    assert "full_clear_main:missing_candidate_full_clear_summary" in report["decision"]["failed_reasons"]


def test_step8_protocol_fails_when_full_clear_incomplete():
    candidate = _full_clear_summary(forget_clf=0.35, forget_gen=0.25)
    candidate["tasks"]["gen_retain"]["num_scored_examples"] = 9
    report = build_step8_protocol_report(
        candidate_pair_eval=_pair_eval(forget=0.4, hard=0.75, counter=0.7),
        baseline_pair_eval=_pair_eval(forget=0.5, hard=0.75, counter=0.7),
        candidate_full_clear_summary=candidate,
        baseline_full_clear_summary=_full_clear_summary(forget_clf=0.45, forget_gen=0.35),
    )
    assert report["decision"]["can_claim_success"] is False
    assert "full_clear_main:full_clear_gen_retain_complete" in report["decision"]["failed_reasons"]


def main():
    test_step8_protocol_passes_when_pair_and_full_clear_pass()
    test_step8_protocol_fails_when_pair_forget_does_not_drop()
    test_step8_protocol_fails_without_full_clear_summary()
    test_step8_protocol_fails_when_full_clear_incomplete()
    print("Step 8 protocol tests passed.")


if __name__ == "__main__":
    main()
