import json
import sys
import tempfile
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from causal_mip.causal_scores.classify_paths import (
    CATEGORY_FORGET,
    CATEGORY_IRRELEVANT,
    CATEGORY_RETAIN,
    CATEGORY_SHARED,
    classify_path_scores,
    load_score_records_jsonl,
    write_classified_paths,
)


def _record(path_id, pair_id, nec, suf, ret, modality="text"):
    return {
        "pair_id": pair_id,
        "path_id": path_id,
        "path_source": "test",
        "path_modality": modality,
        "mip_score": 1.0,
        "num_nodes": 2,
        "num_patchable_nodes": 2,
        "status": "ok",
        "Nec": nec,
        "Suf": suf,
        "Ret": ret,
    }


def test_explicit_threshold_classification():
    records = [
        _record("p_forget", "pair_0", 0.8, 0.2, 0.1),
        _record("p_forget", "pair_1", 0.6, 0.0, 0.0),
        _record("p_shared", "pair_0", 0.8, 0.1, 0.9),
        _record("p_retain", "pair_0", 0.1, 0.0, 0.8),
        _record("p_irrelevant", "pair_0", 0.0, 0.0, 0.0),
        _record("p_skipped", "pair_0", None, None, None),
    ]
    categories, summary = classify_path_scores(
        records,
        alpha=1.0,
        forget_threshold=0.5,
        retain_threshold=0.5,
    )

    assert [record["path_id"] for record in categories[CATEGORY_FORGET]] == ["p_forget"]
    assert [record["path_id"] for record in categories[CATEGORY_SHARED]] == ["p_shared"]
    assert [record["path_id"] for record in categories[CATEGORY_RETAIN]] == ["p_retain"]
    assert [record["path_id"] for record in categories[CATEGORY_IRRELEVANT]] == ["p_irrelevant"]
    assert summary["num_input_score_records"] == 6
    assert summary["num_skipped_score_records"] == 1
    assert summary["num_classified_paths"] == 4


def test_forget_requires_positive_sufficiency_by_default():
    records = [
        _record("p_nec_only", "pair_0", 0.8, 0.0, 0.1),
        _record("p_shared", "pair_0", 0.8, 0.1, 0.9),
        _record("p_retain", "pair_0", 0.1, 0.0, 0.8),
    ]
    categories, summary = classify_path_scores(
        records,
        alpha=1.0,
        forget_threshold=0.5,
        retain_threshold=0.5,
    )

    assert categories[CATEGORY_FORGET] == []
    assert [record["path_id"] for record in categories[CATEGORY_SHARED]] == ["p_shared"]
    assert [record["path_id"] for record in categories[CATEGORY_RETAIN]] == ["p_retain"]
    demoted = [record for record in categories[CATEGORY_IRRELEVANT] if record["path_id"] == "p_nec_only"]
    assert len(demoted) == 1
    assert demoted[0]["demoted_from"] == CATEGORY_FORGET
    assert demoted[0]["forget_eligibility"]["reasons"] == ["sufficiency_not_positive"]
    assert summary["eligibility"]["num_demoted_from_forget"] == 1


def test_forget_compatibility_can_allow_zero_sufficiency():
    records = [_record("p_nec_only", "pair_0", 0.8, 0.0, 0.1)]
    categories, summary = classify_path_scores(
        records,
        alpha=1.0,
        forget_threshold=0.5,
        retain_threshold=0.5,
        require_positive_forget_sufficiency=False,
    )

    assert [record["path_id"] for record in categories[CATEGORY_FORGET]] == ["p_nec_only"]
    assert summary["eligibility"]["num_demoted_from_forget"] == 0


def test_forget_requires_full_patchable_by_default():
    records = [
        {
            **_record("p_partial", "pair_0", 0.8, 0.2, 0.1),
            "num_nodes": 3,
            "num_patchable_nodes": 2,
        }
    ]
    categories, summary = classify_path_scores(
        records,
        alpha=1.0,
        forget_threshold=0.5,
        retain_threshold=0.5,
    )

    assert categories[CATEGORY_FORGET] == []
    demoted = categories[CATEGORY_IRRELEVANT][0]
    assert demoted["path_id"] == "p_partial"
    assert demoted["forget_eligibility"]["reasons"] == ["not_all_score_records_fully_patchable"]
    assert summary["eligibility"]["num_demoted_from_forget"] == 1


def test_forget_can_require_saliency_specificity():
    records = [
        {
            **_record("p_specific", "pair_0", 0.8, 0.2, 0.1),
            "forget_saliency": 0.4,
            "retain_anchor_saliency": 0.1,
            "saliency_specificity_margin": 0.3,
            "fisher_specificity_margin": 0.03,
        },
        _record("p_missing", "pair_0", 0.8, 0.2, 0.1),
    ]
    categories, summary = classify_path_scores(
        records,
        alpha=1.0,
        forget_threshold=0.5,
        retain_threshold=0.5,
        require_saliency_specificity=True,
        min_saliency_specificity=0.05,
    )

    assert [record["path_id"] for record in categories[CATEGORY_FORGET]] == ["p_specific"]
    demoted = [record for record in categories[CATEGORY_IRRELEVANT] if record["path_id"] == "p_missing"]
    assert demoted[0]["forget_eligibility"]["reasons"] == ["missing_saliency_specificity"]
    assert summary["eligibility"]["require_saliency_specificity"] is True


def test_high_retain_anchor_saliency_routes_forget_candidate_to_shared():
    records = [
        {
            **_record("p_high_anchor", "pair_0", 0.8, 0.2, 0.1),
            "forget_saliency": 0.4,
            "retain_anchor_saliency": 0.35,
            "saliency_specificity_margin": 0.05,
        }
    ]
    categories, summary = classify_path_scores(
        records,
        alpha=1.0,
        forget_threshold=0.5,
        retain_threshold=0.5,
        require_saliency_specificity=True,
        min_saliency_specificity=0.0,
        max_retain_anchor_saliency=0.2,
    )

    assert categories[CATEGORY_FORGET] == []
    assert [record["path_id"] for record in categories[CATEGORY_SHARED]] == ["p_high_anchor"]
    assert categories[CATEGORY_SHARED][0]["promoted_shared_by"] == "retain_anchor_saliency"
    assert summary["eligibility"]["num_forget_to_shared_by_retain_anchor"] == 1


def test_jsonl_io():
    records = [
        _record("p_forget", "pair_0", 1.0, 0.1, 0.0),
        _record("p_irrelevant", "pair_0", 0.0, 0.0, 0.0),
    ]
    with tempfile.TemporaryDirectory() as temp_dir:
        input_path = Path(temp_dir) / "scores.jsonl"
        with input_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")

        loaded = load_score_records_jsonl([str(input_path)])
        categories, summary = classify_path_scores(
            loaded,
            forget_threshold=0.5,
            retain_threshold=0.5,
        )
        outputs = write_classified_paths(categories, summary, str(Path(temp_dir) / "classified"))

        assert Path(outputs[CATEGORY_FORGET]).exists()
        assert Path(outputs[CATEGORY_SHARED]).exists()
        assert Path(outputs[CATEGORY_RETAIN]).exists()
        assert Path(outputs[CATEGORY_IRRELEVANT]).exists()
        assert Path(outputs["P_classified"]).exists()
        assert Path(outputs["summary"]).exists()


def main():
    test_explicit_threshold_classification()
    test_forget_requires_positive_sufficiency_by_default()
    test_forget_compatibility_can_allow_zero_sufficiency()
    test_forget_requires_full_patchable_by_default()
    test_forget_can_require_saliency_specificity()
    test_high_retain_anchor_saliency_routes_forget_candidate_to_shared()
    test_jsonl_io()
    print("Step 6 path-classification tests passed.")


if __name__ == "__main__":
    main()
