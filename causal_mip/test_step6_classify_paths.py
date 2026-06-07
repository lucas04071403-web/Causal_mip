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


def _record(path_id, pair_id, nec, suf, ret, modality="text", contains_projector=False):
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
        "contains_projector": contains_projector,
    }


def _name_record(path_id, pair_id, name_nec, name_suf, name_ret, modality="text", contains_projector=False):
    return {
        **_record(
            path_id,
            pair_id,
            nec=0.0,
            suf=0.0,
            ret=0.0,
            modality=modality,
            contains_projector=contains_projector,
        ),
        "NameNec": name_nec,
        "NameSuf": name_suf,
        "NameRet": name_ret,
        "target_name": "Alice Smith",
        "name_match_status": "matched",
    }


def _name_projector_dim_record(path_id, pair_id, name_nec, name_suf, name_ret, salun_score):
    return {
        **_name_record(
            path_id,
            pair_id,
            name_nec=name_nec,
            name_suf=name_suf,
            name_ret=name_ret,
            modality="vision_text",
            contains_projector=True,
        ),
        "candidate_metadata": {
            "selected_projector_dims": [
                {"dim_index": 1, "salun_ssd_score": salun_score, "fisher_specificity_margin": 0.1},
                {"dim_index": 2, "salun_ssd_score": salun_score, "fisher_specificity_margin": 0.1},
            ]
        },
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


def test_forget_can_require_worst_anchor_specificity():
    records = [
        {
            **_record("p_stable", "pair_0", 0.8, 0.2, 0.1),
            "saliency_specificity_margin": 0.3,
            "min_anchor_margin": 0.08,
            "max_anchor_retain_saliency": 0.12,
        },
        {
            **_record("p_mean_only", "pair_0", 0.8, 0.2, 0.1),
            "saliency_specificity_margin": 0.3,
            "min_anchor_margin": -0.01,
            "max_anchor_retain_saliency": 0.45,
        },
    ]
    categories, summary = classify_path_scores(
        records,
        alpha=1.0,
        forget_threshold=0.5,
        retain_threshold=0.5,
        require_saliency_specificity=True,
        saliency_specificity_key="min_anchor_margin",
        min_saliency_specificity=0.0,
        max_retain_anchor_saliency=0.2,
        retain_anchor_saliency_key="max_anchor_retain_saliency",
    )

    assert [record["path_id"] for record in categories[CATEGORY_FORGET]] == ["p_stable"]
    assert [record["path_id"] for record in categories[CATEGORY_SHARED]] == ["p_mean_only"]
    assert categories[CATEGORY_SHARED][0]["promoted_shared_by"] == "retain_anchor_saliency"
    assert summary["eligibility"]["saliency_specificity_key"] == "min_anchor_margin"
    assert summary["eligibility"]["retain_anchor_saliency_key"] == "max_anchor_retain_saliency"


def test_pair_level_classification_preserves_pair_specific_signal():
    records = [
        _record("shared_path_id", "pair_a", 0.8, 0.2, 0.0),
        _record("shared_path_id", "pair_b", 0.0, 0.0, 0.0),
    ]

    global_categories, _ = classify_path_scores(
        records,
        alpha=1.0,
        forget_threshold=0.5,
        retain_threshold=0.5,
    )
    pair_categories, pair_summary = classify_path_scores(
        records,
        alpha=1.0,
        aggregation_key="path_pair",
        forget_threshold=0.5,
        retain_threshold=0.5,
    )

    assert global_categories[CATEGORY_FORGET] == []
    assert [record["pair_path_id"] for record in pair_categories[CATEGORY_FORGET]] == [
        "pair_a::shared_path_id"
    ]
    assert pair_categories[CATEGORY_FORGET][0]["pair_id"] == "pair_a"
    assert pair_summary["aggregation_key"] == "path_pair"


def test_name_aware_classification_uses_name_scores():
    records = [
        _name_record("p_name_forget", "pair_0", 0.2, 0.4, 0.0),
        _name_record("p_name_shared", "pair_0", 0.2, 0.4, 0.5),
        _name_record("p_name_retain", "pair_0", 0.0, 0.0, 0.6),
        _name_record("p_irrelevant", "pair_0", 0.0, 0.0, 0.0),
    ]
    categories, summary = classify_path_scores(
        records,
        name_aware_forget=True,
        name_forget_threshold=0.3,
        name_retain_threshold=0.3,
    )

    assert [record["path_id"] for record in categories[CATEGORY_FORGET]] == ["p_name_forget"]
    assert [record["path_id"] for record in categories[CATEGORY_SHARED]] == ["p_name_shared"]
    assert [record["path_id"] for record in categories[CATEGORY_RETAIN]] == ["p_name_retain"]
    assert summary["eligibility"]["name_aware_forget"] is True
    assert summary["eligibility"]["name_thresholds"]["name_forget_threshold"] == 0.3


def test_name_aware_projector_topk_can_enter_forget():
    records = [
        _name_record(
            "p_projector",
            "pair_0",
            name_nec=0.8,
            name_suf=0.4,
            name_ret=0.3,
            modality="vision_text",
            contains_projector=True,
        ),
        _name_record("p_name_forget", "pair_0", 0.7, 0.4, 0.0),
        _name_record("p_retain", "pair_0", 0.0, 0.0, 0.8),
    ]
    categories, summary = classify_path_scores(
        records,
        name_aware_forget=True,
        name_forget_threshold=0.5,
        name_retain_threshold=0.2,
        max_forget_projector_paths=1,
        projector_name_effect_ratio_threshold=1.2,
    )

    forget_ids = {record["path_id"] for record in categories[CATEGORY_FORGET]}
    assert "p_projector" in forget_ids
    promoted = next(record for record in categories[CATEGORY_FORGET] if record["path_id"] == "p_projector")
    assert promoted["promoted_to_forget_by"] == "projector_name_effect_topk"
    assert promoted["previous_category"] == CATEGORY_SHARED
    assert summary["eligibility"]["num_projector_promoted_to_forget"] == 1
    assert summary["category_projector_counts"][CATEGORY_FORGET]["contains_projector_true"] == 1


def test_name_aware_projector_topk_can_use_editable_score():
    records = [
        _name_record(
            "p_projector_high_forget_high_retain",
            "pair_0",
            name_nec=0.9,
            name_suf=0.4,
            name_ret=0.9,
            modality="vision_text",
            contains_projector=True,
        ),
        _name_record(
            "p_projector_balanced",
            "pair_0",
            name_nec=0.45,
            name_suf=0.2,
            name_ret=0.2,
            modality="vision_text",
            contains_projector=True,
        ),
        _name_record("p_retain", "pair_0", 0.0, 0.0, 1.0),
    ]
    categories, summary = classify_path_scores(
        records,
        name_aware_forget=True,
        name_forget_threshold=0.5,
        name_retain_threshold=0.1,
        max_forget_projector_paths=1,
        projector_name_effect_ratio_threshold=1.2,
        projector_topk_metric="name_editable_score",
    )

    forget_ids = [record["path_id"] for record in categories[CATEGORY_FORGET]]
    assert forget_ids == ["p_projector_balanced"]
    promoted = categories[CATEGORY_FORGET][0]
    assert promoted["promoted_to_forget_by"] == "projector_name_effect_topk"
    assert promoted["name_editable_score"] > 1.0
    assert summary["eligibility"]["projector_topk_metric"] == "name_editable_score"


def test_name_aware_projector_topk_can_use_salun_ssd_editable_score():
    records = [
        _name_projector_dim_record(
            "p_projector_high_name_low_dim",
            "pair_0",
            name_nec=0.9,
            name_suf=0.4,
            name_ret=0.2,
            salun_score=0.1,
        ),
        _name_projector_dim_record(
            "p_projector_balanced_high_dim",
            "pair_0",
            name_nec=0.45,
            name_suf=0.2,
            name_ret=0.2,
            salun_score=1.0,
        ),
        _name_record("p_retain", "pair_0", 0.0, 0.0, 1.0),
    ]
    categories, summary = classify_path_scores(
        records,
        name_aware_forget=True,
        name_forget_threshold=0.5,
        name_retain_threshold=0.1,
        max_forget_projector_paths=1,
        projector_name_effect_ratio_threshold=1.2,
        projector_topk_metric="name_salun_ssd_editable_score",
    )

    forget_ids = [record["path_id"] for record in categories[CATEGORY_FORGET]]
    assert forget_ids == ["p_projector_balanced_high_dim"]
    promoted = categories[CATEGORY_FORGET][0]
    assert promoted["selected_salun_ssd_score"] == 1.0
    assert promoted["name_salun_ssd_editable_score"] > 1.0
    assert summary["eligibility"]["projector_topk_metric"] == "name_salun_ssd_editable_score"


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
    test_forget_can_require_worst_anchor_specificity()
    test_pair_level_classification_preserves_pair_specific_signal()
    test_name_aware_classification_uses_name_scores()
    test_name_aware_projector_topk_can_enter_forget()
    test_name_aware_projector_topk_can_use_editable_score()
    test_name_aware_projector_topk_can_use_salun_ssd_editable_score()
    test_jsonl_io()
    print("Step 6 path-classification tests passed.")


if __name__ == "__main__":
    main()
