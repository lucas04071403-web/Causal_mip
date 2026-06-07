import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from causal_mip.path_localization.node_specific_export import _select_nodes, score_record_nodes
from causal_mip.path_localization.hybrid_projector_dim_export import export_hybrid_projector_dim_candidates
from causal_mip.path_localization.projector_dim_export import _select_dims, score_projector_dims


def _node_score(node_index, saliency, module_kind="llm", status="ok"):
    return {
        "node_index": node_index,
        "module": f"module.{node_index}",
        "layer": node_index,
        "neuron": node_index,
        "token_selector": "answer_tokens",
        "module_kind": module_kind,
        "saliency": saliency,
        "fisher_saliency": saliency / 10.0,
        "status": status,
    }


def test_node_specific_export_uses_worst_anchor_filters():
    record = {
        "saliency_specificity": {
            "forget": {
                "node_scores": [
                    _node_score(0, 0.30),
                    _node_score(1, 0.30),
                ]
            },
            "retain_anchors": {
                "same_topic": {"node_scores": [_node_score(0, 0.10), _node_score(1, 0.10)]},
                "same_reasoning": {"node_scores": [_node_score(0, 0.12), _node_score(1, 0.31)]},
                "counterfactual_retain": {"node_scores": [_node_score(0, 0.08), _node_score(1, 0.08)]},
            },
        }
    }

    node_records = score_record_nodes(record, eps=1e-6)
    by_index = {record["node_index"]: record for record in node_records}
    assert by_index[0]["min_anchor_margin"] > 0.0
    assert by_index[1]["saliency_specificity_margin"] > 0.0
    assert by_index[1]["min_anchor_margin"] < 0.0

    selected = _select_nodes(
        node_records,
        top_k_nodes=4,
        min_specificity_margin=None,
        allow_projector_nodes=False,
        min_anchor_margin=0.0,
        max_anchor_retain_saliency=0.2,
    )
    assert [record["node_index"] for record in selected] == [0]


def test_projector_dim_export_uses_worst_anchor_filters():
    record = {
        "saliency_specificity": {
            "forget": {
                "node_scores": [
                    {
                        **_node_score(0, 0.0, module_kind="projector"),
                        "dim_scores": [
                            {"dim_index": 3, "saliency": 0.30, "fisher_saliency": 0.03, "status": "ok"},
                            {"dim_index": 7, "saliency": 0.30, "fisher_saliency": 0.03, "status": "ok"},
                        ],
                    }
                ]
            },
            "retain_anchors": {
                "same_topic": {
                    "node_scores": [
                        {
                            **_node_score(0, 0.0, module_kind="projector"),
                            "dim_scores": [
                                {"dim_index": 3, "saliency": 0.10, "fisher_saliency": 0.01, "status": "ok"},
                                {"dim_index": 7, "saliency": 0.10, "fisher_saliency": 0.01, "status": "ok"},
                            ],
                        }
                    ]
                },
                "same_reasoning": {
                    "node_scores": [
                        {
                            **_node_score(0, 0.0, module_kind="projector"),
                            "dim_scores": [
                                {"dim_index": 3, "saliency": 0.12, "fisher_saliency": 0.01, "status": "ok"},
                                {"dim_index": 7, "saliency": 0.31, "fisher_saliency": 0.01, "status": "ok"},
                            ],
                        }
                    ]
                },
            },
        }
    }

    dim_records = score_projector_dims(record, eps=1e-6)
    by_dim = {record["dim_index"]: record for record in dim_records}
    assert by_dim[3]["min_anchor_margin"] > 0.0
    assert by_dim[7]["saliency_specificity_margin"] > 0.0
    assert by_dim[7]["min_anchor_margin"] < 0.0

    selected = _select_dims(
        dim_records,
        top_k_dims=4,
        min_specificity_margin=None,
        min_anchor_margin=0.0,
        max_anchor_retain_saliency=0.2,
    )
    assert [record["dim_index"] for record in selected] == [3]


def test_projector_dim_export_can_rank_by_salun_ssd_score():
    record = {
        "saliency_specificity": {
            "forget": {
                "node_scores": [
                    {
                        **_node_score(0, 0.0, module_kind="projector"),
                        "dim_scores": [
                            {"dim_index": 3, "saliency": 0.30, "fisher_saliency": 0.03, "status": "ok"},
                            {"dim_index": 7, "saliency": 0.45, "fisher_saliency": 0.03, "status": "ok"},
                            {"dim_index": 9, "saliency": 0.25, "fisher_saliency": 0.03, "status": "ok"},
                        ],
                    }
                ]
            },
            "retain_anchors": {
                "same_topic": {
                    "node_scores": [
                        {
                            **_node_score(0, 0.0, module_kind="projector"),
                            "dim_scores": [
                                {"dim_index": 3, "saliency": 0.02, "fisher_saliency": 0.001, "status": "ok"},
                                {"dim_index": 7, "saliency": 0.35, "fisher_saliency": 0.100, "status": "ok"},
                                {"dim_index": 9, "saliency": 0.03, "fisher_saliency": 0.001, "status": "ok"},
                            ],
                        }
                    ]
                },
            },
        }
    }

    dim_records = score_projector_dims(record, eps=1e-6, fisher_penalty=1.0)
    selected = _select_dims(
        dim_records,
        top_k_dims=2,
        min_specificity_margin=None,
        max_anchor_retain_fisher_saliency=0.01,
        rank_metric="salun_ssd_score",
    )
    assert [record["dim_index"] for record in selected] == [3, 9]
    assert selected[0]["salun_ssd_score"] > selected[1]["salun_ssd_score"]


def _candidate(path_id, dims, original_path_id=None):
    return {
        "path_id": path_id,
        "source": "test_projector",
        "modality": "vision_text",
        "mip_score": 1.0,
        "nodes": [
            {"module": "mm_projector", "layer": None, "neuron": int(dim), "token_selector": "image_tokens"}
            for dim in dims
        ],
        "source_sample_idx": None,
        "metadata": {
            "projector_dim_level": True,
            "original_path_id": original_path_id or path_id,
            "selected_projector_dims": [
                {
                    "node_index": 0,
                    "module": "visual.merger",
                    "layer": None,
                    "token_selector": "image_tokens",
                    "module_kind": "projector",
                    "dim_index": int(dim),
                    "salun_ssd_score": float(100 - offset),
                    "fisher_specificity_margin": 0.1,
                }
                for offset, dim in enumerate(dims)
            ],
        },
    }


def _p_forget_record(path_id, pair_id, name_suf=0.1, name_ret=0.0):
    return {
        "path_id": path_id,
        "pair_ids": [pair_id],
        "path_source": "test_projector",
        "path_modality": "vision_text",
        "contains_projector": True,
        "projector_patchable": True,
        "NameSuf": name_suf,
        "NameRet": name_ret,
        "name_forget_effect": max(name_suf, 0.0),
        "name_retain_impact": max(name_ret, 0.0),
        "category": "P_forget",
    }


def _write_jsonl(path, records):
    import json

    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def test_hybrid_projector_dim_export_can_replace_drop_and_augment():
    import json
    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        base_candidates = temp / "base_candidates.jsonl"
        base_forget = temp / "base_forget.jsonl"
        replace_candidates = temp / "replace_candidates.jsonl"
        replace_forget = temp / "replace_forget.jsonl"
        augment_candidates = temp / "augment_candidates.jsonl"
        augment_forget = temp / "augment_forget.jsonl"

        _write_jsonl(
            base_candidates,
            [
                _candidate("top16_a", [1, 2, 3]),
                _candidate("top16_b", [10, 11, 12]),
                _candidate("top16_drop", [20, 21, 22]),
            ],
        )
        _write_jsonl(
            base_forget,
            [
                _p_forget_record("top16_a", "pair_a"),
                _p_forget_record("top16_b", "pair_b"),
                _p_forget_record("top16_drop", "pair_drop"),
            ],
        )
        _write_jsonl(replace_candidates, [_candidate("top12_b", [10, 11])])
        _write_jsonl(replace_forget, [_p_forget_record("top12_b", "pair_b")])
        _write_jsonl(augment_candidates, [_candidate("ssd_a", [2, 4, 5, 6])])
        _write_jsonl(augment_forget, [_p_forget_record("ssd_a", "pair_a")])

        output_candidates = temp / "hybrid_candidates.jsonl"
        output_forget = temp / "P_forget.jsonl"
        output_shared = temp / "P_shared.jsonl"
        output_bindings = temp / "hybrid_bound.jsonl"
        summary_path = temp / "summary.json"

        summary = export_hybrid_projector_dim_candidates(
            base_candidates_path=str(base_candidates),
            base_p_forget_path=str(base_forget),
            replace_candidates_path=str(replace_candidates),
            replace_p_forget_path=str(replace_forget),
            augment_candidates_path=str(augment_candidates),
            augment_p_forget_path=str(augment_forget),
            output_candidates_path=str(output_candidates),
            output_p_forget_path=str(output_forget),
            output_p_shared_path=str(output_shared),
            output_bindings_path=str(output_bindings),
            summary_path=str(summary_path),
            path_id_prefix="hybrid_test",
            drop_path_ids=["top16_drop"],
            replace_path_ids=["top16_b"],
            augment_pair_ids=["pair_a"],
            augment_top_k_dims=2,
        )

        candidates = [json.loads(line) for line in output_candidates.read_text().splitlines()]
        forget = [json.loads(line) for line in output_forget.read_text().splitlines()]
        bindings = [json.loads(line) for line in output_bindings.read_text().splitlines()]

        assert summary["num_output_candidates"] == 2
        assert output_shared.read_text() == ""
        assert [record["path_id"] for record in candidates] == ["hybrid_test_p000000", "hybrid_test_p000001"]
        assert [node["neuron"] for node in candidates[0]["nodes"]] == [1, 2, 3, 4, 5]
        assert [node["neuron"] for node in candidates[1]["nodes"]] == [10, 11]
        assert forget[1]["hybrid_replaced_by_path_id"] == "top12_b"
        assert bindings[0]["augmented_by_path_id"] == "ssd_a"
        assert bindings[0]["selected_dim_indices"] == [1, 2, 3, 4, 5]


def main():
    test_node_specific_export_uses_worst_anchor_filters()
    test_projector_dim_export_uses_worst_anchor_filters()
    test_projector_dim_export_can_rank_by_salun_ssd_score()
    test_hybrid_projector_dim_export_can_replace_drop_and_augment()
    print("Retain-aware localization tests passed.")


if __name__ == "__main__":
    main()
