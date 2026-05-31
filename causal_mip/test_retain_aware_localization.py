import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from causal_mip.path_localization.node_specific_export import _select_nodes, score_record_nodes
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


def main():
    test_node_specific_export_uses_worst_anchor_filters()
    test_projector_dim_export_uses_worst_anchor_filters()
    print("Retain-aware localization tests passed.")


if __name__ == "__main__":
    main()
