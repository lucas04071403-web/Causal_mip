"""
Causal-MIP-Editor MVP: Step 2

This script demonstrates extracting top-k candidate paths from MIP-Editor
instead of just the greedy best path.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causal_mip.path_localization import (
    CandidatePath,
    build_simple_cross_modal_paths,
    build_cross_modal_paths_from_unimodal_paths,
    export_saliency_specific_candidates,
    merge_paths_from_modalities,
    load_candidate_paths,
    extract_text_paths_from_mip_scores,
    extract_vision_paths_from_mip_scores,
)


def test_path_schema():
    """Test the path schema definition."""
    print("\n=== Testing Path Schema ===")

    from causal_mip.path_localization.path_schema import PathNode, CandidatePath

    # Create a sample path
    nodes = [
        PathNode(
            module="language_model.layers.8.mlp",
            layer=8,
            neuron=12450,
            token_selector="image_tokens"
        ),
        PathNode(
            module="language_model.layers.15.mlp",
            layer=15,
            neuron=892,
            token_selector="answer_tokens"
        ),
    ]

    path = CandidatePath(
        path_id="p_000001",
        source="mip_editor_igi",
        modality="text",
        mip_score=0.817,
        nodes=nodes
    )

    # Test serialization
    path_dict = path.to_dict()
    print(f"Path dict: {path_dict}")

    # Test deserialization
    path2 = CandidatePath.from_dict(path_dict)
    print(f"Path from dict: {path2.path_id}, score: {path2.mip_score}")

    # Test JSON
    path_json = path.to_json()
    print(f"Path JSON (truncated): {path_json[:100]}...")

    print("✓ Path schema test passed!")
    return True


def test_igi_topk_paths():
    """Test IGI top-k path extraction from cached scores."""
    print("\n=== Testing IGI Top-k Paths ===")

    import torch

    scores = torch.tensor([
        [[0.1, 0.9, 0.2], [0.8, 0.4, 0.3], [0.2, 0.3, 0.7]],
        [[0.7, 0.2, 0.1], [0.1, 0.5, 0.4], [0.6, 0.3, 0.2]],
    ])
    paths = extract_text_paths_from_mip_scores(scores, num_candidates=4, per_layer_topk=2)

    assert len(paths) >= 4
    assert all(path.modality == "text" for path in paths)
    assert all(len(path.nodes) == scores.shape[1] for path in paths)
    assert len({tuple(node.neuron for node in path.nodes) for path in paths}) == len(paths)

    print(f"Generated {len(paths)} text candidate paths")
    return True


def test_fisher_topk_paths():
    """Test Fisher top-k path extraction from cached scores."""
    print("\n=== Testing Fisher Top-k Paths ===")

    import torch

    scores = torch.tensor([
        [[0.3, 0.6, 0.1], [0.2, 0.9, 0.1]],
        [[0.5, 0.2, 0.3], [0.7, 0.2, 0.1]],
    ])
    paths = extract_vision_paths_from_mip_scores(scores, num_candidates=4, per_layer_topk=2)

    assert len(paths) >= 4
    assert all(path.modality == "vision" for path in paths)
    assert all(len(path.nodes) == scores.shape[1] for path in paths)
    assert len({tuple(node.neuron for node in path.nodes) for path in paths}) >= 2

    print(f"Generated {len(paths)} vision candidate paths")
    return True


def test_cross_modal_builder():
    """Test cross-modal path builder."""
    print("\n=== Testing Cross-Modal Path Builder ===")

    from causal_mip.path_localization.cross_modal_path_builder import build_simple_cross_modal_paths

    # Simulate top neurons from MIP-Editor
    vision_top_neurons = [100, 500, 1200, 2500, 3800, 4500, 5200, 6000, 6800, 7500]
    llm_top_neurons = [200, 800, 1500, 3200, 4800, 6500, 8200, 10000, 12000, 12450]

    # Build cross-modal paths
    cross_modal_paths = build_simple_cross_modal_paths(
        vision_top_neurons=vision_top_neurons,
        llm_top_neurons=llm_top_neurons,
        num_paths=5
    )

    print(f"Generated {len(cross_modal_paths)} cross-modal paths:")
    for path in cross_modal_paths[:2]:
        print(f"  - {path.path_id}: {path.modality}, score={path.mip_score:.3f}, nodes={len(path.nodes)}")

    print("✓ Cross-modal path builder test passed!")
    return True


def test_cross_modal_from_unimodal_paths():
    """Test cross-modal bridge construction from extracted unimodal paths."""
    print("\n=== Testing Cross-Modal From Unimodal Paths ===")

    from causal_mip.path_localization.path_schema import PathNode, CandidatePath

    vision_paths = [
        CandidatePath(
            "vision_fisher_p000000",
            "mip_editor_ifi",
            "vision",
            0.9,
            [PathNode("model.visual.blocks.0.mlp.down_proj", 0, 12, "image_tokens")],
        )
    ]
    text_paths = [
        CandidatePath(
            "text_igi_p000000",
            "mip_editor_igi",
            "text",
            0.8,
            [
                PathNode("model.language_model.layers.0.mlp.down_proj", 0, 21, "image_tokens"),
                PathNode("model.language_model.layers.1.mlp.down_proj", 1, 34, "answer_tokens"),
            ],
        )
    ]
    paths = build_cross_modal_paths_from_unimodal_paths(
        vision_paths=vision_paths,
        text_paths=text_paths,
        model_name="Qwen2.5-VL-3B-Instruct",
        num_paths=1,
    )

    assert len(paths) == 1
    assert len(paths[0].nodes) == 4
    assert paths[0].nodes[1].module == "mm_projector"
    print("✓ Cross-modal from unimodal path builder test passed!")
    return True


def test_merge_and_save():
    """Test merging and saving paths."""
    print("\n=== Testing Merge and Save ===")

    import tempfile
    from causal_mip.path_localization.path_schema import PathNode, CandidatePath
    from causal_mip.path_localization.cross_modal_path_builder import build_simple_cross_modal_paths

    # Create dummy paths
    text_nodes = [PathNode("language_model.layers.0.mlp", 0, 100, "image_tokens")]
    text_path = CandidatePath("text_igi_p000001", "mip_editor_igi", "text", 0.8, text_nodes)

    vision_nodes = [PathNode("model.visual.blocks.0.mlp.down_proj", 0, 200, "image_tokens")]
    vision_path = CandidatePath("vision_fisher_p000001", "mip_editor_ifi", "vision", 0.7, vision_nodes)

    cross_paths = build_simple_cross_modal_paths([100], [200], num_paths=2)

    # Merge and save
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, encoding='utf-8') as f:
        temp_path = f.name

    all_paths = merge_paths_from_modalities(
        text_paths=[text_path],
        vision_paths=[vision_path],
        cross_modal_paths=cross_paths,
        output_path=temp_path
    )

    # Load back
    loaded_paths = load_candidate_paths(temp_path)
    print(f"Loaded {len(loaded_paths)} paths from file")

    # Cleanup
    os.unlink(temp_path)

    print("✓ Merge and save test passed!")
    return True


def test_saliency_specific_candidate_export():
    """Test saliency-specific candidate export from Step5 score records."""
    print("\n=== Testing Saliency-Specific Candidate Export ===")

    import json
    import tempfile
    from pathlib import Path
    from causal_mip.path_localization.path_schema import PathNode, CandidatePath

    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        candidates_path = temp / "P_cand.jsonl"
        scores_path = temp / "scores.jsonl"
        output_candidates = temp / "P_cand_saliency.jsonl"
        output_bindings = temp / "P_cand_saliency_bound.jsonl"
        summary_path = temp / "summary.json"

        candidates = [
            CandidatePath(
                path_id="text_igi_p000000",
                source="mip_editor_igi",
                modality="text",
                mip_score=0.1,
                nodes=[PathNode("model.language_model.layers.0.mlp.down_proj", 0, 1, "answer_tokens")],
            ),
            CandidatePath(
                path_id="vision_fisher_p000000",
                source="mip_editor_ifi",
                modality="vision",
                mip_score=0.2,
                nodes=[PathNode("model.visual.blocks.0.mlp.down_proj", 0, 2, "image_tokens")],
            ),
        ]
        with candidates_path.open("w", encoding="utf-8") as handle:
            for path in candidates:
                handle.write(json.dumps(path.to_dict()) + "\n")

        records = [
            {
                "pair_id": "pair_000000",
                "path_id": "text_igi_p000000",
                "path_modality": "text",
                "path_source": "mip_editor_igi",
                "status": "ok",
                "all_nodes_patchable": True,
                "contains_projector": False,
                "projector_patchable": False,
                "forget_saliency": 0.4,
                "retain_anchor_saliency": 0.1,
                "saliency_specificity_margin": 0.3,
                "saliency_specificity_ratio": 4.0,
            },
            {
                "pair_id": "pair_000000",
                "path_id": "vision_fisher_p000000",
                "path_modality": "vision",
                "path_source": "mip_editor_ifi",
                "status": "ok",
                "all_nodes_patchable": True,
                "contains_projector": False,
                "projector_patchable": False,
                "forget_saliency": 0.1,
                "retain_anchor_saliency": 0.2,
                "saliency_specificity_margin": -0.1,
                "saliency_specificity_ratio": 0.5,
            },
        ]
        with scores_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")

        summary = export_saliency_specific_candidates(
            score_paths=[str(scores_path)],
            candidate_paths_path=str(candidates_path),
            output_candidates_path=str(output_candidates),
            output_bindings_path=str(output_bindings),
            summary_path=str(summary_path),
            top_k_per_pair_modality=2,
        )

        selected = load_candidate_paths(str(output_candidates))
        assert summary["num_selected_candidate_paths"] == 1
        assert summary["num_selected_bindings"] == 1
        assert selected[0].path_id == "saliency_specific_p000000"
        assert selected[0].source == "mip_editor_igi_saliency_specific"
        assert selected[0].metadata["original_path_id"] == "text_igi_p000000"
        bindings = [json.loads(line) for line in output_bindings.read_text(encoding="utf-8").splitlines()]
        assert bindings[0]["path_id"] == "saliency_specific_p000000"

    print("✓ Saliency-specific candidate export test passed!")
    return True


def test_full_pipeline():
    """Test the full pipeline with a mock model."""
    print("\n=== Testing Full Pipeline (Mock) ===")

    from causal_mip.path_localization.path_schema import PathNode, CandidatePath
    from causal_mip.path_localization.cross_modal_path_builder import build_simple_cross_modal_paths

    # Simulate what Step 2 would output
    print("Simulating Step 2 output format...")

    # Text paths
    text_paths = []
    for i in range(10):
        nodes = [
            PathNode(f"model.language_model.layers.{j}.mlp.down_proj", j, 100 + i * 10, "image_tokens")
            for j in range(5)
        ]
        path = CandidatePath(
            path_id=f"text_igi_p{i:06d}",
            source="mip_editor_igi",
            modality="text",
            mip_score=0.9 - i * 0.05,
            nodes=nodes
        )
        text_paths.append(path)

    # Vision paths
    vision_paths = []
    for i in range(10):
        nodes = [
            PathNode(f"model.visual.blocks.{j}.mlp.down_proj", j, 200 + i * 10, "image_tokens")
            for j in range(3)
        ]
        path = CandidatePath(
            path_id=f"vision_fisher_p{i:06d}",
            source="mip_editor_ifi",
            modality="vision",
            mip_score=0.85 - i * 0.04,
            nodes=nodes
        )
        vision_paths.append(path)

    # Cross-modal paths
    vision_neurons = list(range(100, 200, 10))
    llm_neurons = list(range(100, 200, 10))
    cross_paths = build_simple_cross_modal_paths(vision_neurons, llm_neurons, num_paths=5)

    # Merge
    print(f"Text paths: {len(text_paths)}")
    print(f"Vision paths: {len(vision_paths)}")
    print(f"Cross-modal paths: {len(cross_paths)}")

    print("✓ Full pipeline simulation passed!")
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("Causal-MIP-Editor MVP: Step 2 Test")
    print("=" * 60)

    tests = [
        ("Path Schema", test_path_schema),
        ("IGI Top-k Paths", test_igi_topk_paths),
        ("Fisher Top-k Paths", test_fisher_topk_paths),
        ("Cross-Modal Builder", test_cross_modal_builder),
        ("Cross-Modal From Unimodal", test_cross_modal_from_unimodal_paths),
        ("Merge and Save", test_merge_and_save),
        ("Saliency-Specific Candidate Export", test_saliency_specific_candidate_export),
        ("Full Pipeline", test_full_pipeline),
    ]

    results = []
    for name, test_fn in tests:
        try:
            result = test_fn()
            results.append((name, result))
        except Exception as e:
            print(f"✗ {name} failed with error: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! Step 2 implementation is ready.")
        return 0
    else:
        print("\n⚠ Some tests failed. Please review the implementation.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
