import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from causal_mip.data_pairs.build_pairs import build_pairs_from_samples


def _sample(sample_id, question, answer, caption, name, dataset_name="clear"):
    return {
        "source_dataset": dataset_name,
        "dataset_path": "/tmp/mock",
        "row_idx": sample_id,
        "id": sample_id,
        "name": name,
        "question": question,
        "answer": answer,
        "caption": caption,
        "image_ref": {
            "dataset_path": "/tmp/mock",
            "row_idx": sample_id,
            "item_id": sample_id,
            "image_path": None,
            "has_bytes": False,
        },
        "question_template": question.lower(),
    }


def main():
    forget_clean = [
        _sample(1, "What can you see in this image?", "Alice is shown with books.", "Alice is shown with books.", "Alice"),
        _sample(2, "What can you see in this image?", "Bob is standing near a car.", "Bob is standing near a car.", "Bob"),
    ]
    forget_corrupt = [
        _sample(1, "What is the correct caption for this image?", "Alice is shown with books.", "Carol is shown with flowers.", "Alice", "clear_corrupt"),
    ]
    retain_clean = [
        _sample(11, "What can you see in this image?", "Carol is shown with flowers.", "Carol is shown with flowers.", "Carol", "clear_retain"),
        _sample(12, "What can you see in this image?", "Dave is standing near a river.", "Dave is standing near a river.", "Dave", "clear_retain"),
    ]

    pairs = build_pairs_from_samples(
        forget_clean_samples=forget_clean,
        retain_clean_samples=retain_clean,
        forget_corrupt_samples=forget_corrupt,
        max_pairs=2,
        seed=7,
    )

    assert len(pairs) == 2
    assert pairs[0]["forget_corrupt"]["corruption_type"] == "explicit_perturbed_pair"
    assert pairs[1]["forget_corrupt"]["corruption_type"] in {"semantic_minimal_pair", "symmetric_token_replacement"}
    assert pairs[0]["hard_retain"][0]["type"] == "same_topic"
    assert pairs[0]["counterfactual_retain"]["type"] == "counterfactual_retain"

    same_input_corrupt = [
        {
            **_sample(
                1,
                "What can you see in this image?",
                "Alice is shown with books.",
                "Carol is shown with flowers.",
                "Alice",
                "clear_corrupt",
            ),
            "image_ref": {
                "dataset_path": "/tmp/mock",
                "row_idx": 1,
                "item_id": 1,
                "image_path": None,
                "has_bytes": False,
            },
        }
    ]
    same_input_pairs = build_pairs_from_samples(
        forget_clean_samples=forget_clean[:1],
        retain_clean_samples=retain_clean,
        forget_corrupt_samples=same_input_corrupt,
        max_pairs=1,
        seed=7,
    )
    assert same_input_pairs[0]["forget_corrupt"]["corruption_type"] in {
        "semantic_minimal_pair",
        "symmetric_token_replacement",
    }
    print("Step 3 data-pair tests passed.")


if __name__ == "__main__":
    main()
