import json
import sys
import tempfile
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from causal_mip.data_pairs.bind_paths_to_pairs import build_path_pair_bindings, load_path_pair_bindings
from causal_mip.path_localization.path_schema import CandidatePath, PathNode


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def _pair(pair_id: str, row_idx: int) -> dict:
    return {
        "pair_id": pair_id,
        "forget_clean": {
            "id": row_idx,
            "row_idx": row_idx,
            "image_ref": {
                "dataset_path": "/tmp/mock",
                "row_idx": row_idx,
                "item_id": row_idx,
            },
        },
    }


def main():
    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        paths_path = temp / "P_cand.jsonl"
        pairs_path = temp / "pairs.jsonl"
        output_path = temp / "P_cand_bound.jsonl"

        paths = [
            CandidatePath(
                path_id="text_igi_p000000",
                source="mip_editor_igi",
                modality="text",
                mip_score=1.0,
                nodes=[PathNode("model.language_model.layers.0.mlp.down_proj", 0, 1, "answer_tokens")],
                source_sample_idx=7,
            ),
            CandidatePath(
                path_id="text_igi_p000001",
                source="mip_editor_igi",
                modality="text",
                mip_score=1.0,
                nodes=[PathNode("model.language_model.layers.0.mlp.down_proj", 0, 2, "answer_tokens")],
                source_sample_idx=8,
            ),
        ]
        _write_jsonl(paths_path, [path.to_dict() for path in paths])
        _write_jsonl(pairs_path, [_pair("pair_000007", 7)])

        summary = build_path_pair_bindings(
            candidate_paths_path=str(paths_path),
            pairs_path=str(pairs_path),
            output_path=str(output_path),
        )
        assert summary["num_bindings"] == 1
        assert summary["num_unbound_paths"] == 1

        bindings = load_path_pair_bindings(str(output_path))
        assert bindings == {"pair_000007": {"text_igi_p000000"}}

    print("Step 3 path-pair binding tests passed.")


if __name__ == "__main__":
    main()
