import argparse
import json
import random
import re
from pathlib import Path
from typing import Any

from .hard_retain_builder import (
    build_counterfactual_retain,
    build_same_reasoning_retain,
    build_same_topic_retain,
)
from .image_corruption import (
    build_corrupt_index,
    select_explicit_corruption,
    select_semantic_minimal_image_corruption,
)
from .text_corruption import build_text_corrupt_sample, normalize_question_template


def _import_pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.ipc as ipc
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError(
            "Step 3 pair building requires pyarrow. Please run with the mip-editor environment."
        ) from exc
    return pa, ipc, pq


def _load_arrow_dataset(dataset_dir: str) -> list[dict[str, Any]]:
    pa, ipc, _ = _import_pyarrow()
    rows: list[dict[str, Any]] = []
    for shard_path in sorted(Path(dataset_dir).glob("data-*.arrow")):
        with pa.memory_map(str(shard_path), "r") as source:
            reader = ipc.open_stream(source)
            table = reader.read_all()
        rows.extend(table.to_pylist())
    return rows


def _load_parquet_dataset(parquet_path: str) -> list[dict[str, Any]]:
    _, _, pq = _import_pyarrow()
    table = pq.read_table(parquet_path)
    return table.to_pylist()


def _infer_name(raw_row: dict[str, Any]) -> str | None:
    if raw_row.get("name"):
        return raw_row["name"]

    biography = raw_row.get("biography")
    if isinstance(biography, str):
        match = re.search(r'"Name"\s*:\s*"([^"]+)"', biography)
        if match:
            return match.group(1)

    answer = raw_row.get("answer")
    if isinstance(answer, str):
        match = re.match(r"([A-Z][a-zA-Z'.-]+(?:\s+[A-Z][a-zA-Z'.-]+){0,3})\s+is\b", answer)
        if match:
            return match.group(1)
    return None


def _normalize_image_ref(image: Any, dataset_path: str, row_idx: int, item_id: Any) -> dict[str, Any]:
    image_path = None
    has_bytes = False
    if isinstance(image, dict):
        image_path = image.get("path")
        has_bytes = image.get("bytes") is not None
    return {
        "dataset_path": dataset_path,
        "row_idx": row_idx,
        "item_id": item_id,
        "image_path": image_path,
        "has_bytes": has_bytes,
    }


def _normalize_row(raw_row: dict[str, Any], dataset_path: str, dataset_name: str, row_idx: int) -> dict[str, Any]:
    name = _infer_name(raw_row)
    item_id = raw_row.get("id", raw_row.get("ID", row_idx))
    question = raw_row.get("question") or ""
    answer = raw_row.get("answer") or raw_row.get("caption") or ""
    caption = raw_row.get("caption") or answer
    sample = {
        "source_dataset": dataset_name,
        "dataset_path": dataset_path,
        "row_idx": row_idx,
        "id": item_id,
        "name": name,
        "question": question,
        "answer": answer,
        "caption": caption,
        "image_ref": _normalize_image_ref(raw_row.get("image"), dataset_path, row_idx, item_id),
    }
    sample["question_template"] = normalize_question_template(question, name)
    return sample


def _load_normalized_samples(dataset_path: str, dataset_name: str) -> list[dict[str, Any]]:
    path = Path(dataset_path)
    if path.is_dir():
        rows = _load_arrow_dataset(str(path))
    elif path.suffix == ".parquet":
        rows = _load_parquet_dataset(str(path))
    else:
        raise ValueError(f"Unsupported dataset path: {dataset_path}")

    return [
        _normalize_row(raw_row, str(path), dataset_name, row_idx)
        for row_idx, raw_row in enumerate(rows)
    ]


def _export_sample(sample: dict[str, Any], sample_type: str | None = None) -> dict[str, Any]:
    exported = {
        "source_dataset": sample.get("source_dataset"),
        "row_idx": sample.get("row_idx"),
        "id": sample.get("id"),
        "name": sample.get("name"),
        "question": sample.get("question"),
        "answer": sample.get("answer"),
        "caption": sample.get("caption"),
        "image_ref": dict(sample.get("image_ref", {})),
        "question_template": sample.get("question_template"),
    }
    if sample_type is not None:
        exported["type"] = sample_type
    return exported


def build_pairs_from_samples(
    forget_clean_samples: list[dict[str, Any]],
    retain_clean_samples: list[dict[str, Any]],
    forget_corrupt_samples: list[dict[str, Any]] | None = None,
    max_pairs: int | None = None,
    seed: int = 42,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    corrupt_index = build_corrupt_index(forget_corrupt_samples or [])
    pairs: list[dict[str, Any]] = []

    for pair_idx, clean_sample in enumerate(forget_clean_samples):
        if max_pairs is not None and len(pairs) >= max_pairs:
            break

        forget_corrupt = select_explicit_corruption(clean_sample, corrupt_index)
        if forget_corrupt is None:
            forget_corrupt = select_semantic_minimal_image_corruption(clean_sample, retain_clean_samples)
        if forget_corrupt is None:
            forget_corrupt = build_text_corrupt_sample(clean_sample)

        same_topic = build_same_topic_retain(clean_sample)
        same_reasoning = build_same_reasoning_retain(clean_sample, retain_clean_samples, rng)
        used_ids = {sample_id for sample_id in [clean_sample.get("id"), same_reasoning.get("id") if same_reasoning else None] if sample_id is not None}
        counterfactual = build_counterfactual_retain(clean_sample, retain_clean_samples, rng, used_ids=used_ids)

        hard_retain = [same_topic]
        if same_reasoning is not None:
            hard_retain.append(same_reasoning)

        pair = {
            "pair_id": f"pair_{pair_idx:06d}",
            "dataset": clean_sample.get("source_dataset"),
            "forget_clean": _export_sample(clean_sample),
            "forget_corrupt": forget_corrupt,
            "hard_retain": hard_retain,
            "counterfactual_retain": counterfactual,
            "metadata": {
                "question_template": clean_sample.get("question_template"),
                "corruption_type": forget_corrupt.get("corruption_type"),
                "builder": "causal_mip_step3_notice_inspired",
            },
        }
        pairs.append(pair)

    return pairs


def write_pairs_jsonl(pairs: list[dict[str, Any]], output_path: str) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for pair in pairs:
            handle.write(json.dumps(pair, ensure_ascii=False) + "\n")


def split_pairs_train_val(
    pairs: list[dict[str, Any]],
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError(f"val_ratio must be in [0, 1), got {val_ratio}")
    if not pairs or val_ratio == 0.0:
        return list(pairs), []

    shuffled = list(pairs)
    random.Random(seed).shuffle(shuffled)
    val_size = max(1, int(len(shuffled) * val_ratio))
    if val_size >= len(shuffled):
        val_size = len(shuffled) - 1
    train_pairs = shuffled[:-val_size]
    val_pairs = shuffled[-val_size:]
    return train_pairs, val_pairs


def _default_dataset_paths(dataset: str, base_path: str, forget_ratio: int) -> dict[str, str | None]:
    base = Path(base_path)
    if dataset == "clear":
        return {
            "forget_clean": str(base / "CLEAR" / f"forget{forget_ratio}+tofu"),
            "forget_corrupt": str(base / "CLEAR" / f"forget{forget_ratio}_perturbed"),
            "retain_clean": str(base / "CLEAR" / f"retain{100 - forget_ratio}+tofu"),
        }
    if dataset == "mllmu":
        return {
            "forget_clean": str(base / "MLLMU-Bench" / f"forget_{forget_ratio}" / "train-00000-of-00001.parquet"),
            "forget_corrupt": None,
            "retain_clean": str(base / "MLLMU-Bench" / f"retain_{100 - forget_ratio}" / "train-00000-of-00001.parquet"),
        }
    raise ValueError(f"Unsupported dataset: {dataset}")


def build_causal_pairs_for_project(
    dataset: str,
    base_path: str,
    forget_ratio: int,
    output_path: str,
    val_output_path: str | None = None,
    val_ratio: float = 0.0,
    max_pairs: int | None = None,
    seed: int = 42,
    forget_clean_path: str | None = None,
    forget_corrupt_path: str | None = None,
    retain_clean_path: str | None = None,
) -> dict[str, Any]:
    if val_ratio > 0.0 and val_output_path is None:
        raise ValueError("val_output_path must be provided when val_ratio > 0.")

    defaults = _default_dataset_paths(dataset, base_path, forget_ratio)
    forget_clean_path = forget_clean_path or defaults["forget_clean"]
    forget_corrupt_path = forget_corrupt_path or defaults["forget_corrupt"]
    retain_clean_path = retain_clean_path or defaults["retain_clean"]

    forget_clean_samples = _load_normalized_samples(forget_clean_path, f"{dataset}_forget_clean")
    retain_clean_samples = _load_normalized_samples(retain_clean_path, f"{dataset}_retain_clean")
    forget_corrupt_samples = (
        _load_normalized_samples(forget_corrupt_path, f"{dataset}_forget_corrupt")
        if forget_corrupt_path
        else None
    )

    pairs = build_pairs_from_samples(
        forget_clean_samples=forget_clean_samples,
        retain_clean_samples=retain_clean_samples,
        forget_corrupt_samples=forget_corrupt_samples,
        max_pairs=max_pairs,
        seed=seed,
    )
    train_pairs, val_pairs = split_pairs_train_val(pairs, val_ratio=val_ratio, seed=seed)
    write_pairs_jsonl(train_pairs, output_path)
    if val_output_path is not None:
        write_pairs_jsonl(val_pairs, val_output_path)
    return {
        "dataset": dataset,
        "num_pairs": len(pairs),
        "num_train_pairs": len(train_pairs),
        "num_val_pairs": len(val_pairs),
        "output_path": output_path,
        "val_output_path": val_output_path,
        "forget_clean_path": forget_clean_path,
        "forget_corrupt_path": forget_corrupt_path,
        "retain_clean_path": retain_clean_path,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Step 3 causal pairs for CHIP-Editor.")
    parser.add_argument("--dataset", type=str, choices=["clear", "mllmu"], required=True)
    parser.add_argument("--base_path", type=str, required=True)
    parser.add_argument("--forget_ratio", type=int, default=5)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--val_output", type=str, default=None)
    parser.add_argument("--val_ratio", type=float, default=0.0)
    parser.add_argument("--max_pairs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--forget_clean_path", type=str, default=None)
    parser.add_argument("--forget_corrupt_path", type=str, default=None)
    parser.add_argument("--retain_clean_path", type=str, default=None)
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    summary = build_causal_pairs_for_project(
        dataset=args.dataset,
        base_path=args.base_path,
        forget_ratio=args.forget_ratio,
        output_path=args.output,
        val_output_path=args.val_output,
        val_ratio=args.val_ratio,
        max_pairs=args.max_pairs,
        seed=args.seed,
        forget_clean_path=args.forget_clean_path,
        forget_corrupt_path=args.forget_corrupt_path,
        retain_clean_path=args.retain_clean_path,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
