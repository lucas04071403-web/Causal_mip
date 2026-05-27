from __future__ import annotations

import argparse
import json

from causal_mip.causal_scores.metrics import (
    build_pair_prepared_batches,
    compute_path_causal_score_record,
    filter_candidate_paths_for_step5,
    write_path_score_records_jsonl,
)
from causal_mip.data_pairs.bind_paths_to_pairs import load_path_pair_bindings
from causal_mip.interventions.activation_cache import (
    SampleReferenceResolver,
    load_candidate_paths_jsonl,
    load_pairs_jsonl,
)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Step 5 causal path scores.")
    parser.add_argument("--dataset", type=str, default="clear")
    parser.add_argument("--model", type=str, default="Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--llm_directory", type=str, required=True)
    parser.add_argument("--output_file_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--image_resize", type=int, default=224)
    parser.add_argument("--pairs_path", type=str, required=True)
    parser.add_argument("--candidate_paths_path", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--peft_checkpoint", type=str, default=None)
    parser.add_argument("--num_pairs", type=int, default=None)
    parser.add_argument("--num_paths", type=int, default=None)
    parser.add_argument("--pair_start", type=int, default=0)
    parser.add_argument("--path_start", type=int, default=0)
    parser.add_argument("--path_modality", type=str, default="all", choices=["all", "text", "vision_text", "vision"])
    parser.add_argument("--path_pair_bindings", type=str, default=None)
    parser.add_argument("--strict_patchable", action="store_true", default=False)
    return parser


def _load_model_for_step5(args):
    from load_model import load_base_model, load_peft_model

    if args.peft_checkpoint is None:
        return load_peft_model(args, trainable=False)

    from peft import PeftModel

    base_model = load_base_model(args)
    model = PeftModel.from_pretrained(base_model, args.peft_checkpoint, is_trainable=False)
    model.to(args.device)
    return model


def _slice_records(records, start: int, limit: int | None):
    sliced = records[start:]
    if limit is not None:
        sliced = sliced[:limit]
    return sliced


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    try:
        from transformers import AutoProcessor
    except ImportError as exc:
        raise ImportError(
            "Step 5 score runner requires transformers and its runtime dependencies. "
            "Please use a complete environment before running build_scores.py."
        ) from exc

    model_path = f"{args.llm_directory}{args.model}"
    processor = AutoProcessor.from_pretrained(model_path, padding_side="left")
    model = _load_model_for_step5(args)
    model.eval()

    pairs = load_pairs_jsonl(args.pairs_path)
    candidate_paths = load_candidate_paths_jsonl(args.candidate_paths_path)
    candidate_paths = filter_candidate_paths_for_step5(candidate_paths, modality_filter=args.path_modality)
    pairs = _slice_records(pairs, args.pair_start, args.num_pairs)
    candidate_paths = _slice_records(candidate_paths, args.path_start, args.num_paths)
    candidate_paths_by_id = {path.path_id: path for path in candidate_paths}
    bindings = load_path_pair_bindings(args.path_pair_bindings) if args.path_pair_bindings else None

    resolver = SampleReferenceResolver()
    records = []
    for pair in pairs:
        pair_candidate_paths = candidate_paths
        if bindings is not None:
            allowed_path_ids = bindings.get(pair.get("pair_id"), set())
            pair_candidate_paths = [
                candidate_paths_by_id[path_id]
                for path_id in sorted(allowed_path_ids)
                if path_id in candidate_paths_by_id
            ]
        prepared_batches = build_pair_prepared_batches(
            pair=pair,
            processor=processor,
            model=model,
            image_resize=args.image_resize,
            resolver=resolver,
        )
        for candidate_path in pair_candidate_paths:
            records.append(
                compute_path_causal_score_record(
                    model=model,
                    candidate_path=candidate_path,
                    pair=pair,
                    prepared_batches=prepared_batches,
                    strict=args.strict_patchable,
                )
            )

    write_path_score_records_jsonl(records, args.output)
    summary = {
        "num_pairs": len(pairs),
        "num_paths": len(candidate_paths),
        "num_records": len(records),
        "path_pair_bindings": args.path_pair_bindings,
        "output_path": args.output,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
