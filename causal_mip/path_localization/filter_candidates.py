from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_jsonl(path: str) -> list[dict[str, Any]]:
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _write_jsonl(path: str | Path, records: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def filter_candidates(
    candidates_path: str,
    output_candidates_path: str,
    keep_path_ids: set[str],
    bindings_path: str | None = None,
    output_bindings_path: str | None = None,
    keep_pair_ids: set[str] | None = None,
) -> dict[str, Any]:
    candidates = [
        record
        for record in _load_jsonl(candidates_path)
        if str(record.get("path_id")) in keep_path_ids
    ]
    _write_jsonl(output_candidates_path, candidates)

    bindings = []
    if bindings_path and output_bindings_path:
        bindings = [
            record
            for record in _load_jsonl(bindings_path)
            if str(record.get("path_id")) in keep_path_ids
            and (keep_pair_ids is None or str(record.get("pair_id")) in keep_pair_ids)
        ]
        _write_jsonl(output_bindings_path, bindings)

    return {
        "candidates_path": candidates_path,
        "output_candidates_path": output_candidates_path,
        "bindings_path": bindings_path,
        "output_bindings_path": output_bindings_path,
        "keep_path_ids": sorted(keep_path_ids),
        "keep_pair_ids": sorted(keep_pair_ids) if keep_pair_ids else None,
        "num_candidates": len(candidates),
        "num_bindings": len(bindings),
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Filter candidate paths and optional pair bindings by path/pair id.")
    parser.add_argument("--candidates_path", required=True)
    parser.add_argument("--output_candidates", required=True)
    parser.add_argument("--keep_path_ids", nargs="+", required=True)
    parser.add_argument("--bindings_path", default=None)
    parser.add_argument("--output_bindings", default=None)
    parser.add_argument("--keep_pair_ids", nargs="*", default=None)
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    summary = filter_candidates(
        candidates_path=args.candidates_path,
        output_candidates_path=args.output_candidates,
        keep_path_ids=set(args.keep_path_ids),
        bindings_path=args.bindings_path,
        output_bindings_path=args.output_bindings,
        keep_pair_ids=set(args.keep_pair_ids) if args.keep_pair_ids else None,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
