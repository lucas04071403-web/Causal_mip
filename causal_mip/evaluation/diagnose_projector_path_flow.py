from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from causal_mip.project_paths import WORKSPACE_ROOT


PROJECTOR_MODULE_ALIASES = {
    "mm_projector",
    "model.mm_projector",
    "base_model.model.mm_projector",
    "visual.merger",
    "model.visual.merger",
    "base_model.model.visual.merger",
    "base_model.model.model.visual.merger",
}


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def count_by(records: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(record.get(key)) for record in records).items()))


def node_is_projector(node: dict[str, Any]) -> bool:
    module = node.get("module")
    module_kind = node.get("module_kind")
    return module_kind == "projector" or module in PROJECTOR_MODULE_ALIASES


def record_contains_projector_nodes(record: dict[str, Any]) -> bool:
    return any(node_is_projector(node) for node in record.get("nodes") or [])


def restored_nodes(record: dict[str, Any]) -> list[dict[str, Any]]:
    sufficiency = record.get("sufficiency") or {}
    return list(sufficiency.get("restored_nodes") or [])


def sufficiency_scores(record: dict[str, Any]) -> tuple[float, float, float] | None:
    sufficiency = record.get("sufficiency") or {}
    try:
        clean = float(sufficiency["clean_score"])
        corrupt = float(sufficiency["corrupt_score"])
        restored = float(sufficiency["restored_score"])
    except (KeyError, TypeError, ValueError):
        return None
    return clean, corrupt, restored


def score_contains_projector(record: dict[str, Any]) -> bool:
    if record_contains_projector_nodes(record):
        return True
    return any(node_is_projector(node) for node in restored_nodes(record))


def summarize_sufficiency_deltas(records: list[dict[str, Any]]) -> dict[str, Any]:
    values = [scores for record in records if (scores := sufficiency_scores(record)) is not None]
    if not values:
        return {
            "num_records_with_scores": 0,
            "num_clean_equals_corrupt": 0,
            "num_restored_equals_corrupt": 0,
            "mean_clean_minus_corrupt": None,
            "mean_restored_minus_corrupt": None,
        }
    clean_minus_corrupt = [clean - corrupt for clean, corrupt, _ in values]
    restored_minus_corrupt = [restored - corrupt for _, corrupt, restored in values]
    return {
        "num_records_with_scores": len(values),
        "num_clean_equals_corrupt": sum(1 for value in clean_minus_corrupt if abs(value) <= 1e-12),
        "num_restored_equals_corrupt": sum(1 for value in restored_minus_corrupt if abs(value) <= 1e-12),
        "mean_clean_minus_corrupt": mean(clean_minus_corrupt),
        "mean_restored_minus_corrupt": mean(restored_minus_corrupt),
    }


def summarize_candidates(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"path": None, "exists": False}
    records = read_jsonl(path)
    unique_ids = {record.get("path_id") for record in records if record.get("path_id") is not None}
    by_modality = count_by(records, "modality")
    by_source = count_by(records, "source")
    projector_ids = {
        record.get("path_id")
        for record in records
        if record.get("path_id") is not None and record_contains_projector_nodes(record)
    }
    vision_text_ids = {
        record.get("path_id")
        for record in records
        if record.get("path_id") is not None and record.get("modality") == "vision_text"
    }
    projector_neurons = sorted(
        {
            int(node["neuron"])
            for record in records
            for node in record.get("nodes") or []
            if node_is_projector(node) and node.get("neuron") is not None
        }
    )
    return {
        "path": str(path),
        "exists": True,
        "num_records": len(records),
        "num_unique_paths": len(unique_ids),
        "records_by_modality": by_modality,
        "records_by_source": by_source,
        "num_unique_vision_text_paths": len(vision_text_ids),
        "num_unique_projector_paths": len(projector_ids),
        "projector_path_ids_sample": sorted(projector_ids)[:20],
        "projector_neurons": projector_neurons,
    }


def summarize_bound_candidates(paths: list[Path]) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    all_records: list[dict[str, Any]] = []
    for path in paths:
        records = read_jsonl(path)
        all_records.extend(records)
        summaries[path.name] = {
            "path": str(path),
            "num_records": len(records),
            "num_unique_paths": len({record.get("path_id") for record in records}),
            "records_by_modality": count_by(records, "path_modality"),
            "records_by_source": count_by(records, "path_source"),
            "binding_strategies": count_by(records, "binding_strategy"),
            "num_vision_text_records": sum(1 for record in records if record.get("path_modality") == "vision_text"),
            "num_unique_vision_text_paths": len(
                {record.get("path_id") for record in records if record.get("path_modality") == "vision_text"}
            ),
            "num_vision_text_pairs": len(
                {record.get("pair_id") for record in records if record.get("path_modality") == "vision_text"}
            ),
        }

    return {
        "files": summaries,
        "combined": {
            "num_records": len(all_records),
            "num_unique_paths": len({record.get("path_id") for record in all_records}),
            "records_by_modality": count_by(all_records, "path_modality"),
            "records_by_source": count_by(all_records, "path_source"),
            "binding_strategies": count_by(all_records, "binding_strategy"),
            "num_vision_text_records": sum(1 for record in all_records if record.get("path_modality") == "vision_text"),
            "num_unique_vision_text_paths": len(
                {record.get("path_id") for record in all_records if record.get("path_modality") == "vision_text"}
            ),
            "num_vision_text_pairs": len(
                {record.get("pair_id") for record in all_records if record.get("path_modality") == "vision_text"}
            ),
        },
    }


def summarize_scores(path: Path) -> dict[str, Any]:
    records = read_jsonl(path)
    by_modality = defaultdict(list)
    projector_records = []
    for record in records:
        by_modality[str(record.get("path_modality"))].append(record)
        if score_contains_projector(record):
            projector_records.append(record)

    modality_summary: dict[str, Any] = {}
    for modality, items in sorted(by_modality.items()):
        full_patchable = [
            record
            for record in items
            if record.get("num_nodes") is not None
            and record.get("num_patchable_nodes") is not None
            and int(record["num_nodes"]) == int(record["num_patchable_nodes"])
        ]
        positive_suf = [record for record in items if float(record.get("Suf") or 0.0) > 0.0]
        positive_nec = [record for record in items if float(record.get("Nec") or 0.0) > 0.0]
        positive_ret = [record for record in items if float(record.get("Ret") or 0.0) > 0.0]
        modality_summary[modality] = {
            "num_records": len(items),
            "num_unique_paths": len({record.get("path_id") for record in items}),
            "sufficiency_corrupt_sources": count_by(items, "sufficiency_corrupt_source"),
            "num_full_patchable_records": len(full_patchable),
            "full_patchable_rate": len(full_patchable) / len(items) if items else 0.0,
            "num_positive_Nec_records": len(positive_nec),
            "num_positive_Suf_records": len(positive_suf),
            "num_positive_Ret_records": len(positive_ret),
            "mean_Nec": mean(float(record.get("Nec") or 0.0) for record in items) if items else 0.0,
            "mean_Suf": mean(float(record.get("Suf") or 0.0) for record in items) if items else 0.0,
            "mean_Ret": mean(float(record.get("Ret") or 0.0) for record in items) if items else 0.0,
            "sufficiency_deltas": summarize_sufficiency_deltas(items),
        }

    full_projector = [
        record
        for record in projector_records
        if record.get("num_nodes") is not None
        and record.get("num_patchable_nodes") is not None
        and int(record["num_nodes"]) == int(record["num_patchable_nodes"])
    ]
    positive_suf_projector = [record for record in projector_records if float(record.get("Suf") or 0.0) > 0.0]
    projector_node_modules = Counter()
    projector_node_neurons = Counter()
    for record in projector_records:
        for node in restored_nodes(record):
            if node_is_projector(node):
                projector_node_modules[str(node.get("module"))] += 1
                projector_node_neurons[str(node.get("neuron"))] += 1

    return {
        "path": str(path),
        "num_records": len(records),
        "num_unique_paths": len({record.get("path_id") for record in records}),
        "records_by_modality": modality_summary,
        "projector": {
            "num_records": len(projector_records),
            "num_unique_paths": len({record.get("path_id") for record in projector_records}),
            "sufficiency_corrupt_sources": count_by(projector_records, "sufficiency_corrupt_source"),
            "num_full_patchable_records": len(full_projector),
            "full_patchable_rate": len(full_projector) / len(projector_records) if projector_records else 0.0,
            "num_positive_Suf_records": len(positive_suf_projector),
            "path_ids_sample": sorted({record.get("path_id") for record in projector_records})[:20],
            "restored_projector_modules": dict(sorted(projector_node_modules.items())),
            "restored_projector_neurons": dict(sorted(projector_node_neurons.items())),
            "sufficiency_deltas": summarize_sufficiency_deltas(projector_records),
        },
    }


def summarize_step6(step6_dir: Path) -> dict[str, Any]:
    category_files = {
        "P_forget": step6_dir / "P_forget.jsonl",
        "P_shared": step6_dir / "P_shared.jsonl",
        "P_retain": step6_dir / "P_retain.jsonl",
        "P_irrelevant": step6_dir / "P_irrelevant.jsonl",
    }
    categories: dict[str, Any] = {}
    all_records: list[dict[str, Any]] = []
    for category, path in category_files.items():
        records = read_jsonl(path)
        all_records.extend(records)
        projector_records = [record for record in records if record.get("path_modality") == "vision_text"]
        categories[category] = {
            "path": str(path),
            "num_paths": len(records),
            "paths_by_modality": count_by(records, "path_modality"),
            "num_vision_text_paths": len(projector_records),
            "vision_text_path_ids": sorted(record.get("path_id") for record in projector_records),
            "num_full_patchable_vision_text_paths": sum(
                1
                for record in projector_records
                if record.get("num_nodes") is not None
                and record.get("num_patchable_nodes") is not None
                and int(record["num_nodes"]) == int(record["num_patchable_nodes"])
            ),
            "num_positive_Suf_paths": sum(1 for record in records if float(record.get("Suf") or 0.0) > 0.0),
            "num_positive_Suf_vision_text_paths": sum(
                1 for record in projector_records if float(record.get("Suf") or 0.0) > 0.0
            ),
            "mean_forget_effect": mean(float(record.get("forget_effect") or 0.0) for record in records)
            if records
            else 0.0,
            "mean_retain_impact": mean(float(record.get("retain_impact") or 0.0) for record in records)
            if records
            else 0.0,
        }

    summary_path = step6_dir / "classification_summary.json"
    return {
        "path": str(step6_dir),
        "classification_summary": read_json(summary_path) if summary_path.exists() else None,
        "categories": categories,
        "combined": {
            "num_paths": len(all_records),
            "paths_by_modality": count_by(all_records, "path_modality"),
            "num_vision_text_paths": sum(1 for record in all_records if record.get("path_modality") == "vision_text"),
            "num_positive_Suf_paths": sum(1 for record in all_records if float(record.get("Suf") or 0.0) > 0.0),
        },
    }


def summarize_step7(path: Path) -> dict[str, Any]:
    data = read_json(path)
    mask_summary = data.get("mask_summary") or {}
    modules = list(mask_summary.get("modules") or [])
    skipped = list(mask_summary.get("skipped_modules") or [])
    projector_modules = [module for module in modules if module.get("module_kind") == "projector"]
    projector_skipped = [
        module
        for module in skipped
        if module.get("module") in PROJECTOR_MODULE_ALIASES or "projector" in str(module.get("module"))
    ]
    skipped_reasons = Counter(str(module.get("reason")) for module in skipped)
    return {
        "path": str(path),
        "num_modules": mask_summary.get("num_modules"),
        "num_editable_neurons": mask_summary.get("num_editable_neurons"),
        "num_projector_modules": len(projector_modules),
        "num_projector_editable_neurons": sum(len(module.get("editable_neurons") or []) for module in projector_modules),
        "projector_modules": projector_modules,
        "projector_skipped_modules": projector_skipped,
        "skipped_reasons": dict(sorted(skipped_reasons.items())),
        "num_loss_records": data.get("num_loss_records"),
        "checkpoint_dir": data.get("checkpoint_dir"),
        "merged_partial_linear_modules": data.get("merged_partial_linear_modules"),
    }


def summarize_step8(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    data = read_json(path)
    decision = data.get("decision") or {}
    pair_screen = data.get("pair_screen") or {}
    full_clear = data.get("full_clear_main") or {}
    return {
        "path": str(path),
        "decision": decision,
        "pair_screen_passed": pair_screen.get("passed"),
        "pair_screen_metrics": pair_screen.get("metrics"),
        "pair_screen_failed_checks": [
            check for check in pair_screen.get("checks") or [] if not check.get("passed")
        ],
        "full_clear_passed": full_clear.get("passed"),
    }


def infer_bottleneck(report: dict[str, Any]) -> list[str]:
    findings: list[str] = []
    candidates = report.get("candidate_paths") or {}
    bound = report.get("bound_candidate_paths") or {}
    scores = report.get("step5_scores") or {}
    step6 = report.get("step6") or {}
    step7 = report.get("step7") or {}

    projector_neurons = candidates.get("projector_neurons") or []
    if projector_neurons == [0]:
        findings.append("Candidate projector nodes all use neuron=0; projector dimensions look like placeholders.")

    bound_combined = bound.get("combined") or {}
    binding_strategies = bound_combined.get("binding_strategies") or {}
    if binding_strategies.get("global_fallback_vision_text", 0) > 0:
        findings.append(
            "Bound vision_text paths use global_fallback_vision_text; cross-modal paths are not sample-local."
        )

    score_projector = scores.get("projector") or {}
    if score_projector.get("num_records", 0) == 0:
        findings.append("Step5 did not score any projector/cross-modal records.")
    elif score_projector.get("full_patchable_rate", 0.0) < 1.0:
        findings.append("Step5 has projector records, but not all projector paths are fully patchable.")
    else:
        findings.append("Step5 projector/cross-modal records are present and fully patchable.")

    if score_projector.get("num_positive_Suf_records", 0) == 0:
        findings.append("Step5 projector/cross-modal sufficiency is zero; classification is effectively Nec/Ret-driven.")
    deltas = score_projector.get("sufficiency_deltas") or {}
    if (
        deltas.get("num_records_with_scores", 0) > 0
        and deltas.get("num_clean_equals_corrupt") == deltas.get("num_records_with_scores")
    ):
        findings.append(
            "For projector/cross-modal Step5 records, clean_score equals corrupt_score for every record; "
            "the corrupt input is not creating a sufficiency contrast."
        )

    categories = (step6.get("categories") or {})
    p_forget = categories.get("P_forget") or {}
    p_shared = categories.get("P_shared") or {}
    if p_forget.get("num_vision_text_paths", 0) == 0 and p_shared.get("num_vision_text_paths", 0) > 0:
        findings.append(
            "Step6 places vision_text/projector paths in P_shared, not P_forget; Step7 will preserve them instead of editing them."
        )
    elif p_forget.get("num_vision_text_paths", 0) > 0:
        findings.append("Step6 includes vision_text/projector paths in P_forget.")
    else:
        findings.append("Step6 does not include vision_text/projector paths in P_forget or P_shared.")

    if step7.get("num_projector_modules", 0) == 0:
        skipped = step7.get("projector_skipped_modules") or []
        reasons = sorted({str(item.get("reason")) for item in skipped})
        reason_text = ", ".join(reasons) if reasons else "unknown"
        findings.append(f"Step7 edited zero projector modules; projector skip reason(s): {reason_text}.")
    else:
        findings.append("Step7 edited projector modules.")

    return findings


def default_paths(run_id: str, workspace_root: Path) -> dict[str, Path]:
    outputs = workspace_root / "outputs"
    return {
        "candidate_paths": outputs / "paths" / "P_cand.jsonl",
        "bound_train": outputs / "paths" / f"P_cand_bound_train_{run_id}.jsonl",
        "bound_val": outputs / "paths" / f"P_cand_bound_val_{run_id}.jsonl",
        "scores": outputs / "scores" / f"path_scores_bound_projector_{run_id}.jsonl",
        "step6_dir": outputs / "paths" / f"step6_bound_suf_projector_{run_id}",
        "step7_summary": outputs / f"masked_rmisu_{run_id}.json",
        "step8_protocol": outputs / f"step8_protocol_{run_id}.json",
        "output": outputs / "diagnostics" / f"projector_path_flow_{run_id}.json",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose where projector / vision_text paths drop out of the causal selective editing pipeline."
    )
    parser.add_argument("--run_id", default="step9_bound_projector_0526_214028")
    parser.add_argument("--workspace_root", default=str(WORKSPACE_ROOT))
    parser.add_argument("--candidate_paths", default=None)
    parser.add_argument("--bound_train", default=None)
    parser.add_argument("--bound_val", default=None)
    parser.add_argument("--scores", default=None)
    parser.add_argument("--step6_dir", default=None)
    parser.add_argument("--step7_summary", default=None)
    parser.add_argument("--step8_protocol", default=None)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workspace_root = Path(args.workspace_root)
    paths = default_paths(args.run_id, workspace_root)
    for key in [
        "candidate_paths",
        "bound_train",
        "bound_val",
        "scores",
        "step6_dir",
        "step7_summary",
        "step8_protocol",
        "output",
    ]:
        override = getattr(args, key)
        if override is not None:
            paths[key] = Path(override)

    required = ["bound_train", "bound_val", "scores", "step6_dir", "step7_summary"]
    missing = [key for key in required if not paths[key].exists()]
    if missing:
        details = "\n".join(f"{key}: {paths[key]}" for key in missing)
        raise FileNotFoundError(f"Missing required diagnostics inputs:\n{details}")

    candidate_path = paths["candidate_paths"] if paths["candidate_paths"].exists() else None
    report: dict[str, Any] = {
        "run_id": args.run_id,
        "workspace_root": str(workspace_root),
        "candidate_paths": summarize_candidates(candidate_path),
        "bound_candidate_paths": summarize_bound_candidates([paths["bound_train"], paths["bound_val"]]),
        "step5_scores": summarize_scores(paths["scores"]),
        "step6": summarize_step6(paths["step6_dir"]),
        "step7": summarize_step7(paths["step7_summary"]),
        "step8": summarize_step8(paths["step8_protocol"]),
    }
    report["findings"] = infer_bottleneck(report)

    write_json(paths["output"], report)

    print(f"Wrote diagnostics: {paths['output']}")
    print("\nKey findings:")
    for finding in report["findings"]:
        print(f"- {finding}")

    step6_categories = report["step6"]["categories"]
    print("\nVision_text distribution in Step6:")
    for category in ["P_forget", "P_shared", "P_retain", "P_irrelevant"]:
        item = step6_categories[category]
        print(f"- {category}: {item['num_vision_text_paths']} vision_text paths")

    step7 = report["step7"]
    print("\nStep7 projector editing:")
    print(f"- num_projector_modules: {step7['num_projector_modules']}")
    print(f"- projector_skipped_modules: {step7['projector_skipped_modules']}")


if __name__ == "__main__":
    main()
