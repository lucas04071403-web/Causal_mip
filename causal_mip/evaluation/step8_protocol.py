from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PAIR_EVAL_SETS = ("forget_clean", "hard_retain", "counterfactual_retain")
FULL_CLEAR_REMOTE_ACC_METRICS = (
    "forget_classification_remote_acc",
    "forget_generation_remote_acc",
    "retain_classification_remote_acc",
    "retain_generation_remote_acc",
)
LOWER_IS_BETTER_FULL_CLEAR = {
    "forget_classification_remote_acc",
    "forget_generation_remote_acc",
}
HIGHER_IS_BETTER_FULL_CLEAR = {
    "retain_classification_remote_acc",
    "retain_generation_remote_acc",
}


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def _metric(data: dict[str, Any], *keys: str) -> float | None:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    if current is None:
        return None
    return float(current)


def _lower_is_better_check(
    name: str,
    candidate: float | None,
    baseline: float | None,
    min_drop: float,
) -> dict[str, Any]:
    if candidate is None or baseline is None:
        return {
            "name": name,
            "passed": False,
            "reason": "missing_metric",
            "candidate": candidate,
            "baseline": baseline,
            "required": None,
        }
    required = baseline - min_drop
    passed = candidate < baseline if min_drop == 0.0 else candidate <= required
    return {
        "name": name,
        "passed": passed,
        "direction": "lower_is_better",
        "candidate": candidate,
        "baseline": baseline,
        "delta": candidate - baseline,
        "min_required_drop": min_drop,
        "required_max": required,
    }


def _higher_is_better_check(
    name: str,
    candidate: float | None,
    baseline: float | None,
    max_drop: float,
) -> dict[str, Any]:
    if candidate is None or baseline is None:
        return {
            "name": name,
            "passed": False,
            "reason": "missing_metric",
            "candidate": candidate,
            "baseline": baseline,
            "required": None,
        }
    required = baseline - max_drop
    return {
        "name": name,
        "passed": candidate >= required,
        "direction": "higher_is_better_with_drop_tolerance",
        "candidate": candidate,
        "baseline": baseline,
        "delta": candidate - baseline,
        "max_allowed_drop": max_drop,
        "required_min": required,
    }


def evaluate_pair_screen(
    candidate_pair_eval: dict[str, Any],
    baseline_pair_eval: dict[str, Any],
    min_forget_name_hit_drop: float = 0.0,
    max_retain_name_hit_drop: float = 0.05,
) -> dict[str, Any]:
    candidate_summary = candidate_pair_eval.get("summary", {})
    baseline_summary = baseline_pair_eval.get("summary", {})
    checks = []
    metrics: dict[str, dict[str, float | None]] = {}

    for eval_set in PAIR_EVAL_SETS:
        candidate = _metric(candidate_summary, "by_eval_set", eval_set, "name_hit_rate")
        baseline = _metric(baseline_summary, "by_eval_set", eval_set, "name_hit_rate")
        metrics[eval_set] = {
            "candidate_name_hit_rate": candidate,
            "baseline_name_hit_rate": baseline,
        }
        if eval_set == "forget_clean":
            checks.append(
                _lower_is_better_check(
                    "pair_forget_clean_name_hit_rate",
                    candidate,
                    baseline,
                    min_forget_name_hit_drop,
                )
            )
        else:
            checks.append(
                _higher_is_better_check(
                    f"pair_{eval_set}_name_hit_rate",
                    candidate,
                    baseline,
                    max_retain_name_hit_drop,
                )
            )

    return {
        "protocol": "pair_based_step8_quick_screen_v1",
        "passed": all(check["passed"] for check in checks),
        "metrics": metrics,
        "checks": checks,
    }


def _full_clear_tasks_complete(summary: dict[str, Any]) -> dict[str, Any]:
    tasks = summary.get("tasks", {})
    checks = []
    for task_name, task in sorted(tasks.items()):
        num_examples = task.get("num_examples")
        num_scored = task.get("num_scored_examples")
        passed = (
            isinstance(num_examples, int)
            and isinstance(num_scored, int)
            and num_examples > 0
            and num_scored == num_examples
        )
        checks.append(
            {
                "name": f"full_clear_{task_name}_complete",
                "passed": passed,
                "num_examples": num_examples,
                "num_scored_examples": num_scored,
            }
        )
    if not tasks:
        checks.append(
            {
                "name": "full_clear_tasks_present",
                "passed": False,
                "reason": "missing_tasks",
            }
        )
    return {
        "passed": all(check["passed"] for check in checks),
        "checks": checks,
    }


def evaluate_full_clear_main(
    candidate_summary: dict[str, Any] | None,
    baseline_summary: dict[str, Any] | None,
    min_forget_remote_acc_drop: float = 0.0,
    max_retain_remote_acc_drop: float = 0.05,
) -> dict[str, Any]:
    if candidate_summary is None:
        return {
            "protocol": "full_clear_remote_main_protocol_v1",
            "passed": False,
            "status": "missing_candidate_full_clear_summary",
            "checks": [],
            "metrics": {},
        }
    if baseline_summary is None:
        return {
            "protocol": "full_clear_remote_main_protocol_v1",
            "passed": False,
            "status": "missing_baseline_full_clear_summary",
            "checks": [],
            "metrics": {},
        }

    candidate_metrics = candidate_summary.get("metrics", {})
    baseline_metrics = baseline_summary.get("metrics", {})
    checks = []
    metrics: dict[str, dict[str, float | None]] = {}

    for metric_name in FULL_CLEAR_REMOTE_ACC_METRICS:
        candidate = _metric(candidate_metrics, metric_name)
        baseline = _metric(baseline_metrics, metric_name)
        metrics[metric_name] = {
            "candidate": candidate,
            "baseline": baseline,
        }
        if metric_name in LOWER_IS_BETTER_FULL_CLEAR:
            checks.append(
                _lower_is_better_check(
                    f"full_clear_{metric_name}",
                    candidate,
                    baseline,
                    min_forget_remote_acc_drop,
                )
            )
        elif metric_name in HIGHER_IS_BETTER_FULL_CLEAR:
            checks.append(
                _higher_is_better_check(
                    f"full_clear_{metric_name}",
                    candidate,
                    baseline,
                    max_retain_remote_acc_drop,
                )
            )

    completion = _full_clear_tasks_complete(candidate_summary)
    checks.extend(completion["checks"])

    return {
        "protocol": "full_clear_remote_main_protocol_v1",
        "passed": all(check["passed"] for check in checks),
        "status": "ok",
        "metrics": metrics,
        "checks": checks,
    }


def _failed_reasons(section: str, checks: list[dict[str, Any]]) -> list[str]:
    return [
        f"{section}:{check['name']}"
        for check in checks
        if not check.get("passed", False)
    ]


def build_step8_protocol_report(
    candidate_pair_eval: dict[str, Any],
    baseline_pair_eval: dict[str, Any],
    candidate_full_clear_summary: dict[str, Any] | None,
    baseline_full_clear_summary: dict[str, Any] | None,
    min_pair_forget_name_hit_drop: float = 0.0,
    max_pair_retain_name_hit_drop: float = 0.05,
    min_full_forget_remote_acc_drop: float = 0.0,
    max_full_retain_remote_acc_drop: float = 0.05,
) -> dict[str, Any]:
    pair_screen = evaluate_pair_screen(
        candidate_pair_eval=candidate_pair_eval,
        baseline_pair_eval=baseline_pair_eval,
        min_forget_name_hit_drop=min_pair_forget_name_hit_drop,
        max_retain_name_hit_drop=max_pair_retain_name_hit_drop,
    )
    full_clear_main = evaluate_full_clear_main(
        candidate_summary=candidate_full_clear_summary,
        baseline_summary=baseline_full_clear_summary,
        min_forget_remote_acc_drop=min_full_forget_remote_acc_drop,
        max_retain_remote_acc_drop=max_full_retain_remote_acc_drop,
    )
    reasons = []
    reasons.extend(_failed_reasons("pair_screen", pair_screen["checks"]))
    reasons.extend(_failed_reasons("full_clear_main", full_clear_main["checks"]))
    if not full_clear_main["checks"] and not full_clear_main["passed"]:
        reasons.append(f"full_clear_main:{full_clear_main['status']}")

    can_claim_success = pair_screen["passed"] and full_clear_main["passed"]
    return {
        "protocol": {
            "name": "chip_editor_step8_main_protocol_v1",
            "decision_rule": "pair_screen_must_pass_and_full_clear_remote_main_must_pass",
            "pair_screen": {
                "min_forget_name_hit_drop": min_pair_forget_name_hit_drop,
                "max_retain_name_hit_drop": max_pair_retain_name_hit_drop,
            },
            "full_clear_main": {
                "min_forget_remote_acc_drop": min_full_forget_remote_acc_drop,
                "max_retain_remote_acc_drop": max_full_retain_remote_acc_drop,
            },
        },
        "pair_screen": pair_screen,
        "full_clear_main": full_clear_main,
        "decision": {
            "status": "pass" if can_claim_success else "do_not_claim_success",
            "can_claim_success": can_claim_success,
            "failed_reasons": reasons,
        },
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gate Step8 pair eval and Full CLEAR remote eval into one fixed protocol decision.")
    parser.add_argument("--pair_eval", required=True, help="Candidate Step8 pair eval JSON.")
    parser.add_argument("--pair_baseline", required=True, help="Baseline Step8 pair eval JSON.")
    parser.add_argument("--full_clear_summary", required=True, help="Candidate Full CLEAR remote protocol summary JSON.")
    parser.add_argument("--full_clear_baseline", required=True, help="Baseline Full CLEAR remote protocol summary JSON.")
    parser.add_argument("--output", required=True, help="Output Step8 protocol report JSON.")
    parser.add_argument("--min_pair_forget_name_hit_drop", type=float, default=0.0)
    parser.add_argument("--max_pair_retain_name_hit_drop", type=float, default=0.05)
    parser.add_argument("--min_full_forget_remote_acc_drop", type=float, default=0.0)
    parser.add_argument("--max_full_retain_remote_acc_drop", type=float, default=0.05)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    report = build_step8_protocol_report(
        candidate_pair_eval=read_json(args.pair_eval),
        baseline_pair_eval=read_json(args.pair_baseline),
        candidate_full_clear_summary=read_json(args.full_clear_summary),
        baseline_full_clear_summary=read_json(args.full_clear_baseline),
        min_pair_forget_name_hit_drop=args.min_pair_forget_name_hit_drop,
        max_pair_retain_name_hit_drop=args.max_pair_retain_name_hit_drop,
        min_full_forget_remote_acc_drop=args.min_full_forget_remote_acc_drop,
        max_full_retain_remote_acc_drop=args.max_full_retain_remote_acc_drop,
    )
    write_json(args.output, report)
    print(json.dumps(report["decision"], ensure_ascii=False, indent=2))
    print(f"Saved Step8 protocol report to {args.output}")


if __name__ == "__main__":
    main()
