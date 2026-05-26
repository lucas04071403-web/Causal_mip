from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from clear import CLEAR_Clf_Dataset, CLEAR_Gen_Dataset
from metrics.bleu.bleu import Bleu
from metrics.rouge.rouge import Rouge
from test_data import ClearClfDataset, ClearGenDataset, collator


TASKS = ("clf_forget", "clf_retain", "gen_forget", "gen_retain")


class TimeoutRemoteLLMClient:
    """OpenAI-compatible remote judge client with bounded request time."""

    def __init__(
        self,
        base_url: str,
        api_key: str = "empty",
        model_name: str = "Qwen2.5-VL-7B-Instruct",
        timeout: float = 60.0,
    ) -> None:
        from openai import OpenAI
        import httpx

        os.environ["no_proxy"] = "*"
        http_client = httpx.Client(
            timeout=httpx.Timeout(timeout, connect=min(10.0, timeout)),
            trust_env=False,
        )
        self.client = OpenAI(api_key=api_key, base_url=base_url, http_client=http_client)
        self.model_name = model_name

    def chat(self, messages: list[dict[str, Any]], max_tokens: int = 50, temperature: float = 0.1) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, default=_json_default)
    tmp.replace(output)


def extract_assistant_text(decoded: str, model_name: str) -> str:
    if "Qwen" in model_name:
        marker = "assistant\n"
        idx = decoded.lower().find(marker)
        return decoded[idx + len(marker) :].strip() if idx >= 0 else decoded.strip()
    if "llava" in model_name:
        marker = "ASSISTANT:"
        idx = decoded.find(marker)
        pred = decoded[idx + len(marker) :] if idx >= 0 else decoded
        return pred.split("ASSISTANT:")[0].strip().split("\n")[0]
    return decoded.strip()


def build_args_dict(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "model": args.model,
        "model_path": args.model_path,
        "dataset": "clear",
        "forget_ratio": args.forget_ratio,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "finetune_epochs": args.finetune_epochs,
        "image_resize": args.image_resize,
        "device": args.device,
        "score_llm": args.score_llm,
        "use_remote_scoring": True,
        "remote_scoring_url": args.remote_scoring_url,
        "this_run_id": args.run_id,
        "llm_directory": args.llm_directory,
        "base_path": args.base_path,
        "output_file_path": args.output_dir,
    }


def load_clear_dataset(task: str, processor, args: argparse.Namespace):
    base = Path(args.base_path) / "CLEAR"
    if task == "clf_forget":
        return CLEAR_Clf_Dataset(str(base / f"forget{args.forget_ratio}_perturbed"), processor, args.image_resize)
    if task == "clf_retain":
        return CLEAR_Clf_Dataset(str(base / "retain_perturbed"), processor, args.image_resize)
    if task == "gen_forget":
        return CLEAR_Gen_Dataset(str(base / f"forget{args.forget_ratio}+tofu"), processor, args.image_resize)
    if task == "gen_retain":
        return CLEAR_Gen_Dataset(str(base / f"retain{100 - args.forget_ratio}+tofu"), processor, args.image_resize)
    raise ValueError(f"Unknown task: {task}")


def maybe_limit_dataset(raw_dataset, max_examples: int | None):
    if max_examples is None:
        return raw_dataset

    class LimitedDataset(torch.utils.data.Dataset):
        def __init__(self, dataset, limit: int) -> None:
            self.dataset = dataset
            self.limit = min(limit, len(dataset))

        def __len__(self) -> int:
            return self.limit

        def __getitem__(self, idx: int):
            return self.dataset[idx]

    return LimitedDataset(raw_dataset, max_examples)


def prediction_path(args: argparse.Namespace, task: str) -> Path:
    split = "forget" if "forget" in task else "retain"
    kind = "clf" if task.startswith("clf") else "gen"
    return Path(args.output_dir) / args.run_id / f"{kind}_{split}set_multi_preds.json"


def scored_path(args: argparse.Namespace, task: str) -> Path:
    split = "forget" if "forget" in task else "retain"
    kind = "clf" if task.startswith("clf") else "gen"
    return Path(args.output_dir) / args.run_id / f"{kind}_{split}set_multi_preds_remote_scored.json"


def generate_predictions_for_task(model, processor, args: argparse.Namespace, task: str) -> dict[str, Any]:
    path = prediction_path(args, task)
    if path.exists() and not args.force_generate:
        print(f"Using existing predictions: {path}")
        return read_json(path)

    raw_dataset = maybe_limit_dataset(load_clear_dataset(task, processor, args), args.max_examples_per_task)
    is_clf = task.startswith("clf")
    processed = (
        ClearClfDataset(raw_dataset, processor, args)
        if is_clf
        else ClearGenDataset(raw_dataset, processor, args, modality="multi")
    )
    loader = torch.utils.data.DataLoader(
        processed,
        batch_size=args.pred_batch_size,
        shuffle=False,
        collate_fn=lambda batch: collator(batch, processor, args),
    )

    preds: list[dict[str, Any]] = []
    model.eval()
    for batch_idx, sample in enumerate(tqdm(loader, desc=f"Generate {task}")):
        with torch.no_grad():
            generate_kwargs = {
                **sample["inputs"],
                "max_new_tokens": args.clf_max_new_tokens if is_clf else args.gen_max_new_tokens,
            }
            if is_clf:
                generate_kwargs["min_new_tokens"] = 1
            generated_ids = model.generate(**generate_kwargs)
        decoded = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        for i, text in enumerate(decoded):
            row = {
                "question": sample["question"][i],
                "gt": sample["ground_truth"][i],
                "pred": extract_assistant_text(text, args.model),
            }
            if is_clf:
                row["options"] = sample["options"][i]
            preds.append(row)

        if batch_idx % args.save_every_batches == 0:
            data = build_prediction_json(args, task, preds, partial=True)
            write_json(path, data)
            tqdm.write(f"Saved partial predictions: {path} ({len(preds)} rows)")

    data = build_prediction_json(args, task, preds, partial=False)
    add_text_metrics(data, is_clf=is_clf)
    write_json(path, data)
    print(f"Saved predictions: {path} ({len(preds)} rows)")
    return data


def build_prediction_json(args: argparse.Namespace, task: str, preds: list[dict[str, Any]], partial: bool) -> dict[str, Any]:
    is_clf = task.startswith("clf")
    split = "forget" if "forget" in task else "retain"
    return {
        "remark": "full CLEAR remote scoring predictions",
        "args": build_args_dict(args),
        "dataset_split": split,
        "data_modality": "multi",
        "task": "clf" if is_clf else "gen",
        "partial": partial,
        "preds": preds,
    }


def safe_bleu(preds: list[str], refs: list[str]) -> float:
    try:
        return float(Bleu().compute(predictions=preds, references=refs)["bleu"])
    except Exception:
        return 0.0


def safe_rouge(preds: list[str], refs: list[str]) -> dict[str, float]:
    try:
        scores = Rouge().compute(predictions=preds, references=refs)
        return {key: float(scores[key]) for key in ("rouge1", "rouge2", "rougeL", "rougeLsum")}
    except Exception:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "rougeLsum": 0.0}


def add_text_metrics(data: dict[str, Any], is_clf: bool) -> None:
    if is_clf:
        return
    preds = data["preds"]
    pred_texts = [row["pred"] for row in preds]
    refs = [row["gt"] for row in preds]
    data["bleu"] = safe_bleu(pred_texts, refs)
    data.update(safe_rouge(pred_texts, refs))


def parse_remote_answer(response: str | None) -> bool | None:
    if response is None:
        return None
    text = response.strip()
    text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text).strip()
    text = text.replace("'", '"')
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict) and "answer" in parsed:
            value = parsed["answer"]
        else:
            value = parsed
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "yes", "correct"}:
                return True
            if lowered in {"false", "no", "incorrect"}:
                return False
    except Exception:
        lowered = text.lower()
        if '"answer": true' in lowered or '"answer":true' in lowered:
            return True
        if '"answer": false' in lowered or '"answer":false' in lowered:
            return False
    return None


def build_conversation_for_gen_task(item: dict[str, Any]) -> list[dict[str, str]]:
    system_prompt = """You are a helpful assistant that determines whether the prediction is correct.
Given a question with correct answer and a predicted answer, you will output a JSON object with a boolean field 'answer' indicating
whether the 'pred' answer is semantically the same (true) or different (false) as the 'truth' option.
The sentences will be provided in the fields 'input', 'question', 'truth', and 'pred'.
Please respond only with the JSON object, without any additional text or explanation."""
    data = {
        "question": item["question"],
        "truth": item["gt"],
        "pred": item["pred"][:500],
    }
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(data, ensure_ascii=False, indent=2)},
    ]


def build_conversation_for_clf_task(item: dict[str, Any]) -> list[dict[str, str]]:
    system_prompt = """You are a helpful assistant that determines whether the prediction is correct.
Given a multiple-choice question with correct answer and a predicted answer, you will output a JSON object with a boolean field 'answer' indicating
whether the 'pred' sentence has correctly answered the question (true) or not (false) according to the 'truth' option.
The 'pred' may be a single option or repeated characters of the same option or a complete sentence. It's okay if the 'pred' is not exactly the same as the 'truth' option, as long as it conveys the same meaning.
But it should not be a combination of multiple options, as that would be incorrect.
The sentences will be provided in the fields 'input', 'question', 'truth', and 'options'.
Please respond only with the JSON object, without any additional text or explanation."""
    data = {
        "question": item["question"],
        "options": item["options"],
        "truth": item["gt"],
        "pred": item["pred"][:500],
    }
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(data, ensure_ascii=False, indent=2)},
    ]


def score_one(remote_client, item: dict[str, Any], task: str, max_retries: int, retry_sleep: float) -> tuple[bool | None, str | None]:
    conversation = (
        build_conversation_for_clf_task(item)
        if task == "clf"
        else build_conversation_for_gen_task(item)
    )
    last_response = None
    for attempt in range(max_retries + 1):
        try:
            response = remote_client.chat(conversation, max_tokens=50, temperature=0.1)
            last_response = response
            parsed = parse_remote_answer(response)
            if parsed is not None:
                return parsed, response
        except Exception as exc:
            last_response = f"ERROR: {type(exc).__name__}: {exc}"
        if attempt < max_retries:
            time.sleep(retry_sleep)
    return None, last_response


def remote_score_predictions(args: argparse.Namespace, task: str, prediction_data: dict[str, Any]) -> dict[str, Any]:
    output = scored_path(args, task)
    if output.exists() and not args.force_score:
        scored = read_json(output)
    else:
        scored = dict(prediction_data)
        scored["preds"] = [dict(row) for row in prediction_data["preds"]]
        scored["remote_scoring_url"] = args.remote_scoring_url
        scored["score_llm"] = args.score_llm

    task_kind = "clf" if task.startswith("clf") else "gen"
    preds = scored["preds"]
    pending_indices = [
        idx
        for idx, row in enumerate(preds)
        if row.get("correct") is None
    ]

    if not pending_indices:
        add_score_metrics(scored, is_clf=task_kind == "clf")
        write_json(output, scored)
        print(f"All rows already scored: {output}")
        return scored

    def score_index(idx: int) -> tuple[int, bool | None, str | None]:
        remote_client = TimeoutRemoteLLMClient(
            args.remote_scoring_url,
            model_name=args.score_llm,
            timeout=args.remote_timeout,
        )
        correct, raw = score_one(
            remote_client=remote_client,
            item=preds[idx],
            task=task_kind,
            max_retries=args.remote_retries,
            retry_sleep=args.remote_retry_sleep,
        )
        return idx, correct, raw

    completed_since_save = 0
    max_workers = max(1, args.remote_workers)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(score_index, idx) for idx in pending_indices]
        progress = tqdm(as_completed(futures), total=len(futures), desc=f"Remote score {task}")
        for future in progress:
            idx, correct, raw = future.result()
            preds[idx]["correct"] = correct
            preds[idx]["remote_raw_response"] = raw
            completed_since_save += 1
            if completed_since_save >= args.save_every_scores:
                add_score_metrics(scored, is_clf=task_kind == "clf")
                write_json(output, scored)
                completed_since_save = 0
                progress.set_postfix(scored=scored["num_scored_examples"], total=len(preds))

    add_score_metrics(scored, is_clf=task_kind == "clf")
    write_json(output, scored)
    print(f"Saved remote scored predictions: {output}")
    return scored


def add_score_metrics(data: dict[str, Any], is_clf: bool) -> None:
    preds = data["preds"]
    valid = [row for row in preds if row.get("correct") is not None]
    acc = sum(int(row["correct"]) for row in valid) / len(valid) if valid else 0.0
    if is_clf:
        data["acc_by_llm"] = acc
    else:
        data["acc"] = acc
        add_text_metrics(data, is_clf=False)
    data["num_examples"] = len(preds)
    data["num_scored_examples"] = len(valid)


def build_protocol_summary(args: argparse.Namespace, scored_by_task: dict[str, dict[str, Any]]) -> dict[str, Any]:
    metrics: dict[str, float] = {}
    tasks: dict[str, Any] = {}
    for task, data in scored_by_task.items():
        is_clf = task.startswith("clf")
        split = "forget" if "forget" in task else "retain"
        metric_key = f"{split}_{'classification' if is_clf else 'generation'}_remote_acc"
        value = data.get("acc_by_llm") if is_clf else data.get("acc")
        metrics[metric_key] = float(value if value is not None else 0.0)
        if not is_clf:
            metrics[f"{split}_generation_bleu"] = float(data.get("bleu", 0.0))
            metrics[f"{split}_generation_rougeL"] = float(data.get("rougeL", 0.0))
        tasks[task] = {
            "path": str(scored_path(args, task)),
            "task": "clf" if is_clf else "gen",
            "split": split,
            "modality": "multi",
            "num_examples": data.get("num_examples", len(data.get("preds", []))),
            "num_scored_examples": data.get("num_scored_examples"),
            "remote_correct_rate": value,
            "score_llm": args.score_llm,
            "remote_scoring_url": args.remote_scoring_url,
            "bleu": data.get("bleu"),
            "rougeL": data.get("rougeL"),
        }
    return {
        "protocol": {
            "name": "mip_full_clear_remote_eval_v1",
            "modality": "multi",
            "required_remote_judge": args.score_llm,
        },
        "source": {
            "type": "checkpoint",
            "model_path": args.model_path,
            "run_id": args.run_id,
        },
        "metrics": metrics,
        "tasks": tasks,
    }


def run(args: argparse.Namespace) -> None:
    os.environ["no_proxy"] = "*"
    output_root = Path(args.output_dir) / args.run_id
    output_root.mkdir(parents=True, exist_ok=True)

    processor = AutoProcessor.from_pretrained(args.model_path, padding_side="left")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.to(args.device)

    scored_by_task = {}
    for task in args.tasks:
        prediction_data = generate_predictions_for_task(model, processor, args, task)
        if args.generate_only:
            continue
        model.cpu()
        torch.cuda.empty_cache()
        scored_by_task[task] = remote_score_predictions(args, task, prediction_data)
        model.to(args.device)

    if not args.generate_only:
        summary = build_protocol_summary(args, scored_by_task)
        summary_path = output_root / "full_clear_remote_protocol_summary.json"
        write_json(summary_path, summary)
        print(json.dumps(summary["metrics"], ensure_ascii=False, indent=2))
        print(f"Saved summary: {summary_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Full CLEAR prediction + resumable remote scoring.")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--model", default="Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--base_path", default="/home/lucas/Desktop/CurrentReacher/MIP_fusion7/mip_workspace/datasets/")
    parser.add_argument("--llm_directory", default="/home/lucas/Desktop/CurrentReacher/MIP_fusion7/mip_workspace/llms/")
    parser.add_argument("--output_dir", default="/home/lucas/Desktop/CurrentReacher/MIP_fusion7/mip_workspace/outputs/full_clear_remote_eval")
    parser.add_argument("--run_id", default="chip_full_train_0526_1348_full_remote")
    parser.add_argument("--forget_ratio", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--finetune_epochs", type=int, default=1)
    parser.add_argument("--image_resize", type=int, default=224)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pred_batch_size", type=int, default=28)
    parser.add_argument("--clf_max_new_tokens", type=int, default=60)
    parser.add_argument("--gen_max_new_tokens", type=int, default=100)
    parser.add_argument("--remote_scoring_url", default="http://210.40.56.85:21936/v1")
    parser.add_argument("--score_llm", default="Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--remote_timeout", type=float, default=60.0)
    parser.add_argument("--remote_retries", type=int, default=2)
    parser.add_argument("--remote_retry_sleep", type=float, default=2.0)
    parser.add_argument("--remote_workers", type=int, default=4)
    parser.add_argument("--save_every_batches", type=int, default=5)
    parser.add_argument("--save_every_scores", type=int, default=20)
    parser.add_argument("--max_examples_per_task", type=int, default=None)
    parser.add_argument("--force_generate", action="store_true")
    parser.add_argument("--force_score", action="store_true")
    parser.add_argument("--generate_only", action="store_true")
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    return parser


def main() -> None:
    parser = build_arg_parser()
    run(parser.parse_args())


if __name__ == "__main__":
    main()
