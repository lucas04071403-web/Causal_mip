from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from datasets import load_dataset, load_from_disk
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from metrics.bleu.bleu import Bleu
from metrics.rouge.rouge import Rouge
from causal_mip.project_paths import resolve_workspace_dataset_path


IMAGE_CAPTION_QUESTION = "Describe the visible scene and objects in this image without identifying the person."


@dataclass
class EvalExample:
    pair_id: str
    split: str
    eval_set: str
    sample_type: str
    question: str
    target: str
    name: str
    image: Image.Image | None


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_dataset_item(image_ref: dict[str, Any] | None) -> dict[str, Any] | None:
    if not image_ref:
        return None
    dataset_path = resolve_workspace_dataset_path(image_ref["dataset_path"])
    row_idx = int(image_ref["row_idx"])
    try:
        dataset = load_from_disk(dataset_path)
        if hasattr(dataset, "keys"):
            dataset = dataset["train"]
    except Exception:
        dataset = load_dataset(dataset_path, split="train")
    return dataset[row_idx]


def load_image(image_ref: dict[str, Any] | None, image_resize: int) -> Image.Image | None:
    item = load_dataset_item(image_ref)
    if item is None:
        return None
    image = item.get("image")
    if image is None:
        return None
    return image.convert("RGB").resize((image_resize, image_resize))


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip().lower()


def name_hit(prediction: str, name: str) -> bool:
    pred = normalize_text(prediction)
    target = normalize_text(name)
    if not target:
        return False
    if target in pred:
        return True
    parts = [part for part in re.split(r"[^a-zA-Z]+", target) if len(part) > 2]
    return bool(parts) and all(part.lower() in pred for part in parts)


def build_conversation(question: str, has_image: bool) -> list[dict[str, Any]]:
    content = []
    if has_image:
        content.append({"type": "image"})
    content.append({"type": "text", "text": question})
    return [{"role": "user", "content": content}]


def extract_assistant_text(decoded: str) -> str:
    marker = "assistant\n"
    idx = decoded.lower().find(marker)
    if idx >= 0:
        return decoded[idx + len(marker) :].strip()
    return decoded.strip()


def build_examples(pair_records: list[dict[str, Any]], split: str, image_resize: int) -> list[EvalExample]:
    examples: list[EvalExample] = []
    for record in pair_records:
        pair_id = record["pair_id"]
        forget_clean = record["forget_clean"]
        examples.append(
            EvalExample(
                pair_id=pair_id,
                split=split,
                eval_set="forget_clean",
                sample_type=forget_clean.get("type", "forget_clean"),
                question=forget_clean["question"],
                target=forget_clean.get("answer") or forget_clean.get("caption") or "",
                name=forget_clean.get("name", ""),
                image=load_image(forget_clean.get("image_ref"), image_resize),
            )
        )

        for hard in record.get("hard_retain", []):
            examples.append(
                EvalExample(
                    pair_id=pair_id,
                    split=split,
                    eval_set="hard_retain",
                    sample_type=hard.get("type", "hard_retain"),
                    question=hard.get("question") or IMAGE_CAPTION_QUESTION,
                    target=hard.get("answer") or hard.get("caption") or "",
                    name=hard.get("name", ""),
                    image=load_image(hard.get("image_ref"), image_resize),
                )
            )

        counterfactual = record.get("counterfactual_retain")
        if counterfactual is not None:
            examples.append(
                EvalExample(
                    pair_id=pair_id,
                    split=split,
                    eval_set="counterfactual_retain",
                    sample_type=counterfactual.get("type", "counterfactual_retain"),
                    question=counterfactual["question"],
                    target=counterfactual.get("answer") or counterfactual.get("caption") or "",
                    name=counterfactual.get("name", ""),
                    image=load_image(counterfactual.get("image_ref"), image_resize),
                )
            )
    return examples


def limit_examples(examples: list[EvalExample], max_per_set: int | None) -> list[EvalExample]:
    if max_per_set is None:
        return examples
    counts: dict[str, int] = {}
    limited = []
    for example in examples:
        key = f"{example.split}:{example.eval_set}:{example.sample_type}"
        if counts.get(key, 0) >= max_per_set:
            continue
        counts[key] = counts.get(key, 0) + 1
        limited.append(example)
    return limited


def generate_predictions(
    model,
    processor,
    examples: list[EvalExample],
    device: str,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    predictions = []
    model.eval()
    for example in tqdm(examples, desc="Step8 final eval"):
        chat = processor.apply_chat_template(
            build_conversation(example.question, has_image=example.image is not None),
            tokenize=False,
            add_generation_prompt=True,
        )
        if example.image is not None:
            inputs = processor(text=[chat], images=example.image, padding=True, return_tensors="pt").to(device)
        else:
            inputs = processor(text=[chat], padding=True, return_tensors="pt").to(device)

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, min_new_tokens=1)
        decoded = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        pred = extract_assistant_text(decoded)
        predictions.append(
            {
                "pair_id": example.pair_id,
                "split": example.split,
                "eval_set": example.eval_set,
                "sample_type": example.sample_type,
                "question": example.question,
                "gt": example.target,
                "name": example.name,
                "pred": pred,
                "name_hit": name_hit(pred, example.name),
            }
        )
    return predictions


def safe_bleu(predictions: list[str], references: list[str]) -> float:
    if not predictions:
        return 0.0
    try:
        return float(Bleu().compute(predictions=predictions, references=references)["bleu"])
    except Exception:
        return 0.0


def safe_rouge(predictions: list[str], references: list[str]) -> dict[str, float]:
    if not predictions:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "rougeLsum": 0.0}
    try:
        scores = Rouge().compute(predictions=predictions, references=references)
        return {key: float(scores[key]) for key in ("rouge1", "rouge2", "rougeL", "rougeLsum")}
    except Exception:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "rougeLsum": 0.0}


def summarize_predictions(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "overall": {"num_examples": len(predictions)},
        "by_eval_set": {},
        "by_sample_type": {},
    }
    groups: dict[str, list[dict[str, Any]]] = {}
    sample_type_groups: dict[str, list[dict[str, Any]]] = {}
    for pred in predictions:
        key = pred["eval_set"]
        groups.setdefault(key, []).append(pred)
        sample_type_key = f"{pred['eval_set']}::{pred['sample_type']}"
        sample_type_groups.setdefault(sample_type_key, []).append(pred)

    def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
        pred_texts = [row["pred"] for row in rows]
        refs = [row["gt"] for row in rows]
        rouge = safe_rouge(pred_texts, refs)
        bleu = safe_bleu(pred_texts, refs)
        name_hits = sum(1 for row in rows if row["name_hit"])
        return {
            "num_examples": len(rows),
            "name_hit_rate": name_hits / len(rows) if rows else 0.0,
            "bleu": bleu,
            **rouge,
        }

    for key, rows in sorted(groups.items()):
        summary["by_eval_set"][key] = summarize_rows(rows)
    for key, rows in sorted(sample_type_groups.items()):
        summary["by_sample_type"][key] = summarize_rows(rows)

    return summary


def load_model(model_path: str, device: str, peft_checkpoint: str | None = None):
    checkpoint = Path(peft_checkpoint) if peft_checkpoint is not None else None
    full_checkpoint = checkpoint is not None and (checkpoint / "config.json").exists()
    adapter_checkpoint = checkpoint is not None and (checkpoint / "adapter_config.json").exists()
    load_path = str(checkpoint) if full_checkpoint and not adapter_checkpoint else model_path
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        load_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    if adapter_checkpoint:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, peft_checkpoint, is_trainable=False)
        model = model.merge_and_unload()
    return model.to(device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--peft_checkpoint", type=str, default=None)
    parser.add_argument("--pair_jsonl", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--image_resize", type=int, default=224)
    parser.add_argument("--max_per_set", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=80)
    args = parser.parse_args()

    records = read_jsonl(args.pair_jsonl)
    examples = limit_examples(build_examples(records, args.split, args.image_resize), args.max_per_set)

    processor = AutoProcessor.from_pretrained(args.model_path, padding_side="left")
    model = load_model(args.model_path, args.device, args.peft_checkpoint)

    predictions = generate_predictions(
        model=model,
        processor=processor,
        examples=examples,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
    )
    summary = summarize_predictions(predictions)
    result = {
        "model_path": args.model_path,
        "peft_checkpoint": args.peft_checkpoint,
        "pair_jsonl": args.pair_jsonl,
        "split": args.split,
        "max_per_set": args.max_per_set,
        "max_new_tokens": args.max_new_tokens,
        "summary": summary,
        "predictions": predictions,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved Step8 evaluation to {output}")


if __name__ == "__main__":
    main()
