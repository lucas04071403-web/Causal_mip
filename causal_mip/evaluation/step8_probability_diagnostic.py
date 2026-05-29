from __future__ import annotations

import argparse
import gc
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


IMAGE_CAPTION_QUESTION = "Describe the visible scene and objects in this image without identifying the person."
MODEL_LABEL_BASELINE = "baseline"
MODEL_LABEL_CANDIDATE = "candidate"


@dataclass
class DiagnosticExample:
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


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def load_dataset_item(image_ref: dict[str, Any] | None) -> dict[str, Any] | None:
    if not image_ref:
        return None
    dataset_path = image_ref["dataset_path"]
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


def build_conversation(question: str, has_image: bool, answer_text: str | None = None) -> list[dict[str, Any]]:
    content = []
    if has_image:
        content.append({"type": "image"})
    content.append({"type": "text", "text": question})
    conversation = [{"role": "user", "content": content}]
    if answer_text is not None:
        conversation.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": answer_text}],
            }
        )
    return conversation


def extract_assistant_text(decoded: str) -> str:
    marker = "assistant\n"
    idx = decoded.lower().find(marker)
    if idx >= 0:
        return decoded[idx + len(marker) :].strip()
    return decoded.strip()


def extract_leading_name(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text or "").strip()
    if not normalized:
        return ""
    prefix = re.split(r"[,.\n]", normalized, maxsplit=1)[0].strip()
    words = prefix.split()
    if len(words) > 4:
        return ""
    if not words:
        return ""
    capitalized = [word for word in words if re.match(r"^[A-Z][A-Za-z'.-]*$", word)]
    if len(capitalized) < 2:
        return ""
    return " ".join(words)


def _sample_to_example(
    pair_id: str,
    split: str,
    eval_set: str,
    sample: dict[str, Any],
    image_resize: int,
) -> DiagnosticExample:
    return DiagnosticExample(
        pair_id=pair_id,
        split=split,
        eval_set=eval_set,
        sample_type=sample.get("type", eval_set),
        question=sample.get("question") or IMAGE_CAPTION_QUESTION,
        target=sample.get("answer") or sample.get("caption") or "",
        name=sample.get("name", ""),
        image=load_image(sample.get("image_ref"), image_resize),
    )


def build_examples(pair_records: list[dict[str, Any]], split: str, image_resize: int) -> list[DiagnosticExample]:
    examples: list[DiagnosticExample] = []
    for record in pair_records:
        pair_id = record["pair_id"]
        examples.append(
            _sample_to_example(
                pair_id=pair_id,
                split=split,
                eval_set="forget_clean",
                sample=record["forget_clean"],
                image_resize=image_resize,
            )
        )
        for hard in record.get("hard_retain", []):
            examples.append(
                _sample_to_example(
                    pair_id=pair_id,
                    split=split,
                    eval_set="hard_retain",
                    sample=hard,
                    image_resize=image_resize,
                )
            )
        counterfactual = record.get("counterfactual_retain")
        if counterfactual is not None:
            examples.append(
                _sample_to_example(
                    pair_id=pair_id,
                    split=split,
                    eval_set="counterfactual_retain",
                    sample=counterfactual,
                    image_resize=image_resize,
                )
            )
    return examples


def limit_examples(examples: list[DiagnosticExample], max_per_set: int | None) -> list[DiagnosticExample]:
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


def build_prompt_text(processor, example: DiagnosticExample) -> str:
    return processor.apply_chat_template(
        build_conversation(example.question, has_image=example.image is not None),
        tokenize=False,
        add_generation_prompt=True,
    )


def build_answer_text(processor, example: DiagnosticExample, answer_text: str) -> str:
    return processor.apply_chat_template(
        build_conversation(example.question, has_image=example.image is not None, answer_text=answer_text),
        tokenize=False,
        add_generation_prompt=False,
    )


def build_model_inputs(processor, text: str, image: Image.Image | None, device: str) -> dict[str, torch.Tensor]:
    if image is not None:
        inputs = processor(text=[text], images=image, padding=True, return_tensors="pt")
    else:
        inputs = processor(text=[text], padding=True, return_tensors="pt")
    return {key: value.to(device) for key, value in inputs.items() if torch.is_tensor(value)}


def _token_ids(processor, text: str) -> list[int]:
    if not text:
        return []
    return processor.tokenizer(text, add_special_tokens=False)["input_ids"]


def find_subsequence_positions(sequence: list[int], subsequence: list[int]) -> list[int]:
    if not subsequence or len(subsequence) > len(sequence):
        return []
    limit = len(sequence) - len(subsequence) + 1
    for start in range(limit):
        if sequence[start : start + len(subsequence)] == subsequence:
            return list(range(start, start + len(subsequence)))
    return []


def compute_logprob_stats(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    positions: list[int],
) -> dict[str, Any]:
    positions = [position for position in positions if position > 0 and position < input_ids.shape[1]]
    if not positions:
        return {
            "token_count": 0,
            "mean_logprob": None,
            "sum_logprob": None,
            "mean_prob": None,
            "ce": None,
            "token_logprobs": [],
        }

    log_probs = torch.log_softmax(logits[:, :-1, :].float(), dim=-1)
    token_values = []
    for position in positions:
        token_id = input_ids[0, position]
        score_position = position - 1
        value = log_probs[0, score_position, token_id]
        token_values.append(value)
    stacked = torch.stack(token_values)
    token_logprobs = [float(value.detach().cpu().item()) for value in stacked]
    mean_logprob = float(stacked.mean().detach().cpu().item())
    sum_logprob = float(stacked.sum().detach().cpu().item())
    return {
        "token_count": len(token_logprobs),
        "mean_logprob": mean_logprob,
        "sum_logprob": sum_logprob,
        "mean_prob": float(torch.exp(stacked).mean().detach().cpu().item()),
        "ce": -mean_logprob,
        "token_logprobs": token_logprobs,
    }


def score_target_text(
    model,
    processor,
    example: DiagnosticExample,
    target_text: str,
    device: str,
    name_text: str | None = None,
) -> dict[str, Any]:
    if not target_text:
        return {
            "text": target_text,
            "status": "empty_target_text",
            "answer": _empty_scope_score("answer"),
            "name": _empty_scope_score("name"),
        }

    prompt_text = build_prompt_text(processor, example)
    full_text = build_answer_text(processor, example, target_text)
    prompt_inputs = build_model_inputs(processor, prompt_text, example.image, device)
    full_inputs = build_model_inputs(processor, full_text, example.image, device)
    prompt_length = int(prompt_inputs["input_ids"].shape[1])
    input_ids = full_inputs["input_ids"]
    answer_positions = list(range(prompt_length, int(full_inputs["attention_mask"][0].sum().item())))

    scored_name_text = (name_text if name_text is not None else example.name) or ""
    name_token_ids = _token_ids(processor, scored_name_text)
    full_token_ids = input_ids[0].detach().cpu().tolist()
    name_positions = find_subsequence_positions(full_token_ids, name_token_ids)
    scoped_name_positions = [position for position in name_positions if position in set(answer_positions)]
    if (
        not scoped_name_positions
        and scored_name_text
        and normalize_text(target_text).startswith(normalize_text(scored_name_text))
    ):
        scoped_name_positions = answer_positions[: len(name_token_ids)] if name_token_ids else []
    name_match_status = "matched" if scoped_name_positions else "name_not_in_target_text"

    with torch.no_grad():
        outputs = model(**full_inputs)

    return {
        "text": target_text,
        "scored_name_text": scored_name_text,
        "name_match_status": name_match_status,
        "prompt_length": prompt_length,
        "num_input_tokens": int(input_ids.shape[1]),
        "answer": compute_logprob_stats(outputs.logits, input_ids, answer_positions),
        "name": compute_logprob_stats(outputs.logits, input_ids, scoped_name_positions),
        "answer_token_positions": answer_positions,
        "name_token_positions": scoped_name_positions,
        "name_token_count_expected": len(name_token_ids),
    }


def _empty_scope_score(scope: str) -> dict[str, Any]:
    return {
        "scope": scope,
        "token_count": 0,
        "mean_logprob": None,
        "sum_logprob": None,
        "mean_prob": None,
        "ce": None,
        "token_logprobs": [],
    }


def generate_prediction(
    model,
    processor,
    example: DiagnosticExample,
    device: str,
    max_new_tokens: int,
) -> str:
    prompt_text = build_prompt_text(processor, example)
    inputs = build_model_inputs(processor, prompt_text, example.image, device)
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, min_new_tokens=1)
    decoded = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return extract_assistant_text(decoded)


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


def score_examples_for_model(
    label: str,
    model_path: str,
    peft_checkpoint: str | None,
    processor,
    examples: list[DiagnosticExample],
    device: str,
    max_new_tokens: int,
    generated_name_source: str,
) -> dict[str, Any]:
    model = load_model(model_path, device, peft_checkpoint)
    model.eval()
    records = []
    for example in tqdm(examples, desc=f"Step8 probability diagnostic: {label}"):
        prediction = ""
        generated_name = ""
        generated_name_score = None
        if generated_name_source == "self":
            prediction = generate_prediction(model, processor, example, device, max_new_tokens)
            generated_name = extract_leading_name(prediction)
            if generated_name:
                generated_name_score = score_target_text(
                    model,
                    processor,
                    example,
                    generated_name,
                    device,
                    name_text=generated_name,
                )
        target_score = score_target_text(model, processor, example, example.target, device)
        records.append(
            {
                "model_label": label,
                "pair_id": example.pair_id,
                "split": example.split,
                "eval_set": example.eval_set,
                "sample_type": example.sample_type,
                "question": example.question,
                "target": example.target,
                "target_name": example.name,
                "prediction": prediction,
                "generated_name": generated_name,
                "target_score": target_score,
                "generated_name_score": generated_name_score,
            }
        )
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "label": label,
        "model_path": model_path,
        "peft_checkpoint": peft_checkpoint,
        "records": records,
        "summary": summarize_records(records),
    }


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def summarize_record_group(records: list[dict[str, Any]]) -> dict[str, Any]:
    target_name_ce = [
        record["target_score"]["name"]["ce"]
        for record in records
        if record["target_score"]["name"]["ce"] is not None
    ]
    target_name_logprob = [
        record["target_score"]["name"]["mean_logprob"]
        for record in records
        if record["target_score"]["name"]["mean_logprob"] is not None
    ]
    target_answer_ce = [
        record["target_score"]["answer"]["ce"]
        for record in records
        if record["target_score"]["answer"]["ce"] is not None
    ]
    margins = []
    for record in records:
        generated_score = record.get("generated_name_score")
        target_logprob = record["target_score"]["name"]["mean_logprob"]
        generated_logprob = (
            generated_score.get("name", {}).get("mean_logprob")
            if isinstance(generated_score, dict)
            else None
        )
        if target_logprob is not None and generated_logprob is not None:
            margins.append(target_logprob - generated_logprob)
    return {
        "num_examples": len(records),
        "target_name_ce": _mean(target_name_ce),
        "target_name_mean_logprob": _mean(target_name_logprob),
        "target_answer_ce": _mean(target_answer_ce),
        "target_vs_generated_name_logprob_margin": _mean(margins),
    }


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_eval_set: dict[str, list[dict[str, Any]]] = {}
    by_sample_type: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_eval_set.setdefault(record["eval_set"], []).append(record)
        by_sample_type.setdefault(f"{record['eval_set']}::{record['sample_type']}", []).append(record)

    return {
        "overall": summarize_record_group(records),
        "by_eval_set": {
            key: summarize_record_group(rows)
            for key, rows in sorted(by_eval_set.items())
        },
        "by_sample_type": {
            key: summarize_record_group(rows)
            for key, rows in sorted(by_sample_type.items())
        },
    }


def _metric_at(summary: dict[str, Any], section: str, group: str, metric: str) -> float | None:
    if section == "overall":
        value = summary.get("overall", {}).get(metric)
    else:
        value = summary.get(section, {}).get(group, {}).get(metric)
    return None if value is None else float(value)


def compare_model_summaries(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    baseline_summary = baseline["summary"]
    candidate_summary = candidate["summary"]
    comparisons = []
    group_specs: list[tuple[str, str]] = [("overall", "overall")]
    for section in ("by_eval_set", "by_sample_type"):
        groups = set(baseline_summary.get(section, {})) | set(candidate_summary.get(section, {}))
        group_specs.extend((section, group) for group in sorted(groups))

    metrics = (
        "target_name_ce",
        "target_name_mean_logprob",
        "target_answer_ce",
        "target_vs_generated_name_logprob_margin",
    )
    for section, group in group_specs:
        row: dict[str, Any] = {"section": section, "group": group}
        for metric in metrics:
            base_value = _metric_at(baseline_summary, section, group, metric)
            cand_value = _metric_at(candidate_summary, section, group, metric)
            row[metric] = {
                "baseline": base_value,
                "candidate": cand_value,
                "delta": None if base_value is None or cand_value is None else cand_value - base_value,
            }
        comparisons.append(row)

    return {
        "interpretation": {
            "forget_signal": "For forget_clean, target_name_ce delta > 0 or target_name_mean_logprob delta < 0 indicates target-name suppression.",
            "retain_signal": "For hard_retain/counterfactual_retain, target_name_ce delta <= 0 or target_name_mean_logprob delta >= 0 indicates no target-name degradation.",
        },
        "groups": comparisons,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute Step8 target-name probability diagnostics for baseline and candidate checkpoints."
    )
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--pair_jsonl", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--baseline_peft_checkpoint", default=None)
    parser.add_argument("--candidate_peft_checkpoint", required=True)
    parser.add_argument("--split", default="diagnostic")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image_resize", type=int, default=224)
    parser.add_argument("--max_per_set", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=80)
    parser.add_argument(
        "--generated_name_source",
        choices=["none", "self"],
        default="self",
        help="Score the model's generated leading name as a target-vs-generated margin reference.",
    )
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    pair_records = read_jsonl(args.pair_jsonl)
    examples = limit_examples(build_examples(pair_records, args.split, args.image_resize), args.max_per_set)
    processor = AutoProcessor.from_pretrained(args.model_path, padding_side="left")

    baseline = score_examples_for_model(
        label=MODEL_LABEL_BASELINE,
        model_path=args.model_path,
        peft_checkpoint=args.baseline_peft_checkpoint,
        processor=processor,
        examples=examples,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        generated_name_source=args.generated_name_source,
    )
    candidate = score_examples_for_model(
        label=MODEL_LABEL_CANDIDATE,
        model_path=args.model_path,
        peft_checkpoint=args.candidate_peft_checkpoint,
        processor=processor,
        examples=examples,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        generated_name_source=args.generated_name_source,
    )
    return {
        "protocol": "step8_probability_diagnostic_v1",
        "model_path": args.model_path,
        "pair_jsonl": args.pair_jsonl,
        "split": args.split,
        "max_per_set": args.max_per_set,
        "max_new_tokens": args.max_new_tokens,
        "image_resize": args.image_resize,
        "baseline": baseline,
        "candidate": candidate,
        "comparison": compare_model_summaries(baseline, candidate),
    }


def main() -> None:
    args = build_arg_parser().parse_args()
    result = run(args)
    write_json(args.output, result)
    print(json.dumps(result["comparison"]["groups"], ensure_ascii=False, indent=2))
    print(f"Saved Step8 probability diagnostic to {args.output}")


if __name__ == "__main__":
    main()
