from __future__ import annotations

import re
from typing import Any

import torch

from causal_mip.interventions.ablation import ablate_candidate_path
from causal_mip.interventions.activation_cache import (
    PreparedSampleBatch,
    cache_candidate_path_activations,
    resolve_candidate_path_targets,
)
from causal_mip.interventions.restoration import restore_path_activations
from causal_mip.path_localization.path_schema import CandidatePath


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip().lower()


def get_token_ids(processor_or_tokenizer, text: str) -> list[int]:
    if not text:
        return []
    tokenizer = getattr(processor_or_tokenizer, "tokenizer", processor_or_tokenizer)
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def find_subsequence_positions(sequence: list[int], subsequence: list[int]) -> list[int]:
    if not subsequence or len(subsequence) > len(sequence):
        return []
    limit = len(sequence) - len(subsequence) + 1
    for start in range(limit):
        if sequence[start : start + len(subsequence)] == subsequence:
            return list(range(start, start + len(subsequence)))
    return []


def find_name_token_positions(
    processor_or_tokenizer,
    input_ids: torch.Tensor,
    answer_positions: list[int],
    target_text: str | None,
    name_text: str | None,
) -> dict[str, Any]:
    name_text = name_text or ""
    name_token_ids = get_token_ids(processor_or_tokenizer, name_text)
    input_id_list = input_ids[0].detach().cpu().tolist()
    all_name_positions = find_subsequence_positions(input_id_list, name_token_ids)
    answer_position_set = set(answer_positions)
    scoped_name_positions = [
        position
        for position in all_name_positions
        if position in answer_position_set
    ]
    if (
        not scoped_name_positions
        and name_text
        and normalize_text(target_text or "").startswith(normalize_text(name_text))
        and name_token_ids
    ):
        scoped_name_positions = list(answer_positions[: len(name_token_ids)])

    return {
        "name_text": name_text,
        "name_token_ids": name_token_ids,
        "name_token_positions": scoped_name_positions,
        "name_token_count_expected": len(name_token_ids),
        "name_match_status": "matched" if scoped_name_positions else "name_not_in_target_text",
    }


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
        token_values.append(log_probs[0, score_position, token_id])
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


def score_name_tokens_from_logits(
    logits: torch.Tensor,
    prepared_batch: PreparedSampleBatch,
    processor_or_tokenizer,
    name_text: str | None,
) -> dict[str, Any]:
    location = find_name_token_positions(
        processor_or_tokenizer=processor_or_tokenizer,
        input_ids=prepared_batch.input_ids,
        answer_positions=list(prepared_batch.answer_token_positions),
        target_text=prepared_batch.target_answer_text,
        name_text=name_text,
    )
    stats = compute_logprob_stats(
        logits=logits,
        input_ids=prepared_batch.input_ids,
        positions=location["name_token_positions"],
    )
    return {
        "target_answer_text": prepared_batch.target_answer_text,
        "target_name": location["name_text"],
        "name_match_status": location["name_match_status"],
        "name_token_positions": location["name_token_positions"],
        "name_token_count_expected": location["name_token_count_expected"],
        "name_token_ids": location["name_token_ids"],
        **stats,
    }


def _mean_logprob(score: dict[str, Any] | None) -> float | None:
    if not score:
        return None
    value = score.get("mean_logprob")
    return None if value is None else float(value)


def _delta(left: dict[str, Any] | None, right: dict[str, Any] | None) -> float | None:
    left_value = _mean_logprob(left)
    right_value = _mean_logprob(right)
    if left_value is None or right_value is None:
        return None
    return left_value - right_value


def _positive(value: float | None) -> float:
    return max(0.0, float(value)) if value is not None else 0.0


def _score_model_outputs(model, prepared_batch: PreparedSampleBatch) -> torch.Tensor:
    with torch.no_grad():
        outputs = model(**prepared_batch.model_inputs)
    return outputs.logits.detach()


def compute_path_name_token_scores(
    model,
    processor_or_tokenizer,
    clean_batch: PreparedSampleBatch,
    corrupt_batch: PreparedSampleBatch,
    retain_batches: dict[str, PreparedSampleBatch],
    candidate_path: CandidatePath,
    strict: bool = False,
) -> dict[str, Any]:
    resolved_nodes = resolve_candidate_path_targets(candidate_path, clean_batch, strict=strict, model=model)
    target_name = clean_batch.sample.get("name", "")
    if not resolved_nodes:
        return {
            "name_score_status": "no_patchable_nodes",
            "target_name": target_name,
            "NameNec": None,
            "NameSuf": None,
            "NameRet": None,
            "name_forget_effect": None,
            "name_retain_impact": None,
            "name_match_status": "not_scored",
            "retain_name_details": {},
        }

    clean_logits = _score_model_outputs(model, clean_batch)
    clean_name_score = score_name_tokens_from_logits(
        clean_logits,
        prepared_batch=clean_batch,
        processor_or_tokenizer=processor_or_tokenizer,
        name_text=target_name,
    )

    ablated_outputs, _ = ablate_candidate_path(
        model=model,
        prepared_batch=clean_batch,
        candidate_path=candidate_path,
        strict=strict,
        no_grad=True,
    )
    ablated_name_score = score_name_tokens_from_logits(
        ablated_outputs.logits.detach(),
        prepared_batch=clean_batch,
        processor_or_tokenizer=processor_or_tokenizer,
        name_text=target_name,
    )

    cached_path = cache_candidate_path_activations(
        model=model,
        prepared_batch=clean_batch,
        candidate_path=candidate_path,
        strict=strict,
        no_grad=True,
    )
    corrupt_logits = _score_model_outputs(model, corrupt_batch)
    corrupt_name_score = score_name_tokens_from_logits(
        corrupt_logits,
        prepared_batch=corrupt_batch,
        processor_or_tokenizer=processor_or_tokenizer,
        name_text=target_name,
    )
    restored_outputs, _ = restore_path_activations(
        model=model,
        prepared_batch=corrupt_batch,
        cached_path=cached_path,
        no_grad=True,
    )
    restored_name_score = score_name_tokens_from_logits(
        restored_outputs.logits.detach(),
        prepared_batch=corrupt_batch,
        processor_or_tokenizer=processor_or_tokenizer,
        name_text=target_name,
    )

    name_nec = _delta(clean_name_score, ablated_name_score)
    name_suf = _delta(restored_name_score, corrupt_name_score)

    retain_details: dict[str, dict[str, Any]] = {}
    retain_impacts: list[float] = []
    for retain_name, retain_batch in retain_batches.items():
        retain_target_name = retain_batch.sample.get("name", "")
        baseline_logits = _score_model_outputs(model, retain_batch)
        baseline_name_score = score_name_tokens_from_logits(
            baseline_logits,
            prepared_batch=retain_batch,
            processor_or_tokenizer=processor_or_tokenizer,
            name_text=retain_target_name,
        )
        retain_ablated_outputs, _ = ablate_candidate_path(
            model=model,
            prepared_batch=retain_batch,
            candidate_path=candidate_path,
            strict=strict,
            no_grad=True,
        )
        retain_ablated_name_score = score_name_tokens_from_logits(
            retain_ablated_outputs.logits.detach(),
            prepared_batch=retain_batch,
            processor_or_tokenizer=processor_or_tokenizer,
            name_text=retain_target_name,
        )
        impact = _delta(baseline_name_score, retain_ablated_name_score)
        retain_details[retain_name] = {
            "target_name": retain_target_name,
            "baseline_name_score": baseline_name_score,
            "ablated_name_score": retain_ablated_name_score,
            "impact": impact,
        }
        if impact is not None:
            retain_impacts.append(float(impact))

    name_ret = sum(retain_impacts) / len(retain_impacts) if retain_impacts else None
    name_match_status = (
        "matched"
        if clean_name_score["name_match_status"] == "matched"
        and corrupt_name_score["name_match_status"] == "matched"
        else "name_not_in_target_text"
    )

    return {
        "name_score_status": "ok",
        "target_name": target_name,
        "name_match_status": name_match_status,
        "NameNec": name_nec,
        "NameSuf": name_suf,
        "NameRet": name_ret,
        "name_forget_effect": _positive(name_nec) + _positive(name_suf),
        "name_retain_impact": _positive(name_ret),
        "clean_target_name_score": clean_name_score,
        "ablated_target_name_score": ablated_name_score,
        "corrupt_target_name_score": corrupt_name_score,
        "restored_target_name_score": restored_name_score,
        "clean_name_token_positions": clean_name_score["name_token_positions"],
        "corrupt_name_token_positions": corrupt_name_score["name_token_positions"],
        "retain_name_details": retain_details,
        "num_retain_name_scores": len(retain_impacts),
    }
