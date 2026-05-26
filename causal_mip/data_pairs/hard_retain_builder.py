import random
from typing import Any

from .text_corruption import build_same_topic_answer, build_same_topic_question, normalize_question_template


def _choose_best_template_match(
    clean_sample: dict[str, Any],
    retain_pool: list[dict[str, Any]],
    rng: random.Random,
    exclude_ids: set[Any] | None = None,
) -> dict[str, Any] | None:
    exclude_ids = exclude_ids or set()
    clean_template = clean_sample.get("question_template") or normalize_question_template(
        clean_sample.get("question", ""),
        clean_sample.get("name"),
    )
    same_template = [
        sample for sample in retain_pool
        if sample.get("id") not in exclude_ids
        and sample.get("id") != clean_sample.get("id")
        and sample.get("question_template") == clean_template
    ]
    if same_template:
        return rng.choice(same_template)

    fallback = [
        sample for sample in retain_pool
        if sample.get("id") not in exclude_ids and sample.get("id") != clean_sample.get("id")
    ]
    if not fallback:
        return None
    return rng.choice(fallback)


def build_same_topic_retain(clean_sample: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "same_topic",
        "source_dataset": clean_sample.get("source_dataset"),
        "row_idx": clean_sample.get("row_idx"),
        "id": clean_sample.get("id"),
        "name": clean_sample.get("name"),
        "question": build_same_topic_question(clean_sample.get("question")),
        "answer": build_same_topic_answer(clean_sample.get("caption"), clean_sample.get("name")),
        "caption": build_same_topic_answer(clean_sample.get("caption"), clean_sample.get("name")),
        "image_ref": dict(clean_sample.get("image_ref", {})),
        "question_template": "same_topic_visual_description",
    }


def build_same_reasoning_retain(
    clean_sample: dict[str, Any],
    retain_pool: list[dict[str, Any]],
    rng: random.Random,
) -> dict[str, Any] | None:
    matched = _choose_best_template_match(clean_sample, retain_pool, rng)
    if matched is None:
        return None
    return {
        "type": "same_reasoning",
        "source_dataset": matched.get("source_dataset"),
        "row_idx": matched.get("row_idx"),
        "id": matched.get("id"),
        "name": matched.get("name"),
        "question": matched.get("question"),
        "answer": matched.get("answer"),
        "caption": matched.get("caption"),
        "image_ref": dict(matched.get("image_ref", {})),
        "question_template": matched.get("question_template"),
    }


def build_counterfactual_retain(
    clean_sample: dict[str, Any],
    retain_pool: list[dict[str, Any]],
    rng: random.Random,
    used_ids: set[Any] | None = None,
) -> dict[str, Any] | None:
    used_ids = used_ids or set()
    matched = _choose_best_template_match(clean_sample, retain_pool, rng, exclude_ids=used_ids)
    if matched is None:
        return None
    return {
        "type": "counterfactual_retain",
        "source_dataset": matched.get("source_dataset"),
        "row_idx": matched.get("row_idx"),
        "id": matched.get("id"),
        "name": matched.get("name"),
        "question": matched.get("question"),
        "answer": matched.get("answer"),
        "caption": matched.get("caption"),
        "image_ref": dict(matched.get("image_ref", {})),
        "question_template": matched.get("question_template"),
    }
