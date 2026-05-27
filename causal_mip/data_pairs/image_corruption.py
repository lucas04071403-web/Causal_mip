from typing import Any

from .text_corruption import normalize_question_template


def build_corrupt_index(samples: list[dict[str, Any]]) -> dict[Any, dict[str, Any]]:
    return {sample.get("id"): sample for sample in samples}


def _looks_like_same_prompt_image(clean_sample: dict[str, Any], corrupt_sample: dict[str, Any]) -> bool:
    """Detect perturbation rows that change only labels/captions, not model inputs."""
    clean_ref = clean_sample.get("image_ref") or {}
    corrupt_ref = corrupt_sample.get("image_ref") or {}
    same_question = (clean_sample.get("question") or "") == (corrupt_sample.get("question") or clean_sample.get("question") or "")
    same_dataset = clean_ref.get("dataset_path") == corrupt_ref.get("dataset_path")
    same_item = clean_sample.get("id") == corrupt_sample.get("id")
    same_row = clean_sample.get("row_idx") == corrupt_sample.get("row_idx")
    same_ref_item = clean_ref.get("item_id") == corrupt_ref.get("item_id")
    same_ref_row = clean_ref.get("row_idx") == corrupt_ref.get("row_idx")
    same_path = clean_ref.get("image_path") is not None and clean_ref.get("image_path") == corrupt_ref.get("image_path")
    return same_question and (same_path or (same_dataset and ((same_item and same_row) or (same_ref_item and same_ref_row))))


def select_explicit_corruption(clean_sample: dict[str, Any], corrupt_index: dict[Any, dict[str, Any]]) -> dict[str, Any] | None:
    sample_id = clean_sample.get("id")
    if sample_id not in corrupt_index:
        return None

    corrupt_sample = corrupt_index[sample_id]
    if _looks_like_same_prompt_image(clean_sample, corrupt_sample):
        return None

    corrupt_answer = corrupt_sample.get("caption") or corrupt_sample.get("answer") or "unknown"
    return {
        "type": "image_corrupt",
        "corruption_type": "explicit_perturbed_pair",
        "source_dataset": corrupt_sample.get("source_dataset"),
        "row_idx": corrupt_sample.get("row_idx"),
        "id": corrupt_sample.get("id"),
        "name": corrupt_sample.get("name"),
        "question": clean_sample.get("question"),
        "answer": corrupt_answer,
        "caption": corrupt_sample.get("caption"),
        "image_ref": dict(corrupt_sample.get("image_ref", {})),
        "question_template": clean_sample.get("question_template"),
    }


def select_semantic_minimal_image_corruption(
    clean_sample: dict[str, Any],
    candidate_pool: list[dict[str, Any]],
) -> dict[str, Any] | None:
    clean_template = clean_sample.get("question_template") or normalize_question_template(
        clean_sample.get("question", ""),
        clean_sample.get("name"),
    )
    clean_answer_len = len((clean_sample.get("answer") or "").split())
    clean_caption_len = len((clean_sample.get("caption") or "").split())

    scored_candidates = []
    for candidate in candidate_pool:
        if candidate.get("id") == clean_sample.get("id"):
            continue
        candidate_template = candidate.get("question_template") or normalize_question_template(
            candidate.get("question", ""),
            candidate.get("name"),
        )
        template_penalty = 0 if candidate_template == clean_template else 1
        answer_penalty = abs(len((candidate.get("answer") or "").split()) - clean_answer_len)
        caption_penalty = abs(len((candidate.get("caption") or "").split()) - clean_caption_len)
        score = (template_penalty, answer_penalty, caption_penalty, str(candidate.get("id")))
        scored_candidates.append((score, candidate))

    if not scored_candidates:
        return None

    scored_candidates.sort(key=lambda item: item[0])
    selected = scored_candidates[0][1]
    return {
        "type": "image_corrupt",
        "corruption_type": "semantic_minimal_pair",
        "source_dataset": selected.get("source_dataset"),
        "row_idx": selected.get("row_idx"),
        "id": selected.get("id"),
        "name": selected.get("name"),
        "question": clean_sample.get("question"),
        "answer": selected.get("caption") or selected.get("answer") or "unknown",
        "caption": selected.get("caption"),
        "image_ref": dict(selected.get("image_ref", {})),
        "question_template": clean_template,
    }
