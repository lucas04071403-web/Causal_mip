import re
from typing import Any


_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str | None) -> str:
    if not text:
        return ""
    return _WHITESPACE_RE.sub(" ", text).strip()


def mask_named_entity(text: str | None, name: str | None, replacement: str = "[MASK_NAME]") -> str:
    normalized = normalize_text(text)
    if not normalized or not name:
        return normalized
    pattern = re.compile(re.escape(name), re.IGNORECASE)
    return pattern.sub(replacement, normalized)


def normalize_question_template(question: str | None, name: str | None = None) -> str:
    normalized = normalize_text(question).lower()
    if not normalized:
        return ""

    if name:
        normalized = re.sub(re.escape(name.lower()), "<name>", normalized)

    normalized = re.sub(r"\b(this|the)\s+person\b", "<person>", normalized)
    normalized = re.sub(r"\b[a-z]+(?:\s+[a-z]+){0,2}'s\b", "<entity>'s", normalized)
    normalized = re.sub(r"\d+", "<num>", normalized)
    normalized = re.sub(r"[^\w\s<>\[\]']+", " ", normalized)
    normalized = _WHITESPACE_RE.sub(" ", normalized).strip()
    return normalized


def build_same_topic_question(question: str | None = None) -> str:
    normalized = normalize_text(question).lower()
    if "caption" in normalized:
        return "Describe the visible scene in this image without identifying the person by name."
    return "Describe the visible scene and objects in this image without identifying the person."


def build_same_topic_answer(caption: str | None, name: str | None = None) -> str:
    normalized = normalize_text(caption)
    if not normalized:
        return "The image shows a person in a scene with visible objects and surroundings."

    redacted = mask_named_entity(normalized, name, "the person")
    redacted = re.sub(r"^the person,\s*[^,]+,\s*", "The person ", redacted, flags=re.IGNORECASE)
    redacted = re.sub(r"^the person\s+is\s+", "The person is ", redacted, flags=re.IGNORECASE)
    redacted = redacted.strip()

    if not redacted:
        return "The image shows a person in a scene with visible objects and surroundings."
    if redacted[0].islower():
        redacted = redacted[0].upper() + redacted[1:]
    return redacted


def build_text_corrupt_sample(sample: dict[str, Any]) -> dict[str, Any]:
    corrupted_question = mask_named_entity(sample.get("question", ""), sample.get("name"))
    if not corrupted_question or corrupted_question == normalize_text(sample.get("question", "")):
        template = normalize_question_template(sample.get("question", ""), sample.get("name"))
        if template:
            corrupted_question = template.replace("<name>", "[MASK_NAME]")
        else:
            corrupted_question = "What can be inferred about [MASK_NAME] from this image?"

    return {
        "type": "text_corrupt",
        "corruption_type": "symmetric_token_replacement",
        "source_dataset": sample.get("source_dataset"),
        "row_idx": sample.get("row_idx"),
        "id": sample.get("id"),
        "name": sample.get("name"),
        "question": corrupted_question,
        "answer": "unknown",
        "caption": build_same_topic_answer(sample.get("caption"), sample.get("name")),
        "image_ref": dict(sample.get("image_ref", {})),
        "question_template": normalize_question_template(corrupted_question, sample.get("name")),
    }
