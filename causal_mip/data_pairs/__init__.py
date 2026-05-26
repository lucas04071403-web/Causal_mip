from importlib import import_module

__all__ = [
    "build_causal_pairs_for_project",
    "build_pairs_from_samples",
    "build_same_topic_answer",
    "build_same_topic_question",
    "build_text_corrupt_sample",
    "normalize_question_template",
    "split_pairs_train_val",
    "write_pairs_jsonl",
]


def __getattr__(name: str):
    if name in {
        "build_causal_pairs_for_project",
        "build_pairs_from_samples",
        "split_pairs_train_val",
        "write_pairs_jsonl",
    }:
        module = import_module(f"{__name__}.build_pairs")
        value = getattr(module, name)
        globals()[name] = value
        return value

    if name in {
        "build_same_topic_answer",
        "build_same_topic_question",
        "build_text_corrupt_sample",
        "normalize_question_template",
    }:
        module = import_module(f"{__name__}.text_corruption")
        value = getattr(module, name)
        globals()[name] = value
        return value

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
