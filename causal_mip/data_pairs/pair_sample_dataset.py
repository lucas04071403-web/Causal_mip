from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from causal_mip.interventions.activation_cache import SampleReferenceResolver


class PairSampleDataset(Dataset):
    """Expose selected Step3 pair samples as CLEAR-style training rows."""

    def __init__(
        self,
        pair_jsonl: str,
        sample_key: str = "counterfactual_retain",
        max_examples: int | None = None,
        resolver: SampleReferenceResolver | None = None,
    ) -> None:
        self.pair_jsonl = pair_jsonl
        self.sample_key = sample_key
        self.resolver = resolver or SampleReferenceResolver()
        self.records = self._load_samples(pair_jsonl, sample_key)
        if max_examples is not None:
            self.records = self.records[:max_examples]

    def _load_samples(self, pair_jsonl: str, sample_key: str) -> list[dict[str, Any]]:
        samples: list[dict[str, Any]] = []
        with Path(pair_jsonl).open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                pair = json.loads(line)
                sample = pair.get(sample_key)
                if sample is None:
                    continue
                sample = copy.deepcopy(sample)
                sample["pair_id"] = pair.get("pair_id")
                sample.setdefault("type", sample_key)
                samples.append(sample)
        return samples

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = copy.deepcopy(self.records[index])
        image = self.resolver.resolve_image(sample)
        if image is not None:
            sample["image"] = image
        return sample
