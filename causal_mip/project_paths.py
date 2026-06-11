from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REPO_ROOT.parent
WORKSPACE_ROOT = Path(os.environ.get("MIP_WORKSPACE_ROOT", PROJECT_ROOT / "mip_workspace"))


def workspace_path(*parts: str) -> str:
    return str(WORKSPACE_ROOT.joinpath(*parts))


def workspace_path_with_sep(*parts: str) -> str:
    return str(WORKSPACE_ROOT.joinpath(*parts)) + "/"


def resolve_workspace_dataset_path(dataset_path: str) -> str:
    """Map stale absolute mip_workspace/datasets paths to this workspace."""
    path = Path(dataset_path)
    if path.exists():
        return str(path)

    parts = path.parts
    if "datasets" not in parts:
        return str(path)
    datasets_index = parts.index("datasets")
    suffix = parts[datasets_index + 1 :]
    if not suffix:
        return str(path)
    relocated = WORKSPACE_ROOT.joinpath("datasets", *suffix)
    if relocated.exists():
        return str(relocated)
    return str(path)
