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
