from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv


def load_env() -> None:
    # Loads .env if present; no-op otherwise
    load_dotenv(override=False)


def get_env(key: str, default: str | None = None) -> str:
    v = os.getenv(key)
    if v is None or v == "":
        return default if default is not None else ""
    return v


def ensure_dirs(paths: Iterable[Path]) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)
