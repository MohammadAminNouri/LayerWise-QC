"""Small app-state dataclasses for future Streamlit refactor."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path


@dataclass
class AppMode:
    mode: str = "Demo / proxy mode"
    manifest_path: Path | None = None
    checkpoint_path: Path | None = None
