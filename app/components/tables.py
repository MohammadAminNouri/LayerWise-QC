"""Table formatting helpers."""
from __future__ import annotations
import pandas as pd


def issues_to_table(issues) -> pd.DataFrame:
    return pd.DataFrame([getattr(i, "__dict__", dict(i)) for i in issues])
