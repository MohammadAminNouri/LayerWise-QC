"""Chart helpers kept outside the main Streamlit app."""
from __future__ import annotations
import pandas as pd


def safe_bar_data(values: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame({"label": list(values.keys()), "value": list(values.values())})
