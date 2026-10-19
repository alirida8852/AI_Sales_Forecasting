"""Small, deterministic normalizations for CRM dimension keys."""

from __future__ import annotations

import pandas as pd


PRODUCT_ALIASES = {
    "GTXPro": "GTX Pro",
}


def normalize_product_names(data: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with known pipeline product aliases standardized."""
    normalized = data.copy()
    normalized["product"] = normalized["product"].replace(PRODUCT_ALIASES)
    return normalized