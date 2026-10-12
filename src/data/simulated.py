from __future__ import annotations

from pathlib import Path

import pandas as pd

GROUND_TRUTH = "ground_truth.csv"
COMPARISON = "deal_comparison.csv"
SIGNALS_FILE = "deal_signals.csv"

TREND_COLUMNS = ["sector_match", "trend_5y_avg_growth", "trend_5y_slope", "trend_last_growth"]
NUMERIC = TREND_COLUMNS + ["recycled_pct", "longevity_years", "made_in_usa", "competitor_price",
                           "recycled_diff", "longevity_diff", "usa_diff"]
SIGNALS = ["discount_pct", "quote_gap", "engagement_score", "days_since_contact"]


def available(data_dir: Path) -> bool:
    return (Path(data_dir) / GROUND_TRUTH).exists() and (Path(data_dir) / COMPARISON).exists()


def extra_numeric(data_dir: Path) -> list[str]:
    return NUMERIC + (SIGNALS if (Path(data_dir) / SIGNALS_FILE).exists() else [])


def side_table(data_dir: Path) -> pd.DataFrame:
    data_dir = Path(data_dir)
    extra = (pd.read_csv(data_dir / GROUND_TRUTH, usecols=["opportunity_id", *TREND_COLUMNS])
             .merge(pd.read_csv(data_dir / COMPARISON).drop(columns="competitor_product"),
                    on="opportunity_id", validate="one_to_one"))
    if (data_dir / SIGNALS_FILE).exists():
        extra = extra.merge(pd.read_csv(data_dir / SIGNALS_FILE), on="opportunity_id", validate="one_to_one")
    return extra


def add_features(frame: pd.DataFrame, extra: pd.DataFrame) -> pd.DataFrame:
    out = frame.merge(extra, on="opportunity_id", how="left", validate="many_to_one")
    if len(out) != len(frame):
        raise AssertionError("simulated feature join changed the row count")
    missing = out.loc[out[TREND_COLUMNS[0]].isna(), "opportunity_id"]
    if len(missing):
        raise ValueError(f"no simulated features for {len(missing)} deals, e.g. {list(missing[:3])}")
    out.index = frame.index
    return out
