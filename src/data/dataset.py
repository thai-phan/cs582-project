from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


@dataclass
class Dataset:

    name: str
    frame: pd.DataFrame
    target: str
    categorical: list[str]
    numeric: list[str]
    report: dict[str, object] = field(default_factory=dict)
    extra: dict[str, object] = field(default_factory=dict)

    @property
    def feature_columns(self) -> list[str]:
        return self.categorical + self.numeric

    def features(self) -> pd.DataFrame:
        return self.frame[self.feature_columns]

    def labels(self) -> pd.Series:
        return self.frame[self.target].astype(int)

    def __post_init__(self) -> None:
        missing = [c for c in self.feature_columns + [self.target] if c not in self.frame]
        if missing:
            raise ValueError(f"{self.name}: columns missing from frame: {missing}")
        overlap = set(self.categorical) & set(self.numeric)
        if overlap:
            raise ValueError(f"{self.name}: columns listed as both categorical and numeric: {overlap}")
        bad = set(self.frame[self.target].dropna().unique()) - {0, 1}
        if bad:
            raise ValueError(f"{self.name}: target must be 0/1, found {bad}")


def is_text(series: pd.Series) -> bool:
    return series.dtype == object or isinstance(series.dtype, pd.StringDtype)


def strip_strings(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in out.columns:
        if is_text(out[column]):
            out[column] = out[column].map(lambda v: v.strip() if isinstance(v, str) else v)
    return out


def blank_to_na(frame: pd.DataFrame, placeholders: tuple[str, ...] = ()) -> pd.DataFrame:
    out = frame.replace(r"^\s*$", np.nan, regex=True)
    if placeholders:
        out = out.replace(list(placeholders), np.nan)
    return out


def collapse_rare(series: pd.Series, top_k: int, other: str = "Other") -> pd.Series:
    keep = series.value_counts().head(top_k).index
    return series.where(series.isin(keep) | series.isna(), other)


def fill_missing_category(frame: pd.DataFrame, columns: list[str], token: str = "Missing") -> pd.DataFrame:
    out = frame.copy()
    for column in columns:
        out[column] = out[column].astype("object").where(out[column].notna(), token).astype(str)
    return out


def yes_no_to_int(series: pd.Series) -> pd.Series:
    return series.map({"Yes": 1, "No": 0, "yes": 1, "no": 0}).astype("Int64")


def basic_report(frame: pd.DataFrame, target: str) -> dict[str, object]:
    positives = int(frame[target].sum())
    return {
        "rows": len(frame),
        "positives": positives,
        "negatives": len(frame) - positives,
        "positive_rate": round(positives / len(frame), 4) if len(frame) else float("nan"),
    }
