from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data.dataset import Dataset


def make_preprocessor(dataset: Dataset, scale_numeric: bool) -> ColumnTransformer:
    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scale", StandardScaler()))
    return ColumnTransformer(
        transformers=[
            ("cat", Pipeline([('impute',SimpleImputer(strategy='constant',fill_value='Missing')),
                               ('encode',OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), dataset.categorical),
            ("num", Pipeline(numeric_steps), dataset.numeric),
        ],
        verbose_feature_names_out=False,
    )
