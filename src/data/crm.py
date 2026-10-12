from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data import simulated
from src.data.dataset import DATA_DIR, Dataset, basic_report, blank_to_na, strip_strings

CRM_DIR = DATA_DIR / "crm"
DATE_FORMAT = "%m/%d/%y"
TARGET = "is_won"
CLOSED_STAGES = ("Won", "Lost")

LEAKAGE_COLUMNS = ("close_date", "close_value", "deal_stage", "opportunity_id")

PRODUCT_NAME_FIXES = {"GTXPro": "GTX Pro"}
SECTOR_FIXES = {"technolgy": "technology"}

CATEGORICAL = ["product", "series", "sector", "office_location", "regional_office", "manager", "sales_agent"]
NUMERIC = [
    "sales_price", "revenue", "employees", "revenue_per_employee", "year_established",
    "account_age_at_engage", "engage_year", "engage_month", "engage_quarter", "engage_dayofweek",
    "is_subsidiary",
]

REQUIRED = {
    'sales_pipeline': ['opportunity_id','sales_agent','product','account','deal_stage','engage_date','close_date','close_value'],
    'accounts': ['account','sector','year_established','revenue','employees','office_location','subsidiary_of'],
    'products': ['product','series','sales_price'],
    'sales_teams': ['sales_agent','manager','regional_office'],
}


def validate_raw(raw: dict[str, pd.DataFrame]) -> None:
    for name, required in REQUIRED.items():
        missing = set(required) - set(raw[name].columns)
        if missing:
            raise ValueError(f'{name}: missing required columns {sorted(missing)}')
        key = required[0]
        if raw[name][key].isna().any() or raw[name][key].duplicated().any():
            raise ValueError(f'{name}: {key} must be nonempty and unique')
    pipeline = raw['sales_pipeline']
    if not pipeline['deal_stage'].isin(['Won','Lost','Engaging','Prospecting']).all():
        raise ValueError('sales_pipeline: unknown or missing deal_stage')
    for name, columns in [('accounts',['year_established','revenue','employees']),('products',['sales_price']),('sales_pipeline',['close_value'])]:
        for column in columns:
            values = raw[name][column]
            numeric = pd.to_numeric(values, errors='coerce')
            if (values.notna() & numeric.isna()).any() or np.isinf(numeric).any() or (numeric.dropna() < 0).any():
                raise ValueError(f'{name}: invalid numeric {column}')
            raw[name][column] = numeric


def load_raw(data_dir: Path = CRM_DIR) -> dict[str, pd.DataFrame]:
    names = ["sales_pipeline", "accounts", "products", "sales_teams"]
    return {n: blank_to_na(strip_strings(pd.read_csv(data_dir / f"{n}.csv"))) for n in names}


def clean_pipeline(pipeline: pd.DataFrame) -> pd.DataFrame:
    out = pipeline.copy()
    out["product"] = out["product"].replace(PRODUCT_NAME_FIXES)
    for column in ("engage_date", "close_date"):
        parsed = pd.to_datetime(out[column], format=DATE_FORMAT, errors="coerce")
        if (out[column].notna() & parsed.isna()).any():
            raise ValueError(f'invalid {column} date')
        out[column] = parsed
    if 'deal_stage' in out:
        closed = out['deal_stage'].isin(CLOSED_STAGES)
        engaged = closed | out['deal_stage'].eq('Engaging')
        if out.loc[engaged,'engage_date'].isna().any() or out.loc[closed,'close_date'].isna().any():
            raise ValueError('engaged/closed deals require valid dates')
        if (out.loc[closed,'close_date'] < out.loc[closed,'engage_date']).any():
            raise ValueError('close_date cannot precede engage_date')
    return out


def clean_accounts(accounts: pd.DataFrame) -> pd.DataFrame:
    out = accounts.copy()
    out["sector"] = out["sector"].replace(SECTOR_FIXES)
    out["is_subsidiary"] = out["subsidiary_of"].notna().astype(int)
    out["revenue_per_employee"] = (out["revenue"] * 1_000_000 / out["employees"]).replace([np.inf, -np.inf], np.nan)
    return out


def join_tables(pipeline, accounts, products, sales_teams) -> pd.DataFrame:
    products = products.assign(product=products["product"].replace(PRODUCT_NAME_FIXES))
    joined = (
        pipeline.merge(products, on="product", how="left", validate="many_to_one")
        .merge(accounts, on="account", how="left", validate="many_to_one")
        .merge(sales_teams, on="sales_agent", how="left", validate="many_to_one")
    )
    if len(joined) != len(pipeline):
        raise AssertionError("join changed the row count; check for duplicate keys")
    if joined['series'].isna().any():
        raise ValueError('unmatched product after normalization')
    if joined['manager'].isna().any():
        raise ValueError('unmatched sales_agent')
    if (joined['account'].notna() & joined['sector'].isna()).any():
        raise ValueError('unmatched account')
    return joined


def add_derived_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    engage = out["engage_date"]
    out["engage_year"] = engage.dt.year
    out["engage_month"] = engage.dt.month
    out["engage_quarter"] = engage.dt.quarter
    out["engage_dayofweek"] = engage.dt.dayofweek
    out["account_age_at_engage"] = out["engage_year"] - out["year_established"]
    return out


def build(data_dir: Path = CRM_DIR, simulated_features: bool = True) -> Dataset:
    raw = load_raw(data_dir)
    validate_raw(raw)
    joined = join_tables(clean_pipeline(raw["sales_pipeline"]), clean_accounts(raw["accounts"]),
                         raw["products"], raw["sales_teams"])
    joined = add_derived_features(joined)
    numeric = list(NUMERIC)
    extended = simulated_features and simulated.available(data_dir)
    if extended:
        joined = simulated.add_features(joined, simulated.side_table(data_dir))
        numeric += simulated.extra_numeric(data_dir)
    closed = joined["deal_stage"].isin(CLOSED_STAGES)
    joined[TARGET] = np.where(closed, (joined["deal_stage"] == "Won").astype(int), np.nan)

    labeled = joined[closed].sort_values("engage_date", kind="stable").reset_index(drop=True)
    labeled[TARGET] = labeled[TARGET].astype(int)
    open_deals = joined[~closed].drop(columns=TARGET).reset_index(drop=True)
    scorable_open_deals = open_deals[
        open_deals["deal_stage"].eq("Engaging") & open_deals["engage_date"].notna()
    ].reset_index(drop=True)

    report = basic_report(labeled, TARGET) | {
        "raw_rows": {k: len(v) for k, v in raw.items()},
        "open_rows": len(open_deals),
        "scorable_open_rows": len(scorable_open_deals),
        "scorable_missing_account": int(scorable_open_deals['account'].isna().sum()),
        "product_name_fixes": int(raw['sales_pipeline']['product'].isin(PRODUCT_NAME_FIXES).sum()),
        "row_order_is_temporal": True,
        "temporal_column": "engage_date",
        "unmatched_product": int(joined["series"].isna().sum()),
        "unmatched_account_in_labeled": int(labeled["sector"].isna().sum()),
        "feature_set": "standard + simulated extended" if extended else "standard",
    }
    return Dataset(
        "crm",
        labeled,
        TARGET,
        CATEGORICAL,
        numeric,
        report,
        {"open_deals": open_deals, "scorable_open_deals": scorable_open_deals},
    )


def prepare_new_deals(pipeline: pd.DataFrame, data_dir: Path = CRM_DIR) -> pd.DataFrame:
    raw = load_raw(data_dir)
    rows = blank_to_na(strip_strings(pipeline))
    missing = [c for c in ("opportunity_id", "sales_agent", "product", "account", "engage_date") if c not in rows]
    if missing:
        raise ValueError(f"new deals lack columns: {missing}")
    if "close_date" not in rows:
        rows["close_date"] = np.nan
    rows = clean_pipeline(rows)
    if rows["engage_date"].isna().any():
        raise ValueError("every new deal needs an engage_date (Prospecting deals cannot be scored)")
    joined = add_derived_features(join_tables(rows, clean_accounts(raw["accounts"]), raw["products"], raw["sales_teams"]))
    if simulated.available(data_dir):
        joined = simulated.add_features(joined, simulated.side_table(data_dir))
    return joined.reset_index(drop=True)
