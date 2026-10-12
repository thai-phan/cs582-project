from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PRODUCT_FIX = {"GTXPro": "GTX Pro"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, default=ROOT / "reports/crm/final")
    ap.add_argument("--sims", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    open_deals = pd.read_csv(args.run / "predictions" / "open_deal_predictions.csv")
    products = pd.read_csv(ROOT / "data/crm/products.csv")
    products["product"] = products["product"].replace(PRODUCT_FIX)

    deals = open_deals.merge(products[["product", "sales_price"]], on="product", how="left")
    if len(deals) != len(open_deals) or deals["sales_price"].isna().any():
        raise SystemExit("Product join failed: missing price or changed row count.")

    p = deals["win_probability"].to_numpy()
    price = deals["sales_price"].to_numpy(dtype=float)
    rng = np.random.default_rng(args.seed)
    wins = rng.random((args.sims, len(deals))) < p
    sims = wins @ price

    def summary(mask: np.ndarray) -> dict:
        return {
            "deals": int(mask.sum()),
            "pipeline_value": float(price[mask].sum()),
            "expected_revenue": float((p[mask] * price[mask]).sum()),
        }

    missing = deals["account_missing"].astype(bool).to_numpy()
    result = {
        "label": "ILLUSTRATIVE_NOT_A_VALIDATED_FORECAST",
        "source": str(args.run / "predictions" / "open_deal_predictions.csv"),
        "scoring_context": sorted(deals["scoring_context"].unique().tolist()),
        "all_open_deals": summary(np.ones(len(deals), dtype=bool)),
        "account_present_only": summary(~missing),
        "simulation": {
            "n": args.sims,
            "p05": float(np.percentile(sims, 5)),
            "p50": float(np.percentile(sims, 50)),
            "p95": float(np.percentile(sims, 95)),
        },
        "caveats": [
            "Model ROC-AUC ~0.5: probabilities barely rank deals.",
            "List sales_price, not close_value.",
            "Only current Engaging deals; no new deals or Prospecting rows.",
            "Simulation treats deals as independent and ignores model error.",
        ],
    }
    out = args.run / "predictions" / "expected_revenue_3m.json"
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
