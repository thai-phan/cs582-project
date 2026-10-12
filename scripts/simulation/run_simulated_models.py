import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data.crm import build
from src.evaluation.metrics import choose_threshold, metrics
from src.models.zoo import MODEL_NAMES, fit_model
from src.data.split import asof_split

SIM = ROOT / "data" / "crm_simulated"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smoke test: few epochs/trees")
    ap.add_argument("--data-dir", type=Path, default=SIM, help="output dir of make_simulated_deals.py")
    a = ap.parse_args()
    quick, syn = a.quick, a.data_dir
    base = build(syn, simulated_features=False)
    split = asof_split(base)
    truth = pd.read_csv(syn / "ground_truth.csv").set_index("opportunity_id")["win_prob"]
    rows = []
    for label, ds in (("base", base), ("extended", build(syn))):
        X, y = ds.features(), ds.labels()
        xv, yv, xt, yt = X.loc[split.validation], y.loc[split.validation], X.loc[split.test], y.loc[split.test]
        for name in MODEL_NAMES:
            print(f"[{label}] training {name}...", flush=True)
            m = fit_model(name, ds, split.train, split.validation, quick=quick)
            pv, pt = m.predict_proba(xv)[:, 1], m.predict_proba(xt)[:, 1]
            t = choose_threshold(yv, pv)
            rows.append({"features": label, "model": name, "split": "validation", **metrics(yv, pv, t)})
            rows.append({"features": label, "model": name, "split": "test", **metrics(yt, pt, t),
                         "fit_seconds": round(m.fit_seconds, 1)})
    for part, idx in (("validation", split.validation), ("test", split.test)):
        ids = base.frame.loc[idx, "opportunity_id"]
        rows.append({"features": "oracle", "model": "true_win_prob", "split": part,
                     **metrics(base.labels().loc[idx], truth.loc[ids].to_numpy(), .5)})
    res = pd.DataFrame(rows)
    res.to_csv(syn / ("model_results_quick.csv" if quick else "model_results.csv"), index=False)
    print(f"train/val/test rows: {len(split.train)}/{len(split.validation)}/{len(split.test)}")
    cols = [c for c in ("accuracy", "precision", "recall", "f1", "roc_auc", "brier") if c in res]
    print(res[res["split"] == "test"][["features", "model", *cols]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
