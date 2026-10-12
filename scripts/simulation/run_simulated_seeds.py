import argparse
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "crm_simulated"
METRICS = ["roc_auc", "f1", "accuracy", "brier"]


def one_run(seed, k, n, tmp, smooth=0.0):
    d = Path(tmp) / f"k{k}_m{smooth}_s{seed}"
    subprocess.run([sys.executable, str(ROOT / "scripts/simulation/make_simulated_deals.py"), "--n-deals", str(n),
                    "--seed", str(seed), "--interaction-strength", str(k), "--smooth-strength", str(smooth),
                    "--out-dir", str(d)],
                   check=True, capture_output=True)
    subprocess.run([sys.executable, str(ROOT / "scripts/simulation/run_simulated_models.py"), "--data-dir", str(d)],
                   check=True, capture_output=True)
    res = pd.read_csv(d / "model_results.csv")
    print(f"done: strength={k} smooth={smooth} seed={seed}", flush=True)
    return res[res["split"] == "test"].assign(seed=seed, interaction_strength=k, smooth_strength=smooth)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--strengths", type=float, nargs="+", default=[1.0, 3.0])
    ap.add_argument("--smooth", type=float, default=0.0, help="--smooth-strength passed to the generator")
    ap.add_argument("--tag", default="", help="suffix for output file names, e.g. _smooth")
    ap.add_argument("--n-deals", type=int, default=100000)
    ap.add_argument("--workers", type=int, default=2, help="parallel runs (each model uses 2 threads)")
    a = ap.parse_args()
    with tempfile.TemporaryDirectory() as tmp, ThreadPoolExecutor(a.workers) as pool:
        jobs = [pool.submit(one_run, s, k, a.n_deals, tmp, a.smooth) for k in a.strengths for s in a.seeds]
        runs = pd.concat([j.result() for j in jobs], ignore_index=True)
    runs.to_csv(OUT / f"seed_results{a.tag}.csv", index=False)
    summary = (runs.groupby(["interaction_strength", "features", "model"])[METRICS]
               .agg(["mean", "std", "min", "max"]).round(4))
    summary.columns = ["_".join(c) for c in summary.columns]
    summary.to_csv(OUT / f"seed_summary{a.tag}.csv")
    best = (runs[runs["features"] != "oracle"].loc[lambda r: r.groupby(
        ["interaction_strength", "features", "seed"])["roc_auc"].idxmax()]
        .groupby(["interaction_strength", "features", "model"]).size().rename("times_best"))
    print(summary[["roc_auc_mean", "roc_auc_std", "f1_mean", "f1_std"]].to_string())
    print(best.to_string())


if __name__ == "__main__":
    main()
