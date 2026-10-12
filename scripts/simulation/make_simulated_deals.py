import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_industry_trends import build as build_trends, trailing_features

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "crm"
SIM = ROOT / "data" / "crm_simulated"
COMPETITORS_PER_PRODUCT = 2
TARGET_WIN_RATE = 0.63
START, LAST_ENGAGE, SNAPSHOT = pd.Timestamp("2016-10-20"), pd.Timestamp("2017-12-27"), pd.Timestamp("2017-12-31")
PROSPECTING_SHARE, STALLED_SHARE, OPEN_MISSING_ACCOUNT = 0.057, 0.12, 0.68


def effects(rng, names, scale):
    return dict(zip(names, rng.normal(0, scale, len(names))))


def env_weight(d, k=1.0):
    return (1 + k * d["trend_5y_avg_growth"].fillna(0) / 10).clip(0.3 / k, 2.5 * k)


def win_logit(d, prod_fx, agent_fx, k=1.0):
    z = d["product"].map(prod_fx) + d["sales_agent"].map(agent_fx)
    z += 0.6 * d["sector_match"]
    z += 0.05 * d["trend_5y_avg_growth"].fillna(0)
    z += k * 0.5 * d["sector_match"] * (d["trend_5y_slope"].fillna(0) > 0)
    z -= k * 0.8 * ((np.log(d["sales_price"]) > 8) & (d["revenue"] < d["revenue"].median()))
    z += k * 0.7 * ((d["series"] == "GTS") & (d["regional_office"] == "West"))
    cmp_ = (0.02 * d["recycled_diff"] + 0.15 * d["longevity_diff"] + 0.4 * d["usa_diff"]
            - 0.5 * np.log(d["sales_price"] / d["competitor_price"]))
    return z + env_weight(d, k) * cmp_


SIGNALS = ["discount_pct", "quote_gap", "engagement_score", "days_since_contact"]


def deal_signals(n, seed, competitor_price, list_price):
    rng = np.random.default_rng(seed + 1000)
    discount = rng.uniform(0, 30, n)
    competitor_quote = competitor_price * rng.lognormal(0, 0.15, n)
    return pd.DataFrame({
        "discount_pct": discount.round(2),
        "quote_gap": np.log(list_price * (1 - discount / 100) / competitor_quote).round(4),
        "engagement_score": rng.normal(0, 1, n).round(4),
        "days_since_contact": rng.exponential(10, n).round(2),
    })


def smooth_logit(sig, s):
    z = lambda c: (sig[c] - sig[c].mean()) / sig[c].std()
    index = 0.9 * z("discount_pct") - 1.1 * z("quote_gap") + 0.8 * z("engagement_score") - 0.7 * z("days_since_contact")
    return s * (2.0 * np.tanh(0.8 * index) + 0.8 * z("discount_pct") * z("engagement_score"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-deals", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=582)
    ap.add_argument("--out-dir", type=Path, default=SIM)
    ap.add_argument("--interaction-strength", type=float, default=3.0,
                    help="multiplier on planted interactions (3 = default strong setting; 1 = weak)")
    ap.add_argument("--smooth-strength", type=float, default=0.0,
                    help="weight of the smooth continuous-signal effect (0 = signals are pure noise)")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    out = a.out_dir
    out.mkdir(parents=True, exist_ok=True)

    products = pd.read_csv(SIM / "products.csv")
    accounts = pd.read_csv(RAW / "accounts.csv")
    teams = pd.read_csv(RAW / "sales_teams.csv")
    real = pd.read_csv(RAW / "sales_pipeline.csv")
    real["product"] = real["product"].replace({"GTXPro": "GTX Pro"})
    trends = build_trends()

    n = a.n_deals
    prod_w = products["product"].map(real["product"].value_counts()).fillna(real["product"].value_counts().mean())
    prod_w = (prod_w / prod_w.sum()).to_numpy()
    acct_w = rng.dirichlet(np.full(len(accounts), 3.0))

    d = pd.DataFrame({
        "account": rng.choice(accounts["account"], n, p=acct_w),
        "sales_agent": rng.choice(teams["sales_agent"], n),
        "engage_date": START + pd.to_timedelta((rng.beta(1.4, 1.0, n) * (LAST_ENGAGE - START).days).astype(int), unit="D"),
    })
    d = d.merge(accounts[["account", "sector", "revenue"]], on="account", how="left")
    ind = products["industry"].to_numpy()
    w = prod_w[None, :] * np.where(d["sector"].to_numpy()[:, None] == ind[None, :], 2.5, 1.0)
    cum = (w / w.sum(1, keepdims=True)).cumsum(1)
    d["product"] = products["product"].to_numpy()[(rng.random(n)[:, None] > cum).sum(1).clip(max=len(products) - 1)]
    d = (d.merge(products[["product", "series", "sales_price", "industry"]], on="product", how="left")
          .merge(teams[["sales_agent", "regional_office"]], on="sales_agent", how="left"))
    comp = pd.read_csv(SIM / "competitor_products.csv")
    rival = comp.groupby("competes_with").sample(frac=1, random_state=a.seed).groupby("competes_with").head(
        COMPETITORS_PER_PRODUCT)
    rivals = {k: g.reset_index(drop=True) for k, g in rival.groupby("competes_with")}
    idx = rng.integers(0, COMPETITORS_PER_PRODUCT, n)
    cols = ["competitor_product", "sales_price", "recycled_pct", "longevity_years", "made_in_usa"]
    d[["competitor_product", "competitor_price", "competitor_recycled_pct",
       "competitor_longevity_years", "competitor_made_in_usa"]] = [
        rivals[p].loc[i, cols].to_numpy() for p, i in zip(d["product"], idx)]
    own = products.set_index("product")[["recycled_pct", "longevity_years", "made_in_usa"]]
    d["recycled_diff"] = d["product"].map(own["recycled_pct"]) - d["competitor_recycled_pct"].astype(float)
    d["longevity_diff"] = d["product"].map(own["longevity_years"]) - d["competitor_longevity_years"].astype(float)
    d["usa_diff"] = d["product"].map(own["made_in_usa"]) - d["competitor_made_in_usa"].astype(float)
    d["competitor_price"] = d["competitor_price"].astype(float)
    d["sector_match"] = (d["sector"] == d["industry"]).astype(int)

    feats = pd.concat([trailing_features(trends, y).assign(engage_year=y) for y in (2016, 2017)])
    d["engage_year"] = d["engage_date"].dt.year
    d = d.merge(feats.rename(columns={"industry": "sector"}), on=["sector", "engage_year"], how="left")

    prod_fx = effects(rng, products["product"], 0.4)
    agent_fx = effects(rng, teams["sales_agent"], 0.4)
    sig = deal_signals(n, a.seed, d["competitor_price"].to_numpy(), d["sales_price"].to_numpy())
    z = win_logit(d, prod_fx, agent_fx, a.interaction_strength) + smooth_logit(sig, a.smooth_strength)
    lo, hi = -10.0, 10.0
    for _ in range(50):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if (1 / (1 + np.exp(-(z + mid)))).mean() < TARGET_WIN_RATE else (lo, mid)
    d["win_prob"] = 1 / (1 + np.exp(-(z + mid)))
    d["is_won_latent"] = (rng.random(n) < d["win_prob"]).astype(int)

    days = np.clip(rng.gamma(2.2, 21, n) + 1, 1, 138).astype(int) + 4 * d["is_won_latent"]
    d["close_date"] = d["engage_date"] + pd.to_timedelta(days, unit="D")
    stalled = rng.random(n) < STALLED_SHARE
    prospect = rng.random(n) < PROSPECTING_SHARE
    closed = (d["close_date"] <= SNAPSHOT) & ~stalled & ~prospect
    d["deal_stage"] = np.where(closed, np.where(d["is_won_latent"] == 1, "Won", "Lost"), "Engaging")
    d.loc[prospect, "deal_stage"] = "Prospecting"

    won = d["deal_stage"] == "Won"
    d["close_value"] = np.where(won, (d["sales_price"] * rng.uniform(0.85, 1.05, n)).round(), np.where(closed, 0, np.nan))
    d.loc[~closed, "close_date"] = pd.NaT
    d.loc[prospect, "engage_date"] = pd.NaT
    d["opportunity_id"] = [f"{x:08X}" for x in rng.choice(16**8, n, replace=False)]
    d.loc[~closed & (rng.random(n) < OPEN_MISSING_ACCOUNT), "account"] = np.nan

    fmt = lambda s: s.dt.strftime("%m/%d/%y").str.lstrip("0").str.replace("/0", "/", regex=False)
    pipe = d[["opportunity_id", "sales_agent", "product", "account", "deal_stage", "engage_date", "close_date", "close_value"]].copy()
    pipe["engage_date"], pipe["close_date"] = fmt(pipe["engage_date"]), fmt(pipe["close_date"])
    pipe.to_csv(out / "sales_pipeline.csv", index=False)
    for name in ("accounts", "sales_teams"):
        shutil.copy(RAW / f"{name}.csv", out / f"{name}.csv")
    if out != SIM:
        shutil.copy(SIM / "products.csv", out / "products.csv")
    d[["opportunity_id", "competitor_product", "competitor_price", "recycled_diff", "longevity_diff",
       "usa_diff"]].to_csv(out / "deal_comparison.csv", index=False)
    sig.assign(opportunity_id=d["opportunity_id"])[["opportunity_id", *SIGNALS]].to_csv(
        out / "deal_signals.csv", index=False)
    d[["opportunity_id", "win_prob", "is_won_latent", "sector_match", "trend_5y_avg_growth",
       "trend_5y_slope", "trend_last_growth", "industry", "sector"]].to_csv(out / "ground_truth.csv", index=False)

    vc = d["deal_stage"].value_counts()
    print(vc.to_string(), f"\nwin rate among closed: {d.loc[closed, 'is_won_latent'].mean():.3f}")


if __name__ == "__main__":
    main()
