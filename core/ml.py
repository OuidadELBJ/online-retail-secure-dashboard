# core/ml.py
from __future__ import annotations

import os
import json
import numpy as np
import pandas as pd
from joblib import dump

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import HistGradientBoostingRegressor

ARTIFACT_DIR = "core"
MODEL_BUNDLE_PATH = os.path.join(ARTIFACT_DIR, "ml_bundle.pkl")
PRED_CACHE_PATH = os.path.join(ARTIFACT_DIR, "predictions_cache.json")

DATA_PATH = os.path.join("data", "online_retail_clean_data.csv")  # full clean

TOP_K_PRODUCTS = 500
TOP_K_COUNTRIES = 30


# =========================
# Load
# =========================
def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing dataset: {path}")

    df = pd.read_csv(path, low_memory=False)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df = df.dropna(subset=["InvoiceDate"])

    for c in ["net_amount", "net_qty", "return_amount", "return_qty"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    for c in ["Country", "StockCode", "Description"]:
        if c in df.columns:
            df[c] = df[c].astype(str)

    return df


# =========================
# Features
# =========================
def add_month_feats(df: pd.DataFrame, month_col: str) -> pd.DataFrame:
    df = df.copy()
    df["month"] = df[month_col].dt.month.astype(int)
    df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12.0)
    return df


def add_lags_roll3(g: pd.DataFrame, target: str) -> pd.DataFrame:
    g = g.copy()
    g[f"{target}_lag1"] = g[target].shift(1)
    g[f"{target}_lag2"] = g[target].shift(2)
    g[f"{target}_roll3"] = g[target].rolling(3).mean().shift(1)
    if len(g) >= 14:
        g[f"{target}_lag12"] = g[target].shift(12)
    else:
        g[f"{target}_lag12"] = np.nan
    return g


# =========================
# Monthly panels
# =========================
def monthly_global(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["month_start"] = d["InvoiceDate"].dt.to_period("M").apply(lambda r: r.start_time)
    g = d.groupby("month_start", as_index=False).agg(
        net_amount=("net_amount", "sum"),
        net_qty=("net_qty", "sum"),
        return_amount=("return_amount", "sum"),
        return_qty=("return_qty", "sum"),
    ).sort_values("month_start")

    g = add_month_feats(g, "month_start")
    g["t_idx"] = np.arange(len(g), dtype=int)

    for t in ["net_amount", "net_qty", "return_amount", "return_qty"]:
        g = add_lags_roll3(g, t)

    g = g.dropna(subset=["net_amount_lag1", "net_qty_lag1", "return_amount_lag1", "return_qty_lag1"]).reset_index(drop=True)
    return g


def reduce_countries(df: pd.DataFrame, top_k: int = TOP_K_COUNTRIES) -> pd.DataFrame:
    top = (
        df.groupby("Country")["net_amount"].sum()
        .sort_values(ascending=False)
        .head(top_k)
        .index.astype(str)
        .tolist()
    )
    d = df.copy()
    d["Country_grouped"] = np.where(d["Country"].isin(top), d["Country"], "Other")
    return d


def monthly_country(df: pd.DataFrame) -> pd.DataFrame:
    d = reduce_countries(df, TOP_K_COUNTRIES)
    d["month_start"] = d["InvoiceDate"].dt.to_period("M").apply(lambda r: r.start_time)
    c = d.groupby(["month_start", "Country_grouped"], as_index=False).agg(
        net_amount=("net_amount", "sum"),
    ).sort_values(["Country_grouped", "month_start"])

    c = add_month_feats(c, "month_start")
    c = add_lags_roll3(c, "net_amount")
    c = c.dropna(subset=["net_amount_lag1"]).reset_index(drop=True)
    return c


def monthly_product(df: pd.DataFrame, top_k: int = TOP_K_PRODUCTS) -> tuple[pd.DataFrame, dict]:
    top_products = (
        df.groupby("StockCode")["net_amount"].sum()
        .sort_values(ascending=False)
        .head(top_k)
        .index.astype(str)
        .tolist()
    )
    d = df[df["StockCode"].isin(top_products)].copy()

    # map StockCode -> most frequent Description
    desc_map = (
        d.groupby(["StockCode", "Description"]).size()
        .reset_index(name="n")
        .sort_values(["StockCode", "n"], ascending=[True, False])
        .drop_duplicates("StockCode")
        .set_index("StockCode")["Description"]
        .to_dict()
    )

    d["month_start"] = d["InvoiceDate"].dt.to_period("M").apply(lambda r: r.start_time)
    p = d.groupby(["month_start", "StockCode"], as_index=False).agg(
        net_amount=("net_amount", "sum"),
        net_qty=("net_qty", "sum"),
    ).sort_values(["StockCode", "month_start"])

    p = add_month_feats(p, "month_start")
    p = add_lags_roll3(p, "net_amount")

    p["net_qty_lag1"] = p.groupby("StockCode")["net_qty"].shift(1)
    p["net_qty_roll3"] = p.groupby("StockCode")["net_qty"].rolling(3).mean().shift(1).reset_index(level=0, drop=True)

    p = p.dropna(subset=["net_amount_lag1", "net_qty_lag1"]).reset_index(drop=True)
    return p, desc_map


# =========================
# Model builder (simple)
# =========================
def build_model(cat_cols, num_cols) -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
            ("num", "passthrough", num_cols),
        ],
        remainder="drop",
    )
    reg = HistGradientBoostingRegressor(
        max_depth=6,
        learning_rate=0.08,
        max_iter=500,
        random_state=42,
    )
    return Pipeline([("pre", pre), ("reg", reg)])


# =========================
# NEW: anomalies from history only (no ML)
# =========================
def compute_anomalies_from_history(g: pd.DataFrame) -> dict:
    gg = g.sort_values("month_start").copy()
    if len(gg) < 6:
        return {
            "note": "Pas assez de mois pour détecter des anomalies (min 6).",
            "sales_drop": {"is_anomaly": False},
            "return_spike": {"is_anomaly": False},
        }

    last = gg.iloc[-1]
    last_month = str(pd.to_datetime(last["month_start"]).date())

    # -------------------------
    # 1) Sales drop anomaly (net_amount)
    # -------------------------
    last_net = float(last["net_amount"])
    ref6 = gg["net_amount"].tail(6)
    ref12 = gg["net_amount"].tail(12)

    mean6 = float(ref6.mean())
    std12 = float(ref12.std(ddof=1)) if len(ref12) >= 3 else float(ref6.std(ddof=1))
    z = (last_net - mean6) / std12 if std12 and std12 > 0 else None
    drop_vs_mean6 = (last_net - mean6) / mean6 if mean6 > 0 else None

    drop_anom = False
    reasons = []
    if z is not None and z < -2.0:
        drop_anom = True
        reasons.append(f"Z-score={z:.2f} (< -2) : chute rare vs historique.")
    if drop_vs_mean6 is not None and drop_vs_mean6 < -0.20:
        drop_anom = True
        reasons.append(f"Chute de {drop_vs_mean6*100:.1f}% vs moyenne 6 mois.")

    sales_drop = {
        "what": "Chute CA net : comparaison du dernier mois vs moyenne (6 mois) + variabilité (12 mois).",
        "last_month": last_month,
        "last_net_amount": last_net,
        "mean_6m": mean6,
        "std_12m": std12,
        "z_score": z,
        "drop_vs_mean6": drop_vs_mean6,
        "is_anomaly": drop_anom,
        "reason": " | ".join(reasons) if reasons else "Aucune chute anormale détectée.",
        "rules": [
            "Alerte si Z-score < -2 (chute statistiquement rare).",
            "Alerte si CA net < -20% vs moyenne des 6 derniers mois.",
        ],
    }

    # -------------------------
    # 2) Return rate spike anomaly (value)
    # -------------------------
    gg["return_rate_value"] = gg["return_amount"] / (gg["net_amount"] + gg["return_amount"]).replace(0, np.nan)
    last_rr = gg["return_rate_value"].iloc[-1]
    last_rr = float(last_rr) if pd.notna(last_rr) else None

    rr_ref = gg["return_rate_value"].tail(12).dropna()
    rr_mean = float(rr_ref.mean()) if len(rr_ref) else None
    rr_std = float(rr_ref.std(ddof=1)) if len(rr_ref) >= 3 else None
    z2 = (last_rr - rr_mean) / rr_std if (last_rr is not None and rr_mean is not None and rr_std and rr_std > 0) else None

    spike_anom = False
    reasons2 = []
    if z2 is not None and z2 > 2.0:
        spike_anom = True
        reasons2.append(f"Z-score={z2:.2f} (> 2) : hausse rare du taux de retour.")
    if (last_rr is not None) and (rr_mean is not None) and (last_rr > rr_mean * 1.30):
        spike_anom = True
        reasons2.append("Taux de retour > +30% vs moyenne.")

    return_spike = {
        "what": "Hausse taux de retour (valeur) : comparaison du dernier mois vs historique.",
        "last_month": last_month,
        "last_return_rate_value": last_rr,
        "mean_12m": rr_mean,
        "std_12m": rr_std,
        "z_score": z2,
        "is_anomaly": spike_anom,
        "reason": " | ".join(reasons2) if reasons2 else "Aucune hausse anormale des retours détectée.",
        "rules": [
            "Alerte si Z-score > 2 (hausse statistiquement rare).",
            "Alerte si taux de retour > +30% vs moyenne.",
        ],
    }

    return {"sales_drop": sales_drop, "return_spike": return_spike}


# =========================
# Offline training + cache
# =========================
def train_and_cache():
    df = load_data(DATA_PATH)

    # -------- GLOBAL models
    g = monthly_global(df)
    if g["month_start"].nunique() < 4:
        raise ValueError("Pas assez de mois (>=4) pour faire une prévision mensuelle.")

    base = ["t_idx", "sin_month", "cos_month"]

    def feat(target: str):
        return base + [f"{target}_lag1", f"{target}_lag2", f"{target}_roll3", f"{target}_lag12"]

    targets = ["net_amount", "net_qty", "return_amount", "return_qty"]
    models = {}
    for t in targets:
        X = g[feat(t)].copy().replace([np.inf, -np.inf], np.nan).fillna(0.0)
        y = g[t].copy().replace([np.inf, -np.inf], np.nan).fillna(0.0)
        m = build_model([], list(X.columns))
        m.fit(X, y)
        models[t] = {"model": m, "feature_cols": list(X.columns)}

    # -------- Country model (net_amount)
    c = monthly_country(df)
    feat_c = ["Country_grouped", "sin_month", "cos_month", "net_amount_lag1", "net_amount_lag2", "net_amount_roll3", "net_amount_lag12"]
    Xc = c[feat_c].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    yc = c["net_amount"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    m_country = build_model(["Country_grouped"], ["sin_month", "cos_month", "net_amount_lag1", "net_amount_lag2", "net_amount_roll3", "net_amount_lag12"])
    m_country.fit(Xc, yc)

    # -------- Product model (net_amount)
    p, desc_map = monthly_product(df, TOP_K_PRODUCTS)
    feat_p = ["StockCode", "sin_month", "cos_month",
              "net_amount_lag1", "net_amount_lag2", "net_amount_roll3", "net_amount_lag12",
              "net_qty_lag1", "net_qty_roll3"]
    Xp = p[feat_p].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    yp = p["net_amount"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    m_product = build_model(["StockCode"], ["sin_month", "cos_month",
                                           "net_amount_lag1", "net_amount_lag2", "net_amount_roll3", "net_amount_lag12",
                                           "net_qty_lag1", "net_qty_roll3"])
    m_product.fit(Xp, yp)

    # -------- Forecast M+1 and 12 months sum (iterative)
    last_month = pd.to_datetime(g["month_start"].max())
    t_idx_last = int(g["t_idx"].max())

    hist = {t: g[t].tail(12).astype(float).tolist() for t in targets}

    def predict_step(step: int):
        ms = (last_month + pd.offsets.MonthBegin(step)).to_pydatetime()
        sin_m = float(np.sin(2 * np.pi * ms.month / 12.0))
        cos_m = float(np.cos(2 * np.pi * ms.month / 12.0))
        t_idx = t_idx_last + step

        out = {"month": str(pd.to_datetime(ms).date())}
        for t in targets:
            series = hist[t]
            lag12 = series[-12] if len(series) >= 12 else 0.0
            X = pd.DataFrame([{
                "t_idx": t_idx,
                "sin_month": sin_m,
                "cos_month": cos_m,
                f"{t}_lag1": series[-1],
                f"{t}_lag2": series[-2] if len(series) >= 2 else series[-1],
                f"{t}_roll3": float(np.mean(series[-3:])) if len(series) >= 3 else float(np.mean(series)),
                f"{t}_lag12": lag12,
            }])[models[t]["feature_cols"]].replace([np.inf, -np.inf], np.nan).fillna(0.0)

            pred = float(models[t]["model"].predict(X)[0])
            pred = max(pred, 0.0)
            out[f"pred_{t}"] = pred

        # update hist
        for t in targets:
            hist[t].append(out[f"pred_{t}"])
        return out

    m1 = predict_step(1)
    m1["pred_return_rate_value"] = (
        m1["pred_return_amount"] / (m1["pred_net_amount"] + m1["pred_return_amount"])
        if (m1["pred_net_amount"] + m1["pred_return_amount"]) > 0 else None
    )

    sums = {t: 0.0 for t in targets}
    for step in range(1, 13):
        o = predict_step(step)
        for t in targets:
            sums[t] += o[f"pred_{t}"]

    year_obj = {
        "pred_net_amount_12m": sums["net_amount"],
        "pred_net_qty_12m": sums["net_qty"],
        "pred_return_amount_12m": sums["return_amount"],
        "pred_return_qty_12m": sums["return_qty"],
        "pred_return_rate_value_12m": sums["return_amount"] / (sums["net_amount"] + sums["return_amount"])
        if (sums["net_amount"] + sums["return_amount"]) > 0 else None,
    }

    # -------- Country ranking next month
    c_last = c.sort_values(["Country_grouped", "month_start"]).groupby("Country_grouped", as_index=False).tail(1).copy()
    next_month = (pd.to_datetime(c["month_start"].max()) + pd.offsets.MonthBegin(1)).to_pydatetime()
    sin_m = float(np.sin(2 * np.pi * next_month.month / 12.0))
    cos_m = float(np.cos(2 * np.pi * next_month.month / 12.0))

    Xc_next = pd.DataFrame({
        "Country_grouped": c_last["Country_grouped"].astype(str),
        "sin_month": sin_m,
        "cos_month": cos_m,
        "net_amount_lag1": c_last["net_amount_lag1"].astype(float),
        "net_amount_lag2": c_last["net_amount_lag2"].astype(float),
        "net_amount_roll3": c_last["net_amount_roll3"].astype(float),
        "net_amount_lag12": c_last["net_amount_lag12"].astype(float),
    }).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    pred_c = m_country.predict(Xc_next[feat_c]).astype(float)
    pred_c = np.maximum(pred_c, 0.0)

    country_pred = pd.DataFrame({
        "Country": c_last["Country_grouped"].astype(str).values,
        "pred_next_month_net_amount": pred_c,
    }).sort_values("pred_next_month_net_amount", ascending=False)

    tot = float(country_pred["pred_next_month_net_amount"].sum())
    country_pred["pred_share_pct"] = (country_pred["pred_next_month_net_amount"] / tot * 100) if tot > 0 else 0.0

    # -------- Product ranking next month (names)
    p_last = p.sort_values(["StockCode", "month_start"]).groupby("StockCode", as_index=False).tail(1).copy()
    Xp_next = pd.DataFrame({
        "StockCode": p_last["StockCode"].astype(str),
        "sin_month": sin_m,
        "cos_month": cos_m,
        "net_amount_lag1": p_last["net_amount_lag1"].astype(float),
        "net_amount_lag2": p_last["net_amount_lag2"].astype(float),
        "net_amount_roll3": p_last["net_amount_roll3"].astype(float),
        "net_amount_lag12": p_last["net_amount_lag12"].astype(float),
        "net_qty_lag1": p_last["net_qty_lag1"].astype(float),
        "net_qty_roll3": p_last["net_qty_roll3"].astype(float),
    }).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    pred_p = m_product.predict(Xp_next[feat_p]).astype(float)
    pred_p = np.maximum(pred_p, 0.0)

    prod_pred = pd.DataFrame({
        "StockCode": p_last["StockCode"].astype(str).values,
        "pred_next_month_net_amount": pred_p,
    }).sort_values("pred_next_month_net_amount", ascending=False).head(10)

    prod_pred["ProductName"] = prod_pred["StockCode"].map(lambda x: desc_map.get(x, x))
    top_products = prod_pred[["ProductName", "StockCode", "pred_next_month_net_amount"]].to_dict(orient="records")

    # -------- NEW anomalies (history only, no ML)
    anomalies = compute_anomalies_from_history(g)

    # Save bundle (optional)
    bundle = {
        "models_global": models,
        "m_country": m_country,
        "m_product": m_product,
        "feat_c": feat_c,
        "feat_p": feat_p,
        "desc_map": desc_map,
        "top_k_products": TOP_K_PRODUCTS,
        "top_k_countries": TOP_K_COUNTRIES,
    }
    dump(bundle, MODEL_BUNDLE_PATH)

    # Save predictions cache
    cache = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "pred_month_next": m1,
        "pred_year_next_12m": year_obj,
        "country_ranking_next_month": {
            "month": str(pd.to_datetime(next_month).date()),
            "top10": country_pred.head(10).to_dict(orient="records"),
        },
        "product_ranking_next_month": {
            "month": str(pd.to_datetime(next_month).date()),
            "top10": top_products,
        },
        "anomalies": anomalies,
        "quality_note": "Qualité non affichée (désactivée).",
    }

    with open(PRED_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

    print("Saved models:", MODEL_BUNDLE_PATH)
    print("Saved predictions cache:", PRED_CACHE_PATH)


if __name__ == "__main__":
    train_and_cache()