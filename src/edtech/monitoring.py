"""Model monitoring (Module 7): data drift, segment stability and re-clustering triggers.

Signals
-------
* PSI (population stability index) and KS statistic per input feature
* Evidently DataDriftPreset report (HTML) for the same reference/current pair
* Segment-share shift (percentage points and chi-square test)
* Centroid shift: K-Means refitted on the current batch, matched to the production
  centroids with the Hungarian algorithm, distance in standardised units
* Share of atypical learners (distance to nearest centroid above the training p95)
* Silhouette of the production assignment on the current batch

Triggers (documented in docs/MONITORING_PLAYBOOK.md) turn these signals into
GREEN / AMBER / RED decisions, where RED = re-cluster.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import chi2_contingency, ks_2samp
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from . import config as C

TRIGGERS = {
    "psi_feature_red": 0.25,          # PSI > 0.25 = significant shift (industry convention)
    "psi_feature_amber": 0.10,
    "n_features_red": 2,              # >= 2 features in red -> re-cluster
    "segment_share_pp_red": 10.0,     # any segment share moves > 10 percentage points
    "segment_share_pp_amber": 5.0,
    "centroid_shift_red": 0.50,       # mean matched centroid shift > 0.5 SD
    "centroid_shift_amber": 0.30,
    "atypical_share_red": 0.15,       # > 15 % of learners beyond training p95 distance (expected 5 %)
    "atypical_share_amber": 0.08,
    "silhouette_drop_red": 0.05,      # absolute drop vs training silhouette
}


def psi(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
    ref, cur = ref[~np.isnan(ref)], cur[~np.isnan(cur)]
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        edges = np.linspace(ref.min(), ref.max() + 1e-9, bins + 1)
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref)
    c = np.histogram(cur, edges)[0] / len(cur)
    r, c = np.clip(r, 1e-4, None), np.clip(c, 1e-4, None)
    return float(np.sum((c - r) * np.log(c / r)))


def feature_drift(ref: pd.DataFrame, cur: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    for f in features:
        a, b = ref[f].astype(float).to_numpy(), cur[f].astype(float).to_numpy()
        p = psi(a, b)
        ks = ks_2samp(a[~np.isnan(a)], b[~np.isnan(b)])
        status = "RED" if p > TRIGGERS["psi_feature_red"] else "AMBER" if p > TRIGGERS["psi_feature_amber"] else "GREEN"
        rows.append({"feature": f, "ref_mean": np.nanmean(a), "cur_mean": np.nanmean(b),
                     "ref_median": np.nanmedian(a), "cur_median": np.nanmedian(b),
                     "psi": p, "ks_stat": ks.statistic, "ks_pvalue": ks.pvalue, "status": status})
    return pd.DataFrame(rows)


def segment_shift(ref_labels: pd.Series, cur_labels: pd.Series) -> tuple[pd.DataFrame, float]:
    r = ref_labels.value_counts(normalize=True)
    c = cur_labels.value_counts(normalize=True)
    t = pd.DataFrame({"reference_share": r, "current_share": c}).fillna(0)
    t["shift_pp"] = 100 * (t["current_share"] - t["reference_share"])
    counts = pd.DataFrame({"r": ref_labels.value_counts(), "c": cur_labels.value_counts()}).fillna(0)
    p = chi2_contingency(counts.T.to_numpy())[1]
    return t.sort_index(), float(p)


def centroid_shift(prod_centroids: np.ndarray, Z_cur: np.ndarray) -> tuple[float, np.ndarray]:
    k = prod_centroids.shape[0]
    km = KMeans(n_clusters=k, n_init=10, random_state=C.SEED).fit(Z_cur)
    cost = np.linalg.norm(prod_centroids[:, None, :] - km.cluster_centers_[None, :, :], axis=2)
    r, c = linear_sum_assignment(cost)
    shifts = cost[r, c]
    return float(shifts.mean()), shifts


def evaluate(model, ref_df: pd.DataFrame, cur_df: pd.DataFrame, train_silhouette: float,
             batch_name: str) -> dict:
    """Full monitoring run for one incoming batch; returns a JSON-serialisable summary."""
    feats = model.features()
    ref_a, cur_a = ref_df[ref_df["active_day_fraction"] > 0], cur_df[cur_df["active_day_fraction"] > 0]
    drift = feature_drift(ref_a, cur_a, feats)
    ref_seg = model.assign(ref_df)["segment_id"]
    cur_assign = model.assign(cur_df)
    seg, chi_p = segment_shift(ref_seg, cur_assign["segment_id"])
    Z_cur = model.pipeline.transform(cur_a)
    mean_shift, shifts = centroid_shift(model.kmeans.cluster_centers_, Z_cur)
    lab = model.kmeans.predict(Z_cur)
    rng = np.random.default_rng(C.SEED)
    idx = rng.choice(len(Z_cur), min(6000, len(Z_cur)), replace=False)
    sil = float(silhouette_score(Z_cur[idx], lab[idx])) if len(np.unique(lab)) > 1 else np.nan
    atyp = float(cur_assign.loc[cur_df["active_day_fraction"] > 0, "is_atypical"].mean())

    n_red = int((drift["status"] == "RED").sum())
    max_pp = float(seg["shift_pp"].abs().max())
    checks = {
        "feature_drift": "RED" if n_red >= TRIGGERS["n_features_red"] else "AMBER" if (drift["status"] != "GREEN").any() else "GREEN",
        "segment_share": "RED" if max_pp > TRIGGERS["segment_share_pp_red"] else "AMBER" if max_pp > TRIGGERS["segment_share_pp_amber"] else "GREEN",
        "centroid_shift": "RED" if mean_shift > TRIGGERS["centroid_shift_red"] else "AMBER" if mean_shift > TRIGGERS["centroid_shift_amber"] else "GREEN",
        "atypical_share": "RED" if atyp > TRIGGERS["atypical_share_red"] else "AMBER" if atyp > TRIGGERS["atypical_share_amber"] else "GREEN",
        "silhouette": "RED" if (train_silhouette - sil) > TRIGGERS["silhouette_drop_red"] else "GREEN",
    }
    n_red_checks = sum(v == "RED" for v in checks.values())
    decision = ("RE-CLUSTER" if n_red_checks >= 2 else
                "INVESTIGATE" if n_red_checks == 1 or "AMBER" in checks.values() else "NO ACTION")
    return {
        "batch": batch_name, "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_reference": int(len(ref_df)), "n_current": int(len(cur_df)),
        "features_red": n_red, "max_segment_shift_pp": max_pp, "segment_chi2_pvalue": chi_p,
        "mean_centroid_shift_sd": mean_shift, "centroid_shifts_sd": shifts.round(3).tolist(),
        "atypical_share": atyp, "silhouette_current": sil, "silhouette_train": train_silhouette,
        "checks": checks, "decision": decision,
        "feature_drift": drift.round(4).to_dict("records"),
        "segment_shares": seg.reset_index(names="segment_id").round(4).to_dict("records"),
    }


def evidently_report(ref: pd.DataFrame, cur: pd.DataFrame, features: list[str], path) -> dict:
    from evidently import Report
    from evidently.presets import DataDriftPreset

    report = Report([DataDriftPreset()])
    snap = report.run(reference_data=ref[features].reset_index(drop=True),
                      current_data=cur[features].reset_index(drop=True))
    snap.save_html(str(path))
    d = snap.dict()
    out = {"html": str(path.name), "metrics": []}
    for m in d["metrics"]:
        out["metrics"].append({"metric": m["metric_name"], "value": m["value"]})
    return out


def simulate_drift(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Stress scenario: platform redesign + exam-season pressure in a future cohort.

    Multiplicative, documented shocks applied to an incoming batch so that the
    triggers can be demonstrated end-to-end (the real 2014J batch is mostly stable).
    """
    x = df.copy()
    act = x["active_day_fraction"] > 0
    x.loc[act, "active_day_fraction"] *= rng.uniform(0.55, 0.8, act.sum())        # less frequent study
    x.loc[act, "night_login_share"] = np.clip(x.loc[act, "night_login_share"] + rng.normal(0.25, 0.08, act.sum()), 0, 1)
    x.loc[act, "last_minute_submission_share"] = np.clip(x.loc[act, "last_minute_submission_share"] * 1.8 + 0.1, 0, 1)
    x.loc[act, "deadline_click_ratio"] *= rng.uniform(1.3, 1.9, act.sum())
    x.loc[act, "video_completion_rate"] = np.clip(x.loc[act, "video_completion_rate"] - 0.15, 0, 1)
    x["recency_fraction"] = np.clip(x["recency_fraction"] + 0.1 * act, 0, 1)
    return x


def save_summary(summary: dict, name: str) -> None:
    (C.MONITORING_DATA / f"{name}.json").write_text(json.dumps(summary, indent=2, default=float))
