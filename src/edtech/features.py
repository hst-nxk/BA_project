"""Feature engineering (Module 4, Lecture 7 transformations, Lecture 9 feature selection).

1. Aggregate the cleaned synthetic logs to the enrollment grain and integrate with OULAD.
2. Build RFM-style behavioural features (Recency, Frequency, intensity/"Monetary").
3. Transform skewed features (log1p / Yeo-Johnson), scale (Z-score), encode categoricals.
4. Select a compact, non-redundant set for distance-based clustering.

``FeaturePipeline`` is the single object used by notebooks, the API and monitoring, so
training and serving share exactly the same transformation code.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer, StandardScaler

from . import config as C

NIGHT_HOURS = {22, 23, 0, 1, 2, 3, 4, 5}


def session_aggregates(s: pd.DataFrame, enr: pd.DataFrame) -> pd.DataFrame:
    s = s.merge(enr[C.KEY + ["module_presentation_length"]], on=C.KEY, how="left")
    w = s[(s["course_day"] >= 0) & (s["course_day"] <= s["module_presentation_length"])].copy()
    w["night"] = w["login_hour"].isin(NIGHT_HOURS).astype(int)
    w["mobile"] = (w["device_type"] == "mobile").astype(int)
    w["week"] = w["course_day"] // 7
    g = w.groupby(C.KEY)
    out = pd.DataFrame({
        "logins_total": g.size(),
        "login_days": g["course_day"].nunique(),
        "login_weeks": g["week"].nunique(),
        "night_login_share": g["night"].mean(),
        "mobile_share": g["mobile"].mean(),
        "median_login_hour": g["login_hour"].median(),
        "mean_session_minutes": g["session_minutes"].mean(),
        "videos_started": g["videos_started"].sum(),
        "videos_completed": g["videos_completed"].sum(),
        "forum_posts": g["forum_posts"].sum(),
        "forum_replies": g["forum_replies"].sum(),
    })
    period = pd.cut(w["login_hour"], [-1, 5, 11, 17, 21, 23], labels=["night", "morning", "afternoon", "evening", "night2"])
    w["period"] = period.astype(str).replace({"night2": "night"})
    out["preferred_login_period"] = w.groupby(C.KEY)["period"].agg(lambda x: x.value_counts().idxmax())
    # first login over the whole log (pre-course logins included)
    out["first_login_day"] = s.groupby(C.KEY)["course_day"].min()
    return out.reset_index()


def submission_aggregates(sub: pd.DataFrame) -> pd.DataFrame:
    sub = sub.copy()
    h = sub["hours_before_deadline"]
    sub["last_minute"] = ((h >= 0) & (h < 24)).astype(int)
    sub["late"] = (h < 0).astype(int)
    g = sub.groupby(C.KEY)
    return pd.DataFrame({
        "median_hours_before_deadline": g["hours_before_deadline"].median(),
        "last_minute_submission_share": g["last_minute"].mean(),
        "late_submission_share": g["late"].mean(),
        "deadline_day_night_submissions": g.apply(lambda x: ((x["submit_hour"] >= 21) & (x["last_minute"] == 1)).mean(),
                                                  include_groups=False),
    }).reset_index()


def integrate(enr: pd.DataFrame, sess_agg: pd.DataFrame, sub_agg: pd.DataFrame) -> pd.DataFrame:
    df = enr.merge(sess_agg, on=C.KEY, how="left", validate="one_to_one").merge(sub_agg, on=C.KEY, how="left", validate="one_to_one")
    for c in ["logins_total", "login_days", "login_weeks", "videos_started", "videos_completed", "forum_posts", "forum_replies"]:
        df[c] = df[c].fillna(0)
    weeks = df["observation_weeks"]
    # ---- RFM-style behavioural features --------------------------------------------------
    df["logins_per_week"] = df["logins_total"] / weeks                      # F (frequency)
    df["login_week_coverage"] = df["login_weeks"] / weeks                    # F (regularity)
    df["clicks_per_week"] = df["total_clicks"] / weeks                       # M (intensity)
    df["forum_posts_per_week"] = df["forum_posts"] / weeks
    # Bayesian-smoothed completion rate: (completed + 2*prior) / (started + 2)
    prior = df["videos_completed"].sum() / max(df["videos_started"].sum(), 1)
    df["video_completion_rate"] = (df["videos_completed"] + 2 * prior) / (df["videos_started"] + 2)
    df.loc[df["logins_total"] == 0, "video_completion_rate"] = 0.0
    df["days_to_first_login"] = (df["first_login_day"] - df["date_registration"]).clip(lower=0)
    # R (recency) already computed from OULAD: recency_fraction
    df["night_login_share"] = df["night_login_share"].fillna(0)
    df["mobile_share"] = df["mobile_share"].fillna(0)
    # learners with no submissions: neutral timeliness values + flag
    df["no_submission"] = (df["submitted_assessments"] == 0).astype(int)
    df["last_minute_submission_share"] = df["last_minute_submission_share"].fillna(0)
    df["late_submission_share"] = df["late_submission_share"].fillna(0)
    # ---- composite KPI: engagement score 0-100 (Module 1 definition) ----------------------
    pct = lambda x: x.rank(pct=True)
    w = C.ENGAGEMENT_SCORE_WEIGHTS
    comp = (w["active_day_fraction"] * pct(df["active_day_fraction"])
            + w["logins_per_week_pct"] * pct(df["logins_per_week"])
            + w["video_completion_rate"] * df["video_completion_rate"]
            + w["submission_rate"] * df["submission_rate"]
            + w["forum_posts_pct"] * pct(df["forum_posts_per_week"]))
    df["engagement_score"] = (100 * comp).round(2)
    return df


# Candidate features (before selection) with their RFM / business role
CANDIDATES = {
    "recency_fraction": "R - share of course elapsed since last activity",
    "active_day_fraction": "F - share of course days with VLE activity",
    "logins_per_week": "F - synthetic login frequency",
    "login_week_coverage": "F - share of weeks with at least one login",
    "total_clicks": "M - recorded resource interactions",
    "clicks_per_week": "M - interaction intensity",
    "clicks_per_active_day": "M - depth per study day",
    "weekly_click_cv": "Consistency - variability of weekly clicks (whole course)",
    "weekly_cv_active_span": "Consistency - variability of weekly clicks within own active span",
    "late_click_share": "Pacing - share of clicks in last third",
    "deadline_click_ratio": "Pacing - click concentration on deadline days (1 = uniform)",
    "submission_rate": "Performance - share of scheduled assessments submitted",
    "avg_score_scheduled": "Performance - mean score over scheduled assessments",
    "on_time_rate_scheduled": "Timeliness - on-time submissions / scheduled",
    "last_minute_submission_share": "Timeliness - submitted <24h before deadline",
    "median_hours_before_deadline": "Timeliness - typical lead time",
    "video_completion_rate": "Content - synthetic video completion",
    "forum_posts_per_week": "Social - synthetic forum posts",
    "night_login_share": "Rhythm - logins 22:00-05:59",
    "days_to_first_login": "Onboarding - days from registration to first login",
    "mobile_share": "Context - mobile sessions",
}

SKEWED = ["logins_per_week", "total_clicks", "clicks_per_week", "clicks_per_active_day",
          "forum_posts_per_week", "days_to_first_login", "weekly_click_cv", "weekly_cv_active_span",
          "deadline_click_ratio"]

# Final clustering feature set (chosen in notebook 04 by correlation filter + interpretability)
SELECTED = [
    "recency_fraction",              # R
    "active_day_fraction",           # F
    "clicks_per_active_day",         # M (depth, low corr with F)
    "weekly_cv_active_span",         # consistency (burstiness inside own active span)
    "deadline_click_ratio",          # pacing / cramming (real OULAD)
    "avg_score_scheduled",           # performance
    "last_minute_submission_share",  # timeliness (synthetic, hour level)
    "video_completion_rate",         # synthetic content engagement
    "forum_posts_per_week",          # synthetic social engagement
    "night_login_share",             # synthetic rhythm
]


class FeaturePipeline:
    """Yeo-Johnson for skewed inputs, then Z-score on everything (fit on training cohorts)."""

    def __init__(self, features: list[str] | None = None) -> None:
        self.features = list(features or SELECTED)
        self.skewed = [f for f in self.features if f in SKEWED]
        self.power = PowerTransformer(method="yeo-johnson", standardize=False)
        self.scaler = StandardScaler()
        self.medians_: dict[str, float] = {}
        self.clip_: dict[str, tuple[float, float]] = {}

    def _prep(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[self.features].astype(float).copy()
        for c in self.features:
            X[c] = X[c].fillna(self.medians_.get(c, 0.0))
            if c in self.clip_:
                lo, hi = self.clip_[c]
                X[c] = X[c].clip(lo, hi)
        return X

    def fit(self, df: pd.DataFrame) -> "FeaturePipeline":
        X = df[self.features].astype(float)
        self.medians_ = X.median().to_dict()
        # winsorise at 0.5% / 99.5% of the training distribution to limit k-means outlier leverage
        self.clip_ = {c: (float(X[c].quantile(0.005)), float(X[c].quantile(0.995))) for c in self.features}
        X = self._prep(df)
        if self.skewed:
            X[self.skewed] = self.power.fit_transform(X[self.skewed])
        self.scaler.fit(X)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        X = self._prep(df)
        if self.skewed:
            X[self.skewed] = self.power.transform(X[self.skewed])
        return self.scaler.transform(X)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)

    def inverse_centroids(self, Z: np.ndarray) -> pd.DataFrame:
        X = pd.DataFrame(self.scaler.inverse_transform(Z), columns=self.features)
        if self.skewed:
            X[self.skewed] = self.power.inverse_transform(X[self.skewed])
        return X


def correlation_filter(df: pd.DataFrame, cols: list[str], threshold: float = 0.80) -> pd.DataFrame:
    """Pairs whose |Spearman rho| exceeds the threshold (redundancy, Lecture 9 p.6)."""
    corr = df[cols].corr(method="spearman")
    pairs = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            if abs(corr.loc[a, b]) >= threshold:
                pairs.append({"feature_a": a, "feature_b": b, "spearman_rho": round(corr.loc[a, b], 3)})
    return pd.DataFrame(pairs)


def one_hot_context(df: pd.DataFrame) -> pd.DataFrame:
    """Nominal -> binary (1-to-N) encoding of context variables used for profiling only."""
    cats = ["course_domain", "code_module", "start_month", "age_band", "highest_education", "gender", "disability"]
    dummies = pd.get_dummies(df[cats], columns=cats, dtype=int)
    return pd.concat([df[C.KEY].reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)


IMD_ORDER = ["0-10%", "10-20%", "20-30%", "30-40%", "40-50%", "50-60%", "60-70%", "70-80%", "80-90%", "90-100%"]
EDU_ORDER = ["No Formal quals", "Lower Than A Level", "A Level or Equivalent", "HE Qualification", "Post Graduate Qualification"]


def ordinal_context(df: pd.DataFrame) -> pd.DataFrame:
    """Ordinal -> numeric conversion (Lecture 7 p.11) for ordered context variables."""
    out = df[C.KEY].copy()
    out["imd_decile"] = df["imd_band"].map({b: i + 1 for i, b in enumerate(IMD_ORDER)})
    out["education_level"] = df["highest_education"].map({b: i for i, b in enumerate(EDU_ORDER)})
    out["age_level"] = df["age_band"].map({"0-35": 0, "35-55": 1, "55<=": 2})
    return out
