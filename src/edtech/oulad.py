"""Load OULAD source tables and build the enrollment-level integration table.

Grain: (id_student, code_module, code_presentation). Activity and assessment records
are aggregated separately and joined on the key, so no join multiplies rows.
Observation window: course days 0..L, where L = module_presentation_length.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def read_raw() -> dict[str, pd.DataFrame]:
    na = ["?", ""]
    t = {
        "courses": pd.read_csv(C.RAW / "courses.csv"),
        "assessments": pd.read_csv(C.RAW / "assessments.csv", na_values=na),
        "studentInfo": pd.read_csv(C.RAW / "studentInfo.csv", na_values=na),
        "studentRegistration": pd.read_csv(C.RAW / "studentRegistration.csv", na_values=na),
        "studentAssessment": pd.read_csv(C.RAW / "studentAssessment.csv", na_values=na),
        "vle": pd.read_csv(C.RAW / "vle.csv", na_values=na),
    }
    return t


def activity_features(day: pd.DataFrame, courses: pd.DataFrame) -> pd.DataFrame:
    """Engagement features from the Spark enrollment_day table (all within [0, L])."""
    d = day.merge(courses, on=["code_module", "code_presentation"], how="left")
    L = d["module_presentation_length"]
    pre = d[d["date"] < 0].groupby(C.KEY)["clicks"].sum().rename("pre_course_clicks")
    w = d[(d["date"] >= 0) & (d["date"] <= L) & (d["clicks"] > 0)].copy()
    w["week"] = w["date"] // 7
    w["third"] = np.minimum((3 * w["date"]) // (w["module_presentation_length"] + 1), 2)
    g = w.groupby(C.KEY)
    out = pd.DataFrame({
        "total_clicks": g["clicks"].sum(),
        "active_days": g["date"].nunique(),
        "first_active_day": g["date"].min(),
        "last_active_day": g["date"].max(),
        "forum_clicks": g["forum_clicks"].sum(),
        "content_clicks": g["content_clicks"].sum(),
        "quiz_clicks": g["quiz_clicks"].sum(),
        "max_daily_clicks": g["clicks"].max(),
        "max_daily_records": g["n_records"].max(),
    })
    thirds = w.pivot_table(index=C.KEY, columns="third", values="clicks", aggfunc="sum", fill_value=0)
    thirds.columns = ["early_clicks", "mid_clicks", "late_clicks"][: thirds.shape[1]]
    weekly = w.groupby(C.KEY + ["week"])["clicks"].sum().reset_index()
    weekly_stats = weekly.groupby(C.KEY)["clicks"].agg(["sum", lambda s: (s ** 2).sum(), "size"])
    weekly_stats.columns = ["wk_sum", "wk_sumsq", "wk_active"]
    out = out.join(thirds).join(weekly_stats).join(pre, how="outer")
    return out.reset_index()


def deadline_activity(day: pd.DataFrame, assessments: pd.DataFrame, courses: pd.DataFrame) -> pd.DataFrame:
    """Share of in-window clicks made on a deadline day or the 2 days before it.

    ``deadline_click_ratio`` = observed share / share expected if clicks were spread
    uniformly over the course days (1 = no deadline concentration, >1 = cramming).
    """
    dl = assessments[(assessments["assessment_type"] != "Exam") & assessments["date"].notna()]
    dl = dl[["code_module", "code_presentation", "date"]].astype({"date": int})
    near = pd.concat([dl.assign(date=dl["date"] - k) for k in (0, 1, 2)]).drop_duplicates()
    n_near = near.groupby(["code_module", "code_presentation"]).size().rename("near_days").reset_index()
    near = near.assign(is_near=1)
    d = day.merge(courses, on=["code_module", "code_presentation"])
    d = d[(d["date"] >= 0) & (d["date"] <= d["module_presentation_length"])]
    d = d.merge(near, on=["code_module", "code_presentation", "date"], how="left")
    d["near_clicks"] = d["clicks"] * d["is_near"].fillna(0)
    g = d.groupby(C.KEY)[["clicks", "near_clicks"]].sum().reset_index()
    g = g.merge(n_near, on=["code_module", "code_presentation"]).merge(courses, on=["code_module", "code_presentation"])
    g["deadline_click_share"] = g["near_clicks"] / g["clicks"].where(g["clicks"] > 0)
    expected = g["near_days"] / (g["module_presentation_length"] + 1)
    g["deadline_click_ratio"] = g["deadline_click_share"] / expected
    return g[C.KEY + ["deadline_click_share", "deadline_click_ratio"]]


def assessment_features(t: dict[str, pd.DataFrame]) -> pd.DataFrame:
    a = t["assessments"]
    sa = t["studentAssessment"].merge(a, on="id_assessment", how="left", validate="many_to_one")
    sa = sa.merge(t["courses"], on=["code_module", "code_presentation"], how="left")
    cur = sa[(sa["assessment_type"] != "Exam") & (sa["is_banked"] == 0)
             & (sa["date_submitted"] <= sa["module_presentation_length"])].copy()
    cur["days_early"] = cur["date"] - cur["date_submitted"]          # >0 early, <0 late
    cur["on_time"] = (cur["days_early"] >= 0).astype(float)
    cur["score_x_weight"] = cur["score"] * cur["weight"]
    g = cur.groupby(C.KEY)
    out = pd.DataFrame({
        "submitted_assessments": g["id_assessment"].nunique(),
        "score_sum": g["score"].sum(min_count=1),
        "scored_count": g["score"].count(),
        "mean_score": g["score"].mean(),
        "on_time_submissions": g["on_time"].sum(),
        "mean_days_early": g["days_early"].mean(),
        "weighted_score_sum": g["score_x_weight"].sum(),
        "submitted_weight": g["weight"].sum(),
    })
    # Scheduled non-exam assessments with a deadline inside the window: the denominator
    sched = a[(a["assessment_type"] != "Exam")].merge(t["courses"], on=["code_module", "code_presentation"])
    sched = sched[sched["date"].isna() | (sched["date"] <= sched["module_presentation_length"])]
    n_sched = sched.groupby(["code_module", "code_presentation"]).size().rename("scheduled_assessments")
    banked = sa[sa["is_banked"] == 1].groupby(C.KEY).size().rename("banked_assessments")
    exam = sa[sa["assessment_type"] == "Exam"].groupby(C.KEY)["score"].mean().rename("exam_score")
    return out.join(banked, how="outer").join(exam, how="outer").reset_index(), n_sched.reset_index()


def build_enrollment_table(t: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    t = t or read_raw()
    day = pd.read_parquet(C.INTERIM / "enrollment_day.parquet")
    base = (t["studentInfo"]
            .merge(t["studentRegistration"], on=C.KEY, how="left", validate="one_to_one")
            .merge(t["courses"], on=["code_module", "code_presentation"], how="left", validate="many_to_one"))
    act = activity_features(day, t["courses"])
    ass, n_sched = assessment_features(t)
    dla = deadline_activity(day, t["assessments"], t["courses"])
    df = (base.merge(act, on=C.KEY, how="left", validate="one_to_one")
              .merge(dla, on=C.KEY, how="left", validate="one_to_one")
              .merge(ass, on=C.KEY, how="left", validate="one_to_one")
              .merge(n_sched, on=["code_module", "code_presentation"], how="left"))
    zero_cols = ["total_clicks", "active_days", "forum_clicks", "content_clicks", "quiz_clicks",
                 "early_clicks", "mid_clicks", "late_clicks", "pre_course_clicks", "wk_sum", "wk_sumsq",
                 "wk_active", "max_daily_clicks", "max_daily_records", "submitted_assessments",
                 "on_time_submissions", "banked_assessments", "weighted_score_sum", "submitted_weight",
                 "scored_count"]
    df[zero_cols] = df[zero_cols].fillna(0)
    df["score_sum"] = df["score_sum"].fillna(0)
    df["course_domain"] = df["code_module"].map(C.MODULE_DOMAIN)
    df["start_month"] = np.where(df["code_presentation"].str.endswith("B"), "February", "October")
    df["presentation_year"] = df["code_presentation"].str[:4].astype(int)
    return df


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Derived engagement / performance measures computed on the integrated table."""
    df = df.copy()
    L = df["module_presentation_length"]
    obs_days = L + 1
    weeks = np.ceil(obs_days / 7)
    df["observation_weeks"] = weeks
    df["active_day_fraction"] = df["active_days"] / obs_days
    df["clicks_per_active_day"] = np.where(df["active_days"] > 0, df["total_clicks"] / df["active_days"].clip(lower=1), 0.0)
    mean_w = df["wk_sum"] / weeks
    var_w = (df["wk_sumsq"] / weeks) - mean_w ** 2
    df["weekly_click_cv"] = np.where(mean_w > 0, np.sqrt(var_w.clip(lower=0)) / mean_w.replace(0, np.nan), 0.0)
    df["active_week_fraction"] = df["wk_active"] / weeks
    # Burstiness inside the learner's own active span (first to last active week), which
    # separates irregular study from low volume (zero weeks inside the span count as 0).
    span = (df["last_active_day"] // 7 - df["first_active_day"] // 7 + 1).where(df["active_days"] > 0)
    m_s = df["wk_sum"] / span
    v_s = (df["wk_sumsq"] / span) - m_s ** 2
    df["active_span_weeks"] = span.fillna(0)
    df["weekly_cv_active_span"] = (np.sqrt(v_s.clip(lower=0)) / m_s).fillna(0)
    df["deadline_click_ratio"] = df["deadline_click_ratio"].fillna(0)
    df["deadline_click_share"] = df["deadline_click_share"].fillna(0)
    df["recency_days"] = np.where(df["active_days"] > 0, L - df["last_active_day"], obs_days)
    df["recency_fraction"] = df["recency_days"] / obs_days
    df["late_click_share"] = np.where(df["total_clicks"] > 0, df["late_clicks"] / df["total_clicks"].clip(lower=1), 0.0)
    df["forum_click_share"] = np.where(df["total_clicks"] > 0, df["forum_clicks"] / df["total_clicks"].clip(lower=1), 0.0)
    df["submission_rate"] = (df["submitted_assessments"] / df["scheduled_assessments"]).clip(upper=1)
    df["on_time_rate_scheduled"] = (df["on_time_submissions"] / df["scheduled_assessments"]).clip(upper=1)
    # Average score across all scheduled assessments, non-submission counted as 0 (OU convention)
    df["avg_score_scheduled"] = df["score_sum"] / df["scheduled_assessments"]
    df["days_registration_to_first_activity"] = df["first_active_day"] - df["date_registration"]
    df["no_activity"] = (df["active_days"] == 0).astype(int)
    df["is_withdrawn"] = (df["final_result"] == "Withdrawn").astype(int)
    df["is_pass"] = df["final_result"].isin(["Pass", "Distinction"]).astype(int)
    df["at_risk_outcome"] = df["final_result"].isin(["Fail", "Withdrawn"]).astype(int)
    return df
