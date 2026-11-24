"""Data-quality assessment and cleaning for both sources (Module 3, Lecture 7).

Quality dimensions: completeness, uniqueness, validity, consistency, referential integrity.
Every cleaning operation appends a row to a log with the rule, justification, the
number of affected records and the action taken, so before/after comparisons are
reproducible.
"""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from . import config as C

DEVICE_MAP = {
    "desktop": "desktop", "pc": "desktop", "computer": "desktop",
    "mobile": "mobile", "phone": "mobile", "smartphone": "mobile",
    "tablet": "tablet", "tab": "tablet", "ipad": "tablet",
}
AUTOMATION_UA = re.compile(r"python-requests|curl|wget|bot|spider|headless", re.I)


class CleaningLog:
    def __init__(self) -> None:
        self.rows: list[dict] = []

    def add(self, source: str, issue: str, rule: str, action: str, affected: int, denominator: int,
            justification: str) -> None:
        self.rows.append({"source": source, "issue": issue, "rule": rule, "action": action,
                          "affected": int(affected), "denominator": int(denominator),
                          "affected_pct": round(100 * affected / max(denominator, 1), 3),
                          "justification": justification})

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


# ----------------------------------------------------------------------------------------
# Quality assessment
# ----------------------------------------------------------------------------------------
def quality_scorecard(tables: dict[str, pd.DataFrame], keys: dict[str, list[str]]) -> pd.DataFrame:
    rows = []
    for name, df in tables.items():
        cells = df.size
        missing = int(df.isna().sum().sum())
        k = keys.get(name)
        dup_full = int(df.duplicated().sum())
        dup_key = int(df.duplicated(k).sum()) if k else np.nan
        rows.append({"table": name, "rows": len(df), "columns": df.shape[1],
                     "completeness_pct": round(100 * (1 - missing / max(cells, 1)), 3),
                     "missing_cells": missing, "exact_duplicate_rows": dup_full,
                     "duplicate_keys": dup_key, "key": ",".join(k) if k else "-"})
    return pd.DataFrame(rows)


def missing_profile(df: pd.DataFrame, name: str) -> pd.DataFrame:
    m = df.isna().sum()
    m = m[m > 0]
    return pd.DataFrame({"table": name, "column": m.index, "missing": m.values,
                         "missing_pct": (100 * m.values / len(df)).round(3)})


def referential_checks(t: dict[str, pd.DataFrame], sessions: pd.DataFrame) -> pd.DataFrame:
    enr_keys = t["studentInfo"][C.KEY].drop_duplicates()
    checks = []

    def chk(name, child, parent, on):
        m = child[on].drop_duplicates().merge(parent[on].drop_duplicates(), on=on, how="left", indicator=True)
        bad = int((m["_merge"] == "left_only").sum())
        checks.append({"relationship": name, "child_keys": len(m), "orphan_keys": bad,
                       "integrity_pct": round(100 * (1 - bad / max(len(m), 1)), 3)})

    chk("studentRegistration -> studentInfo", t["studentRegistration"], enr_keys, C.KEY)
    chk("studentAssessment -> assessments", t["studentAssessment"], t["assessments"], ["id_assessment"])
    chk("studentInfo -> courses", t["studentInfo"], t["courses"], ["code_module", "code_presentation"])
    chk("vle -> courses", t["vle"], t["courses"], ["code_module", "code_presentation"])
    chk("synthetic sessions -> studentInfo", sessions, enr_keys, C.KEY)
    return pd.DataFrame(checks)


def validity_checks_sessions(s: pd.DataFrame) -> pd.DataFrame:
    canon = s["device_type"].astype("string").str.strip().str.lower().map(DEVICE_MAP)
    rules = {
        "session_minutes < 0": (s["session_minutes"] < 0).sum(),
        "session_minutes > 1440 (longer than a day)": (s["session_minutes"] > 1440).sum(),
        "videos_completed > videos_started": (s["videos_completed"] > s["videos_started"]).sum(),
        "login_hour outside 0..23": (~s["login_hour"].between(0, 23)).sum(),
        "device_type not in canonical vocabulary": (s["device_type"].notna() & ~s["device_type"].isin(["desktop", "mobile", "tablet"])).sum(),
        "device_type unmappable after standardisation": (s["device_type"].notna() & canon.isna()).sum(),
        "forum_replies > forum_posts": (s["forum_replies"] > s["forum_posts"]).sum(),
    }
    return pd.DataFrame({"rule": list(rules), "violations": [int(v) for v in rules.values()],
                         "denominator": len(s)})


# ----------------------------------------------------------------------------------------
# Bot / automation detection (multi-rule, enrollment level)
# ----------------------------------------------------------------------------------------
def bot_signals(s: pd.DataFrame, enr: pd.DataFrame) -> pd.DataFrame:
    tmp = s[C.KEY + ["login_hour"]].copy()
    tmp["sub_minute"] = (s["session_minutes"] < 1).astype(float)
    tmp["auto_ua"] = s["user_agent"].str.contains(AUTOMATION_UA).astype(float)
    tmp["minutes"] = s["session_minutes"]
    g = tmp.groupby(C.KEY)
    f = pd.DataFrame({"sessions": g.size(), "median_minutes": g["minutes"].median(),
                      "share_sub_minute": g["sub_minute"].mean(), "automation_ua_share": g["auto_ua"].mean()})
    hc = tmp.groupby(C.KEY + ["login_hour"]).size().unstack(fill_value=0)
    p = hc.div(hc.sum(1), axis=0).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -np.nansum(np.where(p > 0, p * np.log2(p), 0), axis=1)
    f["hour_entropy_bits"] = pd.Series(ent, index=hc.index)
    f = f.reset_index()
    f = f.merge(enr[C.KEY + ["module_presentation_length", "active_days"]], on=C.KEY, how="left")
    f["sessions_per_active_day"] = f["sessions"] / f["active_days"].clip(lower=1)
    # course-specific extreme outlier fence (Q3 + 3 IQR, Lecture 7) on session counts
    def fence(x):
        q1, q3 = x.quantile([0.25, 0.75])
        return q3 + 3 * (q3 - q1)
    f["session_fence"] = f.groupby(["code_module", "code_presentation"])["sessions"].transform(fence)
    f["r1_extreme_session_count"] = (f["sessions"] > f["session_fence"]).astype(int)
    f["r2_sub_minute_sessions"] = (f["share_sub_minute"] > 0.30).astype(int)
    f["r3_automation_user_agent"] = (f["automation_ua_share"] > 0.10).astype(int)
    f["r4_uniform_hours"] = (f["hour_entropy_bits"] > 4.2).astype(int)   # max = log2(24) = 4.585
    f["rules_triggered"] = f[[c for c in f if c.startswith("r") and c[1].isdigit()]].sum(1)
    f["bot_flag"] = (f["rules_triggered"] >= 2).astype(int)
    return f


def evaluate_bot_detection(signals: pd.DataFrame, defects_path=C.SYNTHETIC / "_injected_defects.json") -> dict:
    truth = pd.DataFrame(json.loads(defects_path.read_text())["bot_enrollments"]).assign(is_bot=1)
    m = signals.merge(truth, on=C.KEY, how="left").fillna({"is_bot": 0})
    tp = int(((m.bot_flag == 1) & (m.is_bot == 1)).sum())
    fp = int(((m.bot_flag == 1) & (m.is_bot == 0)).sum())
    fn = int(((m.bot_flag == 0) & (m.is_bot == 1)).sum())
    return {"true_positive": tp, "false_positive": fp, "false_negative": fn,
            "precision": round(tp / max(tp + fp, 1), 4), "recall": round(tp / max(tp + fn, 1), 4)}


# ----------------------------------------------------------------------------------------
# Cleaning
# ----------------------------------------------------------------------------------------
def clean_oulad(t: dict[str, pd.DataFrame], enr: pd.DataFrame, log: CleaningLog) -> pd.DataFrame:
    df = enr.copy()
    n = len(df)
    # Inconsistent categorical label: '10-20' vs every other band carrying '%'
    bad = df["imd_band"].eq("10-20")
    df.loc[bad, "imd_band"] = "10-20%"
    log.add("OULAD", "inconsistent categorical label", "imd_band == '10-20'", "relabel to '10-20%'",
            bad.sum(), n, "All other IMD bands use the 'x-y%' format; same decile, two spellings.")
    miss = df["imd_band"].isna()
    df["imd_band_missing"] = miss.astype(int)
    df["imd_band"] = df["imd_band"].fillna("Unknown")
    log.add("OULAD", "missing value", "imd_band is null", "category 'Unknown' + flag", miss.sum(), n,
            "Deprivation band is not imputable from behaviour; explicit category avoids bias (Lecture 7: global constant).")
    # Whitespace / case consistency of all string columns
    for c in ["gender", "region", "highest_education", "age_band", "disability", "final_result"]:
        before = df[c].copy()
        df[c] = df[c].str.strip()
        changed = int((before != df[c]).sum())
        if changed:
            log.add("OULAD", "whitespace", f"{c} has padding", "strip", changed, n, "Formatting only.")
    reg_miss = df["date_registration"].isna()
    df["date_registration_missing"] = reg_miss.astype(int)
    log.add("OULAD", "missing value", "date_registration is null", "keep NA + flag", reg_miss.sum(), n,
            "Registration day is unknown, not zero; only used for days-to-first-login, which becomes NA.")
    # Consistency: Withdrawn outcome without unregistration date or the reverse
    w = df["final_result"].eq("Withdrawn")
    inc = (w & df["date_unregistration"].isna()) | (~w & df["date_unregistration"].notna())
    df["withdrawal_inconsistent"] = inc.astype(int)
    log.add("OULAD", "cross-field inconsistency", "Withdrawn XOR date_unregistration present", "flag only",
            inc.sum(), n, "Administrative fields disagree; neither can be verified, so no value is invented.")
    # Outliers on clicks: course-specific IQR fences, flag only (high activity is plausible)
    grp = df.groupby(["code_module", "code_presentation"])["total_clicks"]
    q1, q3 = grp.transform(lambda x: x.quantile(0.25)), grp.transform(lambda x: x.quantile(0.75))
    df["clicks_extreme_outlier"] = (df["total_clicks"] > q3 + 3 * (q3 - q1)).astype(int)
    log.add("OULAD", "outlier", "total_clicks > Q3 + 3 IQR within course", "flag; Yeo-Johnson + winsorising in features",
            df["clicks_extreme_outlier"].sum(), n, "Very active learners are real; Yeo-Johnson transform + winsorising limit leverage (Lecture 7: do nothing / transform).")
    log.add("OULAD", "possible duplicate events", "identical studentVle rows",
            "retain (documented sensitivity)", json.loads((C.INTERIM / "etl_stats.json").read_text()).get("studentVle_exact_duplicate_rows", 0),
            10_655_280, "OULAD has no event id; identical daily rows may be distinct interactions. Removing them changes window clicks by ~3%.")
    return df


def clean_sessions(s: pd.DataFrame, enr: pd.DataFrame, log: CleaningLog) -> tuple[pd.DataFrame, pd.DataFrame]:
    n0 = len(s)
    s = s.copy()
    # 1. exact duplicates (double logging)
    dup = s.duplicated()
    s = s[~dup]
    log.add("Synthetic sessions", "duplicate rows", "all columns identical", "drop", dup.sum(), n0,
            "Same session_id and identical payload = logging retry.")
    # 2. session_id uniqueness after exact dedup
    dupid = s["session_id"].duplicated()
    log.add("Synthetic sessions", "duplicate key", "session_id repeated with different payload", "drop later copy",
            dupid.sum(), len(s), "Primary key must be unique.")
    s = s[~dupid]
    n = len(s)
    # 3. device label standardisation
    raw = s["device_type"].astype("string")
    canon = raw.str.strip().str.lower().map(DEVICE_MAP)
    changed = int((raw.notna() & (raw != canon)).sum())
    s["device_type"] = canon.fillna("unknown")
    log.add("Synthetic sessions", "inconsistent categorical labels", "device_type spelling/synonyms",
            "map to {desktop, mobile, tablet}", changed, n, "e.g. 'PC', ' desktop', 'DESKTOP' are the same device class.")
    log.add("Synthetic sessions", "missing value", "device_type is null", "category 'unknown'",
            int(raw.isna().sum()), n, "Device cannot be inferred reliably from a free-text user agent.")
    # 4. impossible values -> NA, then impute
    neg = s["session_minutes"] < 0
    big = s["session_minutes"] > 1440
    s.loc[neg | big, "session_minutes"] = np.nan
    log.add("Synthetic sessions", "invalid value", "session_minutes < 0", "set NA then impute", neg.sum(), n,
            "Negative duration is impossible (clock error).")
    log.add("Synthetic sessions", "invalid value", "session_minutes > 1440", "set NA then impute", big.sum(), n,
            "Session longer than 24h = unclosed session; true length unknown.")
    vid = s["videos_completed"] > s["videos_started"]
    s.loc[vid, "videos_completed"] = s.loc[vid, "videos_started"]
    log.add("Synthetic sessions", "invalid value", "videos_completed > videos_started", "cap at videos_started",
            vid.sum(), n, "A video cannot be completed without being started.")
    # 5. bot-like accounts (multi-rule detection) -> remove automated sessions only
    sig = bot_signals(s, enr)
    flagged = sig.loc[sig["bot_flag"] == 1, C.KEY]
    m = s.merge(flagged.assign(_bot=1), on=C.KEY, how="left")
    auto = (m["_bot"] == 1) & (m["user_agent"].str.contains(AUTOMATION_UA) | (m["session_minutes"] < 1))
    s = s[~auto.to_numpy()]
    log.add("Synthetic sessions", "bot-like activity", ">=2 of 4 rules (IQR session count, sub-minute, automation UA, uniform hours)",
            "drop automated sessions of flagged enrollments", int(auto.sum()), n,
            f"{len(flagged)} enrollments flagged; human sessions of those learners kept.")
    # 6. missing durations: learner-median imputation, global median fallback (+flag)
    na = s["session_minutes"].isna()
    s["session_minutes_imputed"] = na.astype(int)
    learner_med = s.groupby(C.KEY)["session_minutes"].transform("median")
    s["session_minutes"] = s["session_minutes"].fillna(learner_med).fillna(s["session_minutes"].median())
    log.add("Synthetic sessions", "missing value", "session_minutes is null", "learner median (+flag)",
            na.sum(), len(s), "Lecture 7: attribute mean/median within the same group harms distribution least; flag preserves the information.")
    return s.reset_index(drop=True), sig


def clean_submissions(sub: pd.DataFrame, log: CleaningLog) -> pd.DataFrame:
    n = len(sub)
    dup = sub.duplicated(["id_assessment", "id_student"])
    sub = sub[~dup]
    log.add("Synthetic submissions", "duplicate key", "(id_assessment, id_student) repeated", "drop", dup.sum(), n,
            "One submission per learner per assessment.")
    bad = ~sub["submit_hour"].between(0, 23)
    log.add("Synthetic submissions", "invalid value", "submit_hour outside 0..23", "none needed", bad.sum(), n, "Validity check passed.")
    return sub


def before_after(before: pd.DataFrame, after: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    for c in cols:
        for stage, d in (("before", before), ("after", after)):
            x = pd.to_numeric(d[c], errors="coerce")
            rows.append({"column": c, "stage": stage, "rows": len(d), "missing": int(x.isna().sum()),
                         "mean": x.mean(), "median": x.median(), "std": x.std(), "min": x.min(),
                         "p99": x.quantile(0.99), "max": x.max()})
    return pd.DataFrame(rows)
