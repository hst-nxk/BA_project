"""Synthetic behavioural extension (Module 1).

OULAD records daily resource clicks and assessment submissions but not login sessions,
time of day, video completion, forum posts or hour-level submission timeliness.
This module simulates those behaviours with Faker + numpy using *conditional* business
logic anchored to each learner's real OULAD record:

* login sessions are placed on the learner's real active VLE days (plus a few
  click-less "check-in" logins), so login frequency inherits the real engagement rhythm;
* the hour of day follows a chronotype drawn from age band, shifted later for learners
  whose real data show deadline-driven ("cramming") activity;
* video completion is a learner propensity driven by real engagement and real scores;
* forum posts are proportional to real forum-resource clicks and a sociability trait;
* submission hours refine the real OULAD submission day relative to the real deadline.

Realistic data-quality defects are injected on purpose (duplicates, inconsistent
labels, missing values, impossible values, bot-like accounts). They are recorded in
``_injected_defects.json`` so that Module 3 cleaning rules can be validated against
known ground truth. Nothing synthetic is written back to the OULAD tables.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from faker import Faker

from . import config as C

CHRONOTYPES = np.array(["morning", "afternoon", "evening", "night"])
CHRONO_CENTER = np.array([8.5, 14.0, 20.0, 23.5])
CHRONO_PROBS = {  # by OULAD age_band
    "0-35": [0.15, 0.25, 0.35, 0.25],
    "35-55": [0.30, 0.30, 0.30, 0.10],
    "55<=": [0.45, 0.35, 0.15, 0.05],
}
DEVICES = np.array(["desktop", "mobile", "tablet"])
DEVICE_PROBS = {"0-35": [0.45, 0.45, 0.10], "35-55": [0.62, 0.26, 0.12], "55<=": [0.70, 0.12, 0.18]}

MESSY_DEVICE_LABELS = {
    "desktop": ["Desktop", "DESKTOP", " desktop", "PC", "computer"],
    "mobile": ["Mobile", "MOBILE", "mobile ", "phone", "Smartphone"],
    "tablet": ["Tablet", "TABLET", "tab", "iPad"],
}


def _rank(s: pd.Series) -> np.ndarray:
    return s.rank(pct=True, method="average").fillna(0).to_numpy()


def learner_traits(enr: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Latent learner-level traits (kept only for audit; never used as model features)."""
    n = len(enr)
    eng = _rank(enr["active_day_fraction"])
    perf = _rank(enr["avg_score_scheduled"])
    # Real-data signals of deadline-driven study: late-period click share, bursty weeks,
    # few early submissions.  Blend + noise -> cramming propensity in [0, 1].
    cram_signal = (0.50 * _rank(enr["deadline_click_ratio"]) + 0.25 * _rank(enr["weekly_cv_active_span"])
                   + 0.25 * (1 - _rank(enr["mean_days_early"].fillna(0))))
    cram = np.clip(cram_signal + rng.normal(0, 0.12, n), 0, 1)

    probs = np.vstack([CHRONO_PROBS.get(a, CHRONO_PROBS["0-35"]) for a in enr["age_band"]])
    shift = 0.20 * (cram > 0.7)                                  # crammers drift to night
    probs[:, 3] += shift
    probs[:, 0] -= shift / 2
    probs[:, 1] -= shift / 2
    probs = np.clip(probs, 0.01, None)
    probs /= probs.sum(1, keepdims=True)
    chrono_idx = (probs.cumsum(1) > rng.random(n)[:, None]).argmax(1)

    dprobs = np.vstack([DEVICE_PROBS.get(a, DEVICE_PROBS["0-35"]) for a in enr["age_band"]])
    primary_device = (dprobs.cumsum(1) > rng.random(n)[:, None]).argmax(1)

    # Video-completion propensity: Beta with mean driven by real engagement and score.
    m = np.clip(0.30 + 0.35 * eng + 0.25 * perf - 0.12 * cram, 0.05, 0.97)
    kappa = 10.0
    video_p = rng.beta(m * kappa, (1 - m) * kappa)
    sociability = rng.gamma(shape=1.2, scale=1.0, size=n) * (0.5 + eng)
    # Session habit: some learners open many short sessions per study day, others stay
    # logged in (log-normal multiplier on sessions per active day).
    session_habit = rng.lognormal(mean=0.0, sigma=0.6, size=n)
    return pd.DataFrame({
        **{k: enr[k].to_numpy() for k in C.KEY},
        "trait_session_habit": session_habit,
        "trait_engagement_rank": eng, "trait_performance_rank": perf,
        "trait_cramming": cram, "trait_chronotype": CHRONOTYPES[chrono_idx],
        "trait_primary_device": DEVICES[primary_device], "trait_video_propensity": video_p,
        "trait_sociability": sociability,
    })


def generate_sessions(enr: pd.DataFrame, day: pd.DataFrame, traits: pd.DataFrame,
                      deadlines: pd.DataFrame, rng: np.random.Generator, fake: Faker) -> pd.DataFrame:
    tr = traits.set_index(C.KEY)
    # ---------- 1. sessions on real active VLE days ----------------------------------------
    d = day[day["clicks"] > 0][C.KEY + ["date", "clicks", "content_clicks", "forum_clicks"]].copy()
    d = d.join(tr["trait_session_habit"], on=C.KEY)
    lam = np.clip(d["trait_session_habit"].to_numpy() * (0.25 + 0.004 * d["clicks"].to_numpy()), 0, 6)
    n_sess = 1 + rng.poisson(lam)
    rep = np.repeat(np.arange(len(d)), n_sess)
    s = d.iloc[rep].reset_index(drop=True)
    k = np.repeat(n_sess, n_sess)
    s["clicks_in_session"] = s["clicks"] / k
    s["content_in_session"] = s["content_clicks"] / k
    s["forum_in_session"] = s["forum_clicks"] / k
    s["source"] = "vle_day"
    s = s.drop(columns=["trait_session_habit"])

    # ---------- 2. click-less "check-in" logins --------------------------------------------
    e = enr[C.KEY + ["module_presentation_length", "date_registration", "date_unregistration",
                     "active_day_fraction", "no_activity"]].copy()
    mu = np.where(e["no_activity"] == 1, 1.5,
                  0.05 * np.ceil((e["module_presentation_length"] + 1) / 7) * (0.5 + e["active_day_fraction"]))
    n_extra = rng.poisson(mu)
    rep2 = np.repeat(np.arange(len(e)), n_extra)
    x = e.iloc[rep2].reset_index(drop=True)
    lo = np.minimum(x["date_registration"].fillna(0), 0).clip(lower=-30).to_numpy()
    hi = x["date_unregistration"].fillna(x["module_presentation_length"]).clip(upper=x["module_presentation_length"]).to_numpy()
    hi = np.where(x["no_activity"] == 1, np.minimum(hi, lo + 40), hi)   # ghosts log in early only
    hi = np.maximum(hi, lo + 1)
    x["date"] = np.floor(lo + rng.random(len(x)) * (hi - lo)).astype(int)
    for c in ["clicks", "content_clicks", "forum_clicks", "clicks_in_session", "content_in_session", "forum_in_session"]:
        x[c] = 0.0
    x["source"] = "check_in"
    s = pd.concat([s, x[s.columns]], ignore_index=True)

    # ---------- 3. attach traits ------------------------------------------------------------
    s = s.join(tr[["trait_cramming", "trait_chronotype", "trait_primary_device",
                   "trait_video_propensity", "trait_sociability"]], on=C.KEY)
    n = len(s)

    # proximity to a real assessment deadline (within 2 days before it)
    dl = deadlines[["code_module", "code_presentation", "date"]].dropna().rename(columns={"date": "deadline"}).astype({"deadline": int})
    near_days = pd.concat([dl.assign(date=dl["deadline"] - k) for k in (0, 1, 2)])
    near_days = near_days[["code_module", "code_presentation", "date"]].drop_duplicates().assign(near_deadline=1)
    s = s.merge(near_days, on=["code_module", "code_presentation", "date"], how="left")
    s["near_deadline"] = s["near_deadline"].fillna(0).astype(int)

    # ---------- 4. login hour ---------------------------------------------------------------
    center = pd.Series(CHRONO_CENTER, index=CHRONOTYPES)[s["trait_chronotype"]].to_numpy()
    center = center + 2.5 * s["near_deadline"].to_numpy() * s["trait_cramming"].to_numpy()
    hour_f = np.mod(center + rng.normal(0, 2.0, n), 24)
    s["login_hour"] = np.floor(hour_f).astype(int)
    minute = np.floor((hour_f % 1) * 60).astype(int)

    start = pd.to_datetime(s["code_presentation"].map(C.PRESENTATION_START))
    s["login_ts"] = start + pd.to_timedelta(s["date"], unit="D") + pd.to_timedelta(s["login_hour"], unit="h") \
        + pd.to_timedelta(minute, unit="m") + pd.to_timedelta(rng.integers(0, 60, n), unit="s")

    # ---------- 5. device, duration, videos, forum -----------------------------------------
    keep_primary = rng.random(n) < 0.75
    s["device_type"] = np.where(keep_primary, s["trait_primary_device"], DEVICES[rng.integers(0, 3, n)])
    base = 6 + 1.1 * s["clicks_in_session"].to_numpy()
    s["session_minutes"] = np.round(np.exp(rng.normal(np.log(base), 0.45)), 1)
    lam_v = np.clip(0.15 + 0.06 * s["content_in_session"].to_numpy(), 0, 8)
    s["videos_started"] = rng.poisson(lam_v)
    p = np.clip(s["trait_video_propensity"].to_numpy() + rng.normal(0, 0.08, n), 0.01, 0.99)
    s["videos_completed"] = rng.binomial(s["videos_started"].to_numpy(), p)
    lam_f = np.clip(0.02 * s["forum_in_session"].to_numpy() * s["trait_sociability"].to_numpy(), 0, 6)
    s["forum_posts"] = rng.poisson(lam_f)
    s["forum_replies"] = rng.binomial(s["forum_posts"].to_numpy(), 0.55)

    pool = np.array([fake.user_agent() for _ in range(3000)])
    s["user_agent"] = pool[rng.integers(0, len(pool), n)]
    s["session_id"] = [fake.uuid4() for _ in range(n)]
    cols = ["session_id", *C.KEY, "date", "login_ts", "login_hour", "device_type", "user_agent",
            "session_minutes", "videos_started", "videos_completed", "forum_posts", "forum_replies",
            "near_deadline", "source"]
    s = s[cols].rename(columns={"date": "course_day"})
    return s.sort_values(["id_student", "code_module", "code_presentation", "login_ts"]).reset_index(drop=True)


def generate_submissions(sa: pd.DataFrame, assessments: pd.DataFrame, traits: pd.DataFrame,
                         rng: np.random.Generator) -> pd.DataFrame:
    """Hour-level timestamp for every real OULAD non-exam, non-banked submission.

    Deadline is assumed to be 23:59 on the OULAD deadline day.
    """
    x = sa.merge(assessments, on="id_assessment", how="left")
    x = x[(x["assessment_type"] != "Exam") & (x["is_banked"] == 0) & x["date"].notna()].copy()
    x = x.merge(traits[C.KEY + ["trait_cramming", "trait_chronotype"]], on=C.KEY, how="left")
    n = len(x)
    on_deadline_day = (x["date_submitted"] == x["date"]).to_numpy()
    cram = x["trait_cramming"].fillna(0.5).to_numpy()
    chrono_center = pd.Series(CHRONO_CENTER, index=CHRONOTYPES)[x["trait_chronotype"].fillna("evening")].to_numpy()
    hour = np.where(on_deadline_day,
                    np.clip(rng.normal(17 + 5 * cram, 2.5), 0, 23.98),     # deadline-day rush
                    np.mod(chrono_center + rng.normal(0, 2.5, n), 24))
    x["submit_hour"] = np.floor(hour).astype(int)
    x["hours_before_deadline"] = np.round((x["date"] - x["date_submitted"]) * 24 + (23.983 - hour), 2)
    start = pd.to_datetime(x["code_presentation"].map(C.PRESENTATION_START))
    x["submitted_ts"] = start + pd.to_timedelta(x["date_submitted"], unit="D") + pd.to_timedelta(hour, unit="h")
    x["deadline_ts"] = start + pd.to_timedelta(x["date"], unit="D") + pd.Timedelta(hours=23, minutes=59)
    return x[["id_assessment", *C.KEY, "assessment_type", "submitted_ts", "deadline_ts", "submit_hour",
              "hours_before_deadline"]].reset_index(drop=True)


def inject_defects(s: pd.DataFrame, enr: pd.DataFrame, rng: np.random.Generator, fake: Faker) -> tuple[pd.DataFrame, dict]:
    s = s.copy()
    n = len(s)
    log: dict = {}
    # 1. bot-like accounts: scripted logins at all hours, near-zero duration
    active = enr[enr["no_activity"] == 0]
    bots = active.sample(frac=0.004, random_state=C.SEED)[C.KEY + ["module_presentation_length"]]
    rows = []
    for _, b in bots.iterrows():
        m = int(rng.integers(400, 1500))
        days = np.sort(rng.integers(0, int(b["module_presentation_length"]) + 1, m))
        hours = rng.integers(0, 24, m)
        start = pd.Timestamp(C.PRESENTATION_START[b["code_presentation"]])
        rows.append(pd.DataFrame({
            "session_id": [fake.uuid4() for _ in range(m)], "id_student": b["id_student"],
            "code_module": b["code_module"], "code_presentation": b["code_presentation"],
            "course_day": days, "login_ts": start + pd.to_timedelta(days, "D") + pd.to_timedelta(hours, "h"),
            "login_hour": hours, "device_type": "desktop", "user_agent": "python-requests/2.31.0",
            "session_minutes": np.round(rng.uniform(0, 0.4, m), 2), "videos_started": 0, "videos_completed": 0,
            "forum_posts": 0, "forum_replies": 0, "near_deadline": 0, "source": "bot"}))
    bot_df = pd.concat(rows, ignore_index=True)
    log["bot_enrollments"] = bots[C.KEY].astype({"id_student": int}).to_dict("records")
    log["bot_sessions"] = int(len(bot_df))
    s = pd.concat([s, bot_df], ignore_index=True)
    n = len(s)
    # 2. inconsistent device labels (8 % of rows)
    idx = rng.choice(n, int(0.08 * n), replace=False)
    s.loc[idx, "device_type"] = [rng.choice(MESSY_DEVICE_LABELS[v]) for v in s.loc[idx, "device_type"]]
    log["messy_device_labels"] = int(len(idx))
    # 3. missing values
    i_dur = rng.choice(n, int(0.02 * n), replace=False)
    s.loc[i_dur, "session_minutes"] = np.nan
    i_dev = rng.choice(n, int(0.01 * n), replace=False)
    s.loc[i_dev, "device_type"] = None
    log["missing_session_minutes"] = int(len(i_dur))
    log["missing_device_type"] = int(len(i_dev))
    # 4. impossible values
    i_neg = rng.choice(n, int(0.002 * n), replace=False)
    s.loc[i_neg, "session_minutes"] = -s.loc[i_neg, "session_minutes"].fillna(5).abs()
    i_big = rng.choice(n, int(0.001 * n), replace=False)
    s.loc[i_big, "session_minutes"] = rng.uniform(1500, 9000, len(i_big)).round(1)
    i_vid = rng.choice(np.flatnonzero(s["videos_started"].to_numpy() > 0), int(0.003 * n), replace=False)
    s.loc[i_vid, "videos_completed"] = s.loc[i_vid, "videos_started"] + rng.integers(1, 4, len(i_vid))
    log.update(negative_minutes=int(len(i_neg)), over_24h_minutes=int(len(i_big)), completed_gt_started=int(len(i_vid)))
    # 5. exact duplicate rows from double logging (0.8 %)
    dup = s.sample(frac=0.008, random_state=C.SEED + 1)
    s = pd.concat([s, dup], ignore_index=True).sample(frac=1, random_state=C.SEED + 2).reset_index(drop=True)
    log["exact_duplicate_rows"] = int(len(dup))
    log["final_rows"] = int(len(s))
    return s, log


def run(enr: pd.DataFrame, day: pd.DataFrame, t: dict[str, pd.DataFrame]) -> dict:
    rng = np.random.default_rng(C.SEED)
    Faker.seed(C.SEED)
    fake = Faker("en_GB")
    traits = learner_traits(enr, rng)
    sessions = generate_sessions(enr, day, traits, t["assessments"], rng, fake)
    subs = generate_submissions(t["studentAssessment"], t["assessments"], traits, rng)
    sessions_dirty, defects = inject_defects(sessions, enr, rng, fake)

    # generation-only helper columns are kept in the audit file, not in the published log
    sessions_dirty[["session_id", "source", "near_deadline"]].to_parquet(C.SYNTHETIC / "_session_source_audit.parquet", index=False)
    sessions_dirty = sessions_dirty.drop(columns=["source", "near_deadline"])
    sessions_dirty.to_parquet(C.SYNTHETIC / "synthetic_login_sessions.parquet", index=False)
    subs.to_parquet(C.SYNTHETIC / "synthetic_submission_events.parquet", index=False)
    traits.to_parquet(C.SYNTHETIC / "_latent_traits_audit.parquet", index=False)
    (C.SYNTHETIC / "_injected_defects.json").write_text(json.dumps(defects, indent=2, default=int))
    # small human-readable sample
    sessions_dirty.head(2000).to_csv(C.SYNTHETIC / "sample_login_sessions.csv", index=False)
    return {"sessions_clean_rows": len(sessions), "sessions_with_defects": len(sessions_dirty),
            "submission_events": len(subs), "defects": {k: v for k, v in defects.items() if k != "bot_enrollments"}}
