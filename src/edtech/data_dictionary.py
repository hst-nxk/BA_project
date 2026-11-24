"""Data dictionary (required deliverable): every synthetic field + every engineered feature."""
from __future__ import annotations

import pandas as pd

from . import config as C

COLS = ["table", "column", "source", "dtype", "unit", "valid_range", "generation_logic_or_definition", "business_assumption", "role"]

SYN = "Synthetic"
D = []


def add(table, column, source, dtype, unit, valid, logic, assumption="", role="raw"):
    D.append(dict(zip(COLS, [table, column, source, dtype, unit, valid, logic, assumption, role])))


# ------------------------------------------------------------------ synthetic_login_sessions
T = "synthetic_login_sessions"
add(T, "session_id", SYN, "string (UUID4)", "-", "unique", "Faker.uuid4() (seeded, Faker.seed(42)).", "One row per authenticated session.", "primary key")
add(T, "id_student", "OULAD key", "int", "-", "existing studentInfo id", "Copied from OULAD enrollment.", "", "foreign key")
add(T, "code_module", "OULAD key", "string", "-", "AAA..GGG", "Copied from OULAD enrollment.", "", "foreign key")
add(T, "code_presentation", "OULAD key", "string", "-", "2013B..2014J", "Copied from OULAD enrollment.", "", "foreign key")
add(T, "course_day", SYN + " (anchored)", "int", "days relative to course start", "-30 .. L", "Days with real VLE clicks receive 1 + Poisson(habit x (0.25 + 0.004 x clicks)) sessions; extra click-less check-in logins ~ Poisson(0.05 x weeks x (0.5 + active_day_fraction)) on uniform random days before unregistration; non-starters ~ Poisson(1.5) in their first 40 days.", "Every day with recorded clicks requires at least one login; heavier days need more sessions; some logins produce no clicks.")
add(T, "login_ts", SYN, "datetime", "local time", "within presentation calendar", "Presentation start (1 Feb for B / 1 Oct for J, assumed) + course_day + login_hour + random minute/second.", "Exact OU start dates are not published; 1st of month assumed.")
add(T, "login_hour", SYN, "int", "hour of day", "0..23", "floor((chronotype centre + 2.5 x cramming x near_deadline + N(0, 2.0)) mod 24). Chronotype centres: morning 8.5, afternoon 14, evening 20, night 23.5; chronotype ~ Categorical(p | age band), shifted +0.20 to night for cramming > 0.7.", "Younger learners study later; deadline pressure pushes study later in the day.")
add(T, "device_type", SYN, "category", "-", "desktop | mobile | tablet (clean)", "75% learner's primary device (Categorical by age band: 0-35 = 45/45/10, 35-55 = 62/26/12, 55+ = 70/12/18), 25% uniform random. 8% of rows deliberately mislabelled ('PC', 'MOBILE', 'tab'...), 1% set missing.", "Device mix by age reflects common LMS usage patterns; label noise mimics multiple client apps.")
add(T, "user_agent", SYN, "string", "-", "-", "Sampled from a pool of 3,000 Faker.user_agent() strings; bot sessions use 'python-requests/2.31.0'.", "")
add(T, "session_minutes", SYN, "float", "minutes", "0 < x <= 1440 (clean)", "exp(N(log(6 + 1.1 x clicks_in_session), 0.45)). Defects: 2% missing, 0.2% negative, 0.1% > 1440.", "Duration grows with interaction volume; log-normal right tail.")
add(T, "videos_started", SYN, "int", "count", ">= 0", "Poisson(0.15 + 0.06 x content clicks in session), capped by clipping lambda at 8.", "Video lectures are part of content resources (oucontent/resource/url...).")
add(T, "videos_completed", SYN, "int", "count", "0..videos_started", "Binomial(videos_started, clip(video propensity + N(0, 0.08))). Propensity ~ Beta(m x 10, (1 - m) x 10), m = 0.30 + 0.35 x engagement rank + 0.25 x performance rank - 0.12 x cramming. Defect: 0.3% rows completed > started.", "More engaged and better-performing learners finish what they start.")
add(T, "forum_posts", SYN, "int", "count", ">= 0", "Poisson(0.02 x forum clicks in session x sociability), sociability ~ Gamma(1.2, 1) x (0.5 + engagement rank).", "OULAD 'forumng' clicks measure reading; posting is a minority behaviour proportional to forum presence.")
add(T, "forum_replies", SYN, "int", "count", "0..forum_posts", "Binomial(forum_posts, 0.55).", "About half of posts are replies to others.")
add(T, "(bot accounts)", SYN + " defect", "-", "-", "-", "0.4% of active enrollments receive 400-1,500 extra sessions: uniform hours, 0-0.4 minutes, automation user agent, no videos/posts.", "Scripted scraping / credential-sharing traffic present on real platforms.", "quality test")
add(T, "(duplicate rows)", SYN + " defect", "-", "-", "-", "0.8% of rows duplicated exactly and shuffled.", "Client retry / double logging.", "quality test")

# --------------------------------------------------------------- synthetic_submission_events
T = "synthetic_submission_events"
add(T, "id_assessment", "OULAD key", "int", "-", "existing assessments id", "One event per real OULAD non-exam, non-banked submission with a known deadline.", "", "foreign key")
add(T, "submitted_ts", SYN + " (anchored)", "datetime", "local time", "-", "Real OULAD submission day + synthetic hour.", "")
add(T, "deadline_ts", "OULAD + assumption", "datetime", "local time", "-", "Real OULAD deadline day at 23:59.", "Deadlines are assumed to close at 23:59.")
add(T, "submit_hour", SYN, "int", "hour", "0..23", "If submitted on the deadline day: clip(N(17 + 5 x cramming, 2.5), 0, 23.98); otherwise chronotype centre + N(0, 2.5).", "Deadline-day submissions concentrate in the evening, more so for deadline-driven learners.")
add(T, "hours_before_deadline", SYN + " (anchored)", "float", "hours", "negative = late", "(deadline_day - submitted_day) x 24 + (23.983 - submit_hour_fraction).", "")

# ------------------------------------------------------------------ latent traits (audit)
T = "_latent_traits_audit (not a model input)"
add(T, "trait_engagement_rank", SYN, "float", "percentile", "0..1", "Percentile rank of real OULAD active_day_fraction.", "", "generation driver")
add(T, "trait_performance_rank", SYN, "float", "percentile", "0..1", "Percentile rank of real avg_score_scheduled.", "", "generation driver")
add(T, "trait_cramming", SYN, "float", "index", "0..1", "clip(0.50 x rank(deadline_click_ratio) + 0.25 x rank(weekly_cv_active_span) + 0.25 x (1 - rank(mean_days_early)) + N(0, 0.12), 0, 1).", "Real deadline-concentrated clicking and late submissions indicate cramming.", "generation driver")
add(T, "trait_chronotype", SYN, "category", "-", "morning/afternoon/evening/night", "See login_hour.", "", "generation driver")
add(T, "trait_session_habit", SYN, "float", "multiplier", "> 0", "LogNormal(0, 0.6).", "Some learners open many short sessions, others stay logged in.", "generation driver")
add(T, "trait_video_propensity", SYN, "float", "probability", "0..1", "See videos_completed.", "", "generation driver")
add(T, "trait_sociability", SYN, "float", "index", "> 0", "Gamma(1.2, 1) x (0.5 + engagement rank).", "", "generation driver")

# ------------------------------------------------------------ enrollment-level features
T = "learner_features (enrollment grain)"
F = [
    ("logins_per_week", SYN + " aggregate", "float", "logins / week", ">= 0", "Cleaned sessions in [0, L] / ceil((L + 1) / 7).", "profiling, API"),
    ("login_week_coverage", SYN + " aggregate", "float", "share", "0..1", "Weeks with >= 1 login / course weeks.", "profiling"),
    ("night_login_share", SYN + " aggregate", "float", "share", "0..1", "Share of sessions starting 22:00-05:59.", "model feature"),
    ("preferred_login_period", SYN + " aggregate", "category", "-", "morning/afternoon/evening/night", "Modal period of login_hour.", "profiling"),
    ("mobile_share", SYN + " aggregate", "float", "share", "0..1", "Share of sessions on mobile.", "profiling"),
    ("mean_session_minutes", SYN + " aggregate", "float", "minutes", "> 0", "Mean cleaned session duration.", "profiling"),
    ("video_completion_rate", SYN + " aggregate", "float", "share", "0..1", "(videos_completed + 2 x global rate) / (videos_started + 2) - Bayesian smoothing; 0 for learners with no logins.", "model feature"),
    ("forum_posts_per_week", SYN + " aggregate", "float", "posts / week", ">= 0", "forum_posts / course weeks.", "model feature"),
    ("days_to_first_login", SYN + " aggregate", "float", "days", ">= 0", "First login course_day - date_registration (clipped at 0).", "profiling"),
    ("last_minute_submission_share", SYN + " aggregate", "float", "share", "0..1", "Share of submissions made 0-24 h before the deadline (0 if none submitted).", "model feature"),
    ("median_hours_before_deadline", SYN + " aggregate", "float", "hours", "any", "Median hours_before_deadline.", "profiling"),
    ("recency_fraction", "OULAD derived", "float", "share of course", "0..1", "(L - last active day) / (L + 1); 1 if never active.", "model feature (R)"),
    ("active_day_fraction", "OULAD derived", "float", "share", "0..1", "Active VLE days in [0, L] / (L + 1).", "model feature (F)"),
    ("clicks_per_active_day", "OULAD derived", "float", "clicks / active day", ">= 0", "total_clicks / active_days.", "model feature (M)"),
    ("weekly_cv_active_span", "OULAD derived", "float", "ratio", ">= 0", "SD / mean of weekly clicks between first and last active week.", "model feature"),
    ("deadline_click_ratio", "OULAD derived", "float", "ratio", ">= 0 (1 = uniform)", "Share of clicks on deadline day or 2 days before / share of course days that are such days.", "model feature"),
    ("avg_score_scheduled", "OULAD derived", "float", "points", "0..100", "Sum of current non-exam scores / scheduled non-exam assessments (non-submission = 0).", "model feature"),
    ("submission_rate", "OULAD derived", "float", "share", "0..1", "Current non-exam submissions / scheduled.", "profiling"),
    ("on_time_rate_scheduled", "OULAD derived", "float", "share", "0..1", "On-time submissions / scheduled.", "profiling"),
    ("total_clicks", "OULAD derived", "int", "clicks", ">= 0", "Sum of sum_click in [0, L].", "profiling"),
    ("late_click_share", "OULAD derived", "float", "share", "0..1", "Clicks in final third / total clicks.", "profiling"),
    ("engagement_score", "KPI", "float", "points", "0..100", "100 x (0.30 pct(active_day_fraction) + 0.20 pct(logins_per_week) + 0.20 video_completion_rate + 0.20 submission_rate + 0.10 pct(forum_posts_per_week)).", "KPI"),
    ("at_risk_outcome", "OULAD outcome", "int", "flag", "0/1", "final_result in {Fail, Withdrawn}. Never a model input.", "KPI / external validation"),
]
for c, s, dt, u, v, logic, role in F:
    add(T, c, s, dt, u, v, logic, "", role)


def build() -> pd.DataFrame:
    df = pd.DataFrame(D, columns=COLS)
    df.to_excel(C.DOCS / "DATA_DICTIONARY.xlsx", index=False, sheet_name="data_dictionary")
    lines = ["# Data Dictionary", "",
             "Every synthetically generated field and every engineered feature. `L` = module_presentation_length. "
             "Synthetic generation code: `src/edtech/synthetic.py` (seed 42). Anchored = derived from the learner's real OULAD record plus noise.", ""]
    for t, g in df.groupby("table", sort=False):
        lines += [f"## {t}", "", "| Column | Source | Type | Unit | Valid range | Generation logic / definition | Business assumption | Role |",
                  "|---|---|---|---|---|---|---|---|"]
        for _, r in g.iterrows():
            lines.append("| " + " | ".join(str(r[c]).replace("|", "/") for c in ["column", "source", "dtype", "unit", "valid_range",
                                                                                 "generation_logic_or_definition", "business_assumption", "role"]) + " |")
        lines.append("")
    (C.DOCS / "DATA_DICTIONARY.md").write_text("\n".join(lines), encoding="utf-8")
    return df
