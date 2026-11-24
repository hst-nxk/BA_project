# %% [markdown]
# # Module 3 — Data Quality Assessment & Cleaning
# Data Engineer role. Lecture 7: real data is *incomplete*, *noisy* and *inconsistent*; preparation must remove
# errors **without adding bias** ("the more guessing you have to do, the further away from the real data").
#
# 1. Quality assessment by dimension: completeness, uniqueness, validity, consistency, referential integrity
# 2. Missing-value analysis (what does each missing value *mean*?)
# 3. Outliers: course-specific IQR fences (univariate) and bot-like accounts (multi-rule, multivariate)
# 4. Cleaning with a full operation log
# 5. Before/after comparisons

# %%
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent / "src")) if (Path.cwd().parent / "src").exists() else sys.path.insert(0, "src")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from edtech import config as C, oulad, cleaning as K, features as F, utils as U, viz as V
pd.set_option("display.max_columns", 40); pd.set_option("display.width", 200); pd.set_option("display.max_colwidth", 90)

t = oulad.read_raw()
enr = pd.read_parquet(C.INTERIM / "oulad_enrollments_raw.parquet")
sessions = pd.read_parquet(C.SYNTHETIC / "synthetic_login_sessions.parquet")
subs = pd.read_parquet(C.SYNTHETIC / "synthetic_submission_events.parquet")

# %% [markdown]
# ## 1. Quality scorecard

# %%
keys = {"courses": ["code_module", "code_presentation"], "assessments": ["id_assessment"],
        "studentInfo": C.KEY, "studentRegistration": C.KEY, "studentAssessment": ["id_assessment", "id_student"],
        "vle": ["id_site"], "synthetic_login_sessions": ["session_id"],
        "synthetic_submission_events": ["id_assessment", "id_student"]}
tables = {**t, "synthetic_login_sessions": sessions, "synthetic_submission_events": subs}
score = K.quality_scorecard(tables, keys)
U.save_table(score, "m3_quality_scorecard"); score

# %%
ref = K.referential_checks(t, sessions); U.save_table(ref, "m3_referential_integrity"); ref

# %%
val = K.validity_checks_sessions(sessions)
val["violation_pct"] = (100 * val.violations / val.denominator).round(3)
U.save_table(val, "m3_validity_sessions"); val

# %%
cons = pd.DataFrame([
    ("imd_band uses two spellings for one decile ('10-20' vs 'x-y%')", int(t["studentInfo"].imd_band.eq("10-20").sum())),
    ("Withdrawn outcome without unregistration date, or date without Withdrawn",
     int(((enr.final_result == "Withdrawn") != enr.date_unregistration.notna()).sum())),
    ("Submission after course end (excluded from current-behaviour features)",
     int((t["studentAssessment"].merge(t["assessments"], on="id_assessment").merge(t["courses"], on=["code_module", "code_presentation"])
          .eval("date_submitted > module_presentation_length")).sum())),
    ("First recorded activity before registration date", int((enr.first_active_day < enr.date_registration).sum())),
    ("Synthetic device_type: 17 spellings of 3 classes", int(val.loc[val.rule.str.startswith("device_type not"), "violations"].iloc[0])),
], columns=["consistency check", "records"])
U.save_table(cons, "m3_consistency"); cons

# %% [markdown]
# **Summary of quality findings.** OULAD keys are unique and all foreign keys resolve (100% integrity); the
# issues are an inconsistent IMD label, administratively missing dates, 787k identical clickstream rows whose
# status cannot be decided (no event id), and cross-field disagreements. The synthetic log carries the realistic
# defects of an operational system: double-logged rows, inconsistent device labels, missing and impossible
# durations, completed-without-started videos and bot traffic.
#
# ## 2. Missing values — mechanism and treatment

# %%
miss = pd.concat([K.missing_profile(v, k) for k, v in tables.items()], ignore_index=True)
meaning = {
    "date": "Exam deadline not published (structural, MNAR) → keep NA; exams excluded from timeliness",
    "imd_band": "Deprivation band unknown (MAR on region) → explicit 'Unknown' + flag",
    "date_registration": "Registration day not recorded → NA + flag; only affects days-to-first-login",
    "date_unregistration": "Learner did not unregister (structural, not an error) → keep NA",
    "score": "Mark not recorded → excluded from mean; non-submission counted as 0 in avg_score_scheduled",
    "week_from": "Resource not planned for a specific week (structural) → not used",
    "week_to": "Resource not planned for a specific week (structural) → not used",
    "session_minutes": "Session not closed / logging gap (MCAR by design) → learner-median imputation + flag",
    "device_type": "Client did not report device → 'unknown' category",
}
miss["meaning_and_treatment"] = miss.column.map(meaning)
U.save_table(miss, "m3_missing_values"); miss

# %% [markdown]
# Lecture 7 warns against automatic replacement of missing values: most OULAD gaps are **structural** (a learner who
# never withdrew has no withdrawal date), so filling them would invent facts. Only `session_minutes` — a random
# logging gap — is imputed, using the learner's own median (the "same-class mean" method) with an indicator column.
#
# ## 3. Outliers
# ### 3a. Univariate: course-specific IQR fences on clicks
# Click volume depends on course design, so fences are computed within each module-presentation.

# %%
log = K.CleaningLog()
enr_clean = K.clean_oulad(t, enr, log)
o = enr_clean.groupby(["code_module", "code_presentation"]).agg(enrollments=("id_student", "size"),
                                                                 extreme_click_outliers=("clicks_extreme_outlier", "sum"))
print("Extreme click outliers:", int(enr_clean.clicks_extreme_outlier.sum()))
fig, ax = plt.subplots(figsize=(10, 4))
x = enr_clean[enr_clean.total_clicks > 0]
ax.boxplot([np.log1p(x.loc[x.code_module == m, "total_clicks"]) for m in sorted(x.code_module.unique())],
           tick_labels=sorted(x.code_module.unique()), flierprops={"markersize": 2, "alpha": 0.3})
ax.set(title="log(1 + clicks) by module — outliers are high but plausible activity", ylabel="log(1 + total clicks)")
V.save(fig, "m3_click_outliers"); plt.show()

# %% [markdown]
# These learners are genuinely very active (they pass at higher rates); deleting them would remove the most engaged
# learners. Decision: **keep and flag**, and reduce leverage with a power transform + winsorisation inside the
# feature pipeline (Lecture 7 options: "do nothing", "enforce bounds", "binning/transform").
#
# ### 3b. Multivariate: bot-like accounts
# No single rule is reliable: very active humans also have many sessions. We combine four signals and flag an
# enrollment when **at least two** fire:
#
# | Rule | Signal |
# |---|---|
# | R1 | session count > Q3 + 3·IQR within the course (extreme outlier) |
# | R2 | > 30% of sessions shorter than 1 minute |
# | R3 | > 10% of sessions from an automation user agent (python-requests, curl, bot…) |
# | R4 | login-hour entropy > 4.2 bits (humans sleep; uniform 24 h activity ≈ 4.58 bits) |

# %%
s_dedup = sessions.drop_duplicates()
sig = K.bot_signals(s_dedup, enr)
rules = ["r1_extreme_session_count", "r2_sub_minute_sessions", "r3_automation_user_agent", "r4_uniform_hours"]
truth = pd.DataFrame(json.loads((C.SYNTHETIC / "_injected_defects.json").read_text())["bot_enrollments"]).assign(is_bot=1)
m = sig.merge(truth, on=C.KEY, how="left").fillna({"is_bot": 0})
rule_eval = []
for r in rules + ["bot_flag"]:
    tp = int(((m[r] == 1) & (m.is_bot == 1)).sum()); fp = int(((m[r] == 1) & (m.is_bot == 0)).sum()); fn = int(((m[r] == 0) & (m.is_bot == 1)).sum())
    rule_eval.append({"rule": r, "flagged": tp + fp, "true_bots": tp, "false_positives": fp,
                      "precision": round(tp / max(tp + fp, 1), 3), "recall": round(tp / max(tp + fn, 1), 3)})
rule_eval = pd.DataFrame(rule_eval); U.save_table(rule_eval, "m3_bot_rule_evaluation"); rule_eval

# %%
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
hum, bot = m[m.bot_flag == 0], m[m.bot_flag == 1]
ax[0].scatter(hum.sessions, hum.hour_entropy_bits, s=4, alpha=0.2, color=V.CATEGORICAL[0], label="human")
ax[0].scatter(bot.sessions, bot.hour_entropy_bits, s=14, color=V.CATEGORICAL[7], label="flagged bot")
ax[0].axhline(4.2, color=V.TEXT2, ls="--", lw=1); ax[0].set_xscale("log")
ax[0].set(title="Session count vs login-hour entropy", xlabel="sessions (log scale)", ylabel="entropy of login hour (bits)"); ax[0].legend()
ax[1].scatter(hum.sessions, hum.share_sub_minute, s=4, alpha=0.2, color=V.CATEGORICAL[0], label="human")
ax[1].scatter(bot.sessions, bot.share_sub_minute, s=14, color=V.CATEGORICAL[7], label="flagged bot")
ax[1].axhline(0.30, color=V.TEXT2, ls="--", lw=1); ax[1].set_xscale("log")
ax[1].set(title="Session count vs share of sub-minute sessions", xlabel="sessions (log scale)", ylabel="share < 1 minute"); ax[1].legend()
V.save(fig, "m3_bot_detection"); plt.show()

# %% [markdown]
# R1 alone would also flag dozens of genuinely heavy human learners (false positives); requiring two independent
# signals isolates the automated accounts with no false positives against the injected ground truth. Treatment:
# **drop only the automated sessions** of flagged enrollments (their human sessions remain), instead of deleting
# the learner.
#
# ## 4. Cleaning

# %%
sess_clean, signals = K.clean_sessions(sessions, enr_clean, log)
subs_clean = K.clean_submissions(subs, log)
bot_eval = K.evaluate_bot_detection(signals)
cl = log.frame(); U.save_table(cl, "m3_cleaning_log"); cl

# %%
enr_clean.to_parquet(C.INTERIM / "oulad_enrollments_clean.parquet", index=False)
sess_clean.to_parquet(C.INTERIM / "sessions_clean.parquet", index=False)
subs_clean.to_parquet(C.INTERIM / "submissions_clean.parquet", index=False)
cl.to_csv(C.INTERIM / "cleaning_log.csv", index=False)
print(len(sessions), "->", len(sess_clean), "session rows")

# %% [markdown]
# ## 5. Before / after comparison

# %%
ba = K.before_after(sessions, sess_clean, ["session_minutes", "videos_started", "videos_completed", "forum_posts"])
U.save_table(ba.round(3), "m3_before_after_sessions"); ba.round(2)

# %%
dev = pd.DataFrame({"before": sessions.device_type.fillna("<missing>").value_counts(),
                    "after": sess_clean.device_type.value_counts()}).fillna(0).astype(int)
U.save_table(dev.reset_index(names="device_label"), "m3_device_before_after"); dev

# %%
agg_before = F.session_aggregates(sessions, enr)
agg_after = F.session_aggregates(sess_clean, enr_clean)
cmp_ = agg_before.merge(agg_after, on=C.KEY, suffixes=("_before", "_after"), how="outer")
for c in ["logins_total", "mean_session_minutes", "videos_completed", "videos_started"]:
    cmp_[[c + "_before", c + "_after"]] = cmp_[[c + "_before", c + "_after"]].fillna(0)
w = enr_clean.set_index(C.KEY)["observation_weeks"]
cmp_ = cmp_.join(w, on=C.KEY)
fig, ax = plt.subplots(1, 3, figsize=(16, 4.3))
ax[0].hist(np.log10(1 + cmp_.logins_total_before / cmp_.observation_weeks), bins=80, histtype="step", lw=2, label="before", color=V.CATEGORICAL[7])
ax[0].hist(np.log10(1 + cmp_.logins_total_after / cmp_.observation_weeks), bins=80, histtype="step", lw=2, label="after", color=V.CATEGORICAL[0])
ax[0].set(title="Logins per week", xlabel="log10(1 + logins/week)", ylabel="enrollments", yscale="log"); ax[0].legend()
vb = (cmp_.videos_completed_before / cmp_.videos_started_before.replace(0, np.nan)).dropna()
va = (cmp_.videos_completed_after / cmp_.videos_started_after.replace(0, np.nan)).dropna()
ax[1].hist(vb, bins=60, histtype="step", lw=2, label="before", color=V.CATEGORICAL[7], range=(0, 2))
ax[1].hist(va, bins=60, histtype="step", lw=2, label="after", color=V.CATEGORICAL[0], range=(0, 2))
ax[1].axvline(1, color=V.TEXT2, ls="--", lw=1)
ax[1].set(title="Raw video completion ratio", xlabel="completed / started", ylabel="enrollments"); ax[1].legend()
ax[2].hist(sessions.session_minutes.clip(-20, 120).dropna(), bins=70, histtype="step", lw=2, label="before", color=V.CATEGORICAL[7])
ax[2].hist(sess_clean.session_minutes.clip(-20, 120), bins=70, histtype="step", lw=2, label="after", color=V.CATEGORICAL[0])
ax[2].set(title="Session minutes (display clipped −20..120)", xlabel="minutes", ylabel="sessions"); ax[2].legend()
fig.tight_layout(); V.save(fig, "m3_before_after"); plt.show()

# %%
imd = pd.DataFrame({"before": t["studentInfo"].imd_band.fillna("<missing>").value_counts(),
                    "after": enr_clean.imd_band.value_counts()}).fillna(0).astype(int)
U.save_table(imd.reset_index(names="imd_band"), "m3_imd_before_after"); imd

# %%
summary = {
    "session_rows_before": len(sessions), "session_rows_after": len(sess_clean),
    "enrollments_before": len(enr), "enrollments_after": len(enr_clean),
    "bot_detection": bot_eval, "flagged_bot_enrollments": int(signals.bot_flag.sum()),
    "r1_only_false_positives": int(rule_eval.loc[rule_eval.rule == "r1_extreme_session_count", "false_positives"].iloc[0]),
    "imd_relabelled": int(t["studentInfo"].imd_band.eq("10-20").sum()), "imd_missing": int(t["studentInfo"].imd_band.isna().sum()),
    "duplicate_session_rows": int(sessions.duplicated().sum()),
    "click_extreme_outliers": int(enr_clean.clicks_extreme_outlier.sum()),
    "scorecard": score.to_dict("records"),
}
U.save_metrics("m3", summary); summary

# %% [markdown]
# ## Conclusions
# * **No enrollment is removed** (32,593 before and after): every cleaning action works on values or on
#   automated/duplicate *events*, so learner coverage — and therefore the at-risk population — is preserved.
# * The cleaned synthetic log loses ≈ 5% of its rows: double-logged rows and automated bot sessions.
# * Missingness is preserved where it carries meaning (structural dates, unknown IMD) and imputed only where it is
#   a random logging gap; every imputation is flagged.
# * Remaining uncertainty: 787k identical OULAD click rows are kept (no event id to prove duplication; dropping them
#   would change window clicks by ≈ 3%, a documented sensitivity).
