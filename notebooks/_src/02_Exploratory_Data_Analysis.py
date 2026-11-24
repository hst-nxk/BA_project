# %% [markdown]
# # Module 2 — Exploratory Data Analysis
# Data Analyst role. EDA is run on the **integrated, not yet cleaned** data (real OULAD + raw synthetic logs), so
# that anomalies discovered here motivate the cleaning rules in Module 3.
#
# Contents
# 1. Univariate distributions, skewness and outliers (IQR and z-score rules, Lecture 7)
# 2. Correlations among engagement and performance variables
# 3. Comparisons across **course type**, **cohort** (presentation) and **enrollment period**
# 4. Required relationships: login frequency vs final score; video completion vs submission timeliness
# 5. Behaviour vs final outcome
# 6. Time-series view of weekly engagement (moving average, exponential smoothing — Lecture 8.1)
# 7. OLAP on a star schema in DuckDB (roll-up, drill-down, slice, dice, pivot — Lectures 2 & 4)

# %%
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent / "src")) if (Path.cwd().parent / "src").exists() else sys.path.insert(0, "src")
import numpy as np, pandas as pd, matplotlib.pyplot as plt, duckdb
from scipy import stats
from edtech import config as C, features as F, utils as U, viz as V
pd.set_option("display.max_columns", 40); pd.set_option("display.width", 200)

enr = pd.read_parquet(C.INTERIM / "oulad_enrollments_raw.parquet")
sessions_raw = pd.read_parquet(C.SYNTHETIC / "synthetic_login_sessions.parquet")
subs_raw = pd.read_parquet(C.SYNTHETIC / "synthetic_submission_events.parquet")
# raw (uncleaned) synthetic aggregates - deliberately no deduplication / bot removal here
df = F.integrate(enr, F.session_aggregates(sessions_raw, enr), F.submission_aggregates(subs_raw))
print(df.shape)

# %% [markdown]
# ## 1. Univariate distributions and outliers

# %%
num = ["total_clicks", "active_day_fraction", "clicks_per_active_day", "weekly_cv_active_span", "deadline_click_ratio",
       "recency_fraction", "submission_rate", "avg_score_scheduled", "mean_score", "logins_per_week",
       "mean_session_minutes", "video_completion_rate", "forum_posts_per_week", "night_login_share",
       "last_minute_submission_share", "days_to_first_login"]
desc = df[num].describe(percentiles=[.05, .25, .5, .75, .95]).T
desc["skew"] = df[num].skew()
desc["missing_pct"] = 100 * df[num].isna().mean()
U.save_table(desc.round(3).reset_index(names="feature"), "m2_descriptive_stats"); desc.round(2)

# %%
fig, axes = plt.subplots(4, 4, figsize=(16, 12))
for ax, c in zip(axes.flat, num):
    x = df[c].dropna()
    ax.hist(x.clip(upper=x.quantile(0.995)), bins=50, color=V.CATEGORICAL[0])
    ax.set_title(c, fontsize=10); ax.set_ylabel("enrollments")
fig.suptitle("Distributions (upper 0.5% clipped for display only)", x=0.01, ha="left", fontweight="bold")
fig.tight_layout(); V.save(fig, "m2_distributions"); plt.show()

# %% [markdown]
# Activity counts are strongly right-skewed (skew of total clicks ≈ 3.0): a log or Yeo-Johnson transform is needed
# before distance-based clustering (Lecture 7: *spreading the histogram*). Scores are left-skewed with a spike at 0
# for learners who never submitted.
#
# ### Outlier detection: boxplot (IQR) and z-score rules

# %%
rows = []
for c in num:
    x = df[c].dropna()
    q1, q3 = x.quantile([.25, .75]); iqr = q3 - q1
    z = (x - x.mean()) / x.std()
    rows.append({"feature": c, "mild_outliers_1.5IQR": int(((x < q1 - 1.5*iqr) | (x > q3 + 1.5*iqr)).sum()),
                 "extreme_outliers_3IQR": int(((x < q1 - 3*iqr) | (x > q3 + 3*iqr)).sum()),
                 "abs_z_gt_3": int((z.abs() > 3).sum()), "max": x.max()})
outl = pd.DataFrame(rows); U.save_table(outl, "m2_outliers"); outl

# %%
fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
a = df[df.no_activity == 0]
ax[0].scatter(a.active_day_fraction, a.logins_per_week, s=4, alpha=0.25, color=V.CATEGORICAL[0])
ax[0].set(title="Login frequency vs real active days (raw log)", xlabel="active-day fraction (OULAD)", ylabel="logins per week (synthetic, raw)")
sus = a[a.logins_per_week > a.logins_per_week.quantile(0.995)]
ax[0].scatter(sus.active_day_fraction, sus.logins_per_week, s=14, color=V.CATEGORICAL[7], label="top 0.5% login rate")
ax[0].legend()
m = sessions_raw["session_minutes"]
ax[1].hist(m.clip(-50, 300).dropna(), bins=120, color=V.CATEGORICAL[0])
ax[1].set(title="Raw session duration: negatives, zeros and > 24 h values", xlabel="minutes (display clipped to -50..300)", ylabel="sessions", yscale="log")
V.save(fig, "m2_anomalies"); plt.show()
anom = {"negative_minutes": int((m < 0).sum()), "over_1440_minutes": int((m > 1440).sum()), "missing_minutes": int(m.isna().sum()),
        "exact_duplicate_rows": int(sessions_raw.duplicated().sum()),
        "distinct_device_labels": sessions_raw.device_type.dropna().nunique(),
        "enrollments_above_20_logins_per_week": int((df.logins_per_week > 20).sum())}
print(sessions_raw.device_type.value_counts(dropna=False).head(20)); anom

# %% [markdown]
# **Anomalies found (input to Module 3):** a small group of enrollments with login rates far above what their real
# activity explains (bot-like), negative and > 24 h durations, duplicate rows and 15+ spellings of three device types.
#
# ## 2. Correlations among engagement and performance variables
# Spearman (rank) correlation is used because distributions are skewed and relationships monotone but non-linear.

# %%
corr_cols = ["total_clicks", "active_day_fraction", "logins_per_week", "clicks_per_active_day", "weekly_cv_active_span",
             "deadline_click_ratio", "recency_fraction", "submission_rate", "avg_score_scheduled", "on_time_rate_scheduled",
             "video_completion_rate", "forum_posts_per_week", "night_login_share", "last_minute_submission_share"]
cm = df[df.no_activity == 0][corr_cols].corr("spearman")
fig, ax = plt.subplots(figsize=(11, 9))
im = ax.imshow(cm, cmap=V.DIVERGING, vmin=-1, vmax=1)
ax.set_xticks(range(len(corr_cols)), corr_cols, rotation=60, ha="right"); ax.set_yticks(range(len(corr_cols)), corr_cols)
for i in range(len(corr_cols)):
    for j in range(len(corr_cols)):
        ax.text(j, i, f"{cm.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7, color=V.TEXT)
ax.grid(False); fig.colorbar(im, shrink=0.7, label="Spearman ρ")
ax.set_title("Spearman correlations (learners with activity)"); V.save(fig, "m2_correlation_heatmap"); plt.show()
U.save_table(cm.round(3).reset_index(names="feature"), "m2_spearman")

# %% [markdown]
# * A **volume/frequency block** (clicks, active days, logins, submission, score) is highly inter-correlated
#   (ρ 0.7–0.99): one or two representatives are enough for clustering (Lecture 9: minimal redundancy).
# * **Pacing** variables — deadline click ratio, burstiness within the active span, last-minute submissions, night
#   logins — are nearly independent of volume (|ρ| < 0.25). They describe *how* someone studies, not *how much*,
#   which is exactly what distinguishes a "crammer" from a "steady" learner with similar totals.
#
# ## 3. Behaviour across course type, cohort and enrollment period

# %%
df["registration_period"] = pd.cut(df.date_registration, [-1000, -60, -15, 0, 1000],
                                   labels=["early (>60 d before)", "standard (15-60 d)", "late (<15 d)", "after start"])
def comp(by):
    g = df.groupby(by, observed=True)
    return g.agg(enrollments=("id_student", "size"), median_active_day_fraction=("active_day_fraction", "median"),
                 median_logins_per_week=("logins_per_week", "median"), mean_avg_score=("avg_score_scheduled", "mean"),
                 mean_last_minute_share=("last_minute_submission_share", "mean"),
                 withdrawal_rate=("is_withdrawn", "mean"), at_risk_rate=("at_risk_outcome", "mean")).round(3)
by_domain, by_module, by_cohort, by_period = comp("course_domain"), comp("code_module"), comp("code_presentation"), comp("registration_period")
for name, tb in [("domain", by_domain), ("module", by_module), ("cohort", by_cohort), ("registration_period", by_period)]:
    U.save_table(tb.reset_index(), f"m2_by_{name}")
by_domain

# %%
by_cohort

# %%
by_period

# %%
tests = []
for grp in ["course_domain", "code_module", "code_presentation", "registration_period", "start_month"]:
    for v in ["active_day_fraction", "avg_score_scheduled", "last_minute_submission_share"]:
        samples = [s[v].dropna() for _, s in df.groupby(grp, observed=True)]
        h, p = stats.kruskal(*samples)
        n = sum(len(s) for s in samples); k = len(samples)
        tests.append({"grouping": grp, "variable": v, "kruskal_H": h, "p_value": p, "epsilon_squared": (h - k + 1) / (n - k)})
tests = pd.DataFrame(tests).round(4); U.save_table(tests, "m2_kruskal_tests"); tests

# %%
fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
mods = sorted(df.code_module.unique())
ax[0].boxplot([df.loc[df.code_module == m, "active_day_fraction"] for m in mods], tick_labels=mods, showfliers=False)
ax[0].set(title="Active-day fraction by module", ylabel="share of course days active")
cohorts = ["2013B", "2013J", "2014B", "2014J"]
w = by_cohort.reindex(cohorts)
ax[1].bar(cohorts, w.withdrawal_rate, color=V.CATEGORICAL[7], width=0.6)
ax[1].set(title="Withdrawal rate by cohort", ylabel="share withdrawn")
p = by_period
ax[2].bar(range(len(p)), p.at_risk_rate, color=V.CATEGORICAL[1], width=0.6)
ax[2].set_xticks(range(len(p)), p.index, rotation=15); ax[2].set(title="At-risk rate by registration timing", ylabel="share Fail + Withdrawn")
fig.tight_layout(); V.save(fig, "m2_group_comparisons"); plt.show()

# %% [markdown]
# * **Course type:** STEM learners are active on more days (median 17.8% vs 10.6% of course days) yet are more
#   often at risk (55% vs 48%) — STEM modules demand more activity, so raw activity must be read in course context.
#   Module is the grouping with the largest (still small, ε² ≈ 0.05) effect on activity.
# * **Cohort:** withdrawal rose from 27–29% in the 2013 presentations to 34% in 2014 — a real temporal shift that
#   the monitoring module (Module 7) must be able to detect.
# * **Enrollment period:** learners who register very early (> 60 days before start) withdraw *more* often (35.8%)
#   than standard registrants (27.4%), consistent with speculative early sign-ups; the small late-registration
#   group is not more at risk. All group differences are statistically significant (Kruskal–Wallis p < 0.05) but the
#   effect sizes are small (ε² ≤ 0.05): individual behaviour varies far more *within* groups than *between* them —
#   which is exactly why behavioural segmentation is needed rather than course- or cohort-level policies.
#
# ## 4. Required relationships
# *Login frequency vs final score* and *video completion vs submission timeliness*.

# %%
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
a = df[(df.no_activity == 0) & (df.logins_per_week < 20)]
hb = ax[0].hexbin(a.logins_per_week, a.avg_score_scheduled, gridsize=45, cmap=V.SEQUENTIAL, mincnt=1, bins="log")
bins = pd.cut(a.logins_per_week, np.arange(0, 12.5, 0.5))
med = a.groupby(bins, observed=True)["avg_score_scheduled"].median()
ax[0].plot([b.mid for b in med.index], med.values, color=V.CATEGORICAL[1], lw=2.5, label="median score")
ax[0].set(title="Login frequency vs score (all scheduled assessments)", xlabel="logins per week", ylabel="average score (non-submission = 0)", xlim=(0, 12))
ax[0].legend(); fig.colorbar(hb, ax=ax[0], label="enrollments (log)")
b = df[(df.submitted_assessments > 0) & df.median_hours_before_deadline.between(-72, 240)]
hb2 = ax[1].hexbin(b.video_completion_rate, b.median_hours_before_deadline, gridsize=45, cmap=V.SEQUENTIAL, mincnt=1, bins="log")
bins = pd.cut(b.video_completion_rate, np.linspace(0, 1, 21))
med2 = b.groupby(bins, observed=True)["median_hours_before_deadline"].median()
ax[1].plot([x.mid for x in med2.index], med2.values, color=V.CATEGORICAL[1], lw=2.5, label="median lead time")
ax[1].axhline(24, color=V.TEXT2, ls=":", lw=1)
ax[1].set(title="Video completion vs submission timeliness", xlabel="video completion rate", ylabel="median hours before deadline")
ax[1].legend(); fig.colorbar(hb2, ax=ax[1], label="enrollments (log)")
V.save(fig, "m2_required_relationships"); plt.show()
rel = {"rho_logins_score": a[["logins_per_week", "avg_score_scheduled"]].corr("spearman").iloc[0, 1],
       "rho_video_leadtime": b[["video_completion_rate", "median_hours_before_deadline"]].corr("spearman").iloc[0, 1],
       "rho_logins_exam": df[["logins_per_week", "exam_score"]].dropna().corr("spearman").iloc[0, 1]}
rel

# %% [markdown]
# Score rises steeply up to about 3–4 logins per week (median ≈ 75–80) and then flattens — diminishing returns
# (overall ρ ≈ 0.74). Getting low-frequency learners to a *regular minimum* matters more than pushing frequent
# learners further. Video completion is only weakly related to submission lead time (ρ ≈ 0.14): the typical
# learner submits within the last 0–24 hours *whatever* their video completion, so timeliness is a separate
# behavioural dimension, not a by-product of engagement. (Values above 1 on the x-axis are the injected
# "completed > started" defect, removed in Module 3.)
#
# ## 5. Behaviour vs final outcome

# %%
order = ["Distinction", "Pass", "Fail", "Withdrawn"]
out = df.groupby("final_result")[["active_day_fraction", "logins_per_week", "avg_score_scheduled", "video_completion_rate",
                                  "forum_posts_per_week", "last_minute_submission_share", "deadline_click_ratio",
                                  "recency_fraction", "no_activity"]].median().reindex(order)
out.insert(0, "enrollments", df.final_result.value_counts().reindex(order))
U.save_table(out.round(3).reset_index(), "m2_by_outcome"); out.round(3)

# %% [markdown]
# ## 6. Weekly engagement as a time series (Lecture 8.1)
# Average clicks per enrollment per course week, by final outcome, with a 3-week **moving average** and
# **exponential smoothing** (α = 0.3). Vertical lines mark TMA deadlines of module FFF 2014J as an example.

# %%
day = pd.read_parquet(C.INTERIM / "enrollment_day.parquet", columns=C.KEY + ["date", "clicks"])
day = day[day.date.between(0, 270)].merge(df[C.KEY + ["final_result"]], on=C.KEY)
day["week"] = day.date // 7
n_by = df.final_result.value_counts()
ws = day.groupby(["final_result", "week"]).clicks.sum().unstack(0).div(n_by, axis=1).fillna(0)
fig, ax = plt.subplots(1, 2, figsize=(15, 4.5))
for o in order:
    s = ws[o]
    ax[0].plot(s.index, s.ewm(alpha=0.3).mean(), color=V.OUTCOME_COLORS[o], label=o)
ax[0].set(title="Weekly clicks per enrollment (exponential smoothing, α=0.3)", xlabel="course week", ylabel="clicks / enrollment")
ax[0].legend()
f = day[(day.code_module == "FFF") & (day.code_presentation == "2014J")].groupby("date").clicks.sum()
f = f.reindex(range(0, 270), fill_value=0) / (df.code_module.eq("FFF") & df.code_presentation.eq("2014J")).sum()
ax[1].plot(f.index, f.values, color=V.GRID, lw=1, label="daily")
ax[1].plot(f.index, f.rolling(7, center=True).mean(), color=V.CATEGORICAL[0], label="7-day moving average")
asm = pd.read_csv(C.RAW / "assessments.csv", na_values="?")
for d in asm[(asm.code_module == "FFF") & (asm.code_presentation == "2014J") & (asm.assessment_type == "TMA")].date.dropna():
    ax[1].axvline(d, color=V.CATEGORICAL[1], ls="--", lw=1)
ax[1].set(title="FFF 2014J daily clicks per enrollment; dashed = TMA deadlines", xlabel="course day", ylabel="clicks / enrollment")
ax[1].legend(); V.save(fig, "m2_time_series"); plt.show()

# %% [markdown]
# The series has a **downward trend** (attrition), a **"seasonal" component tied to the assessment calendar**
# (peaks right before each TMA deadline) and irregular noise. The deadline peaks are the population-level trace of
# the cramming behaviour measured per learner by `deadline_click_ratio`. Learners who later withdraw or fail are
# already less than half as active as passers in **week 0–2**, so the gap is visible long before the outcome —
# the window in which intervention must happen.
#
# ## 7. OLAP on a star schema (DuckDB, column store)
# A small dimensional model makes multi-dimensional questions one SQL query away:
#
# * **fact_enrollment** — measures: clicks, active days, logins, posts, score, at-risk flag
# * **dim_learner** (age band, education, IMD, gender, disability) · **dim_course** (module, domain) ·
#   **dim_presentation** (year, start month) · **dim_registration** (registration period)
#
# DuckDB stores tables column-wise, which suits these scan-and-aggregate queries (Lecture 4).

# %%
con = duckdb.connect(str(C.WAREHOUSE / "learning_dw.duckdb"))
dim_learner = df[["id_student", "gender", "age_band", "highest_education", "imd_band", "disability", "region"]].drop_duplicates("id_student")
dim_course = df[["code_module", "course_domain"]].drop_duplicates()
dim_presentation = df[["code_presentation", "presentation_year", "start_month"]].drop_duplicates()
fact = df[C.KEY + ["registration_period", "final_result", "total_clicks", "active_days", "logins_total", "forum_posts",
                   "avg_score_scheduled", "at_risk_outcome", "is_withdrawn"]].astype({"registration_period": str})
for name, tb in [("dim_learner", dim_learner), ("dim_course", dim_course), ("dim_presentation", dim_presentation), ("fact_enrollment", fact)]:
    con.register("tmp", tb); con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM tmp"); con.unregister("tmp")
con.execute("SELECT table_name, estimated_size FROM duckdb_tables()").df()

# %%
# ROLL-UP: module -> domain -> all (GROUP BY ROLLUP)
rollup = con.execute("""
SELECT c.course_domain, f.code_module, COUNT(*) AS enrollments,
       ROUND(AVG(f.at_risk_outcome), 3) AS at_risk_rate, ROUND(AVG(f.total_clicks)) AS avg_clicks
FROM fact_enrollment f JOIN dim_course c USING (code_module)
GROUP BY ROLLUP (c.course_domain, f.code_module) ORDER BY 1 NULLS LAST, 2 NULLS LAST""").df()
rollup

# %%
# DRILL-DOWN + DICE: STEM courses, October presentations, by age band and education
dice = con.execute("""
SELECT l.age_band, l.highest_education, COUNT(*) AS n, ROUND(AVG(f.at_risk_outcome), 3) AS at_risk_rate,
       ROUND(AVG(f.logins_total), 1) AS avg_logins
FROM fact_enrollment f JOIN dim_learner l USING (id_student) JOIN dim_course c USING (code_module)
JOIN dim_presentation p USING (code_presentation)
WHERE c.course_domain = 'STEM' AND p.start_month = 'October'
GROUP BY 1, 2 HAVING COUNT(*) >= 50 ORDER BY at_risk_rate DESC""").df()
dice.head(10)

# %%
# SLICE (one presentation) + PIVOT (registration period x outcome)
pivot = con.execute("""
PIVOT (SELECT registration_period, final_result FROM fact_enrollment WHERE code_presentation = '2014J')
ON final_result USING COUNT(*) GROUP BY registration_period ORDER BY registration_period""").df()
con.close(); U.save_table(rollup, "m2_olap_rollup"); U.save_table(pivot, "m2_olap_pivot"); pivot

# %% [markdown]
# ## Key EDA insights
# 1. **Two independent behavioural axes** exist: *volume/regularity* (clicks, active days, logins, submissions,
#    score) and *pacing* (deadline concentration, burstiness, last-minute submission, night study). Segmenting on
#    both is what allows personas such as "last-minute crammer" vs "steady learner".
# 2. **Engagement relates to performance with diminishing returns** — the biggest score gains are between 0 and
#    ~3 logins per week.
# 3. **Risk is visible early**: eventually-withdrawn learners are less than half as active as passers from week 0;
#    withdrawal rose from 2013 to 2014 cohorts (a drift to monitor).
# 4. **Course context matters** (module ε² for activity), so profiles are always reported with their module mix.
# 5. **Data issues** to fix: duplicates, device label inconsistency, impossible durations, bot-like login counts,
#    and 4,093 enrollments without any in-window VLE activity (treated as a rule-based segment, not as noise).

# %%
U.save_metrics("m2", {"anomalies": anom, "relationships": rel, "skew_total_clicks": float(desc.loc["total_clicks", "skew"]),
                      "n_no_activity": int(df.no_activity.sum()),
                      "by_cohort_withdrawal": by_cohort.withdrawal_rate.to_dict(),
                      "by_period_at_risk": {str(k): v for k, v in by_period.at_risk_rate.items()},
                      "by_domain": by_domain.reset_index().to_dict("records")})
