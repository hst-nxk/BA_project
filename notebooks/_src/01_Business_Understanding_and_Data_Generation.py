# %% [markdown]
# # Module 1 — Business Understanding & Data Generation
# **Project:** Learner Behavior Segmentation for an Online Learning Platform
#
# **Business question.** *What distinct behavioural profiles exist among our learners, and how should
# content, pacing or support be tailored to each group?*
#
# This notebook (Data Engineer role) covers:
# 1. Business problem, decision makers and KPIs (engagement score, at-risk learner rate, intervention cost)
# 2. Acquisition and integrity check of the OULAD base data
# 3. Big-data ETL of the 10.6M-row clickstream with **PySpark** (DataFrame API + Pair RDD, Lectures 5–6)
# 4. Integration of the seven OULAD tables at enrollment grain
# 5. Generation of the **synthetic behavioural extension** (Faker + conditional business logic)
# 6. Validation of the synthetic data's realism and the **data dictionary**
#
# | Analytics type (Lecture 1) | Where it appears in this project |
# |---|---|
# | Descriptive | EDA, OLAP cube, cluster profiles (Modules 2, 5) |
# | Predictive | Segment assignment of new learners (API), surrogate decision tree, drift monitoring (Modules 5–7) |
# | Prescriptive | Decision-tree expected value and LP budget allocation of interventions (Module 8) |
#
# The workflow follows the KDD process (Lecture 8): *selection → preprocessing → transformation → data mining → interpretation/evaluation*.

# %%
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent / "src")) if (Path.cwd().parent / "src").exists() else sys.path.insert(0, "src")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from edtech import config as C, etl_spark, oulad, synthetic, data_dictionary, utils as U, viz as V
pd.set_option("display.max_columns", 40); pd.set_option("display.width", 200)
print("Project root:", C.ROOT)

# %% [markdown]
# ## 1. Business understanding
#
# **Decision makers:** Head of Learner Success (support budget), course designers (content & pacing),
# tutors (who to contact and when).
#
# **Decisions supported**
# * Which learners receive which support (nudge, coaching, tutor call, peer group)?
# * How should content and deadlines be paced for each behavioural group?
# * When is the segmentation outdated and must be rebuilt?
#
# **KPIs** (definitions are implemented in `src/edtech/features.py` and documented in the data dictionary)
#
# | KPI | Definition | Target direction |
# |---|---|---|
# | Engagement score (0–100) | 100 × (0.30·pct(active-day fraction) + 0.20·pct(logins/week) + 0.20·video completion + 0.20·submission rate + 0.10·pct(forum posts/week)) | ↑ |
# | At-risk learner rate | share of enrollments ending *Fail* or *Withdrawn* (outcome, never a model input) | ↓ |
# | Intervention cost | Σ learners × unit cost of the intervention they receive (catalogue below) | minimise per extra pass |
# | Cost per additional pass | intervention cost ÷ expected additional passes | ↓ |
# | Segment coverage | share of learners with a confident segment assignment | ↑ |

# %%
catalogue = pd.DataFrame([(k, *v) for k, v in C.INTERVENTION_CATALOGUE.items()],
                         columns=["intervention", "cost_usd_per_learner", "tutor_minutes", "assumed_base_uplift_pass_prob"])
catalogue

# %% [markdown]
# Costs and uplifts are **planning assumptions** (no causal data exists in OULAD); they are used only in the
# prescriptive analysis and are varied in a sensitivity analysis in Module 8. The value of one additional pass
# is assumed to be **USD 400** (contribution margin / retained fee).
#
# ## 2. Data acquisition and integrity
# OULAD (Kuzilek et al., 2017, CC BY 4.0) — the Kaggle mirror required by the brief has identical content to the
# UCI archive used here. File hashes are checked against the manifest recorded at download time.

# %%
manifest = json.loads((C.RAW / "raw_manifest.json").read_text())
rows = []
for name, meta in manifest["files"].items():
    p = C.RAW / name
    if p.suffix == ".csv" and p.exists():
        rows.append({"file": name, "bytes": p.stat().st_size, "sha256_matches_manifest": U.sha256(p) == meta["sha256"]})
integrity = pd.DataFrame(rows); integrity

# %%
t = oulad.read_raw()
shapes = pd.DataFrame([(k, *v.shape) for k, v in t.items()], columns=["table", "rows", "columns"])
shapes.loc[len(shapes)] = ["studentVle", 10_655_280, 6]
U.save_table(shapes, "m1_source_tables"); shapes

# %% [markdown]
# ### Measurement scales (Lecture 1 / Lecture 7)
# Knowing the scale decides the allowed operations and the encoding later on.

# %%
scales = pd.DataFrame([
    ("id_student, id_site, id_assessment", "Nominal (identifier)", "excluded from distances"),
    ("code_module, region, gender, disability, activity_type", "Nominal / categorical", "one-hot (profiling only)"),
    ("final_result", "Nominal outcome", "external validation only"),
    ("age_band, highest_education, imd_band", "Ordinal", "ordinal codes (profiling only)"),
    ("date, date_registration, deadlines (course-relative days)", "Interval", "differences only"),
    ("sum_click, score, session_minutes, counts", "Ratio", "log / power transform + z-score"),
], columns=["fields", "scale", "treatment"]); scales

# %% [markdown]
# ## 3. Big-data ETL with PySpark (Lectures 5–6)
# `studentVle.csv` has 10.6M rows (454 MB). Spark reads it with an explicit schema, broadcasts the small `vle`
# dimension (activity types), and aggregates with the **DataFrame API** to one row per enrollment × day. A
# **Pair RDD** job (`map → reduceByKey`) recomputes total clicks per enrollment as an independent check.

# %%
t0 = time.time()
etl = etl_spark.run(prefer_spark=True)
print(f"ETL finished in {time.time()-t0:.0f}s"); etl

# %%
assert etl["total_clicks_dataframe"] == etl.get("total_clicks_pair_rdd", etl["total_clicks_dataframe"])
day = pd.read_parquet(C.INTERIM / "enrollment_day.parquet")
print(day.shape); day.head()

# %% [markdown]
# ## 4. Integration at enrollment grain
# Grain = (id_student, code_module, code_presentation): 32,593 enrollments of 28,785 learners. Activity and
# assessments are aggregated **separately** and joined one-to-one, so joins never multiply rows
# (`validate="one_to_one"` asserts this).

# %%
enr = oulad.add_derived(oulad.build_enrollment_table(t))
assert enr.duplicated(C.KEY).sum() == 0 and len(enr) == len(t["studentInfo"])
enr.to_parquet(C.INTERIM / "oulad_enrollments_raw.parquet", index=False)
print(enr.shape, "| learners:", enr.id_student.nunique())
enr[C.KEY + ["final_result", "total_clicks", "active_days", "avg_score_scheduled", "deadline_click_ratio"]].head()

# %% [markdown]
# ### Baseline KPIs (before any segmentation)

# %%
at_risk = enr["at_risk_outcome"].mean()
kpi_course = (enr.groupby(["course_domain", "code_module"])
              .agg(enrollments=("id_student", "size"), at_risk_rate=("at_risk_outcome", "mean"),
                   withdrawal_rate=("is_withdrawn", "mean"), no_activity_rate=("no_activity", "mean"))
              .round(3).reset_index())
blanket_cost = enr["at_risk_outcome"].sum() * C.INTERVENTION_CATALOGUE["tutor_outreach_call"][0]
print(f"At-risk learner rate: {at_risk:.1%}")
print(f"Blanket tutor-call cost if every eventually-at-risk learner were called: ${blanket_cost:,.0f} "
      f"(and at-risk status is only known at the end, so this is not actionable)")
U.save_metrics("m1", {"n_enrollments": len(enr), "n_learners": int(enr.id_student.nunique()), "at_risk_rate": at_risk,
                      "withdrawal_rate": enr.is_withdrawn.mean(), "pass_rate": enr.is_pass.mean(),
                      "no_activity_enrollments": int(enr.no_activity.sum()), "blanket_cost": blanket_cost, "etl": etl})
U.save_table(kpi_course, "m1_kpi_by_course"); kpi_course

# %% [markdown]
# ## 5. Synthetic behavioural extension
#
# OULAD lacks login sessions, time of day, video completion, forum **posts** and hour-level submission
# timeliness. We simulate them with **conditional generation anchored to each learner's real record**:
#
# | Required behaviour (brief) | Synthetic field(s) | Anchor in real OULAD data |
# |---|---|---|
# | Login frequency | sessions per active day (log-normal habit × clicks) + check-in logins | real active VLE days and daily clicks |
# | Time-of-day pattern | `login_hour` via chronotype mixture | age band; deadline-driven clicking (cramming trait) |
# | Video completion rate | `videos_started`, `videos_completed` | content clicks; engagement & score ranks |
# | Days enrollment → first activity | first `login_ts` vs `date_registration` | real first active day |
# | Forum / discussion participation | `forum_posts`, `forum_replies` | real `forumng` clicks × sociability |
# | Assignment submission timeliness | `submit_hour`, `hours_before_deadline` | real submission and deadline days |
#
# Faker supplies session UUIDs and user-agent strings; numpy (seed 42) supplies all distributions. Realistic
# **defects are injected on purpose** (duplicates, label noise, missing and impossible values, 0.4% bot accounts)
# so Module 3 can prove its cleaning rules against known ground truth.

# %%
t0 = time.time()
syn = synthetic.run(enr, day, t)
print(f"generated in {time.time()-t0:.0f}s"); syn

# %%
sessions = pd.read_parquet(C.SYNTHETIC / "synthetic_login_sessions.parquet")
subs = pd.read_parquet(C.SYNTHETIC / "synthetic_submission_events.parquet")
traits = pd.read_parquet(C.SYNTHETIC / "_latent_traits_audit.parquet")
audit = pd.read_parquet(C.SYNTHETIC / "_session_source_audit.parquet")
print(sessions.shape, subs.shape); sessions.head()

# %% [markdown]
# ## 6. Validating the realism of the synthetic data
# Checks performed on the log after excluding the injected bot accounts (the other injected defects are still
# present, which is why some plausibility checks are just below 100% — Module 3 removes them):
# 1. **Range / plausibility** — hours 0–23, durations positive and right-skewed, completed ≤ started.
# 2. **Coherence with real behaviour** — every real active VLE day has ≥ 1 login; synthetic aggregates correlate
#    with their real anchors *but not perfectly* (noise keeps them informative rather than copies).
# 3. **Behavioural face validity** — learners who eventually pass complete more videos and submit earlier.

# %%
clean = sessions.merge(audit, on="session_id")
clean = clean[clean["source"] != "bot"].drop_duplicates("session_id")
plaus = pd.DataFrame({
    "check": ["login_hour in 0..23", "session_minutes > 0 (non-missing)", "median session minutes",
              "videos_completed <= videos_started", "sessions per enrollment (median)"],
    "value": [clean.login_hour.between(0, 23).mean(), (clean.session_minutes.dropna() > 0).mean(),
              clean.session_minutes.median(), (clean.videos_completed <= clean.videos_started).mean(),
              clean.groupby(C.KEY).size().median()]})
plaus

# %%
agg = clean.groupby(C.KEY).agg(logins=("session_id", "size"), videos_started=("videos_started", "sum"),
                                videos_completed=("videos_completed", "sum"), posts=("forum_posts", "sum"),
                                night=("login_hour", lambda h: h.isin([22, 23, 0, 1, 2, 3, 4, 5]).mean())).reset_index()
chk = enr.merge(agg, on=C.KEY, how="left").fillna({"logins": 0, "posts": 0})
chk["vcr"] = chk.videos_completed / chk.videos_started.replace(0, np.nan)
sub_agg = subs.groupby(C.KEY)["hours_before_deadline"].median().rename("median_hbd").reset_index()
chk = chk.merge(sub_agg, on=C.KEY, how="left")
anchors = pd.DataFrame([
    ("logins vs real active days", chk[["logins", "active_days"]].corr("spearman").iloc[0, 1]),
    ("forum posts vs real forum clicks", chk[["posts", "forum_clicks"]].corr("spearman").iloc[0, 1]),
    ("video completion vs real avg score", chk[["vcr", "avg_score_scheduled"]].corr("spearman").iloc[0, 1]),
    ("median hours before deadline vs real mean days early", chk[["median_hbd", "mean_days_early"]].corr("spearman").iloc[0, 1]),
    ("night-login share vs real deadline click ratio", chk[["night", "deadline_click_ratio"]].corr("spearman").iloc[0, 1]),
], columns=["synthetic vs real anchor", "spearman_rho"]).round(3)
U.save_table(anchors, "m1_synthetic_anchor_correlations"); anchors

# %%
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
tr = traits.set_index(C.KEY)
h = clean.join(tr["trait_chronotype"], on=C.KEY)
for ch, col in zip(["morning", "afternoon", "evening", "night"], V.CATEGORICAL):
    ax[0].hist(h.loc[h.trait_chronotype == ch, "login_hour"], bins=24, range=(0, 24), histtype="step", lw=2, color=col, label=ch, density=True)
ax[0].set(title="Login hour by simulated chronotype", xlabel="hour of day", ylabel="density"); ax[0].legend()
ax[1].hist(np.log10(clean.session_minutes.dropna().clip(lower=0.1)), bins=60, color=V.CATEGORICAL[0])
ax[1].set(title="Session duration (log10 minutes)", xlabel="log10(minutes)", ylabel="sessions")
order = ["Distinction", "Pass", "Fail", "Withdrawn"]
data = [chk.loc[chk.final_result == o, "vcr"].dropna() for o in order]
b = ax[2].boxplot(data, tick_labels=order, showfliers=False, patch_artist=True)
for patch, o in zip(b["boxes"], order): patch.set_facecolor(V.OUTCOME_COLORS[o]); patch.set_alpha(0.6)
ax[2].set(title="Video completion rate by real final result", ylabel="completed / started")
V.save(fig, "m1_synthetic_validation"); plt.show()

# %%
fig, ax = plt.subplots(figsize=(7, 4))
x = subs.merge(traits[C.KEY + ["trait_cramming"]], on=C.KEY)
x["cram_group"] = pd.cut(x.trait_cramming, [0, 0.33, 0.66, 1.0], labels=["low", "medium", "high"], include_lowest=True)
for g, col in zip(["low", "medium", "high"], [V.CATEGORICAL[0], V.CATEGORICAL[3], V.CATEGORICAL[1]]):
    v = x.loc[(x.cram_group == g) & x.hours_before_deadline.between(-48, 168), "hours_before_deadline"]
    ax.hist(v, bins=72, histtype="step", lw=2, density=True, label=f"{g} cramming propensity", color=col)
ax.axvline(0, color=V.TEXT2, lw=1, ls="--"); ax.axvline(24, color=V.TEXT2, lw=1, ls=":")
ax.set(title="Submission lead time (hours before 23:59 deadline)", xlabel="hours before deadline (negative = late)", ylabel="density")
ax.legend(); V.save(fig, "m1_submission_lead_time"); plt.show()

# %% [markdown]
# **Reading the checks.**
# * **Logins vs real active days (ρ ≈ 0.99)** — by design: a login is required for every active day. Login frequency
#   is therefore consistent with the real data but redundant with `active_day_fraction`; it is kept for profiling and
#   the API, not as a separate clustering input.
# * **Hours before deadline vs real days early (ρ ≈ 0.91)** — the day is real, the synthetic part is the *hour*,
#   which is what separates a submission at 09:00 from one at 23:30 on deadline day (the "last-minute" signal).
# * **Forum posts vs forum clicks (ρ ≈ 0.82)** — posting is proportional to forum presence but varies by sociability.
# * **Video completion vs score (ρ ≈ 0.49)** and **night logins vs deadline clicking (ρ ≈ 0.22)** — moderate links:
#   these fields add behavioural information rather than copying OULAD.
#
# Deadline-driven learners submit inside the last 24 hours far more often, the behaviour the brief asks us to capture.
#
# **Limitations.** Relationships between synthetic fields and outcomes are partly created by the generation logic
# (e.g., video propensity depends on score rank). They are therefore *plausible*, not *evidence*; Module 5 checks that
# the segmentation does not rely on synthetic fields alone (an OULAD-only model is compared).

# %%
U.save_metrics("m1", {"synthetic": syn, "anchor_correlations": dict(zip(anchors.iloc[:, 0], anchors.iloc[:, 1])),
                      "plausibility": dict(zip(plaus.check, plaus.value))})

# %% [markdown]
# ## 7. Data dictionary
# Every synthetic field is documented with name, type, unit, valid range and generation logic / assumption
# (`docs/DATA_DICTIONARY.xlsx` and `docs/DATA_DICTIONARY.md`).

# %%
dd = data_dictionary.build()
print(len(dd), "documented fields"); dd.head(12)
