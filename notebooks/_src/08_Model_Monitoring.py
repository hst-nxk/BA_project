# %% [markdown]
# # Module 7 — Model Monitoring
# MLOps role. After deployment the segmentation can silently go stale: learners change behaviour, courses are
# redesigned, the platform changes. This notebook implements the monitoring loop:
#
# 1. **Data drift** of every model input: PSI, Kolmogorov–Smirnov, and an **Evidently** DataDriftPreset report
# 2. **Cluster stability over time**: segment shares per cohort, centroid shift (Hungarian-matched refit), silhouette,
#    share of atypical learners
# 3. **Re-clustering triggers** (GREEN / AMBER / RED) and the decision for each batch
# 4. A **stress scenario** (simulated platform change) proving the triggers fire
# 5. Monitoring of live API traffic (prediction log)
#
# Reference = training cohorts 2013B–2014B. Incoming batch = the unseen 2014J cohort.

# %%
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent / "src")) if (Path.cwd().parent / "src").exists() else sys.path.insert(0, "src")
import numpy as np, pandas as pd, matplotlib.pyplot as plt, joblib
from sklearn.cluster import KMeans
from edtech import config as C, monitoring as MON, modeling as M, utils as U, viz as V
pd.set_option("display.max_columns", 40); pd.set_option("display.width", 200)

seg = pd.read_parquet(C.PROCESSED / "learner_segments.parquet")
model = joblib.load(C.MODELS / "segmentation_model.joblib")
FEATS = model.features()
m5 = U.load_metrics("m5")
train_sil = m5["temporal"]["silhouette_train"]
ref = seg[seg.split == "train"].reset_index(drop=True)
cur = seg[seg.split == "monitor"].reset_index(drop=True)
pd.DataFrame(MON.TRIGGERS, index=["threshold"]).T

# %% [markdown]
# ### Trigger definitions
# | Signal | AMBER | RED | Rationale |
# |---|---|---|---|
# | PSI per feature | > 0.10 | > 0.25 (and ≥ 2 features RED) | standard credit-risk / ML monitoring bands |
# | Segment share shift | > 5 pp | > 10 pp | a 10 pp move changes staffing of interventions |
# | Mean matched centroid shift | > 0.30 SD | > 0.50 SD | personas' meaning changes |
# | Atypical learners (beyond training p95 distance) | > 8% | > 15% | expected 5% by construction |
# | Silhouette drop vs training | — | > 0.05 | structure weakening |
#
# **Decision rule:** ≥ 2 RED checks → **RE-CLUSTER** (retrain on recent cohorts, re-validate personas);
# 1 RED or any AMBER → **INVESTIGATE**; otherwise **NO ACTION**.
#
# ## 1. Incoming cohort 2014J vs reference

# %%
s14 = MON.evaluate(model, ref, cur, train_sil, "2014J incoming cohort")
print("DECISION:", s14["decision"]); s14["checks"]

# %%
drift14 = pd.DataFrame(s14["feature_drift"]); U.save_table(drift14, "m7_drift_2014J"); drift14

# %%
pd.DataFrame(s14["segment_shares"])

# %%
ev14 = MON.evidently_report(ref[ref.active_day_fraction > 0], cur[cur.active_day_fraction > 0], FEATS,
                            C.MONITORING_REPORTS / "evidently_drift_2014J.html")
pd.DataFrame(ev14["metrics"]).head(12)

# %% [markdown]
# **2014J verdict: NO ACTION.** All PSI values are below 0.10, segment shares move by less than 5 pp, the refitted
# centroids sit 0.27 SD from production (below the 0.30 AMBER line) and 4.7% of learners are atypical — exactly the 5%
# expected. Note that the KS test rejects "same distribution" for almost every feature (p < 0.05): with ~10,000
# learners per batch even negligible differences are "significant". That is why the triggers use **effect sizes**
# (PSI, pp shifts, SD shifts) rather than p-values. Evidently, which uses the Wasserstein distance at this sample size,
# flags 2 of 10 columns individually but not the dataset (share 0.2 < 0.5).

# %% [markdown]
# ## 2. Cluster stability over time (all cohorts)
# Each presentation is scored with the frozen model; K-Means is also refitted on each cohort alone and matched to the
# production centroids to measure how far the "natural" clusters have moved.

# %%
rows = []
for p in ["2013B", "2013J", "2014B", "2014J"]:
    b = seg[seg.code_presentation == p]
    ba = b[b.active_day_fraction > 0]
    Z = model.pipeline.transform(ba)
    shift, _ = MON.centroid_shift(model.kmeans.cluster_centers_, Z)
    r = {"presentation": p, "enrollments": len(b), "silhouette": M.internal_scores(Z, model.kmeans.predict(Z))["silhouette"],
         "mean_centroid_shift_sd": shift, "atypical_share": b.loc[b.active_day_fraction > 0, "is_atypical"].mean(),
         "psi_active_day_fraction": MON.psi(ref.loc[ref.active_day_fraction > 0, "active_day_fraction"].to_numpy(), ba.active_day_fraction.to_numpy())}
    r.update(b.persona.value_counts(normalize=True).to_dict())
    rows.append(r)
over_time = pd.DataFrame(rows).round(4); U.save_table(over_time, "m7_stability_over_time"); over_time

# %%
personas = ["Non-starters", "At-risk Disengaged", "Last-minute Crammers", "Consistent Achievers"]
fig, ax = plt.subplots(1, 3, figsize=(17, 4.3))
for p in personas:
    ax[0].plot(over_time.presentation, over_time[p], marker="o", color=V.PERSONA_COLORS[p], label=p)
ax[0].set(title="Segment share by cohort (frozen model)", ylabel="share of enrollments", ylim=(0, 0.5)); ax[0].legend(fontsize=8)
ax[1].bar(over_time.presentation, over_time.mean_centroid_shift_sd, color=V.CATEGORICAL[0], width=0.6)
ax[1].axhline(MON.TRIGGERS["centroid_shift_amber"], color=V.CATEGORICAL[3], ls="--", lw=1, label="AMBER")
ax[1].axhline(MON.TRIGGERS["centroid_shift_red"], color=V.CATEGORICAL[7], ls="--", lw=1, label="RED")
ax[1].set(title="Mean centroid shift of a per-cohort refit", ylabel="standard deviations"); ax[1].legend()
ax[2].bar(over_time.presentation, over_time.atypical_share, color=V.CATEGORICAL[6], width=0.6)
ax[2].axhline(0.05, color=V.TEXT2, ls=":", lw=1, label="expected 5%")
ax[2].axhline(MON.TRIGGERS["atypical_share_red"], color=V.CATEGORICAL[7], ls="--", lw=1, label="RED")
ax[2].set(title="Share of atypical learners", ylabel="share"); ax[2].legend()
fig.tight_layout(); V.save(fig, "m7_stability_over_time"); plt.show()

# %% [markdown]
# The production segmentation is **stable across all four cohorts** (silhouette 0.19–0.22, atypical share 4–6%,
# refit centroids within 0.15–0.36 SD). There is, however, a slow **trend**: the Consistent Achiever share falls from
# 41% (2013B) to 29% (2014J) while Last-minute Crammers grow from 26% to 38%. Each cohort is within tolerance of the
# pooled reference, but the trend means the next cohort could breach the 10 pp threshold. Monitoring therefore also
# tracks the share series itself; the playbook asks for an investigation when the same segment moves in the same
# direction for three consecutive cohorts.

# %% [markdown]
# ## 3. Stress scenario — simulated platform change
# To demonstrate the triggers end-to-end we apply a documented shock to the 2014J batch: less frequent study
# (active-day fraction × 0.55–0.8), more night logins (+25 pp), more last-minute submissions (×1.8 + 10 pp),
# stronger deadline concentration (×1.3–1.9), lower video completion (−15 pp) and later drop-off (+10 pp recency).

# %%
stress = MON.simulate_drift(cur, np.random.default_rng(C.SEED))
s_stress = MON.evaluate(model, ref, stress, train_sil, "stress scenario (simulated)")
print("DECISION:", s_stress["decision"]); s_stress["checks"]

# %%
drift_s = pd.DataFrame(s_stress["feature_drift"]); U.save_table(drift_s, "m7_drift_stress"); drift_s

# %%
ev_s = MON.evidently_report(ref[ref.active_day_fraction > 0], stress[stress.active_day_fraction > 0], FEATS,
                            C.MONITORING_REPORTS / "evidently_drift_stress_scenario.html")
fig, ax = plt.subplots(1, 2, figsize=(15, 4.5))
x = np.arange(len(FEATS))
ax[0].barh(x - 0.2, drift14.psi, height=0.4, color=V.CATEGORICAL[0], label="2014J (real)")
ax[0].barh(x + 0.2, drift_s.psi, height=0.4, color=V.CATEGORICAL[7], label="stress scenario")
ax[0].axvline(0.10, color=V.CATEGORICAL[3], ls="--", lw=1); ax[0].axvline(0.25, color=V.CATEGORICAL[7], ls="--", lw=1)
ax[0].set_yticks(x, FEATS); ax[0].set(title="PSI per feature vs reference", xlabel="PSI"); ax[0].legend()
sh = pd.DataFrame(s14["segment_shares"]).set_index("segment_id")[["reference_share", "current_share"]]
sh["stress_share"] = pd.DataFrame(s_stress["segment_shares"]).set_index("segment_id")["current_share"]
names = seg.groupby("segment_id").persona.first()
sh.index = [names[i] for i in sh.index]
sh.plot.bar(ax=ax[1], color=[V.GRID, V.CATEGORICAL[0], V.CATEGORICAL[7]], width=0.75, rot=15)
ax[1].set(title="Segment shares: reference vs 2014J vs stress", ylabel="share")
fig.tight_layout(); V.save(fig, "m7_drift_comparison"); plt.show()

# %% [markdown]
# **Stress verdict: RE-CLUSTER.** Six features exceed PSI 0.25, a segment share moves ~20 pp, the refitted centroids
# move 1.8 SD and atypical learners rise to 12%. Four of five checks fire, so the rule requires re-clustering: the
# frozen personas no longer describe the population and interventions planned on them would be mis-targeted.
# The silhouette stays GREEN — a reminder that a model can stay internally "well separated" while becoming wrong for
# the business, which is why several independent signals are combined.

# %% [markdown]
# ## 4. Live API traffic
# Every `/predict` call is appended to `data/monitoring/prediction_log.jsonl`. The same PSI checks run on the log
# once it holds enough requests (≥ 500); the dashboard shows the latest status.

# %%
log_p = C.MONITORING_DATA / "prediction_log.jsonl"
live = pd.read_json(log_p, lines=True) if log_p.exists() else pd.DataFrame()
print(f"logged predictions: {len(live)}")
if len(live) >= 500:
    live_drift = MON.feature_drift(ref[ref.active_day_fraction > 0], live, FEATS); display(live_drift)
else:
    print("Not enough live traffic yet for a statistically meaningful drift check (minimum 500).")

# %%
summary = {"triggers": MON.TRIGGERS, "latest": s14, "stress_test": s_stress,
           "over_time": over_time.to_dict("records"),
           "evidently_reports": {"2014J": "evidently_drift_2014J.html", "stress": "evidently_drift_stress_scenario.html"},
           "evidently_2014J": ev14["metrics"][:1], "evidently_stress": ev_s["metrics"][:1]}
(C.MONITORING_DATA / "monitoring_summary.json").write_text(json.dumps(summary, indent=2, default=float))
U.save_metrics("m7", {"decision_2014J": s14["decision"], "checks_2014J": s14["checks"], "decision_stress": s_stress["decision"],
                      "checks_stress": s_stress["checks"], "features_red_2014J": s14["features_red"],
                      "features_red_stress": s_stress["features_red"], "max_shift_pp_2014J": s14["max_segment_shift_pp"],
                      "max_shift_pp_stress": s_stress["max_segment_shift_pp"], "centroid_shift_2014J": s14["mean_centroid_shift_sd"],
                      "centroid_shift_stress": s_stress["mean_centroid_shift_sd"], "atypical_2014J": s14["atypical_share"],
                      "atypical_stress": s_stress["atypical_share"], "over_time": over_time.to_dict("records"),
                      "evidently_2014J": ev14["metrics"][:1], "evidently_stress": ev_s["metrics"][:1]})
{k: (s14[k], s_stress[k]) for k in ["decision", "features_red", "max_segment_shift_pp", "mean_centroid_shift_sd", "atypical_share", "silhouette_current"]}
