# %% [markdown]
# # Module 4 — Feature Engineering
# ML Engineer (modelling) role.
#
# 1. Aggregate the **cleaned** synthetic logs and integrate them with OULAD (one row per enrollment)
# 2. Build **RFM-style** behavioural features: Recency of last activity, Frequency of activity, engagement
#    intensity ("Monetary"), plus consistency, pacing, performance, content, social and rhythm features
# 3. Encode categorical variables (nominal → binary, ordinal → codes; Lecture 7)
# 4. Transform skewed features and compare scalers (Min-Max vs Z-score vs robust; Lecture 7)
# 5. Select the feature set (variance, redundancy filter, interpretability) and check PCA (dimensionality reduction)
# 6. Fit the transformation pipeline **on training cohorts only** and save it for serving

# %%
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent / "src")) if (Path.cwd().parent / "src").exists() else sys.path.insert(0, "src")
import numpy as np, pandas as pd, matplotlib.pyplot as plt, joblib
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, PowerTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from edtech import config as C, features as F, utils as U, viz as V
pd.set_option("display.max_columns", 40); pd.set_option("display.width", 200); pd.set_option("display.max_colwidth", 80)

enr = pd.read_parquet(C.INTERIM / "oulad_enrollments_clean.parquet")
sess = pd.read_parquet(C.INTERIM / "sessions_clean.parquet")
subs = pd.read_parquet(C.INTERIM / "submissions_clean.parquet")

# %% [markdown]
# ## 1. Integration of cleaned sources

# %%
df = F.integrate(enr, F.session_aggregates(sess, enr), F.submission_aggregates(subs))
assert len(df) == len(enr) and df.duplicated(C.KEY).sum() == 0
df["split"] = np.where(df.code_presentation.isin(C.TRAIN_PRESENTATIONS), "train", "monitor")
df["population"] = np.where(df.no_activity == 1, "non-starter (rule-based)", "clustered")
print(df.shape)
pd.crosstab(df.population, df.split, margins=True)

# %% [markdown]
# **Population decision.** 4,093 enrollments (12.6%) have *no* VLE activity in the course window. They have no
# recency, no pacing and no intensity — any value we put into a distance calculation would be invented, and they
# would form a trivial cluster that distorts K-Means centroids (Lecture 9: K-Means is sensitive to such masses).
# They are therefore assigned by a transparent **business rule** to segment **S0 "Non-starters"**, and clustering is
# run on the 28,500 learners with activity. **Temporal split:** presentations 2013B–2014B train the model; 2014J
# (11,260 enrollments) is held out as the "incoming cohort" for validation and monitoring.
#
# ## 2. Candidate features (RFM-style and beyond)

# %%
cand = pd.DataFrame({"feature": list(F.CANDIDATES), "role": list(F.CANDIDATES.values())})
cand["source"] = np.where(cand.feature.isin(["logins_per_week", "login_week_coverage", "last_minute_submission_share",
                                              "median_hours_before_deadline", "video_completion_rate", "forum_posts_per_week",
                                              "night_login_share", "days_to_first_login", "mobile_share"]), "synthetic", "OULAD")
act = df[df.population == "clustered"]
cand["missing_pct"] = cand.feature.map(lambda c: round(100 * act[c].isna().mean(), 2))
cand["coef_variation"] = cand.feature.map(lambda c: round(act[c].std() / abs(act[c].mean()), 2))
cand["skew"] = cand.feature.map(lambda c: round(act[c].skew(), 2))
U.save_table(cand, "m4_candidate_features"); cand

# %% [markdown]
# ## 3. Encoding categorical variables
# * **Nominal → binary (1-to-N)** for module, domain, start month, gender, disability, age band and education
#   (Lecture 7: integer codes would invent an order and distances between categories).
# * **Ordinal → numeric** for IMD decile, education level and age band where order is meaningful.
#
# Demographics are **not** clustering inputs: segments must describe behaviour that the platform can influence, and
# using protected attributes in a targeting model raises fairness concerns. Encoded context is used to *profile*
# segments (Module 5) and to check for demographic skew.

# %%
onehot = F.one_hot_context(df)
ordinal = F.ordinal_context(df)
onehot.to_parquet(C.PROCESSED / "context_onehot.parquet", index=False)
ordinal.to_parquet(C.PROCESSED / "context_ordinal.parquet", index=False)
print("one-hot columns:", onehot.shape[1] - 3); ordinal.describe().round(2)

# %% [markdown]
# ## 4. Transformation and scaling
# ### 4a. Spreading skewed distributions (Lecture 7: log, Box-Cox, rank transforms)

# %%
skewed = ["total_clicks", "clicks_per_active_day", "weekly_cv_active_span", "deadline_click_ratio", "forum_posts_per_week", "logins_per_week"]
rows = []
for c in skewed:
    x = act[c].astype(float).dropna()
    r = {"feature": c, "raw": stats.skew(x), "log1p": stats.skew(np.log1p(x))}
    r["box_cox(x+1)"] = stats.skew(stats.boxcox(x + 1)[0])
    r["yeo_johnson"] = stats.skew(PowerTransformer(standardize=False).fit_transform(x.to_frame()).ravel())
    r["rank_normal"] = stats.skew(stats.norm.ppf((x.rank() - 0.5) / len(x)))
    rows.append(r)
sk = pd.DataFrame(rows).round(3); U.save_table(sk, "m4_skewness_by_transform"); sk

# %%
fig, ax = plt.subplots(2, 3, figsize=(15, 7))
for i, c in enumerate(["clicks_per_active_day", "deadline_click_ratio", "forum_posts_per_week"]):
    x = act[c].astype(float)
    ax[0, i].hist(x.clip(upper=x.quantile(.995)), bins=60, color=V.CATEGORICAL[7]); ax[0, i].set_title(f"{c} (raw)", fontsize=10)
    ax[1, i].hist(PowerTransformer(standardize=False).fit_transform(x.to_frame()).ravel(), bins=60, color=V.CATEGORICAL[0])
    ax[1, i].set_title(f"{c} (Yeo-Johnson)", fontsize=10)
fig.tight_layout(); V.save(fig, "m4_transformations"); plt.show()

# %% [markdown]
# Yeo-Johnson (the Box-Cox family generalised to zero values) brings skewness closest to 0 for all features and
# handles the many zeros in forum posts; λ is estimated on training data only. Rank-normal scores are excellent
# too but, as Lecture 7 notes, cannot be applied separately to training and new data, so they are unsuitable for a
# deployed model.
#
# ### 4b. Scaling: why Z-score

# %%
sel = F.SELECTED
Xr = act[sel].astype(float).fillna(act[sel].median())
Xt = Xr.copy(); sk_sel = [c for c in sel if c in F.SKEWED]
Xt[sk_sel] = PowerTransformer(standardize=False).fit_transform(Xt[sk_sel])
share = {}
for name, sc in [("none", None), ("min-max", MinMaxScaler()), ("z-score", StandardScaler()), ("robust (IQR)", RobustScaler())]:
    Z = Xt.to_numpy() if sc is None else sc.fit_transform(Xt)
    v = Z.var(0); share[name] = v / v.sum()
contrib = pd.DataFrame(share, index=sel).round(3)
U.save_table(contrib.reset_index(names="feature"), "m4_variance_share_by_scaler"); contrib

# %% [markdown]
# Without scaling, `avg_score_scheduled` (0–100 points) carries more than 99% of the variance and would dominate
# Euclidean distance (Lecture 7: normalisation prevents large-range attributes from outweighing
# small-range ones). Min-Max still leaves unequal weights because of outliers (it "does not eliminate outliers");
# **Z-score after Yeo-Johnson gives every feature equal weight (10%)**, which is the intended neutral starting point.
#
# ## 5. Feature selection
# ### 5a. Redundancy filter (|Spearman ρ| ≥ 0.80, Lecture 9: "minimal information redundancy")

# %%
red = F.correlation_filter(act, list(F.CANDIDATES), 0.80)
U.save_table(red, "m4_redundant_pairs"); red

# %%
decisions = pd.DataFrame([
    ("recency_fraction", "keep", "R of RFM; ρ = −0.80 with active days is at the threshold but measures a distinct concept (drop-off timing)"),
    ("active_day_fraction", "keep", "F of RFM; real data; represents the frequency block"),
    ("logins_per_week / login_week_coverage", "drop", "ρ ≥ 0.96 with active_day_fraction — same information (synthetic anchored to it)"),
    ("total_clicks / clicks_per_week", "drop", "ρ ≥ 0.92 with active_day_fraction"),
    ("clicks_per_active_day", "keep", "M of RFM: depth of each study day, ρ ≈ 0.3 with frequency"),
    ("weekly_click_cv", "replace", "ρ = −0.91 with frequency (low activity ⇒ high CV); replaced by weekly_cv_active_span"),
    ("weekly_cv_active_span", "keep", "burstiness inside the learner's own active period — independent of volume"),
    ("deadline_click_ratio", "keep", "pacing: concentration of study on deadline days (real)"),
    ("late_click_share", "drop", "mostly measures persistence (ρ 0.75 with submission), not cramming"),
    ("avg_score_scheduled", "keep", "performance; combines submission and marks"),
    ("submission_rate / on_time_rate_scheduled", "drop", "ρ 0.79–0.90 with avg_score_scheduled"),
    ("last_minute_submission_share", "keep", "timeliness at hour level (synthetic) — the crammer signal"),
    ("median_hours_before_deadline", "drop", "missing for non-submitters; overlaps last_minute share"),
    ("video_completion_rate", "keep", "content engagement (synthetic)"),
    ("forum_posts_per_week", "keep", "social engagement (synthetic)"),
    ("night_login_share", "keep", "study rhythm (synthetic); independent of all others"),
    ("days_to_first_login / mobile_share", "profile only", "onboarding & context — useful to describe, not to define, behaviour"),
], columns=["feature", "decision", "reason"])
U.save_table(decisions, "m4_selection_decisions"); decisions

# %%
corr_sel = act[sel].corr("spearman")
fig, ax = plt.subplots(figsize=(8.5, 7))
im = ax.imshow(corr_sel, cmap=V.DIVERGING, vmin=-1, vmax=1)
ax.set_xticks(range(len(sel)), sel, rotation=60, ha="right"); ax.set_yticks(range(len(sel)), sel)
for i in range(len(sel)):
    for j in range(len(sel)):
        ax.text(j, i, f"{corr_sel.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
ax.grid(False); fig.colorbar(im, shrink=0.7); ax.set_title("Selected features — Spearman ρ")
V.save(fig, "m4_selected_correlation"); plt.show()
print("max |rho| off-diagonal:", (corr_sel.abs() - np.eye(len(sel))).max().max().round(3))

# %% [markdown]
# ### 5b. Fit the pipeline on training cohorts only

# %%
train = df[(df.split == "train") & (df.population == "clustered")]
pipe = F.FeaturePipeline(sel).fit(train)
Z_train = pipe.transform(train)
print("train matrix:", Z_train.shape, "| mean≈0:", np.abs(Z_train.mean(0)).max().round(6), "| sd≈1:", Z_train.std(0).round(3))
joblib.dump(pipe, C.MODELS / "feature_pipeline.joblib")
params = pd.DataFrame({"feature": sel, "median_fill": [pipe.medians_[c] for c in sel],
                       "clip_low_p0.5": [pipe.clip_[c][0] for c in sel], "clip_high_p99.5": [pipe.clip_[c][1] for c in sel],
                       "yeo_johnson_lambda": [dict(zip(pipe.skewed, pipe.power.lambdas_)).get(c, np.nan) for c in sel],
                       "z_mean": pipe.scaler.mean_, "z_scale": pipe.scaler.scale_}).round(4)
U.save_table(params, "m4_pipeline_parameters"); params

# %% [markdown]
# ### 5c. Dimensionality reduction check (PCA)

# %%
pca = PCA().fit(Z_train)
ev = pd.DataFrame({"component": [f"PC{i+1}" for i in range(len(sel))], "explained_variance_ratio": pca.explained_variance_ratio_,
                   "cumulative": pca.explained_variance_ratio_.cumsum()}).round(3)
load = pd.DataFrame(pca.components_[:4].T, index=sel, columns=["PC1", "PC2", "PC3", "PC4"]).round(2)
fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
ax[0].bar(ev.component, ev.explained_variance_ratio, color=V.CATEGORICAL[0], width=0.6)
ax[0].plot(ev.component, ev.cumulative, color=V.CATEGORICAL[1], marker="o", label="cumulative")
ax[0].axhline(0.8, color=V.TEXT2, ls=":", lw=1); ax[0].set(title="PCA scree plot (training cohorts)", ylabel="explained variance"); ax[0].legend()
idx = np.random.default_rng(0).choice(len(Z_train), 6000, replace=False)
P = pca.transform(Z_train)[idx]
sc = ax[1].scatter(P[:, 0], P[:, 1], c=train.avg_score_scheduled.to_numpy()[idx], cmap=V.SEQUENTIAL, s=5, alpha=0.6)
ax[1].set(title="PC1 vs PC2 coloured by average score", xlabel="PC1", ylabel="PC2"); fig.colorbar(sc, ax=ax[1], label="avg score")
V.save(fig, "m4_pca"); plt.show()
U.save_table(ev, "m4_pca_explained"); U.save_table(load.reset_index(names="feature"), "m4_pca_loadings")
load

# %%
# Does clustering in a reduced PCA space help? Compare K-Means (k=4) on the full matrix vs on PCs reaching 80% variance.
n80 = int((pca.explained_variance_ratio_.cumsum() < 0.80).sum() + 1)
res = []
for name, M in [("all 10 standardised features", Z_train), (f"first {n80} PCs (80% var)", pca.transform(Z_train)[:, :n80])]:
    lab = KMeans(4, n_init=10, random_state=C.SEED).fit_predict(M)
    res.append({"space": name, "dims": M.shape[1], "silhouette_in_own_space": silhouette_score(M[idx], lab[idx]),
                "silhouette_in_original_space": silhouette_score(Z_train[idx], lab[idx])})
pca_cmp = pd.DataFrame(res).round(3); U.save_table(pca_cmp, "m4_pca_vs_full"); pca_cmp

# %% [markdown]
# PC1 is a general "engagement & performance" axis; PC2 is a "pacing" axis (deadline concentration, last-minute
# submissions, burstiness, night logins) — confirming the two behavioural dimensions found in EDA. Ten features are
# few, there is no severe multicollinearity left after selection, and reaching 80% of the variance still needs about
# half of the components (PC1 33%, PC2 19%). Clustering on the first 5 PCs looks better *in its own space*
# (silhouette 0.23 — dropped components shrink distances) but, judged in the original feature space, the partition
# is practically the same quality (0.179 vs 0.171) while centroids would lose their direct meaning. **Decision: cluster on the 10 standardised features; use PCA for
# visualisation** — this also keeps centroids directly interpretable for personas.
#
# ## 6. Save the feature store

# %%
Z_all = pipe.transform(df)
mat = pd.DataFrame(Z_all, columns=[f"z_{c}" for c in sel])
feat = pd.concat([df.reset_index(drop=True), mat], axis=1)
feat.to_parquet(C.PROCESSED / "learner_features.parquet", index=False)
print(feat.shape)

fig, ax = plt.subplots(figsize=(8, 4))
for o in ["Distinction", "Pass", "Fail", "Withdrawn"]:
    ax.hist(df.loc[df.final_result == o, "engagement_score"], bins=50, histtype="step", lw=2, color=V.OUTCOME_COLORS[o], label=o, density=True)
ax.set(title="Engagement score KPI by final result", xlabel="engagement score (0-100)", ylabel="density"); ax.legend()
V.save(fig, "m4_engagement_score"); plt.show()

U.save_metrics("m4", {"selected_features": sel, "n_clustered": int((df.population == "clustered").sum()),
                      "n_non_starters": int((df.population != "clustered").sum()),
                      "n_train_clustered": len(train), "n_monitor": int((df.split == "monitor").sum()),
                      "pca_explained": ev.explained_variance_ratio.tolist(), "pcs_for_80pct": n80,
                      "max_abs_rho_selected": float((corr_sel.abs() - np.eye(len(sel))).max().max()),
                      "engagement_score_by_outcome": df.groupby("final_result").engagement_score.median().round(1).to_dict(),
                      "pca_vs_full": pca_cmp.to_dict("records")})
