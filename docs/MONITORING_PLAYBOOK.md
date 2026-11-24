# Monitoring playbook

Implemented in `src/edtech/monitoring.py`, run by `notebooks/08_Model_Monitoring.ipynb`, shown on the dashboard's
**Monitoring** tab and served by `GET /monitoring/latest`.

## When to run
* **Every new cohort** (presentation) at day 28 and at course end, against the training reference (2013B–2014B).
* **Weekly** on live API traffic (`prediction_log.jsonl`) once at least 500 predictions have been logged.

## Signals and thresholds

| Check | Signal | AMBER | RED |
|---|---|---|---|
| Feature drift | Population Stability Index per model feature (10 quantile bins of the reference) | any feature > 0.10 | ≥ 2 features > 0.25 |
| Segment share | largest absolute change of a segment's share vs reference | > 5 pp | > 10 pp |
| Centroid shift | K-Means refit on the batch, matched to production centroids (Hungarian), mean distance in SD | > 0.30 | > 0.50 |
| Atypical learners | share with distance to nearest centroid > training p95 of that cluster (expected 5%) | > 8% | > 15% |
| Structure | silhouette of the production assignment, drop vs training | — | > 0.05 |

The KS test and the Evidently report (Wasserstein distance) are kept as supporting evidence. Their p-values are not
used as triggers: with batches of about 10,000 learners almost every difference is "significant".

## Decision rule and actions

| Outcome | Rule | Action |
|---|---|---|
| **NO ACTION** | all GREEN | Keep the model, archive the report |
| **INVESTIGATE** | 1 RED or any AMBER, **or** the same segment's share moves in the same direction for 3 consecutive cohorts | Analyst reviews drifted features and course changes (redesign, new assessment calendar, logging change). Check data pipeline errors first |
| **RE-CLUSTER** | ≥ 2 RED | Refit the pipeline and K-Means on the two most recent cohorts. Repeat the k-selection protocol (stability ≥ 0.80). Re-name personas with the archetype matcher. Review recommendations. Version the model and update the reference shares |

## Results so far
* **2014J (real, unseen):** all GREEN → NO ACTION (max PSI 0.07, max share shift 4.9 pp, centroid shift 0.27 SD,
  atypical 4.7%).
* **Trend to watch:** Consistent Achievers fell from 41% (2013B) to 29% (2014J) and Last-minute Crammers rose from 26%
  to 38%. If 2015B continues this, the 3-cohort rule triggers INVESTIGATE.
* **Stress scenario (simulated platform change):** 6 features RED, share shift ≈ 20 pp, centroid shift 1.8 SD →
  **RE-CLUSTER**.
