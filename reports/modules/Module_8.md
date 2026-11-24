# Module 8 — Recommendations, Limitations and Lecture Alignment


## 8.1 Recommendations to the business

1. **Replace blanket support with segment-targeted support.** The LP plan gives 3× the extra passes per dollar of
   segment-blind tutor calls.
2. **Act on Non-starters in the first 10 days.** They are 12.6% of enrollments, almost all withdraw, and they are
   detected with near-perfect recall. Use a welcome sequence and a tutor call.
3. **Fix pacing for Crammers, the largest segment (34%).** Mini-deadlines and study-plan coaching give the best return
   on tutor time. Measure the share of work submitted more than 24 h before the deadline.
4. **Alert on 14-day inactivity** to catch At-risk Disengaged learners before week 10. In month 1 they cannot be told
   apart from Crammers.
5. **Spend nothing extra on Consistent Achievers.** Offer peer-mentor roles that create support capacity for the other
   segments.
6. **Buy tutor hours before raising the budget.** Tutor time is the binding constraint, and a tutor hour is worth
   about $86 in expected retained value.
7. **Validate the uplift assumptions with an A/B test** per segment in the next cohort, then re-optimise.
8. **Watch the Achiever → Crammer trend.** If 2015B continues it, investigate course and assessment design changes.

## 8.2 Limitations

* **Synthetic layer.** Login, video, forum and submission-hour fields are plausible simulations, not observations.
  Their links to outcomes partly reflect the generation logic. Mitigations: they are anchored to real behaviour, every
  field is documented, and the OULAD-only model agrees at ARI 0.69. Replace them with real telemetry when available.
* **Retrospective segments.** They use the whole course. The day-28 early-warning model bridges this gap, but its
  recall for At-risk Disengaged is only 0.55.
* **Clicks are not learning time.** Course design changes activity levels, so profiles should be read in module
  context.
* **Moderate separation (silhouette 0.21).** Behaviour is a continuum. The segments are stable and actionable but not
  sharply separated, which is why the API reports a confidence and an atypical flag.
* **Assumed economics.** Uplifts, costs and the value of a pass are planning assumptions. Sensitivity analysis shows
  where decisions flip.
* **Duplicates in the click log** (7.4%) are kept because OULAD has no event id. Removing them changes window clicks
  by about 3%.

## 8.3 Alignment with the course lectures

| Lecture | Applied in |
|---|---|
| 1 Introduction to BA | analytics types per module; KPIs; measurement scales in the data dictionary |
| 2 Decision analysis, OLAP | EV decision tree per persona, sunk costs; DuckDB star schema with roll-up, drill-down, slice, dice, pivot |
| 3.1 DSS | data / model / UI subsystems: lookup store, model registry, dashboard + API |
| 3.2 Optimisation | LP allocation under budget and tutor-hour constraints; shadow prices; sensitivity curves |
| 4 Data warehouse, column store | Parquet + DuckDB columnar storage, fact and dimension tables |
| 5–6 Spark, Pair RDD, DataFrame | 10.6M-row ETL with DataFrame API; reduceByKey cross-check |
| 7 Data preparation | missing-value mechanisms and treatments, IQR / z outliers, multi-rule bot detection, Box-Cox / Yeo-Johnson, min-max vs z-score, one-hot / ordinal, PCA |
| 8 Data mining / KDD | notebook sequence follows the KDD stages |
| 8.1 Time series | trend and seasonality of weekly clicks, moving average, exponential smoothing, cohort tracking |
| 9 Clustering | K-Means, AGNES linkages, DBSCAN with k-distance; elbow, √(n/2), cross-validated SSE; silhouette, DB, CH, purity, NMI, ARI; stability |
| 10 Classification | surrogate decision tree; random-forest early warning with train/test split and confusion matrix |

A detailed mapping is in `docs/LECTURE_ALIGNMENT.md`.

## 8.4 Deliverables

| Deliverable | Location |
|---|---|
| Notebooks (one per module) | `notebooks/01 … 08_*.ipynb` (executed), sources in `notebooks/_src` |
| Reusable code | `src/edtech/` (ETL, synthetic generator, cleaning, features, modeling, monitoring, serving) |
| Data dictionary | `docs/DATA_DICTIONARY.xlsx`, `docs/DATA_DICTIONARY.md` |
| API / dashboard | `app/api/main.py`, `app/dashboard/app.py`; `render.yaml`, `Dockerfile` |
| Monitoring report | `monitoring/*.html` (Evidently), `docs/MONITORING_PLAYBOOK.md` |
| Model card | `docs/MODEL_CARD.md` |
| Tests | `tests/` (pytest: definitions, cleaning validity, API, dashboard) |
| Report and slides | `reports/Final_Report.pdf / .docx`, `reports/Final_Presentation.pptx`, `reports/modules/` |

## 8.5 Reproducibility

The pipeline runs end to end with `python scripts/run_notebooks.py`, in about 10 minutes on a laptop. Every random
step uses seed 42. Raw files are verified by SHA-256 (`data/raw/raw_manifest.json`). All numbers in this report are
read from `reports/metrics/*.json` and `reports/tables/*.csv`, which the notebooks produce.