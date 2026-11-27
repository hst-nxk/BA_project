# Learner Behavior Segmentation for an Online Learning Platform

Business Analytics capstone. The project finds the behavioural segments of online learners and turns them into
tailored content, pacing and support. The segmentation model is deployed as an API and a dashboard and monitored for
drift.

**Repository:** https://github.com/hst-nxk/BA_project · **Live:** Dashboard https://learner-segmentation-dashboard.onrender.com · API https://learner-segmentation-api.onrender.com/docs

The written report and the slides are submitted separately. `reports/` (figures, tables, metrics) is regenerated
locally by the notebooks and is not tracked in git.

## Results at a glance

| Segment | Persona | Share | Pass rate | Recommended support |
|---|---|---|---|---|
| S0 | Non-starters | 12.6% | 0.1% | onboarding sequence + tutor call by day 10 |
| S1 | Consistent Achievers | 32.1% | 89.6% | enrichment, peer-mentor roles |
| S2 | Last-minute Crammers | 34.4% | 53.3% | mini-deadlines, study-plan coaching |
| S3 | At-risk Disengaged | 20.9% | 0.3% | 14-day inactivity alert → tutor outreach |

The model is K-Means (k = 3) on 10 behavioural features, plus a rule for non-starters.
* Silhouette 0.21; bootstrap ARI 0.96; ARI 0.84 against a refit on the unseen 2014J cohort.
* Segment-targeted support costs **$133 per extra pass**, vs $424 for segment-blind tutor calls (LP with budget and
  tutor-hour constraints).
* Monitoring: 2014J → NO ACTION; simulated platform change → RE-CLUSTER.

## Data

* **OULAD** (Open University Learning Analytics Dataset, CC BY 4.0; Kuzilek et al., 2017): 32,593 enrollments,
  10.6M click records. Kaggle: https://www.kaggle.com/datasets/anlgrbz/student-demographics-online-education-dataoulad.
  `python scripts/download_oulad.py` downloads the same seven tables from the official OU/UCI source and checks them.
* **Synthetic behavioural extension**, generated with Faker and business logic, seed 42:
  * login sessions, device, session length;
  * videos, forum posts;
  * hour-level submission time;
  * injected data defects and bots.

  Each field is documented in `docs/DATA_DICTIONARY.xlsx`.

## Project structure

```
notebooks/           executed notebooks, numbered by project module (sources: notebooks/_src, jupytext percent format)
  01  Module 1  Business understanding & data generation (Spark ETL, KPIs, synthetic data, data dictionary)
  02  Module 2  Exploratory data analysis (distributions, correlations, group comparisons, time series, OLAP)
  03  Module 3  Data quality & cleaning (scorecard, cleaning log, bot detection, before/after)
  04  Module 4  Feature engineering (RFM, encoding, Yeo-Johnson, scaling comparison, PCA, selection)
  05  Module 5  Model development (K-Means k-selection, hierarchical, DBSCAN, validation, stability, surrogate tree)
  05b Module 5 + 8  Personas & recommendations (profiles, early warning, decision EV, LP optimisation, fairness)
  06  Module 6  Model deployment (API tests, consistency, latency, cloud configuration)
  07  Module 7  Model monitoring (PSI/KS, segment shift, centroid shift, Evidently, re-clustering triggers)
      Module 8  final report and slides (submitted separately)
src/edtech/          reusable package: etl_spark, synthetic, cleaning, features, modeling, business, monitoring, serving
app/api/main.py      FastAPI service
app/dashboard/app.py Streamlit dashboard
models/              fitted feature pipeline, segmentation model, surrogate tree, personas.json
monitoring/          Evidently HTML drift reports
docs/                data dictionary, model card, monitoring playbook, deployment guide, lecture alignment
reports/             (generated locally, not in git) figures, tables, metrics
tests/               pytest suite
```

## How to run

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements-dev.txt
.\.venv\Scripts\python scripts\download_oulad.py            # only if data/raw is empty
.\.venv\Scripts\python scripts\run_notebooks.py             # full pipeline (Java 8+ for Spark; falls back to pandas)
.\.venv\Scripts\python -m pytest -q tests
.\.venv\Scripts\python -m uvicorn app.api.main:app --port 8000
.\.venv\Scripts\python -m streamlit run app\dashboard\app.py
```

To run only the API and the dashboard, install `requirements.txt`. The committed artefacts are enough.
Cloud deployment (Render / Streamlit Community Cloud / Hugging Face Spaces) is described in `docs/DEPLOYMENT.md`.

## API example

```bash
curl -X POST https://learner-segmentation-api.onrender.com/predict -H "Content-Type: application/json" -d '{
  "recency_fraction": 0.08, "active_day_fraction": 0.14, "clicks_per_active_day": 15,
  "weekly_cv_active_span": 1.8, "deadline_click_ratio": 4.0, "avg_score_scheduled": 55,
  "last_minute_submission_share": 0.45, "video_completion_rate": 0.5,
  "forum_posts_per_week": 0.0, "night_login_share": 0.4}'
# -> {"segment_id": "S2", "persona": "Last-minute Crammers", "assignment_confidence": 0.73, ...}
```
