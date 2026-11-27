# %% [markdown]
# # Module 6 — Model Deployment
# MLOps role. The segmentation is packaged as a **decision-support system** (Lecture 3.1): a *data subsystem*
# (feature store, learner lookup), a *model subsystem* (feature pipeline + K-Means + persona rules) and a
# *user-interface subsystem* (Streamlit dashboard for instructors), connected through a **FastAPI** service.
#
# ```
#  LMS / instructor ──► Streamlit dashboard ──HTTP──► FastAPI /predict ──► FeaturePipeline ─► K-Means ─► persona + recommendations
#                                   │                         │
#                                   └──── learner lookup ◄────┴──── prediction_log.jsonl ──► monitoring (Module 7)
# ```
#
# This notebook (1) builds the slim serving artefacts, (2) exercises every API endpoint with FastAPI's TestClient,
# including invalid input, (3) checks training/serving consistency and latency, and (4) documents the cloud deployment.

# %%
import sys, time, json, warnings
from pathlib import Path
warnings.filterwarnings("ignore", message=".*httpx.*starlette.testclient.*")
ROOT = Path.cwd().parent if (Path.cwd().parent / "src").exists() else Path.cwd()
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
from fastapi.testclient import TestClient
from edtech import config as C, serving as S, utils as U

S.build_app_data()
lk = S.lookup_table()
print("lookup table:", lk.shape, f"{(C.APP_DATA / 'learner_lookup.parquet').stat().st_size / 1e6:.1f} MB")
pd.DataFrame([(p.name, f"{p.stat().st_size / 1e3:.0f} kB") for p in sorted(C.MODELS.glob("*")) if p.is_file()], columns=["artefact", "size"])

# %% [markdown]
# ## 1. API endpoints

# %%
import os
os.environ["PREDICTION_LOG"] = str(C.MONITORING_DATA / "test_prediction_log.jsonl")   # keep test traffic out of the live log
from app.api.main import app
client = TestClient(app)
pd.DataFrame([(r.path, ",".join(sorted(r.methods))) for r in app.routes if hasattr(r, "methods") and not r.path.startswith(("/openapi", "/docs", "/redoc"))],
             columns=["path", "methods"])

# %%
client.get("/health").json()

# %%
examples = {
    "steady, early submitter": dict(recency_fraction=0.02, active_day_fraction=0.45, clicks_per_active_day=22, weekly_cv_active_span=0.9,
                                    deadline_click_ratio=1.4, avg_score_scheduled=82, last_minute_submission_share=0.05,
                                    video_completion_rate=0.8, forum_posts_per_week=0.3, night_login_share=0.1),
    "deadline-driven": dict(recency_fraction=0.08, active_day_fraction=0.14, clicks_per_active_day=15, weekly_cv_active_span=1.8,
                            deadline_click_ratio=4.0, avg_score_scheduled=55, last_minute_submission_share=0.45,
                            video_completion_rate=0.5, forum_posts_per_week=0.0, night_login_share=0.4),
    "faded early": dict(recency_fraction=0.85, active_day_fraction=0.03, clicks_per_active_day=12, weekly_cv_active_span=0.8,
                        deadline_click_ratio=0.6, avg_score_scheduled=4, last_minute_submission_share=0.0,
                        video_completion_rate=0.5, forum_posts_per_week=0.0, night_login_share=0.1),
    "never started": dict(recency_fraction=1.0, active_day_fraction=0.0, clicks_per_active_day=0, weekly_cv_active_span=0,
                          deadline_click_ratio=0, avg_score_scheduled=0, last_minute_submission_share=0,
                          video_completion_rate=0, forum_posts_per_week=0, night_login_share=0),
}
res = []
for name, ex in examples.items():
    r = client.post("/predict", json=ex); assert r.status_code == 200, r.text
    j = r.json(); res.append({"example": name, "segment": j["segment_id"], "persona": j["persona"],
                              "confidence": j["assignment_confidence"], "top_driver": j["top_drivers"][0]["feature"]})
pd.DataFrame(res)

# %%
raw = {"course_length_days": 240, "active_days": 30, "last_active_day": 235, "total_clicks": 450,
       "weekly_clicks": [5, 0, 40, 2, 0, 60, 3, 0, 0, 55, 1, 0, 70, 0, 2, 45, 0, 0, 80, 5, 0, 30, 0, 52],
       "deadline_window_clicks": 260, "deadline_window_days": 27, "scheduled_assessments": 9,
       "scores": [55, 60, 48, 62, 58, 50, 61], "submissions_last_24h": 5, "videos_started": 60, "videos_completed": 30,
       "forum_posts": 0, "logins": 45, "night_logins": 20}
r = client.post("/predict/raw", json=raw); print(r.status_code); r.json()

# %%
batch = client.post("/predict/batch", json={"learners": list(examples.values())}).json()
print(batch["segment_counts"])
some = lk.sample(1, random_state=1).iloc[0]
look = client.get(f"/learners/{int(some.id_student)}", params={"code_module": some.code_module}).json()
look["enrollments"][0]["persona"], look["enrollments"][0]["final_result"]

# %% [markdown]
# ### Input validation (bad requests are rejected with HTTP 422, unknown learners with 404)

# %%
bad = [("score above 100", "/predict", {**examples["deadline-driven"], "avg_score_scheduled": 140}),
       ("missing field", "/predict", {k: v for k, v in examples["faded early"].items() if k != "recency_fraction"}),
       ("negative share", "/predict", {**examples["faded early"], "night_login_share": -0.2}),
       ("completed > started videos", "/predict/raw", {**raw, "videos_completed": 99}),
       ("more scores than assessments", "/predict/raw", {**raw, "scheduled_assessments": 3})]
checks = [{"case": n, "status": client.post(p, json=b).status_code} for n, p, b in bad]
checks.append({"case": "unknown learner", "status": client.get("/learners/1").status_code})
checks.append({"case": "malformed presentation code", "status": client.get(f"/learners/{int(some.id_student)}", params={"code_presentation": "2014X"}).status_code})
pd.DataFrame(checks)

# %% [markdown]
# ## 2. Training/serving consistency and latency
# The API must reproduce exactly the segments stored by the training notebook.

# %%
seg = pd.read_parquet(C.PROCESSED / "learner_segments.parquet")
sample = seg.sample(2000, random_state=C.SEED)
payload = {"learners": sample[S.load_model().features()].fillna(0).round(10).to_dict("records")}
t0 = time.time(); out = client.post("/predict/batch", json=payload).json(); dt = time.time() - t0
agree = np.mean([o["segment_id"] == s for o, s in zip(out["results"], sample.segment_id)])
lat = []
for _ in range(50):
    t1 = time.time(); client.post("/predict", json=examples["deadline-driven"]); lat.append(1000 * (time.time() - t1))
serve = {"batch_2000_seconds": dt, "agreement_with_training_assignment": agree,
         "single_request_p50_ms": float(np.percentile(lat, 50)), "single_request_p95_ms": float(np.percentile(lat, 95))}
U.save_metrics("m6_deploy", {**serve, "validation_checks": checks, "examples": res}); serve

# %% [markdown]
# Agreement is 100% on learners whose features are complete. Prediction is cheap (nearest centroid on 10 numbers), so a
# single free-tier instance comfortably serves an LMS with thousands of requests per hour.
#
# ## 3. Cloud deployment (free tier)
# The repository contains everything needed; see `docs/DEPLOYMENT.md` for step-by-step instructions.
#
# | Component | Platform (free tier) | Entry point | Config |
# |---|---|---|---|
# | FastAPI service | **Render** web service | `uvicorn app.api.main:app --host 0.0.0.0 --port $PORT` | `render.yaml`, `requirements.txt` |
# | Streamlit dashboard | **Render** web service *or* **Streamlit Community Cloud** | `streamlit run app/dashboard/app.py` | `render.yaml`, env `API_URL` |
# | Alternative (single container) | **Hugging Face Spaces** (Docker) | `Dockerfile` (port 7860) | dashboard with in-process model |
#
# The dashboard calls the API when `API_URL` is configured and falls back to the in-process model otherwise, so a
# cold-starting free API instance never breaks the instructor's view. Every API prediction is logged for monitoring.
