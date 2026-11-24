"""Serving layer shared by the FastAPI service and the Streamlit dashboard.

Two input contracts:
* ``EngineeredProfile`` - the 10 model features (system-to-system integration)
* ``RawActivityProfile`` - counts an LMS can export for one learner in one course;
  converted to the model features by ``features_from_raw`` with the same definitions
  used in training (src/edtech/oulad.py, src/edtech/features.py).
"""
from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, model_validator

from . import config as C

NIGHT = {22, 23, 0, 1, 2, 3, 4, 5}


class EngineeredProfile(BaseModel):
    recency_fraction: float = Field(..., ge=0, le=1, description="(L - last active day) / (L + 1)")
    active_day_fraction: float = Field(..., ge=0, le=1, description="active days / course days")
    clicks_per_active_day: float = Field(..., ge=0, le=2000)
    weekly_cv_active_span: float = Field(..., ge=0, le=50)
    deadline_click_ratio: float = Field(..., ge=0, le=100)
    avg_score_scheduled: float = Field(..., ge=0, le=100)
    last_minute_submission_share: float = Field(..., ge=0, le=1)
    video_completion_rate: float = Field(..., ge=0, le=1)
    forum_posts_per_week: float = Field(..., ge=0, le=200)
    night_login_share: float = Field(..., ge=0, le=1)


class RawActivityProfile(BaseModel):
    course_length_days: int = Field(240, ge=28, le=400, description="L: scheduled course length in days")
    days_elapsed: int | None = Field(None, ge=1, le=400, description="days observed so far (defaults to L)")
    active_days: int = Field(..., ge=0, le=400, description="days with at least one resource click")
    last_active_day: int | None = Field(None, ge=0, le=400, description="course day of last activity")
    total_clicks: int = Field(..., ge=0, le=1_000_000)
    weekly_clicks: list[int] | None = Field(None, description="clicks per course week (optional, for burstiness)")
    deadline_window_clicks: int = Field(0, ge=0, description="clicks on deadline days or the 2 days before")
    deadline_window_days: int = Field(27, ge=1, le=400, description="number of such days in the course")
    scheduled_assessments: int = Field(..., ge=1, le=50)
    scores: list[float] = Field(default_factory=list, description="scores of submitted assessments (0-100)")
    submissions_last_24h: int = Field(0, ge=0, description="submissions made < 24 h before the deadline")
    videos_started: int = Field(0, ge=0)
    videos_completed: int = Field(0, ge=0)
    forum_posts: int = Field(0, ge=0)
    logins: int = Field(0, ge=0)
    night_logins: int = Field(0, ge=0, description="logins starting 22:00-05:59")

    @model_validator(mode="after")
    def _consistency(self):
        horizon = self.days_elapsed or self.course_length_days
        if self.active_days > horizon + 1:
            raise ValueError("active_days cannot exceed observed days")
        if self.videos_completed > self.videos_started:
            raise ValueError("videos_completed cannot exceed videos_started")
        if self.night_logins > self.logins:
            raise ValueError("night_logins cannot exceed logins")
        if len(self.scores) > self.scheduled_assessments:
            raise ValueError("more scores than scheduled assessments")
        if self.submissions_last_24h > len(self.scores):
            raise ValueError("submissions_last_24h cannot exceed number of submissions")
        if any(s < 0 or s > 100 for s in self.scores):
            raise ValueError("scores must be within 0..100")
        if self.active_days > 0 and self.last_active_day is None:
            raise ValueError("last_active_day is required when active_days > 0")
        return self


def features_from_raw(r: RawActivityProfile, global_video_rate: float = 0.55) -> dict:
    L = r.days_elapsed or r.course_length_days
    days = L + 1
    weeks = math.ceil(days / 7)
    active = r.active_days > 0
    f = {
        "recency_fraction": ((L - (r.last_active_day or 0)) / days) if active else 1.0,
        "active_day_fraction": r.active_days / days,
        "clicks_per_active_day": r.total_clicks / r.active_days if active else 0.0,
        "avg_score_scheduled": sum(r.scores) / r.scheduled_assessments,
        "last_minute_submission_share": r.submissions_last_24h / len(r.scores) if r.scores else 0.0,
        "video_completion_rate": ((r.videos_completed + 2 * global_video_rate) / (r.videos_started + 2)) if r.logins > 0 else 0.0,
        "forum_posts_per_week": r.forum_posts / weeks,
        "night_login_share": r.night_logins / r.logins if r.logins else 0.0,
    }
    share = r.deadline_window_clicks / r.total_clicks if r.total_clicks else 0.0
    f["deadline_click_ratio"] = share / (r.deadline_window_days / days)
    if r.weekly_clicks and active:
        w = np.array(r.weekly_clicks, dtype=float)
        nz = np.flatnonzero(w > 0)
        span = w[nz.min(): nz.max() + 1] if len(nz) else w
        f["weekly_cv_active_span"] = float(span.std() / span.mean()) if span.mean() > 0 else 0.0
    else:
        f["weekly_cv_active_span"] = 1.25 if active else 0.0   # training median for active learners
    return f


@lru_cache(maxsize=1)
def load_model():
    return joblib.load(C.MODELS / "segmentation_model.joblib")


@lru_cache(maxsize=1)
def load_personas() -> dict:
    return json.loads((C.MODELS / "personas.json").read_text())


@lru_cache(maxsize=1)
def load_rules() -> str:
    p = C.MODELS / "surrogate_tree_rules.txt"
    return p.read_text() if p.exists() else ""


def predict_features(records: list[dict]) -> list[dict]:
    model = load_model()
    personas = {s["name"]: s for s in load_personas()["segments"]}
    df = pd.DataFrame(records)
    res = model.assign(df)
    Z = model.pipeline.transform(df)
    zdf = pd.DataFrame(Z, columns=model.features())
    out = []
    for i, row in res.iterrows():
        p = personas.get(row["persona"], {})
        drivers = zdf.iloc[i].abs().sort_values(ascending=False).head(3).index.tolist() if row["cluster"] >= 0 else ["active_day_fraction"]
        out.append({
            "segment_id": row["segment_id"], "persona": row["persona"],
            "assignment_confidence": round(float(row["assignment_confidence"]), 3),
            "distance_to_centroid": None if pd.isna(row["distance_to_centroid"]) else round(float(row["distance_to_centroid"]), 3),
            "is_atypical": bool(row["is_atypical"]),
            "top_drivers": [{"feature": d, "value": round(float(df.iloc[i][d]), 4), "z": round(float(zdf.iloc[i][d]), 2)} for d in drivers],
            "recommendations": p.get("recommendations", {}),
            "model_version": model.version,
        })
    return out


def lookup_table() -> pd.DataFrame:
    return pd.read_parquet(C.APP_DATA / "learner_lookup.parquet")


def build_app_data() -> None:
    """Slim tables for the dashboard / API (written by the deployment notebook)."""
    seg = pd.read_parquet(C.PROCESSED / "learner_segments.parquet")
    model = load_model()
    cols = C.KEY + ["course_domain", "final_result", "segment_id", "persona", "assignment_confidence", "is_atypical",
                    "engagement_score", "logins_per_week", "submission_rate", "total_clicks", "split"] + model.features()
    seg[cols].to_parquet(C.APP_DATA / "learner_lookup.parquet", index=False)
