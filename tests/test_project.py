"""Automated checks for the pipeline outputs, the transformation code and the API."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
os.environ.setdefault("PREDICTION_LOG", str(ROOT / "data" / "monitoring" / "test_prediction_log.jsonl"))

from edtech import config as C  # noqa: E402
from edtech.monitoring import psi  # noqa: E402
from edtech.serving import RawActivityProfile, features_from_raw  # noqa: E402

needs_model = pytest.mark.skipif(not (C.MODELS / "segmentation_model.joblib").exists(), reason="run the notebooks first")


def test_psi_zero_for_identical_and_large_for_shift():
    rng = np.random.default_rng(0)
    a = rng.normal(size=5000)
    assert psi(a, a) < 1e-6
    assert psi(a, a + 1.0) > 0.25


def test_features_from_raw_definitions():
    r = RawActivityProfile(course_length_days=209, active_days=21, last_active_day=200, total_clicks=420,
                           deadline_window_clicks=126, deadline_window_days=21, scheduled_assessments=6,
                           scores=[60, 70, 80], submissions_last_24h=1, videos_started=10, videos_completed=10,
                           forum_posts=3, logins=40, night_logins=10)
    f = features_from_raw(r)
    assert f["active_day_fraction"] == pytest.approx(21 / 210)
    assert f["recency_fraction"] == pytest.approx(9 / 210)
    assert f["clicks_per_active_day"] == pytest.approx(20)
    assert f["avg_score_scheduled"] == pytest.approx(35)          # non-submissions count as 0
    assert f["deadline_click_ratio"] == pytest.approx((126 / 420) / (21 / 210))
    assert f["night_login_share"] == pytest.approx(0.25)
    assert f["forum_posts_per_week"] == pytest.approx(3 / 30)


def test_raw_profile_rejects_inconsistent_input():
    with pytest.raises(ValueError):
        RawActivityProfile(active_days=5, last_active_day=3, total_clicks=10, scheduled_assessments=2,
                           videos_started=1, videos_completed=3)


@pytest.mark.skipif(not (C.PROCESSED / "learner_segments.parquet").exists(), reason="run the notebooks first")
def test_segments_cover_every_enrollment_once():
    seg = pd.read_parquet(C.PROCESSED / "learner_segments.parquet")
    assert len(seg) == 32593
    assert seg.duplicated(C.KEY).sum() == 0
    assert seg.segment_id.notna().all()
    assert (seg.loc[seg.no_activity == 1, "segment_id"] == "S0").all()


@pytest.mark.skipif(not (C.INTERIM / "sessions_clean.parquet").exists(), reason="run the notebooks first")
def test_cleaned_sessions_are_valid():
    s = pd.read_parquet(C.INTERIM / "sessions_clean.parquet")
    assert s.session_id.is_unique
    assert s.session_minutes.between(0, 1440).all()
    assert (s.videos_completed <= s.videos_started).all()
    assert set(s.device_type.unique()) <= {"desktop", "mobile", "tablet", "unknown"}
    assert not s.user_agent.str.contains("python-requests").any()


@needs_model
def test_api_end_to_end():
    from fastapi.testclient import TestClient
    from app.api.main import app

    c = TestClient(app)
    assert c.get("/health").json()["status"] == "ok"
    ok = dict(recency_fraction=0.85, active_day_fraction=0.03, clicks_per_active_day=12, weekly_cv_active_span=0.8,
              deadline_click_ratio=0.6, avg_score_scheduled=4, last_minute_submission_share=0.0,
              video_completion_rate=0.5, forum_posts_per_week=0.0, night_login_share=0.1)
    r = c.post("/predict", json=ok)
    assert r.status_code == 200 and r.json()["persona"] == "At-risk Disengaged"
    assert c.post("/predict", json={**ok, "avg_score_scheduled": 101}).status_code == 422
    zero = {k: 0 for k in ok} | {"recency_fraction": 1.0}
    assert c.post("/predict", json=zero).json()["segment_id"] == "S0"
    assert c.get("/learners/1").status_code == 404
