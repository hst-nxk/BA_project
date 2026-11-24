"""Learner Segmentation API (Module 6).

Run locally:  uvicorn app.api.main:app --reload --port 8000   (from the project root)
Docs:         http://127.0.0.1:8000/docs
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from fastapi import FastAPI, HTTPException, Query  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from edtech import config as C  # noqa: E402
from edtech.serving import (EngineeredProfile, RawActivityProfile, features_from_raw, load_model,  # noqa: E402
                            load_personas, load_rules, lookup_table, predict_features)

app = FastAPI(
    title="Learner Segmentation API",
    version="1.0.0",
    description="Assigns an online learner to a behavioural segment (persona) from an activity profile, "
                "returns tailored content/pacing/support recommendations, and exposes monitoring status.",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

LOG = Path(os.environ.get("PREDICTION_LOG", C.MONITORING_DATA / "prediction_log.jsonl"))
_START = time.time()


def _log(features: dict, result: dict, source: str) -> None:
    try:
        LOG.parent.mkdir(parents=True, exist_ok=True)
        with LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "source": source,
                                **features, "segment_id": result["segment_id"]}) + "\n")
    except OSError:  # read-only file systems on some free tiers
        pass


class BatchRequest(BaseModel):
    learners: list[EngineeredProfile] = Field(..., min_length=1, max_length=5000)


@app.get("/", tags=["meta"])
def root():
    return {"service": "learner-segmentation", "docs": "/docs", "endpoints": ["/health", "/segments", "/predict",
            "/predict/raw", "/predict/batch", "/learners/{id_student}", "/explain/rules", "/monitoring/latest"]}


@app.get("/health", tags=["meta"])
def health():
    m = load_model()
    return {"status": "ok", "model_version": m.version, "trained_on": m.trained_on, "n_clusters": int(m.kmeans.n_clusters),
            "features": m.features(), "uptime_seconds": round(time.time() - _START, 1)}


@app.get("/segments", tags=["segments"])
def segments():
    return load_personas()


@app.post("/predict", tags=["predict"])
def predict(profile: EngineeredProfile):
    feats = profile.model_dump()
    res = predict_features([feats])[0]
    _log(feats, res, "engineered")
    return {"input_features": feats, **res}


@app.post("/predict/raw", tags=["predict"])
def predict_raw(profile: RawActivityProfile):
    feats = features_from_raw(profile)
    res = predict_features([feats])[0]
    _log(feats, res, "raw")
    return {"input_features": {k: round(v, 4) for k, v in feats.items()}, **res}


@app.post("/predict/batch", tags=["predict"])
def predict_batch(req: BatchRequest):
    recs = [p.model_dump() for p in req.learners]
    res = predict_features(recs)
    counts: dict[str, int] = {}
    for r in res:
        counts[r["persona"]] = counts.get(r["persona"], 0) + 1
    return {"n": len(res), "segment_counts": counts, "results": res}


@app.get("/learners/{id_student}", tags=["lookup"])
def learner(id_student: int, code_module: str | None = Query(None, min_length=3, max_length=3),
            code_presentation: str | None = Query(None, pattern=r"^20\d\d[BJ]$")):
    t = lookup_table()
    r = t[t.id_student == id_student]
    if code_module:
        r = r[r.code_module == code_module.upper()]
    if code_presentation:
        r = r[r.code_presentation == code_presentation.upper()]
    if r.empty:
        raise HTTPException(404, f"No enrollment found for learner {id_student}")
    personas = {s["name"]: s for s in load_personas()["segments"]}
    out = []
    for _, row in r.iterrows():
        d = {k: (None if (isinstance(v, float) and v != v) else (v.item() if hasattr(v, "item") else v)) for k, v in row.items()}
        d["recommendations"] = personas.get(row["persona"], {}).get("recommendations", {})
        out.append(d)
    return {"id_student": id_student, "enrollments": out}


@app.get("/explain/rules", tags=["explain"])
def rules():
    return {"surrogate_tree_depth": 4, "rules": load_rules()}


@app.get("/monitoring/latest", tags=["monitoring"])
def monitoring():
    p = C.MONITORING_DATA / "monitoring_summary.json"
    if not p.exists():
        raise HTTPException(404, "No monitoring run found")
    return json.loads(p.read_text())
