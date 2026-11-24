"""Central configuration: paths, random seed, course metadata and business constants.

Every notebook and application module imports its paths from here so that the
pipeline can be re-run from any working directory.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
SYNTHETIC = DATA / "synthetic"
WAREHOUSE = DATA / "warehouse"
APP_DATA = DATA / "app"
MONITORING_DATA = DATA / "monitoring"

MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
MODULE_DOCS = REPORTS / "modules"
DOCS = ROOT / "docs"
MONITORING_REPORTS = ROOT / "monitoring"

for _p in (INTERIM, PROCESSED, SYNTHETIC, WAREHOUSE, APP_DATA, MONITORING_DATA,
           MODELS, FIGURES, MODULE_DOCS, DOCS, MONITORING_REPORTS):
    _p.mkdir(parents=True, exist_ok=True)

SEED = 42

# Analytical grain: one row per learner-module-presentation enrollment.
KEY = ["id_student", "code_module", "code_presentation"]

# Open University module domains (Kuzilek et al., 2017, Scientific Data 4:170171).
MODULE_DOMAIN = {
    "AAA": "Social Sciences", "BBB": "Social Sciences", "GGG": "Social Sciences",
    "CCC": "STEM", "DDD": "STEM", "EEE": "STEM", "FFF": "STEM",
}

# Presentation codes: year + B (February start) or J (October start).
# The exact start day is not published; the 1st of the month is an assumption
# used only to place synthetic timestamps on a calendar.
PRESENTATION_START = {
    "2013B": "2013-02-01", "2013J": "2013-10-01",
    "2014B": "2014-02-01", "2014J": "2014-10-01",
}

# Temporal split used for model development and monitoring.
TRAIN_PRESENTATIONS = ["2013B", "2013J", "2014B"]
MONITOR_PRESENTATION = "2014J"

# Business constants for KPI and intervention economics (assumptions, documented
# in docs/DATA_DICTIONARY.md and the Module 1 report).
INTERVENTION_CATALOGUE = {
    # name: (cost per learner in USD, tutor minutes per learner, assumed uplift in pass probability)
    "automated_nudges": (2.0, 0, 0.02),
    "study_plan_coaching": (15.0, 20, 0.06),
    "tutor_outreach_call": (35.0, 45, 0.10),
    "peer_study_group": (8.0, 10, 0.04),
}
VALUE_OF_RETAINED_LEARNER = 400.0  # USD, assumed contribution margin of one additional pass

ENGAGEMENT_SCORE_WEIGHTS = {
    # weights of the composite engagement KPI (sum to 1)
    "active_day_fraction": 0.30,
    "logins_per_week_pct": 0.20,
    "video_completion_rate": 0.20,
    "submission_rate": 0.20,
    "forum_posts_pct": 0.10,
}
