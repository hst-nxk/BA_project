"""Headless test of the Streamlit dashboard (streamlit.testing.v1.AppTest)."""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app" / "dashboard" / "app.py"


@pytest.mark.skipif(not (ROOT / "data" / "app" / "learner_lookup.parquet").exists(), reason="run the notebooks first")
def test_dashboard_renders_and_assigns():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=120).run()
    assert not at.exception, at.exception
    assert len(at.tabs) == 6
    assert any("Enrollments" in m.label for m in at.metric)
    # submit the "assign a new learner" form with its default values
    at.button[0].click().run()
    assert not at.exception, at.exception
    assert any("→" in md.value for md in at.markdown), "no segment shown after submitting the form"
