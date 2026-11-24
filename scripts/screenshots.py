"""Capture dashboard and API screenshots for the report (requires the two local servers running)."""
import sys
from pathlib import Path
from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parents[1] / "reports" / "figures"
DASH = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8502"
API = sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:8765"

with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width": 1440, "height": 1000})
    pg.goto(DASH); pg.wait_for_selector("text=Segment size", timeout=90000); pg.wait_for_timeout(3000)
    pg.screenshot(path=str(OUT / "app_dashboard_overview.png"), full_page=False)
    for tab, name, wait in [("Segment profiles", "app_dashboard_profiles", "Behavioural fingerprint"),
                            ("Learner lookup", "app_dashboard_lookup", "Recommended support"),
                            ("Monitoring", "app_dashboard_monitoring", "Population Stability Index")]:
        pg.get_by_role("tab", name=tab).click(); pg.wait_for_timeout(2500)
        try:
            pg.wait_for_selector(f"text={wait}", timeout=30000)
        except Exception:
            pass
        pg.wait_for_timeout(1500)
        pg.screenshot(path=str(OUT / f"{name}.png"), full_page=False)
    pg.get_by_role("tab", name="Assign a new learner").click(); pg.wait_for_timeout(1500)
    pg.get_by_role("button", name="Assign segment").click(); pg.wait_for_timeout(4000)
    pg.screenshot(path=str(OUT / "app_dashboard_assign.png"), full_page=True)
    pg.goto(API + "/docs"); pg.wait_for_timeout(3000)
    pg.screenshot(path=str(OUT / "app_api_docs.png"), full_page=False)
    b.close()
print("screenshots saved")
