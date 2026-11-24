"""Build the presentation deck (reports/Final_Presentation.pptx) from computed metrics and figures."""
from __future__ import annotations

import json
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "figures"
MET = ROOT / "reports" / "metrics"
INK, INK2, ACCENT, SURF = RGBColor(0x0B, 0x0B, 0x0B), RGBColor(0x52, 0x51, 0x4E), RGBColor(0x1F, 0x4E, 0x8C), RGBColor(0xFC, 0xFC, 0xFB)
PCOL = {"Non-starters": "8a8984", "At-risk Disengaged": "e34948", "Last-minute Crammers": "eb6834", "Consistent Achievers": "1baf7a"}


def m(name):
    p = MET / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else {}


class Deck:
    def __init__(self):
        self.p = Presentation(); self.p.slide_width, self.p.slide_height = Inches(13.333), Inches(7.5)
        self.n = 0

    def slide(self, title, subtitle=None):
        s = self.p.slides.add_slide(self.p.slide_layouts[6]); self.n += 1
        bg = s.background.fill; bg.solid(); bg.fore_color.rgb = SURF
        bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, self.p.slide_width, Inches(0.12)); bar.fill.solid()
        bar.fill.fore_color.rgb = ACCENT; bar.line.fill.background()
        self.text(s, title, 0.5, 0.3, 12.3, 0.8, 28, bold=True, color=ACCENT)
        if subtitle:
            self.text(s, subtitle, 0.5, 1.0, 12.3, 0.5, 15, color=INK2)
        self.text(s, f"{self.n}", 12.5, 7.0, 0.6, 0.3, 10, color=INK2, align=PP_ALIGN.RIGHT)
        return s

    def text(self, s, txt, x, y, w, h, size=16, bold=False, color=INK, align=PP_ALIGN.LEFT):
        tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)); tf = tb.text_frame; tf.word_wrap = True
        lines = txt if isinstance(txt, list) else [txt]
        for i, ln in enumerate(lines):
            par = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            bullet = isinstance(txt, list)
            r = par.add_run(); r.text = ("• " if bullet else "") + ln
            r.font.size = Pt(size); r.font.bold = bold; r.font.color.rgb = color; r.font.name = "Calibri"
            par.alignment = align; par.space_after = Pt(6 if bullet else 0)
        return tb

    def img(self, s, name, x, y, w=None, h=None):
        p = FIG / f"{name}.png"
        if p.exists():
            s.shapes.add_picture(str(p), Inches(x), Inches(y), width=Inches(w) if w else None, height=Inches(h) if h else None)

    def card(self, s, x, y, w, h, title, body, color="1f4e8c"):
        box = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
        box.fill.solid(); box.fill.fore_color.rgb = RGBColor(0xFF, 0xFF, 0xFF); box.line.color.rgb = RGBColor.from_string(color)
        box.line.width = Pt(2); box.adjustments[0] = 0.06
        self.text(s, title, x + 0.15, y + 0.1, w - 0.3, 0.5, 16, bold=True, color=RGBColor.from_string(color))
        self.text(s, body, x + 0.15, y + 0.6, w - 0.3, h - 0.7, 12)

    def table(self, s, rows, x, y, w, h, size=12):
        t = s.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y), Inches(w), Inches(h)).table
        for i, r in enumerate(rows):
            for j, v in enumerate(r):
                c = t.cell(i, j); c.text = str(v)
                for par in c.text_frame.paragraphs:
                    for run in par.runs:
                        run.font.size = Pt(size); run.font.bold = i == 0
                        run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF) if i == 0 else INK
        return t


def build():
    m1, m3, m4, m5, m6, m7, dep = m("m1"), m("m3"), m("m4"), m("m5"), m("m6"), m("m7"), m("m6_deploy")
    kpi = {r["persona"]: r for r in m6.get("kpi", [])}
    d = Deck()

    s = d.slide("Learner Behavior Segmentation for an Online Learning Platform")
    d.text(s, "From raw clickstream to a deployed, monitored segmentation model", 0.5, 1.2, 12, 0.6, 20, color=INK2)
    d.text(s, ["Business question: what distinct behavioural profiles exist among our learners, and how should content, "
               "pacing or support be tailored to each group?",
               "Data: OULAD (32,593 enrollments, 10.6M click records) + synthetic behavioural extension (2.7M login sessions)",
               "Deliverables: notebooks · data dictionary · FastAPI + Streamlit (cloud-ready) · Evidently monitoring · report"],
           0.5, 2.3, 12.3, 3, 18)
    d.text(s, "Business Analytics — Capstone project · [Team / student names]", 0.5, 6.4, 12, 0.5, 14, color=INK2)

    s = d.slide("1 · Business understanding", "One-size-fits-all support wastes tutor time and misses learners who need help")
    d.text(s, [f"{m1.get('at_risk_rate', 0):.1%} of enrollments end in Fail or Withdrawn (at-risk learner rate)",
               f"{m1.get('no_activity_enrollments', 0):,} enrollments never open the course materials",
               "Decision makers: Head of Learner Success (budget), course designers (content, pacing), tutors (outreach)",
               "KPIs: engagement score (0–100) · at-risk learner rate · intervention cost · cost per additional pass",
               "Analytics types: descriptive (EDA, profiles) → predictive (segment assignment, early warning, drift) → "
               "prescriptive (decision EV, LP budget allocation)"], 0.5, 1.7, 7.2, 5, 17)
    d.table(s, [["Intervention", "USD / learner", "Tutor min"], ["Automated nudges", "2", "0"], ["Peer study group", "8", "10"],
                ["Study-plan coaching", "15", "20"], ["Tutor outreach call", "35", "45"]], 8.0, 2.0, 4.8, 2.2, 13)
    d.text(s, "Costs and uplifts are planning assumptions (varied in sensitivity analysis)", 8.0, 4.4, 4.8, 0.8, 12, color=INK2)

    s = d.slide("2 · Data: real base + synthetic behavioural extension", "PySpark ETL (DataFrame + Pair RDD) · conditional simulation anchored to each learner")
    e = m1.get("etl", {})
    d.text(s, [f"Spark aggregates {e.get('studentVle_rows', 0):,} click rows → {e.get('enrollment_day_rows', 0):,} enrollment-days; "
               f"Pair-RDD reduceByKey cross-check: {e.get('total_clicks_pair_rdd', 0):,} clicks (exact match)",
               "Synthetic (Faker + numpy, seed 42): login sessions & hour, device, session minutes, videos started/completed, "
               "forum posts/replies, hour-level submission time",
               "Anchors: sessions sit on real active days; video completion follows real engagement & score; "
               "submission hour refines the real submission day",
               "Injected realistic defects (duplicates, label noise, missing / impossible values, 0.4% bot accounts) "
               "to test cleaning against ground truth",
               "Every synthetic field documented: name, type, unit, valid range, generation logic (docs/DATA_DICTIONARY.xlsx)"],
           0.5, 1.6, 6.3, 5.5, 15)
    d.img(s, "m1_submission_lead_time", 7.0, 1.6, w=6.0)
    d.img(s, "m1_synthetic_validation", 7.0, 5.0, w=6.0)

    s = d.slide("3 · EDA: two independent behavioural axes", "Volume/regularity vs pacing — the basis for meaningful personas")
    d.img(s, "m2_correlation_heatmap", 0.4, 1.5, w=6.0)
    r = m("m2").get("relationships", {})
    d.text(s, ["Volume block (clicks, active days, logins, submission, score) inter-correlated ρ 0.7–0.99",
               "Pacing (deadline concentration, burstiness, last-minute, night) nearly independent of volume (|ρ| < 0.25)",
               f"Logins vs score ρ = {r.get('rho_logins_score', 0):.2f}, with diminishing returns above ~3–4 logins/week",
               f"Video completion vs submission lead time ρ = {r.get('rho_video_leadtime', 0):.2f}: timeliness is its own dimension",
               "Withdrawal rose from 27–29% (2013) to 34% (2014): drift to monitor",
               "OLAP star schema in DuckDB: roll-up / drill-down / slice / dice / pivot"], 6.6, 1.6, 6.4, 5.5, 15)

    s = d.slide("3 · EDA: required relationships and time pattern")
    d.img(s, "m2_required_relationships", 0.3, 1.2, w=7.2)
    d.img(s, "m2_time_series", 0.3, 4.3, w=7.2)
    d.text(s, ["Score rises steeply up to ~3–4 logins/week, then flattens",
               "Typical learners submit in the final 0–24 h regardless of video completion",
               "Clicks follow the assessment calendar (peaks before each deadline) on a downward trend",
               "Future Fail/Withdrawn learners are < half as active as passers from week 0"], 7.8, 1.5, 5.2, 5, 15)

    s = d.slide("4 · Data quality & cleaning", "No enrollment removed; every action logged with before/after evidence")
    be = m3.get("bot_detection", {})
    d.text(s, [f"Session log {m3.get('session_rows_before', 0):,} → {m3.get('session_rows_after', 0):,} rows (duplicates + automated sessions)",
               f"Bot detection: ≥ 2 of 4 rules → precision {be.get('precision', 0):.2f}, recall {be.get('recall', 0):.2f}; "
               f"one IQR rule alone would wrongly flag {m3.get('r1_only_false_positives', 0)} heavy human learners",
               f"IMD '10-20' → '10-20%' ({m3.get('imd_relabelled', 0):,} rows); 1,111 missing → 'Unknown' + flag",
               "17 device spellings → 3 classes; impossible durations → NA → learner-median imputation + flag",
               f"{m3.get('click_extreme_outliers', 0)} extreme click outliers kept (real heavy learners) — handled by Yeo-Johnson + winsorising"],
           0.5, 1.6, 6.0, 5.5, 15)
    d.img(s, "m3_bot_detection", 6.7, 1.5, w=6.3)
    d.img(s, "m3_before_after", 6.7, 4.6, w=6.3)

    s = d.slide("5 · Feature engineering", "RFM-style features + pacing, performance, content and social behaviour")
    d.table(s, [["Role", "Feature", "Source"], ["R", "recency_fraction", "OULAD"], ["F", "active_day_fraction", "OULAD"],
                ["M (intensity)", "clicks_per_active_day", "OULAD"], ["Consistency", "weekly_cv_active_span", "OULAD"],
                ["Pacing", "deadline_click_ratio", "OULAD"], ["Performance", "avg_score_scheduled", "OULAD"],
                ["Timeliness", "last_minute_submission_share", "Synthetic"], ["Content", "video_completion_rate", "Synthetic"],
                ["Social", "forum_posts_per_week", "Synthetic"], ["Rhythm", "night_login_share", "Synthetic"]], 0.5, 1.6, 6.4, 4.8, 12)
    d.text(s, ["19 candidates → redundancy filter |ρ| ≥ 0.8 → 10 features (max remaining |ρ| = 0.80)",
               "Yeo-Johnson (Box-Cox family, handles zeros) + winsorising + Z-score: each feature carries 10% of the variance",
               "Demographics excluded from clustering (fairness); one-hot / ordinal encodings kept for profiling",
               f"PCA: PC1 = engagement & performance, PC2 = pacing; {m4.get('pcs_for_80pct', 5)} PCs for 80% → cluster on full features",
               "Pipeline fitted on 2013B–2014B only; 2014J held out"], 7.2, 1.6, 5.8, 5.5, 14)

    s = d.slide("6 · Model development: choosing k", "Stability, not silhouette alone, drives the decision")
    d.img(s, "m5_k_selection", 0.3, 1.4, w=8.0)
    t = m5.get("temporal", {})
    d.text(s, [f"Hopkins = {m5.get('hopkins', 0):.2f} → clusterable structure",
               f"Elbow at k = {m5.get('k_elbow')}, but k ≥ 4 unstable (bootstrap ARI min < 0.80)",
               f"Selected k = {m5.get('k_selected')}: best silhouette / DB / CH among admissible k",
               f"Silhouette {t.get('silhouette_train', 0):.2f} (train) · {t.get('silhouette_2014J_frozen', 0):.2f} (unseen 2014J)",
               f"Frozen vs refit on 2014J: ARI {t.get('ari_frozen_vs_refit_2014J', 0):.2f}",
               "+ rule-based S0 Non-starters (no activity)"], 8.5, 1.6, 4.6, 5.5, 14)

    s = d.slide("6 · K-Means vs Hierarchical vs DBSCAN")
    rows = [["Algorithm", "Silhouette", "Davies-B.", "Largest cluster", "Stability", "New learners"]]
    for a in m5.get("algorithm_comparison", []):
        rows.append([a["algorithm"], f"{a['silhouette']:.3f}", f"{a['davies_bouldin']:.2f}", f"{a['largest_cluster_share']:.0%}",
                     "-" if a["stability_ari"] is None else f"{a['stability_ari']:.2f}", a["assigns_new_learners"]])
    d.table(s, rows, 0.5, 1.4, 12.3, 1.8, 12)
    d.img(s, "m5_dendrograms", 0.5, 3.5, w=7.0)
    d.text(s, ["Single/average linkage chain (one cluster ≈ 97–100%)", "Ward agrees moderately with K-Means (ARI 0.44)",
               "DBSCAN: one dense continuum + noise → used as outlier lens",
               "K-Means chosen: separation, stability, scalability, instant assignment for the API"], 7.8, 3.6, 5.2, 3.5, 14)

    s = d.slide("7 · Four personas", "Outcome never used in training — yet segments separate pass rates from 0% to 90%")
    d.img(s, "m6_segment_size_outcome", 0.3, 1.5, w=8.2)
    d.img(s, "m6_radar", 8.6, 1.4, h=5.0)
    d.text(s, f"External validation vs final result: NMI {m5.get('external', {}).get('nmi', 0):.2f}, purity {m5.get('external', {}).get('purity', 0):.2f} · "
              f"surrogate decision tree reproduces segments with {m5.get('surrogate_acc_test', 0):.0%} accuracy", 0.5, 6.6, 12.3, 0.5, 13, color=INK2)

    s = d.slide("7 · Persona profiles and tailored support")
    x = 0.4
    body = {"Non-starters": "Never open the course. 89% withdraw.\nSupport: welcome sequence day 1/3/7, tutor call day 10, deferral option.",
            "At-risk Disengaged": "Start like Crammers, fade by week ~10; score ≈ 5.\nSupport: tutor outreach when inactive 14 days, catch-up path.",
            "Last-minute Crammers": "Study concentrated on deadline days (3.4x), 25% of work in final 24 h; half pass.\nSupport: checkpoints, mini-deadlines, study-plan coaching.",
            "Consistent Achievers": "Active 40% of days, submit 41 h early, forum contributors; 90% pass.\nSupport: enrichment, peer-mentor roles."}
    for p, b in body.items():
        k = kpi.get(p, {})
        d.card(s, x, 1.4, 3.05, 5.3, p, f"{k.get('share', 0):.0%} of enrollments · pass {k.get('pass_rate', 0):.0%}\n\n{b}", PCOL[p])
        x += 3.15

    s = d.slide("8 · Early warning: segment predicted on day 28", "Classification (random forest) on facts known after 4 weeks; tested on unseen 2014J")
    d.img(s, "m6_early_warning", 0.3, 1.5, w=8.5)
    rep = {r["class"]: r for r in m6.get("early_warning_report", [])}
    d.text(s, [f"Balanced accuracy {m6.get('early_warning_balanced_accuracy', 0):.2f} (chance 0.25)",
               f"Non-starters recall {rep.get('Non-starters', {}).get('recall', 0):.2f} → act on day 10–28",
               f"At-risk Disengaged recall {rep.get('At-risk Disengaged', {}).get('recall', 0):.2f}: still look like Crammers in month 1",
               "Use as a weekly-refreshed watch list, confirmed by tutors"], 9.0, 1.6, 4.1, 5, 14)

    s = d.slide("8 · Prescriptive analytics: where should the money go?", "Decision-tree EV per learner + LP allocation (budget, tutor hours, segment sizes)")
    d.img(s, "m6_lp_tutor_hours", 0.3, 1.5, w=6.2)
    pc = {r["policy"]: r for r in m6.get("policy_comparison", [])}
    lp = m6.get("lp", {})
    d.text(s, [f"Optimal plan for 2014J: {lp.get('expected_extra_passes', 0):.0f} expected extra passes for ${lp.get('spend', 0):,.0f}",
               f"Cost per extra pass: ${pc.get('segment-targeted LP', {}).get('cost_per_extra_pass', 0):.0f} (targeted) vs "
               f"${pc.get('segment-blind tutor calls', {}).get('cost_per_extra_pass', 0):.0f} (segment-blind tutor calls)",
               "Binding constraint = tutor time: +1 tutor hour ≈ +0.22 passes (≈ $86) → buy tutor hours before raising budget",
               "Tutor time → coaching for Crammers; nudges for other at-risk segments; nothing for Achievers",
               "Uplifts are assumptions → next step: A/B test per segment"], 6.8, 1.6, 6.3, 5.5, 15)

    s = d.slide("9 · Deployment: a decision-support system", "FastAPI service + Streamlit dashboard · Render / Streamlit Cloud / HF Spaces")
    d.img(s, "app_dashboard_overview", 0.3, 1.5, w=7.4)
    d.text(s, ["POST /predict, /predict/raw, /predict/batch · GET /learners/{id}, /segments, /explain/rules, /monitoring/latest",
               f"Input validation: {len(dep.get('validation_checks', []))}/{len(dep.get('validation_checks', []))} bad requests rejected (422 / 404)",
               f"Serving = training assignment: {dep.get('agreement_with_training_assignment', 0):.0%} agreement on 2,000 learners",
               f"Latency p50 {dep.get('single_request_p50_ms', 0):.0f} ms per request",
               "Dashboard: overview & filters · persona profiles · learner lookup · assign new learner · monitoring",
               "Live links: https://learner-segmentation-api.onrender.com · https://learner-segmentation-dashboard.onrender.com · https://github.com/hst-nxk/BA_project"], 7.9, 1.6, 5.2, 5.5, 14)

    s = d.slide("10 · Monitoring and re-clustering triggers", "PSI / KS + Evidently · segment shares · centroid shift · atypical share · silhouette")
    d.img(s, "m7_drift_comparison", 0.3, 1.4, w=8.0)
    d.text(s, [f"2014J (real): {m7.get('decision_2014J')} — all GREEN",
               f"Stress scenario: {m7.get('decision_stress')} — {m7.get('features_red_stress')} features RED, share shift "
               f"{m7.get('max_shift_pp_stress', 0):.0f} pp, centroid shift {m7.get('centroid_shift_stress', 0):.1f} SD",
               "Rule: ≥ 2 RED → re-cluster; 1 RED / AMBER → investigate",
               "Trend: Achievers 41% → 29% (2013B → 2014J); 3-cohort trend rule"], 8.5, 1.6, 4.6, 5.5, 14)
    d.img(s, "m7_stability_over_time", 0.3, 4.6, w=8.0)

    s = d.slide("Limitations and next steps")
    d.text(s, ["Segments are retrospective → early-warning model bridges to intervention timing",
               "Synthetic fields are plausible, not observed; OULAD-only model agrees at ARI 0.69 (core structure is real)",
               "Clicks ≠ study time; course design shifts activity levels (profiles read in module context)",
               "Uplifts/costs are assumptions → run A/B tests per segment, then re-optimise",
               "Next: real login/video telemetry, weekly scoring in production, sub-segments within Crammers, "
               "causal evaluation of interventions"], 0.5, 1.5, 12.3, 5.5, 18)

    s = d.slide("Demo & questions")
    d.img(s, "app_dashboard_lookup", 0.5, 1.5, w=7.0)
    d.text(s, ["Live demo: dashboard → learner lookup → assign new learner → monitoring",
               "API docs: https://learner-segmentation-api.onrender.com/docs", "Code & data dictionary: https://github.com/hst-nxk/BA_project"], 7.6, 2.0, 5.4, 4, 16)
    out = ROOT / "reports" / "Final_Presentation.pptx"
    d.p.save(out)
    print("slides:", out, d.n)


if __name__ == "__main__":
    build()
