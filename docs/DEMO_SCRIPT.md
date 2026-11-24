# Presentation and live-demo script (about 15 minutes)

Before class:
* Open https://learner-segmentation-api.onrender.com/health and https://learner-segmentation-dashboard.onrender.com about 5 minutes before you present, so that free-tier instances wake up.
* Keep `reports/figures/app_*.png` as a fallback. If the network fails, show them, or play a recorded demo.

| Time | Slide / screen | Speaker | What to say |
|---|---|---|---|
| 0:00 | Slides 1–2 | Data Engineer | Business question; 52.8% at-risk rate; blanket tutor calls would cost $602k. KPIs |
| 1:30 | Slide 3 | Data Engineer | OULAD + synthetic layer anchored to each real learner. Realism checks (ρ 0.99 / 0.91 / 0.82). Data dictionary |
| 3:00 | Slides 4–5 | Data Analyst | Two independent behaviour axes: volume and pacing. Logins vs score; video vs timeliness. STEM vs Social Sciences; cohort drift |
| 5:00 | Slide 6 | Data Engineer | Cleaning log. Bot detection: one IQR rule gives 112 false positives; 2-of-4 rules give precision and recall 1.0 |
| 6:00 | Slide 7 | ML Eng. Modeling | RFM features. Why Yeo-Johnson + z-score (score would take 99.8% of the distance). PCA = engagement + pacing |
| 7:30 | Slides 8–9 | ML Eng. Evaluation | Elbow says 4 but k = 4 is unstable → k = 3. K-Means vs Ward vs DBSCAN |
| 9:00 | Slides 10–11 | ML Eng. Evaluation | Four personas; pass rate 0% → 90% without using the outcome. Support per persona |
| 10:30 | Slides 12–13 | ML Eng. Evaluation | Day-28 early warning. LP: $133 vs $424 per extra pass; tutor hours are binding |
| 11:30 | **Live demo** | MLOps Lead | see below |
| 14:00 | Slides 15–16 | MLOps Lead | Monitoring: 2014J NO ACTION, stress scenario RE-CLUSTER. Limitations, next steps |

## Live demo (2–3 minutes)
1. **Dashboard → Overview.** Filter Course type = STEM. KPIs and segment mix update.
2. **Segment profiles.** Radar chart: Crammers spike on deadline ratio and last-minute share. Open the Crammers
   expander to show the recommendations.
3. **Learner lookup.** Type a learner ID (example IDs are listed under the box). Show the segment, confidence, the
   comparison with the segment median and the recommended support.
4. **Assign a new learner.** Click *Assign segment* with the default values → **Consistent Achievers**.
   * Then set Active days = 30, Clicks on deadline days = 800, Submissions in final 24 h = 5, Forum posts = 0,
     Night logins = 40 → **Last-minute Crammers** (confidence ≈ 30%).
   * Then set Active days = 10, Last active day = 60, Total clicks = 150, Clicks on deadline days = 20,
     Scores = "40, 30", Submissions in final 24 h = 0, Videos 10 / 4, Logins = 15, Night logins = 2
     → **At-risk Disengaged** (confidence ≈ 63%).
   * Point out "source: API": the dashboard called the deployed FastAPI service. Confidence =
     1 − d(nearest centroid) / d(second nearest), so 30% means clearly closer to one segment.
5. **API docs** (`https://learner-segmentation-api.onrender.com/docs`). `POST /predict` → *Try it out* → Execute; then send `avg_score_scheduled = 140`
   to show the 422 validation error.
6. **Monitoring tab.** NO ACTION vs RE-CLUSTER, the PSI chart, and the embedded Evidently report.

## Likely questions
* **Why k = 3 when the elbow suggests 4?** At k = 4 the bootstrap ARI minimum is 0.70 and the seed ARI is 0.72:
  the partition changes with the sample. Our acceptance rule is stability ≥ 0.80.
* **Silhouette 0.21 is low?** Behaviour is a continuum. We report confidence and an atypical flag per learner, and
  validate with stability (0.96), temporal ARI (0.84) and outcome separation (pass rate 0–90%).
* **Is the synthetic data realistic, and does it drive the result?** It is anchored to real records (ρ up to 0.99). A
  model built only on real OULAD features agrees at ARI 0.69, so the core structure is real.
* **Why not DBSCAN?** It finds one dense continuum: 91% of learners in a single cluster. That is useful for outliers,
  not for tailoring support.
* **Why exclude demographics?** Fairness. Segments are still only weakly associated with them (Cramér's V ≤ 0.11).
* **When do you re-cluster?** When ≥ 2 checks are RED (PSI, segment share, centroid shift, atypical share,
  silhouette); see the monitoring playbook.
