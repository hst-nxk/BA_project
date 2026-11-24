"""Learner Segments dashboard (Module 6/7) - Streamlit.

Run locally:  streamlit run app/dashboard/app.py
If API_URL is set (env var or Streamlit secret), predictions are made by the deployed FastAPI
service; otherwise the same model is loaded in-process, so the dashboard also runs standalone.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import requests  # noqa: E402
import streamlit as st  # noqa: E402
import streamlit.components.v1 as components  # noqa: E402

from edtech import config as C  # noqa: E402
from edtech.serving import RawActivityProfile, features_from_raw, load_model, load_personas, load_rules, predict_features  # noqa: E402

COLORS = {"Non-starters": "#8a8984", "At-risk Disengaged": "#e34948", "Last-minute Crammers": "#eb6834",
          "Consistent Achievers": "#1baf7a", "Steady Independent Learners": "#2a78d6"}
ORDER = ["Consistent Achievers", "Last-minute Crammers", "At-risk Disengaged", "Non-starters"]
STATUS = {"GREEN": "🟢", "AMBER": "🟠", "RED": "🔴"}

st.set_page_config(page_title="Learner Segments", page_icon="🎓", layout="wide")


def api_url() -> str | None:
    try:
        return st.secrets.get("API_URL") or os.environ.get("API_URL")
    except Exception:
        return os.environ.get("API_URL")


@st.cache_data
def lookup() -> pd.DataFrame:
    return pd.read_parquet(C.APP_DATA / "learner_lookup.parquet")


@st.cache_data
def monitoring() -> dict:
    p = C.MONITORING_DATA / "monitoring_summary.json"
    return json.loads(p.read_text()) if p.exists() else {}


def predict(features: dict) -> dict:
    url = api_url()
    if url:
        try:
            r = requests.post(f"{url.rstrip('/')}/predict", json=features, timeout=20)
            r.raise_for_status()
            return {**r.json(), "_source": f"API {url}"}
        except Exception as exc:
            st.warning(f"API unreachable ({exc}); using the in-process model.")
    return {**predict_features([features])[0], "_source": "in-process model"}


personas = load_personas()
seg_meta = {s["name"]: s for s in personas["segments"]}
df = lookup()
model = load_model()

st.title("🎓 Learner Segmentation — Online Learning Platform")
st.caption("Behavioural segments of 32,593 OULAD enrollments (real clickstream + synthetic behavioural extension). "
           f"Model v{model.version} · K-Means k={model.kmeans.n_clusters} + rule-based Non-starters · trained on {', '.join(model.trained_on)}")

tab_over, tab_prof, tab_look, tab_new, tab_mon, tab_about = st.tabs(
    ["Overview", "Segment profiles", "Learner lookup", "Assign a new learner", "Monitoring", "About the model"])

# ------------------------------------------------------------------------------------- overview
with tab_over:
    f1, f2, f3 = st.columns(3)
    mods = f1.multiselect("Module", sorted(df.code_module.unique()), default=sorted(df.code_module.unique()))
    pres = f2.multiselect("Presentation (cohort)", sorted(df.code_presentation.unique()), default=sorted(df.code_presentation.unique()))
    dom = f3.multiselect("Course type", sorted(df.course_domain.unique()), default=sorted(df.course_domain.unique()))
    d = df[df.code_module.isin(mods) & df.code_presentation.isin(pres) & df.course_domain.isin(dom)]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Enrollments", f"{len(d):,}")
    k2.metric("Median engagement score", f"{d.engagement_score.median():.0f} / 100")
    k3.metric("At-risk learner rate", f"{d.final_result.isin(['Fail', 'Withdrawn']).mean():.1%}")
    k4.metric("Needs support (S0 + At-risk + Crammers)", f"{(~d.persona.eq('Consistent Achievers')).mean():.1%}")
    c1, c2 = st.columns(2)
    sz = d.persona.value_counts().rename_axis("persona").reset_index(name="enrollments")
    fig = px.bar(sz, x="enrollments", y="persona", orientation="h", color="persona", color_discrete_map=COLORS,
                 text=sz.enrollments.map("{:,}".format), category_orders={"persona": ORDER})
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(showlegend=False, title="Segment size", yaxis_title=None, height=360)
    c1.plotly_chart(fig, use_container_width=True)
    out = pd.crosstab(d.persona, d.final_result, normalize="index").reset_index().melt(id_vars="persona", var_name="final_result", value_name="share")
    fig = px.bar(out, x="share", y="persona", color="final_result", orientation="h",
                 color_discrete_map={"Distinction": "#1baf7a", "Pass": "#2a78d6", "Fail": "#eda100", "Withdrawn": "#e34948"},
                 category_orders={"final_result": ["Distinction", "Pass", "Fail", "Withdrawn"], "persona": ORDER})
    fig.update_layout(title="Final result by segment", yaxis_title=None, height=360, xaxis_tickformat=".0%", barmode="stack")
    fig.update_traces(offsetgroup="all")  # px assigns one offsetgroup per colour, which defeats stacking
    c2.plotly_chart(fig, use_container_width=True)
    mix = pd.crosstab(d.code_presentation, d.persona, normalize="index").reset_index().melt(id_vars="code_presentation", value_name="share")
    fig = px.line(mix, x="code_presentation", y="share", color="persona", color_discrete_map=COLORS, markers=True)
    fig.update_layout(title="Segment share by cohort", yaxis_tickformat=".0%", height=330)
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------------------------- profiles
with tab_prof:
    feats = personas["features"]
    rows = [{"persona": s["name"], **{f: s["profile_medians"][f] for f in feats}} for s in personas["segments"] if s["name"] != "Non-starters"]
    prof = pd.DataFrame(rows).set_index("persona")
    pct = df[df.persona != "Non-starters"][feats].rank(pct=True).groupby(df.persona).median()
    fig = go.Figure()
    for p in pct.index:
        fig.add_trace(go.Scatterpolar(r=list(pct.loc[p]) + [pct.loc[p].iloc[0]], theta=feats + [feats[0]], name=p,
                                      line=dict(color=COLORS.get(p)), fill="toself", opacity=0.55))
    fig.update_layout(title="Behavioural fingerprint (median percentile)", polar=dict(radialaxis=dict(range=[0, 1])), height=520)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(prof.T.style.format("{:.2f}"), use_container_width=True)
    for s in personas["segments"]:
        with st.expander(f"{s['segment_id']} · {s['name']} — {s['kpis']['share']:.0%} of enrollments, pass rate {s['kpis']['pass_rate']:.0%}"):
            st.write(s["description"])
            r = s["recommendations"]
            a, b, c = st.columns(3)
            a.markdown(f"**Content**\n\n{r['content']}")
            b.markdown(f"**Pacing**\n\n{r['pacing']}")
            c.markdown(f"**Support**\n\n{r['support']}")
            st.caption(f"Primary intervention: {r['intervention']} · KPI to track: {r['kpi']}")

# --------------------------------------------------------------------------------------- lookup
with tab_look:
    st.subheader("Look up a learner's segment")
    c1, c2, c3 = st.columns([2, 1, 1])
    sid = c1.text_input("Learner ID (id_student)", value=str(df.id_student.iloc[0]))
    try:
        r = df[df.id_student == int(sid)]
    except ValueError:
        r = pd.DataFrame(); st.error("Learner ID must be a number")
    if not r.empty:
        mods_ = c2.selectbox("Module", sorted(r.code_module.unique()))
        r = r[r.code_module == mods_]
        pres_ = c3.selectbox("Presentation", sorted(r.code_presentation.unique()))
        row = r[r.code_presentation == pres_].iloc[0]
        meta = seg_meta[row.persona]
        st.markdown(f"### {row.segment_id} · <span style='color:{COLORS.get(row.persona)}'>{row.persona}</span>", unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Engagement score", f"{row.engagement_score:.0f}")
        m2.metric("Logins / week", f"{row.logins_per_week:.1f}")
        m3.metric("Submission rate", f"{row.submission_rate:.0%}")
        m4.metric("Assignment confidence", f"{row.assignment_confidence:.0%}", help="1 − d(nearest)/d(second nearest centroid)")
        if row.is_atypical:
            st.info("Atypical profile: far from every segment centre — review individually.")
        comp = pd.DataFrame({"this learner": row[feats].astype(float), "segment median": pd.Series(meta["profile_medians"])[feats]})
        st.dataframe(comp.style.format("{:.3f}"), use_container_width=True)
        rec = meta["recommendations"]
        st.success(f"**Recommended support:** {rec['support']}\n\n**Pacing:** {rec['pacing']}\n\n**Content:** {rec['content']}")
        st.caption(f"Historical outcome (not used by the model): {row.final_result}")
    elif sid.isdigit():
        st.warning("No enrollment found for this learner ID.")
    st.caption("Example IDs: " + ", ".join(str(x) for x in df.groupby("persona").id_student.first().tolist()))

# ----------------------------------------------------------------------------------- new learner
with tab_new:
    st.subheader("Assign a new learner from an activity profile")
    st.write("Enter what the LMS records for one learner in one course. The profile is converted to the model "
             "features with the training definitions and sent to the segmentation API.")
    with st.form("raw"):
        a, b, c = st.columns(3)
        L = a.number_input("Course length (days)", 28, 400, 240)
        elapsed = a.number_input("Days observed so far", 1, 400, 240)
        active_days = a.number_input("Active days", 0, 400, 60)
        last_day = a.number_input("Last active course day", 0, 400, 230)
        total_clicks = a.number_input("Total resource clicks", 0, 1_000_000, 1200)
        dl_clicks = b.number_input("Clicks on deadline days (incl. 2 days before)", 0, 1_000_000, 300)
        dl_days = b.number_input("Number of such days in course", 1, 400, 27)
        sched = b.number_input("Scheduled assessments", 1, 50, 9)
        scores = b.text_input("Scores of submitted assessments (comma separated)", "72, 80, 65, 77, 70, 68, 81")
        last24 = b.number_input("Submissions made in final 24 h", 0, 50, 2)
        vs = c.number_input("Videos started", 0, 10_000, 120)
        vc = c.number_input("Videos completed", 0, 10_000, 80)
        posts = c.number_input("Forum posts", 0, 10_000, 6)
        logins = c.number_input("Logins", 0, 100_000, 110)
        night = c.number_input("Logins between 22:00 and 06:00", 0, 100_000, 20)
        go_ = st.form_submit_button("Assign segment", type="primary")
    if go_:
        try:
            sc = [float(x) for x in scores.split(",") if x.strip()]
            raw = RawActivityProfile(course_length_days=L, days_elapsed=elapsed, active_days=active_days, last_active_day=last_day,
                                     total_clicks=total_clicks, deadline_window_clicks=dl_clicks, deadline_window_days=dl_days,
                                     scheduled_assessments=sched, scores=sc, submissions_last_24h=last24, videos_started=vs,
                                     videos_completed=vc, forum_posts=posts, logins=logins, night_logins=night)
            feats_ = features_from_raw(raw)
            res = predict(feats_)
            st.markdown(f"### → {res['segment_id']} · <span style='color:{COLORS.get(res['persona'])}'>{res['persona']}</span>",
                        unsafe_allow_html=True)
            st.caption(f"confidence {res['assignment_confidence']:.0%} · source: {res['_source']}")
            st.write("**Main drivers (standardised):** " + ", ".join(f"{d['feature']} = {d['value']} (z = {d['z']})" for d in res["top_drivers"]))
            if res.get("recommendations"):
                st.success(f"**Support:** {res['recommendations']['support']}\n\n**Pacing:** {res['recommendations']['pacing']}")
            st.json({k: round(v, 4) for k, v in feats_.items()}, expanded=False)
        except Exception as exc:
            st.error(f"Invalid profile: {exc}")

# ------------------------------------------------------------------------------------ monitoring
with tab_mon:
    mon = monitoring()
    if not mon:
        st.warning("No monitoring run yet.")
    else:
        for key, title in [("latest", "Latest incoming cohort"), ("stress_test", "Stress scenario (simulated platform change)")]:
            s = mon[key]
            st.markdown(f"### {title}: **{s['decision']}**")
            cols = st.columns(len(s["checks"]))
            for col, (k, v) in zip(cols, s["checks"].items()):
                col.metric(k.replace("_", " "), f"{STATUS.get(v, '')} {v}")
            dr = pd.DataFrame(s["feature_drift"])
            fig = px.bar(dr, x="psi", y="feature", orientation="h", color="status",
                         color_discrete_map={"GREEN": "#1baf7a", "AMBER": "#eda100", "RED": "#e34948"})
            fig.add_vline(x=0.10, line_dash="dash", line_color="#eda100"); fig.add_vline(x=0.25, line_dash="dash", line_color="#e34948")
            fig.update_layout(height=340, title="Population Stability Index per feature")
            st.plotly_chart(fig, use_container_width=True)
        ot = pd.DataFrame(mon["over_time"])
        long = ot.melt(id_vars="presentation", value_vars=[c for c in ot if c in COLORS], var_name="persona", value_name="share")
        fig = px.line(long, x="presentation", y="share", color="persona", markers=True, color_discrete_map=COLORS)
        fig.update_layout(title="Segment share by cohort (frozen model)", yaxis_tickformat=".0%", height=330)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("**Re-clustering rule:** ≥ 2 RED checks → re-cluster; 1 RED or any AMBER → investigate.")
        st.json(mon["triggers"], expanded=False)
        rep = C.MONITORING_REPORTS / mon["evidently_reports"]["2014J"]
        if rep.exists() and st.toggle("Show Evidently data-drift report (2014J)"):
            components.html(rep.read_text(encoding="utf-8"), height=900, scrolling=True)

# ----------------------------------------------------------------------------------------- about
with tab_about:
    st.markdown((ROOT / "docs" / "MODEL_CARD.md").read_text(encoding="utf-8") if (ROOT / "docs" / "MODEL_CARD.md").exists() else "")
    with st.expander("Decision rules (surrogate tree, depth 4)"):
        st.code(load_rules())
    st.caption(f"API: {api_url() or 'not configured — in-process model'}")
