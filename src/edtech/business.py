"""Personas, recommendations and prescriptive analytics (Modules 1, 5, 8).

* Persona naming: Hungarian matching of cluster profiles to behavioural archetypes,
  so names are reproducible and explainable rather than hand-picked.
* Decision analysis (Lecture 2): expected value of intervening vs not, per segment,
  with break-even uplift.
* Optimisation (Lecture 3.2): linear program allocating a support budget and tutor
  capacity across segments and intervention types (scipy HiGHS, with shadow prices).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment, linprog

from . import config as C

# archetype -> weights on standardised cluster centroids
ARCHETYPES = {
    "Consistent Achievers": {"active_day_fraction": 1.0, "avg_score_scheduled": 1.0, "video_completion_rate": 0.8,
                             "forum_posts_per_week": 1.0, "recency_fraction": -0.6, "last_minute_submission_share": -0.5,
                             "clicks_per_active_day": 0.5},
    "Last-minute Crammers": {"deadline_click_ratio": 1.0, "last_minute_submission_share": 1.2, "weekly_cv_active_span": 0.6,
                             "night_login_share": 0.3, "avg_score_scheduled": -0.2},
    "At-risk Disengaged": {"recency_fraction": 1.2, "avg_score_scheduled": -1.0, "active_day_fraction": -0.8,
                           "video_completion_rate": -0.3},
    "Steady Independent Learners": {"avg_score_scheduled": 0.8, "recency_fraction": -0.8, "forum_posts_per_week": -0.6,
                                    "last_minute_submission_share": -0.4, "weekly_cv_active_span": 0.2, "night_login_share": -0.5},
    "Night-owl Learners": {"night_login_share": 1.5, "avg_score_scheduled": 0.2},
    "Struggling Strivers": {"active_day_fraction": 0.4, "avg_score_scheduled": -0.8, "recency_fraction": -0.4},
    "Fading Starters": {"recency_fraction": 0.8, "avg_score_scheduled": -0.6, "clicks_per_active_day": 0.3},
}

NON_STARTER = {"segment_id": "S0", "name": "Non-starters",
               "tagline": "Registered but no recorded learning activity in the course window"}

RECOMMENDATIONS = {
    "Non-starters": {
        "content": "Short 'first 15 minutes' onboarding module and a visible week-1 checklist.",
        "pacing": "Soft-start deadline: first activity expected within 7 days of course start.",
        "support": "Automated welcome sequence on day 1, 3 and 7; tutor call if still inactive on day 10; offer deferral.",
        "kpi": "Share starting within 14 days; withdrawal before day 30.",
        "intervention": "automated_nudges",
    },
    "At-risk Disengaged": {
        "content": "Condensed 'catch-up path' covering assessed outcomes only; mobile-friendly micro lessons.",
        "pacing": "Re-planned personal schedule with one achievable milestone per week; extension policy explained.",
        "support": "Proactive tutor outreach call when recency > 14 days; refer to student-support services.",
        "kpi": "Recency < 7 days, next assessment submitted.",
        "intervention": "tutor_outreach_call",
    },
    "Last-minute Crammers": {
        "content": "Break assessments into checkpoints; release practice quizzes 7-10 days before each deadline.",
        "pacing": "Interim mini-deadlines and calendar nudges 5 and 2 days before deadlines (not at night).",
        "support": "Study-plan coaching session on time management; peer accountability pairs.",
        "kpi": "Last-minute submission share, deadline click ratio.",
        "intervention": "study_plan_coaching",
    },
    "Steady Independent Learners": {
        "content": "Self-paced enrichment material and optional stretch tasks; keep navigation lean.",
        "pacing": "Learner-controlled pacing; allow early access to next unit.",
        "support": "Light-touch: progress dashboard, invite to forums as contributors (not required).",
        "kpi": "Retention of on-time submission and pass rate.",
        "intervention": "automated_nudges",
    },
    "Consistent Achievers": {
        "content": "Advanced / extension content, recognition badges, pathways to next module.",
        "pacing": "Standard or accelerated pacing.",
        "support": "Recruit as peer mentors / forum moderators; minimal tutor time needed.",
        "kpi": "Distinction rate, progression to next module.",
        "intervention": "peer_study_group",
    },
    "Night-owl Learners": {
        "content": "Ensure on-demand access (recordings, asynchronous forums) for late-hour study.",
        "pacing": "Avoid midnight deadlines; schedule live sessions in the evening.",
        "support": "Wellbeing tips on study timing; asynchronous tutor Q&A.",
        "kpi": "Deadline-day night submissions, pass rate.",
        "intervention": "automated_nudges",
    },
    "Struggling Strivers": {
        "content": "Worked examples and formative quizzes with feedback.",
        "pacing": "Keep standard pacing but add feedback loops after each assessment.",
        "support": "Targeted academic skills support and tutor feedback on assignments.",
        "kpi": "Score improvement between consecutive assessments.",
        "intervention": "study_plan_coaching",
    },
    "Fading Starters": {
        "content": "Re-engagement digest summarising what was missed.",
        "pacing": "Flexible catch-up window.",
        "support": "Automated nudge then tutor check-in.",
        "kpi": "Return within 14 days.",
        "intervention": "tutor_outreach_call",
    },
}


def name_clusters(z_centroids: pd.DataFrame) -> dict[int, str]:
    """Assign each cluster the archetype with best fit (one-to-one, maximise total score)."""
    names = list(ARCHETYPES)
    S = np.zeros((len(z_centroids), len(names)))
    for j, n in enumerate(names):
        w = pd.Series(ARCHETYPES[n]).reindex(z_centroids.columns).fillna(0)
        S[:, j] = z_centroids.to_numpy() @ w.to_numpy() / np.abs(w).sum()
    r, c = linear_sum_assignment(-S)
    return {int(z_centroids.index[i]): names[j] for i, j in zip(r, c)}


# --------------------------------------------------------------------- decision analysis
def segment_uplift_matrix(segments: list[str]) -> pd.DataFrame:
    """Assumed absolute pass-probability uplift of each intervention in each segment.

    Base uplift from config.INTERVENTION_CATALOGUE scaled by a segment fit multiplier.
    These are planning assumptions to be replaced by A/B-test estimates.
    """
    fit = {
        "Non-starters": {"automated_nudges": 1.0, "study_plan_coaching": 0.6, "tutor_outreach_call": 1.2, "peer_study_group": 0.3},
        "At-risk Disengaged": {"automated_nudges": 0.8, "study_plan_coaching": 1.0, "tutor_outreach_call": 1.5, "peer_study_group": 0.6},
        "Last-minute Crammers": {"automated_nudges": 1.2, "study_plan_coaching": 1.6, "tutor_outreach_call": 0.9, "peer_study_group": 1.1},
        "Steady Independent Learners": {"automated_nudges": 0.5, "study_plan_coaching": 0.4, "tutor_outreach_call": 0.3, "peer_study_group": 0.5},
        "Consistent Achievers": {"automated_nudges": 0.2, "study_plan_coaching": 0.2, "tutor_outreach_call": 0.1, "peer_study_group": 0.3},
        "Night-owl Learners": {"automated_nudges": 1.0, "study_plan_coaching": 0.8, "tutor_outreach_call": 0.6, "peer_study_group": 0.7},
        "Struggling Strivers": {"automated_nudges": 0.7, "study_plan_coaching": 1.4, "tutor_outreach_call": 1.1, "peer_study_group": 1.0},
        "Fading Starters": {"automated_nudges": 1.0, "study_plan_coaching": 0.9, "tutor_outreach_call": 1.3, "peer_study_group": 0.6},
    }
    rows = []
    for s in segments:
        for i, (cost, minutes, base) in C.INTERVENTION_CATALOGUE.items():
            rows.append({"persona": s, "intervention": i, "cost": cost, "tutor_minutes": minutes,
                         "uplift": base * fit.get(s, {}).get(i, 0.5)})
    return pd.DataFrame(rows)


def decision_tree_ev(seg: pd.DataFrame, value: float = C.VALUE_OF_RETAINED_LEARNER) -> pd.DataFrame:
    """Per learner: EV(intervene) = (p0+u)V - cost; EV(no action) = p0 V. Sunk costs ignored."""
    um = segment_uplift_matrix(seg["persona"].tolist())
    rows = []
    for _, r in seg.iterrows():
        opts = um[um["persona"] == r["persona"]]
        best = None
        for _, o in opts.iterrows():
            ev_int = (min(r["pass_rate"] + o["uplift"], 1)) * value - o["cost"]
            ev_no = r["pass_rate"] * value
            cand = {"persona": r["persona"], "intervention": o["intervention"], "pass_rate": r["pass_rate"],
                    "uplift": o["uplift"], "cost": o["cost"], "ev_no_action": ev_no, "ev_intervene": ev_int,
                    "net_gain_per_learner": ev_int - ev_no, "break_even_uplift": o["cost"] / value}
            if best is None or cand["net_gain_per_learner"] > best["net_gain_per_learner"]:
                best = cand
        rows.append(best)
    return pd.DataFrame(rows)


def optimise_allocation(seg_sizes: dict[str, int], budget: float, tutor_hours: float,
                        value: float = C.VALUE_OF_RETAINED_LEARNER) -> dict:
    """max sum x_{s,i} * uplift_{s,i}  s.t. budget, tutor minutes, sum_i x_{s,i} <= n_s, x >= 0.

    Only options with uplift x value >= cost (non-negative expected net value) are eligible.
    """
    um = segment_uplift_matrix(list(seg_sizes))
    # economically dominated options (expected value of the uplift below its cost) are not offered
    um = um[um["uplift"] * value >= um["cost"]].reset_index(drop=True)
    n = len(um)
    c = -um["uplift"].to_numpy()                                   # linprog minimises
    A, b, names = [um["cost"].to_numpy(), um["tutor_minutes"].to_numpy()], [budget, tutor_hours * 60], ["budget", "tutor_minutes"]
    for s, size in seg_sizes.items():
        A.append((um["persona"] == s).astype(float).to_numpy())
        b.append(size)
        names.append(f"capacity:{s}")
    res = linprog(c, A_ub=np.vstack(A), b_ub=np.array(b), bounds=[(0, None)] * n, method="highs")
    um["learners_allocated"] = np.round(res.x, 1)
    um["expected_extra_passes"] = um["learners_allocated"] * um["uplift"]
    um["spend"] = um["learners_allocated"] * um["cost"]
    shadow = dict(zip(names, (-res.ineqlin.marginals).round(5)))
    return {"status": res.message, "expected_extra_passes": float(-res.fun),
            "spend": float(um["spend"].sum()), "tutor_hours_used": float((um["learners_allocated"] * um["tutor_minutes"]).sum() / 60),
            "allocation": um[um["learners_allocated"] > 0.05], "shadow_prices": shadow}
