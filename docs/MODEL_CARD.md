### Model card — Learner Behaviour Segmentation v1.0.0

**Purpose.** Assign each learner-course enrollment to a behavioural segment (persona) so that content, pacing and
support can be tailored. It is a decision aid for learner-success teams and tutors, not an automated sanction or
grading tool.

**Data.** OULAD (Open University Learning Analytics Dataset, CC BY 4.0): 32,593 enrollments, 7 modules, 4 cohorts
(2013B–2014J), 10.6M daily click records. It is extended with labelled **synthetic** behaviour (login sessions, time
of day, video completion, forum posts, hour-level submission timing) that is generated from each learner's real record
plus noise (seed 42; see `docs/DATA_DICTIONARY.md`).

**Model.**
* Rule: enrollments with no VLE activity in the course window → **S0 Non-starters**.
* Everyone else → K-Means (k = 3) on 10 standardised behavioural features (Yeo-Johnson for skewed inputs,
  winsorised at p0.5/p99.5, z-score). The pipeline is fitted on 2013B–2014B only.
* Features: recency fraction, active-day fraction, clicks per active day, weekly burstiness within the active span,
  deadline click ratio, average score over scheduled assessments, last-minute submission share*, video completion*,
  forum posts per week*, night-login share* (*synthetic).
* Excluded on purpose: identifiers, final result, and demographics (gender, age, disability, IMD, region, education).

| Segment | Persona | Share | Pass rate |
|---|---|---|---|
| S0 | Non-starters | 12.6% | 0.1% |
| S1 | Consistent Achievers | 32.1% | 89.6% |
| S2 | Last-minute Crammers | 34.4% | 53.3% |
| S3 | At-risk Disengaged | 20.9% | 0.3% |

**Validation.** Hopkins statistic 0.76 (clusterable). Silhouette 0.21 (train) / 0.19 (unseen 2014J). Bootstrap ARI
0.96 (minimum 0.90). Frozen model vs a K-Means refit on 2014J: ARI 0.84. External NMI vs final result: 0.31, which is
expected because segments describe behaviour, not outcomes. A depth-4 decision tree reproduces the segments with 87%
accuracy (explanation rules are available in the dashboard).

**Limitations.**
* The segments are **retrospective**: they use the whole course. For early action use the day-28 early-warning model
  (balanced accuracy 0.74 on 2014J); its weakest class is At-risk Disengaged (recall 0.55).
* Synthetic fields are plausible, not observed; their links to outcomes partly come from the generation logic. An
  OULAD-only model agrees with the full model at ARI 0.69, so the core structure comes from real data.
* Clicks are not study time. Course design affects activity levels, so read profiles in module context.
* The uplifts and costs used in the prescriptive analysis are planning assumptions to replace with A/B-test estimates.

**Fairness.** Demographics are not inputs. The association between segment and each attribute is weak
(Cramér's V ≤ 0.11). Disadvantaged groups are slightly over-represented in the support-needing segments, so support
targeted by segment reaches them without using protected attributes.

**Monitoring.** PSI / KS drift per feature, segment-share shift, centroid shift, atypical share and silhouette, with
GREEN/AMBER/RED triggers (`docs/MONITORING_PLAYBOOK.md`). 2014J: NO ACTION. Simulated platform change: RE-CLUSTER.
