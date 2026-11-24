# Data Dictionary

Every synthetically generated field and every engineered feature. `L` = module_presentation_length. Synthetic generation code: `src/edtech/synthetic.py` (seed 42). Anchored = derived from the learner's real OULAD record plus noise.

## synthetic_login_sessions

| Column | Source | Type | Unit | Valid range | Generation logic / definition | Business assumption | Role |
|---|---|---|---|---|---|---|---|
| session_id | Synthetic | string (UUID4) | - | unique | Faker.uuid4() (seeded, Faker.seed(42)). | One row per authenticated session. | primary key |
| id_student | OULAD key | int | - | existing studentInfo id | Copied from OULAD enrollment. |  | foreign key |
| code_module | OULAD key | string | - | AAA..GGG | Copied from OULAD enrollment. |  | foreign key |
| code_presentation | OULAD key | string | - | 2013B..2014J | Copied from OULAD enrollment. |  | foreign key |
| course_day | Synthetic (anchored) | int | days relative to course start | -30 .. L | Days with real VLE clicks receive 1 + Poisson(habit x (0.25 + 0.004 x clicks)) sessions; extra click-less check-in logins ~ Poisson(0.05 x weeks x (0.5 + active_day_fraction)) on uniform random days before unregistration; non-starters ~ Poisson(1.5) in their first 40 days. | Every day with recorded clicks requires at least one login; heavier days need more sessions; some logins produce no clicks. | raw |
| login_ts | Synthetic | datetime | local time | within presentation calendar | Presentation start (1 Feb for B / 1 Oct for J, assumed) + course_day + login_hour + random minute/second. | Exact OU start dates are not published; 1st of month assumed. | raw |
| login_hour | Synthetic | int | hour of day | 0..23 | floor((chronotype centre + 2.5 x cramming x near_deadline + N(0, 2.0)) mod 24). Chronotype centres: morning 8.5, afternoon 14, evening 20, night 23.5; chronotype ~ Categorical(p / age band), shifted +0.20 to night for cramming > 0.7. | Younger learners study later; deadline pressure pushes study later in the day. | raw |
| device_type | Synthetic | category | - | desktop / mobile / tablet (clean) | 75% learner's primary device (Categorical by age band: 0-35 = 45/45/10, 35-55 = 62/26/12, 55+ = 70/12/18), 25% uniform random. 8% of rows deliberately mislabelled ('PC', 'MOBILE', 'tab'...), 1% set missing. | Device mix by age reflects common LMS usage patterns; label noise mimics multiple client apps. | raw |
| user_agent | Synthetic | string | - | - | Sampled from a pool of 3,000 Faker.user_agent() strings; bot sessions use 'python-requests/2.31.0'. |  | raw |
| session_minutes | Synthetic | float | minutes | 0 < x <= 1440 (clean) | exp(N(log(6 + 1.1 x clicks_in_session), 0.45)). Defects: 2% missing, 0.2% negative, 0.1% > 1440. | Duration grows with interaction volume; log-normal right tail. | raw |
| videos_started | Synthetic | int | count | >= 0 | Poisson(0.15 + 0.06 x content clicks in session), capped by clipping lambda at 8. | Video lectures are part of content resources (oucontent/resource/url...). | raw |
| videos_completed | Synthetic | int | count | 0..videos_started | Binomial(videos_started, clip(video propensity + N(0, 0.08))). Propensity ~ Beta(m x 10, (1 - m) x 10), m = 0.30 + 0.35 x engagement rank + 0.25 x performance rank - 0.12 x cramming. Defect: 0.3% rows completed > started. | More engaged and better-performing learners finish what they start. | raw |
| forum_posts | Synthetic | int | count | >= 0 | Poisson(0.02 x forum clicks in session x sociability), sociability ~ Gamma(1.2, 1) x (0.5 + engagement rank). | OULAD 'forumng' clicks measure reading; posting is a minority behaviour proportional to forum presence. | raw |
| forum_replies | Synthetic | int | count | 0..forum_posts | Binomial(forum_posts, 0.55). | About half of posts are replies to others. | raw |
| (bot accounts) | Synthetic defect | - | - | - | 0.4% of active enrollments receive 400-1,500 extra sessions: uniform hours, 0-0.4 minutes, automation user agent, no videos/posts. | Scripted scraping / credential-sharing traffic present on real platforms. | quality test |
| (duplicate rows) | Synthetic defect | - | - | - | 0.8% of rows duplicated exactly and shuffled. | Client retry / double logging. | quality test |

## synthetic_submission_events

| Column | Source | Type | Unit | Valid range | Generation logic / definition | Business assumption | Role |
|---|---|---|---|---|---|---|---|
| id_assessment | OULAD key | int | - | existing assessments id | One event per real OULAD non-exam, non-banked submission with a known deadline. |  | foreign key |
| submitted_ts | Synthetic (anchored) | datetime | local time | - | Real OULAD submission day + synthetic hour. |  | raw |
| deadline_ts | OULAD + assumption | datetime | local time | - | Real OULAD deadline day at 23:59. | Deadlines are assumed to close at 23:59. | raw |
| submit_hour | Synthetic | int | hour | 0..23 | If submitted on the deadline day: clip(N(17 + 5 x cramming, 2.5), 0, 23.98); otherwise chronotype centre + N(0, 2.5). | Deadline-day submissions concentrate in the evening, more so for deadline-driven learners. | raw |
| hours_before_deadline | Synthetic (anchored) | float | hours | negative = late | (deadline_day - submitted_day) x 24 + (23.983 - submit_hour_fraction). |  | raw |

## _latent_traits_audit (not a model input)

| Column | Source | Type | Unit | Valid range | Generation logic / definition | Business assumption | Role |
|---|---|---|---|---|---|---|---|
| trait_engagement_rank | Synthetic | float | percentile | 0..1 | Percentile rank of real OULAD active_day_fraction. |  | generation driver |
| trait_performance_rank | Synthetic | float | percentile | 0..1 | Percentile rank of real avg_score_scheduled. |  | generation driver |
| trait_cramming | Synthetic | float | index | 0..1 | clip(0.50 x rank(deadline_click_ratio) + 0.25 x rank(weekly_cv_active_span) + 0.25 x (1 - rank(mean_days_early)) + N(0, 0.12), 0, 1). | Real deadline-concentrated clicking and late submissions indicate cramming. | generation driver |
| trait_chronotype | Synthetic | category | - | morning/afternoon/evening/night | See login_hour. |  | generation driver |
| trait_session_habit | Synthetic | float | multiplier | > 0 | LogNormal(0, 0.6). | Some learners open many short sessions, others stay logged in. | generation driver |
| trait_video_propensity | Synthetic | float | probability | 0..1 | See videos_completed. |  | generation driver |
| trait_sociability | Synthetic | float | index | > 0 | Gamma(1.2, 1) x (0.5 + engagement rank). |  | generation driver |

## learner_features (enrollment grain)

| Column | Source | Type | Unit | Valid range | Generation logic / definition | Business assumption | Role |
|---|---|---|---|---|---|---|---|
| logins_per_week | Synthetic aggregate | float | logins / week | >= 0 | Cleaned sessions in [0, L] / ceil((L + 1) / 7). |  | profiling, API |
| login_week_coverage | Synthetic aggregate | float | share | 0..1 | Weeks with >= 1 login / course weeks. |  | profiling |
| night_login_share | Synthetic aggregate | float | share | 0..1 | Share of sessions starting 22:00-05:59. |  | model feature |
| preferred_login_period | Synthetic aggregate | category | - | morning/afternoon/evening/night | Modal period of login_hour. |  | profiling |
| mobile_share | Synthetic aggregate | float | share | 0..1 | Share of sessions on mobile. |  | profiling |
| mean_session_minutes | Synthetic aggregate | float | minutes | > 0 | Mean cleaned session duration. |  | profiling |
| video_completion_rate | Synthetic aggregate | float | share | 0..1 | (videos_completed + 2 x global rate) / (videos_started + 2) - Bayesian smoothing; 0 for learners with no logins. |  | model feature |
| forum_posts_per_week | Synthetic aggregate | float | posts / week | >= 0 | forum_posts / course weeks. |  | model feature |
| days_to_first_login | Synthetic aggregate | float | days | >= 0 | First login course_day - date_registration (clipped at 0). |  | profiling |
| last_minute_submission_share | Synthetic aggregate | float | share | 0..1 | Share of submissions made 0-24 h before the deadline (0 if none submitted). |  | model feature |
| median_hours_before_deadline | Synthetic aggregate | float | hours | any | Median hours_before_deadline. |  | profiling |
| recency_fraction | OULAD derived | float | share of course | 0..1 | (L - last active day) / (L + 1); 1 if never active. |  | model feature (R) |
| active_day_fraction | OULAD derived | float | share | 0..1 | Active VLE days in [0, L] / (L + 1). |  | model feature (F) |
| clicks_per_active_day | OULAD derived | float | clicks / active day | >= 0 | total_clicks / active_days. |  | model feature (M) |
| weekly_cv_active_span | OULAD derived | float | ratio | >= 0 | SD / mean of weekly clicks between first and last active week. |  | model feature |
| deadline_click_ratio | OULAD derived | float | ratio | >= 0 (1 = uniform) | Share of clicks on deadline day or 2 days before / share of course days that are such days. |  | model feature |
| avg_score_scheduled | OULAD derived | float | points | 0..100 | Sum of current non-exam scores / scheduled non-exam assessments (non-submission = 0). |  | model feature |
| submission_rate | OULAD derived | float | share | 0..1 | Current non-exam submissions / scheduled. |  | profiling |
| on_time_rate_scheduled | OULAD derived | float | share | 0..1 | On-time submissions / scheduled. |  | profiling |
| total_clicks | OULAD derived | int | clicks | >= 0 | Sum of sum_click in [0, L]. |  | profiling |
| late_click_share | OULAD derived | float | share | 0..1 | Clicks in final third / total clicks. |  | profiling |
| engagement_score | KPI | float | points | 0..100 | 100 x (0.30 pct(active_day_fraction) + 0.20 pct(logins_per_week) + 0.20 video_completion_rate + 0.20 submission_rate + 0.10 pct(forum_posts_per_week)). |  | KPI |
| at_risk_outcome | OULAD outcome | int | flag | 0/1 | final_result in {Fail, Withdrawn}. Never a model input. |  | KPI / external validation |
