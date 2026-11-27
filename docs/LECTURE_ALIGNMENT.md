# Alignment with the course lectures

Where each concept from Lectures 1–10 is applied in the project (notebook · section).

| Lecture | Concept taught | Applied in this project |
|---|---|---|
| **1 – Introduction to BI & Analytics** | Descriptive / predictive / prescriptive analytics | NB01 §intro maps every module to one of the three types: EDA and profiles (descriptive); segment assignment, early warning and drift detection (predictive); decision-tree EV and LP allocation (prescriptive) |
| | Metrics vs measures; discrete vs continuous metrics | KPI table in NB01 §1: engagement score, at-risk rate, intervention cost, cost per additional pass |
| | Nominal / ordinal / interval / ratio scales | NB01 §2 "Measurement scales" decides how each field is encoded or excluded |
| **2 – Decision analysis** | Decision-making model, decision trees, expected value, sunk costs, framing | NB05b §6: EV(intervene) vs EV(no action) per segment, break-even uplift, sensitivity of net gain to the true uplift, sunk costs excluded |
| **2 / 4 – OLAP, data warehouse, column store** | Star schema, fact and dimension tables, roll-up, drill-down, slice, dice, pivot; column-oriented storage | NB02 §7: DuckDB (columnar) warehouse `data/warehouse/learning_dw.duckdb` with `fact_enrollment` + `dim_learner`, `dim_course`, `dim_presentation`; `GROUP BY ROLLUP`, drill-down + dice query, slice + `PIVOT`. All intermediate data stored as columnar Parquet |
| **3.1 – Decision support systems** | DSS components: data management, model management, user interface | NB06 + `app/`: learner lookup / feature store (data), feature pipeline + K-Means + persona rules (model), Streamlit dashboard and FastAPI (UI); semi-structured decision "who gets which support" |
| **3.2 – Optimisation** | Decision variables, objective, constraints, LP, feasibility, sensitivity analysis (shadow prices) | NB05b §7: LP allocating interventions across segments under budget, tutor-hour and segment-size constraints (scipy HiGHS); shadow prices identify tutor hours as the binding constraint; budget and capacity sensitivity curves |
| **5 – Spark introduction** | Spark architecture, local mode, lazy transformations and actions | NB01 §3 / `src/edtech/etl_spark.py`: 10.6M-row clickstream read with an explicit schema, broadcast join with the `vle` dimension, aggregation to enrollment × day |
| **6 – Pair RDD and DataFrame** | `map`, `reduceByKey`, DataFrame `groupBy/agg/pivot` | Same module: DataFrame API for the main aggregation and pivot by activity type; an independent Pair-RDD job (`map → reduceByKey`) recomputes total clicks per enrollment and must match exactly (39,605,099 clicks both ways) |
| **7 – Data preparation** | Dirty data: incomplete, noisy, inconsistent | NB03 quality scorecard by dimension and consistency checks |
| | Missing data: mechanisms, ignore / global constant / class mean / inference; avoid bias, keep the information that a value was missing | NB03 §2: structural missing dates kept as NA; IMD → "Unknown" + flag (global constant); session minutes → learner median (same-class mean) + flag |
| | Outliers: z-score (k = 2–3), boxplot mild (1.5 IQR) / extreme (3 IQR), multivariate (clustering / distance-based) | NB02 §1 outlier table; NB03 §3a course-specific 3-IQR fences; §3b multi-rule bot detection; NB05 DBSCAN noise and `is_atypical` distance flag |
| | Treatments: do nothing, enforce bounds, binning/transform | Extreme but plausible learners kept; winsorisation at p0.5/p99.5 + Yeo-Johnson inside the feature pipeline |
| | Normalisation: min-max, z-score, decimal scaling | NB04 §4b compares none / min-max / z-score / robust by each feature's share of variance in the distance |
| | Transformations: log, Box-Cox (λ search), rank → normal scores, "spreading the histogram" | NB04 §4a skewness table for log1p, Box-Cox, Yeo-Johnson and rank-normal; rank-normal rejected because it cannot be applied separately to train/new data (as the lecture warns) |
| | Nominal → binary (1-to-N), ordinal → numeric | NB04 §3 one-hot context table and ordinal codes for IMD, education and age |
| | Data reduction: PCA | NB04 §5c scree plot, loadings, PCA-space vs full-space clustering comparison |
| **8 – Data mining / KDD** | KDD stages; descriptive vs predictive tasks; clustering vs classification | Notebook sequence follows KDD: selection (NB01) → cleaning/integration (NB03) → transformation (NB04) → mining (NB05) → interpretation/evaluation (NB05b–07) |
| **8.1 – Time series** | Components (trend, seasonal, cyclical, irregular), moving average, exponential smoothing | NB02 §6 weekly clicks by outcome with exponential smoothing; daily series with a 7-day moving average against assessment deadlines; NB05b §3 per-persona series; NB07 segment shares tracked across cohorts |
| **9 – Clustering** | Steps: feature selection → proximity → criterion → algorithm → validation → interpretation | NB04 (selection, redundancy filter, scaling) and NB05 (all remaining steps) |
| | K-Means (and its weaknesses), k-medoids idea | NB05 §2; weaknesses mitigated by transform + winsorisation; centroid medians reported in original units |
| | Hierarchical AGNES; single / complete / average / centroid linkage; dendrogram | NB05 §3 compares single / complete / average / Ward linkage, cophenetic correlation, dendrograms showing single-link chaining |
| | DBSCAN: Eps, MinPts, core/border/noise, parameter sensitivity | NB05 §4 k-distance graph, Eps × MinPts grid, noise-share analysis |
| | Number of clusters: empirical √(n/2), elbow, cross-validation | NB05 §2: √(n/2) discussed, elbow by chord distance, 5-fold held-out SSE |
| | Quality: internal (silhouette), external (purity, NMI, ARI…), relative | Silhouette, Davies–Bouldin, Calinski–Harabasz; purity / NMI / ARI vs `final_result`; relative comparison across k, linkages, algorithms; plus bootstrap and temporal stability |
| **10 – Classification** | Decision-tree induction, train/test split, applying the model to unseen data, confusion matrix; ensembles | NB05 §8 depth-4 surrogate decision tree explains segments (rules shown in the dashboard); NB05b §4 day-28 early-warning random forest (ensemble), tested on the unseen 2014J cohort with confusion matrix and per-class precision/recall |
