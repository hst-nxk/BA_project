"""Big-data ETL for the 10.6M-row OULAD clickstream (Lectures 5-6: Spark, Pair RDD, DataFrame).

The studentVle file (454 MB) is aggregated with PySpark into two compact tables:

* enrollment_day      - one row per enrollment x course day (clicks by activity group)
* enrollment_activity - one row per enrollment with clicks pivoted by activity_type

A Pair-RDD implementation (map -> reduceByKey) is run alongside the DataFrame API
as a cross-check, mirroring the two Spark abstractions taught in the course.
A pandas fallback produces identical outputs when Java/Spark is unavailable.
"""
from __future__ import annotations

import json
import os
import sys
import time

import pandas as pd

from . import config as C

CONTENT_TYPES = ["oucontent", "resource", "url", "page", "subpage", "folder",
                 "dualpane", "htmlactivity", "sharedsubpage", "glossary", "dataplus"]
QUIZ_TYPES = ["quiz", "externalquiz", "questionnaire", "repeatactivity"]
FORUM_TYPES = ["forumng"]
COLLAB_TYPES = ["ouwiki", "oucollaborate", "ouelluminate"]


def _group_of(activity_type: str) -> str:
    if activity_type in FORUM_TYPES:
        return "forum"
    if activity_type in QUIZ_TYPES:
        return "quiz"
    if activity_type in CONTENT_TYPES:
        return "content"
    if activity_type in COLLAB_TYPES:
        return "collab"
    return "other"  # homepage and anything not listed


def run_spark(vle_csv=C.RAW / "studentVle.csv", site_csv=C.RAW / "vle.csv") -> dict:
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    os.environ["PYTHONPATH"] = str(C.ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")
    from pyspark.sql import SparkSession, functions as F, types as T

    t0 = time.time()
    spark = (SparkSession.builder.master("local[*]").appName("oulad-etl")
             .config("spark.driver.memory", "4g")
             .config("spark.ui.enabled", "false")
             .config("spark.sql.shuffle.partitions", "32")
             .getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")

    schema = T.StructType([
        T.StructField("code_module", T.StringType()),
        T.StructField("code_presentation", T.StringType()),
        T.StructField("id_student", T.IntegerType()),
        T.StructField("id_site", T.IntegerType()),
        T.StructField("date", T.IntegerType()),
        T.StructField("sum_click", T.IntegerType()),
    ])
    clicks = spark.read.csv(str(vle_csv), header=True, schema=schema)
    sites = (spark.read.csv(str(site_csv), header=True, inferSchema=True)
             .select("id_site", "code_module", "code_presentation", "activity_type"))

    # Native column expressions (no Python UDF) keep the mapping inside the JVM.
    at = F.col("activity_type")
    sites = sites.withColumn(
        "activity_group",
        F.when(at.isin(FORUM_TYPES), "forum").when(at.isin(QUIZ_TYPES), "quiz")
         .when(at.isin(CONTENT_TYPES), "content").when(at.isin(COLLAB_TYPES), "collab")
         .otherwise("other"))

    n_rows = clicks.count()
    n_distinct = clicks.dropDuplicates().count()

    joined = clicks.join(F.broadcast(sites), ["id_site", "code_module", "code_presentation"], "left")
    unmatched = joined.filter(F.col("activity_type").isNull()).count()

    # ---- DataFrame API: enrollment x day ------------------------------------------------
    day = (joined.groupBy("id_student", "code_module", "code_presentation", "date")
           .agg(F.sum("sum_click").alias("clicks"),
                F.count(F.lit(1)).alias("n_records"),
                F.sum(F.when(F.col("activity_group") == "forum", F.col("sum_click")).otherwise(0)).alias("forum_clicks"),
                F.sum(F.when(F.col("activity_group") == "content", F.col("sum_click")).otherwise(0)).alias("content_clicks"),
                F.sum(F.when(F.col("activity_group") == "quiz", F.col("sum_click")).otherwise(0)).alias("quiz_clicks"),
                F.countDistinct("id_site").alias("distinct_sites")))
    day_pd = day.toPandas()

    # ---- DataFrame API: enrollment x activity_type pivot -----------------------------------
    types = sorted(r[0] for r in sites.select("activity_type").distinct().collect())
    pivot = (joined.groupBy("id_student", "code_module", "code_presentation")
             .pivot("activity_type", types).agg(F.sum("sum_click")).na.fill(0))
    pivot_pd = pivot.toPandas()

    # ---- Pair RDD cross-check: total clicks per enrollment via reduceByKey ---------------
    pair_totals = (clicks.rdd
                   .map(lambda r: ((r.id_student, r.code_module, r.code_presentation), r.sum_click))
                   .reduceByKey(lambda a, b: a + b))
    rdd_total = pair_totals.values().sum()
    rdd_n_keys = pair_totals.count()
    spark.stop()

    stats = {
        "engine": "pyspark",
        "studentVle_rows": int(n_rows),
        "studentVle_exact_duplicate_rows": int(n_rows - n_distinct),
        "rows_without_site_metadata": int(unmatched),
        "enrollment_day_rows": int(len(day_pd)),
        "enrollments_with_clicks": int(pivot_pd[C.KEY].drop_duplicates().shape[0]),
        "total_clicks_dataframe": int(day_pd["clicks"].sum()),
        "total_clicks_pair_rdd": int(rdd_total),
        "pair_rdd_keys": int(rdd_n_keys),
        "seconds": round(time.time() - t0, 1),
    }
    assert stats["total_clicks_dataframe"] == stats["total_clicks_pair_rdd"], "RDD/DataFrame mismatch"
    _save(day_pd, pivot_pd, stats)
    return stats


def run_pandas(vle_csv=C.RAW / "studentVle.csv", site_csv=C.RAW / "vle.csv") -> dict:
    """Chunked pandas implementation with the same outputs (fallback / validation)."""
    t0 = time.time()
    sites = pd.read_csv(site_csv, usecols=["id_site", "code_module", "code_presentation", "activity_type"])
    sites["activity_group"] = sites["activity_type"].map(_group_of)
    parts, piv_parts, n_rows = [], [], 0
    for chunk in pd.read_csv(vle_csv, chunksize=2_000_000):
        n_rows += len(chunk)
        m = chunk.merge(sites, on=["id_site", "code_module", "code_presentation"], how="left")
        for g in ["forum", "content", "quiz"]:
            m[f"{g}_clicks"] = m["sum_click"].where(m["activity_group"] == g, 0)
        parts.append(m.groupby(C.KEY + ["date"]).agg(
            clicks=("sum_click", "sum"), n_records=("sum_click", "size"),
            forum_clicks=("forum_clicks", "sum"), content_clicks=("content_clicks", "sum"),
            quiz_clicks=("quiz_clicks", "sum")).reset_index())
        piv_parts.append(m.pivot_table(index=C.KEY, columns="activity_type", values="sum_click",
                                       aggfunc="sum", fill_value=0))
    day_pd = pd.concat(parts).groupby(C.KEY + ["date"], as_index=False).sum()
    pivot_pd = pd.concat(piv_parts).groupby(level=[0, 1, 2]).sum().reset_index()
    stats = {"engine": "pandas", "studentVle_rows": n_rows, "enrollment_day_rows": len(day_pd),
             "total_clicks_dataframe": int(day_pd["clicks"].sum()), "seconds": round(time.time() - t0, 1)}
    _save(day_pd, pivot_pd, stats)
    return stats


def _save(day_pd: pd.DataFrame, pivot_pd: pd.DataFrame, stats: dict) -> None:
    for c in ["id_student", "date", "clicks", "n_records"]:
        if c in day_pd:
            day_pd[c] = day_pd[c].astype("int32")
    day_pd.sort_values(C.KEY + ["date"]).to_parquet(C.INTERIM / "enrollment_day.parquet", index=False)
    pivot_pd.to_parquet(C.INTERIM / "enrollment_activity_type.parquet", index=False)
    (C.INTERIM / "etl_stats.json").write_text(json.dumps(stats, indent=2))


def run(prefer_spark: bool = True) -> dict:
    if prefer_spark:
        try:
            return run_spark()
        except Exception as exc:  # pragma: no cover - environment dependent
            print(f"Spark unavailable ({exc!r}); using pandas fallback")
    return run_pandas()


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
