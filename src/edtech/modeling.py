"""Clustering model development and validation (Module 5, Lecture 9).

* Partitioning: K-Means (elbow / SSE, silhouette, Davies-Bouldin, Calinski-Harabasz)
* Hierarchical: AGNES with single / complete / average / Ward linkage, dendrogram, cophenetic r
* Density-based: DBSCAN with k-distance heuristic for Eps and a MinPts grid
* Validation: internal (silhouette...), relative (comparison across settings), external
  (purity / NMI / ARI against final_result), m-fold cross-validation of SSE (Lecture 9 p.43),
  bootstrap stability (ARI) and temporal hold-out (train 2013B-2014B -> score 2014J).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import cophenet, linkage
from scipy.spatial.distance import pdist
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score,
                             normalized_mutual_info_score, silhouette_score)
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors

from . import config as C


def _sample_idx(n: int, size: int, seed: int = C.SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, min(size, n), replace=False))


def internal_scores(X: np.ndarray, labels: np.ndarray, sample: int = 8000) -> dict:
    mask = labels >= 0                                 # DBSCAN noise excluded
    Xm, lm = X[mask], labels[mask]
    if len(np.unique(lm)) < 2:
        return {"silhouette": np.nan, "davies_bouldin": np.nan, "calinski_harabasz": np.nan}
    idx = _sample_idx(len(Xm), sample)
    return {"silhouette": float(silhouette_score(Xm[idx], lm[idx])),
            "davies_bouldin": float(davies_bouldin_score(Xm, lm)),
            "calinski_harabasz": float(calinski_harabasz_score(Xm, lm))}


def purity(labels: np.ndarray, truth: np.ndarray) -> float:
    ct = pd.crosstab(labels, truth)
    return float(ct.max(axis=1).sum() / ct.values.sum())


def external_scores(labels: np.ndarray, truth: np.ndarray) -> dict:
    return {"purity": purity(labels, truth),
            "nmi": float(normalized_mutual_info_score(truth, labels)),
            "ari": float(adjusted_rand_score(truth, labels))}


# ------------------------------------------------------------------------------- K-Means
def kmeans_scan(X: np.ndarray, ks=range(2, 11), n_init: int = 10) -> pd.DataFrame:
    rows = []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=n_init, random_state=C.SEED).fit(X)
        sizes = np.bincount(km.labels_)
        rows.append({"k": k, "sse": km.inertia_, **internal_scores(X, km.labels_),
                     "min_cluster_share": sizes.min() / len(X)})
    out = pd.DataFrame(rows)
    # elbow by maximum distance to the chord between first and last SSE point
    x = (out["k"] - out["k"].min()) / (out["k"].max() - out["k"].min())
    y = (out["sse"] - out["sse"].min()) / (out["sse"].max() - out["sse"].min())
    out["elbow_distance"] = np.abs(x + y - 1) / np.sqrt(2)
    return out


def cv_sse(X: np.ndarray, ks=range(2, 11), folds: int = 5) -> pd.DataFrame:
    """Lecture 9 cross-validation method: fit on m-1 parts, SSE of held-out part to nearest centroid."""
    rows = []
    kf = KFold(folds, shuffle=True, random_state=C.SEED)
    for k in ks:
        vals = []
        for tr, te in kf.split(X):
            km = KMeans(n_clusters=k, n_init=5, random_state=C.SEED).fit(X[tr])
            d = km.transform(X[te]).min(1)
            vals.append(float((d ** 2).mean()))
        rows.append({"k": k, "heldout_mse_mean": np.mean(vals), "heldout_mse_std": np.std(vals)})
    return pd.DataFrame(rows)


def bootstrap_stability(X: np.ndarray, k: int, n_boot: int = 20, sample_frac: float = 0.8) -> dict:
    """Fit on bootstrap samples, predict on full data, ARI vs the reference fit."""
    ref = KMeans(n_clusters=k, n_init=10, random_state=C.SEED).fit(X)
    rng = np.random.default_rng(C.SEED)
    aris = []
    for b in range(n_boot):
        idx = rng.choice(len(X), int(sample_frac * len(X)), replace=True)
        km = KMeans(n_clusters=k, n_init=5, random_state=b).fit(X[idx])
        aris.append(adjusted_rand_score(ref.labels_, km.predict(X)))
    return {"k": k, "ari_mean": float(np.mean(aris)), "ari_min": float(np.min(aris)), "ari_std": float(np.std(aris))}


def seed_stability(X: np.ndarray, k: int, seeds=range(10)) -> float:
    labs = [KMeans(n_clusters=k, n_init=1, random_state=s).fit_predict(X) for s in seeds]
    return float(np.mean([adjusted_rand_score(labs[0], l) for l in labs[1:]]))


# --------------------------------------------------------------------------- Hierarchical
def hierarchical_compare(X: np.ndarray, k: int, sample: int = 6000) -> tuple[pd.DataFrame, dict]:
    idx = _sample_idx(len(X), sample)
    Xs = X[idx]
    d = pdist(Xs)
    rows, Z_all = [], {}
    for method in ["single", "complete", "average", "ward"]:
        Z = linkage(Xs, method=method)
        Z_all[method] = Z
        coph, _ = cophenet(Z, d)
        lab = AgglomerativeClustering(n_clusters=k, linkage=method).fit_predict(Xs)
        sizes = np.bincount(lab)
        rows.append({"linkage": method, "cophenetic_corr": float(coph), **internal_scores(Xs, lab),
                     "largest_cluster_share": sizes.max() / len(Xs), "smallest_cluster_share": sizes.min() / len(Xs)})
    return pd.DataFrame(rows), {"Z": Z_all, "idx": idx}


# --------------------------------------------------------------------------------- DBSCAN
def k_distance(X: np.ndarray, min_pts: int, sample: int = 15000) -> np.ndarray:
    Xs = X[_sample_idx(len(X), sample)]
    nn = NearestNeighbors(n_neighbors=min_pts).fit(Xs)
    d, _ = nn.kneighbors(Xs)
    return np.sort(d[:, -1])


def dbscan_grid(X: np.ndarray, eps_values, min_pts_values, sample: int = 15000) -> pd.DataFrame:
    idx = _sample_idx(len(X), sample)
    Xs = X[idx]
    rows = []
    for mp in min_pts_values:
        for eps in eps_values:
            lab = DBSCAN(eps=eps, min_samples=mp).fit_predict(Xs)
            n_cl = len(set(lab) - {-1})
            sizes = np.bincount(lab[lab >= 0]) if n_cl else np.array([0])
            sc = internal_scores(Xs, lab) if n_cl >= 2 else {"silhouette": np.nan, "davies_bouldin": np.nan, "calinski_harabasz": np.nan}
            rows.append({"min_pts": mp, "eps": eps, "n_clusters": n_cl, "noise_share": float((lab == -1).mean()),
                         "largest_cluster_share": float(sizes.max() / len(Xs)), **sc})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------- Final model
@dataclass
class SegmentationModel:
    """Serving bundle: feature pipeline + K-Means + persona metadata + drift reference."""
    pipeline: object
    kmeans: KMeans
    persona_by_cluster: dict = field(default_factory=dict)
    distance_p95: dict = field(default_factory=dict)
    reference_shares: dict = field(default_factory=dict)
    version: str = "1.0.0"
    trained_on: list = field(default_factory=lambda: list(C.TRAIN_PRESENTATIONS))

    def features(self) -> list[str]:
        return self.pipeline.features

    def assign(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rule-based non-starter segment first, K-Means for everyone else."""
        out = pd.DataFrame(index=df.index)
        no_act = df.get("active_day_fraction", pd.Series(1, index=df.index)).fillna(0) <= 0
        Z = self.pipeline.transform(df)
        dist = self.kmeans.transform(Z)
        cl = dist.argmin(1)
        out["cluster"] = np.where(no_act, -1, cl)
        out["distance_to_centroid"] = np.where(no_act, np.nan, dist.min(1))
        # relative margin between nearest and second-nearest centroid -> assignment confidence
        srt = np.sort(dist, 1)
        out["assignment_confidence"] = np.where(no_act, 1.0, 1 - srt[:, 0] / np.maximum(srt[:, 1], 1e-9))
        p95 = out["cluster"].map(lambda c: self.distance_p95.get(int(c), np.inf))
        out["is_atypical"] = (out["distance_to_centroid"] > p95).astype(int)
        out["segment_id"] = out["cluster"].map(lambda c: self.persona_by_cluster.get(int(c), {}).get("segment_id", f"C{c}"))
        out["persona"] = out["cluster"].map(lambda c: self.persona_by_cluster.get(int(c), {}).get("name", f"Cluster {c}"))
        return out
