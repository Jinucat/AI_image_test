# services/clustering.py
from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from app.includes.config import settings

def _run_dir(run_name: str) -> Path:
    return settings.artifacts_path() / run_name

def _load_run(run_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    rd = _run_dir(run_name)
    npz = np.load(rd / "embeddings.npz", allow_pickle=True)
    X, y, paths = npz["X"], npz["y"], npz["paths"]
    coords = pd.read_csv(rd / "tsne_coords.csv")
    return X, y, paths, coords

def _save_cluster_plot(run_name: str, coords: pd.DataFrame, cluster_ids: np.ndarray, title: str) -> Path:
    rd = _run_dir(run_name)
    coords = coords.copy()
    coords["cluster"] = cluster_ids.astype(int)
    k = coords["cluster"].nunique()
    cmap = plt.cm.get_cmap("tab20", max(k, 2))

    plt.figure(figsize=(11, 9))
    for i, g in coords.groupby("cluster"):
        plt.scatter(g["x"], g["y"], s=8, alpha=0.85, color=cmap(i % 20), label=f"C{i}")
    if k <= 20:
        plt.legend(title="Cluster", fontsize=8, markerscale=2, ncol=2)
    plt.title(title); plt.tight_layout()
    out = rd / "tsne_scatter_clusters.png"
    plt.savefig(out, dpi=200); plt.close()

    coords.to_csv(rd / "tsne_coords_clustered.csv", index=False, encoding="utf-8")
    return out

def cluster_run_auto(
    run_name: str,
    method: str = "kmeans_auto",        # "kmeans", "kmeans_auto", "dbscan"
    k: Optional[int] = None,
    k_min: int = 2,
    k_max: int = 12,
    eps: float = 0.6,
    min_samples: int = 10,
) -> Tuple[int, Dict[int, int], Path, Path]:
    """
    반환: (n_clusters, counts, csv_path, png_path)
    - csv_path: tsne_coords_clustered.csv
    - png_path: tsne_scatter_clusters.png
    """
    X, y, paths, coords = _load_run(run_name)

    if method in ("kmeans_auto", "kmeans"):
        if method == "kmeans_auto" or (method == "kmeans" and not k):
            best_k, best_score = None, -1.0
            for kk in range(max(2, k_min), max(k_min, k_max) + 1):
                km = KMeans(n_clusters=kk, n_init="auto", random_state=42)
                lab = km.fit_predict(X)
                # 샘플 수 < 클러스터 수 방지
                if len(set(lab)) < 2: 
                    continue
                score = silhouette_score(X, lab, metric="euclidean")
                if score > best_score:
                    best_k, best_score = kk, score
            k = best_k or 2
        km = KMeans(n_clusters=k, n_init="auto", random_state=42)
        cids = km.fit_predict(X)
        title = f"t-SNE colored by KMeans (k={len(set(cids))})"

    elif method == "dbscan":
        db = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean")
        cids = db.fit_predict(X)  # -1 = 노이즈
        # 노이즈는 가장 큰 인덱스 다음으로 재맵핑(시각화 편의)
        if np.any(cids == -1):
            cids = cids.copy()
            cids[cids == -1] = cids.max() + 1
        title = f"t-SNE colored by DBSCAN (k={len(set(cids))})"

    else:
        raise ValueError("method must be one of: kmeans_auto, kmeans, dbscan")

    png = _save_cluster_plot(run_name, coords, cids, title)
    # 저장된 CSV 경로
    csvp = _run_dir(run_name) / "tsne_coords_clustered.csv"

    # 클러스터별 개수
    unique, counts = np.unique(cids, return_counts=True)
    count_map = {int(u): int(c) for u, c in zip(unique, counts)}
    return len(unique), count_map, csvp, png