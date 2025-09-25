from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["Malgun Gothic", "NanumGothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

def run_umap(X: np.ndarray, n_neighbors: int = 30, min_dist: float = 0.1,
             metric: str = "euclidean", seed: int = 42) -> np.ndarray:
    import umap
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=int(n_neighbors),
        min_dist=float(min_dist),
        metric=metric,
        random_state=seed,
        verbose=True,
    )
    return reducer.fit_transform(X)

def save_outputs_umap(run_dir: Path, X: np.ndarray, y: np.ndarray,
                      paths: np.ndarray, Z: np.ndarray):
    """UMAP 좌표/플롯 저장 (파일명: umap_coords.csv / umap_scatter.png)"""
    run_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(run_dir / "embeddings.npz", X=X, y=y, paths=paths)

    df = pd.DataFrame({"x": Z[:, 0], "y": Z[:, 1], "label": y, "path": paths})
    coords_csv = run_dir / "umap_coords.csv"
    df.to_csv(coords_csv, index=False, encoding="utf-8")

    # scatter
    plt.figure(figsize=(11, 9))
    ax = plt.gca()
    ax.set_facecolor("#f7f7f7")

    y_str = np.asarray(y).astype(str)
    uniq = np.unique(y_str)

    PALETTE = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#17becf", "#bcbd22"
    ]

    if uniq.size == 1:
        m = (y_str == uniq[0])
        plt.scatter(
            Z[m, 0], Z[m, 1],
            s=18, alpha=0.95, color="C0",
            edgecolors="k", linewidths=0.3, label=str(uniq[0])
        )
    else:
        for i, lab in enumerate(uniq):
            m = (y_str == lab)
            color = PALETTE[i % len(PALETTE)]
            plt.scatter(
                Z[m, 0], Z[m, 1],
                s=14, alpha=0.95, color=color,
                edgecolors="k", linewidths=0.25, label=lab
            )
        if len(uniq) <= 20:
            plt.legend(title="라벨", fontsize=8, markerscale=1.8, ncol=2,
                       frameon=True, facecolor="white")

    plt.title("UMAP of Face Skin Disease")
    plt.tight_layout()
    scatter_png = run_dir / "umap_scatter.png"
    plt.savefig(scatter_png, dpi=200)
    plt.close()

    return coords_csv, scatter_png