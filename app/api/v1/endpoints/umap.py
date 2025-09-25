# app/api/v1/endpoints/umap.py
from __future__ import annotations
import logging
from pathlib import Path
import numpy as np
from fastapi import APIRouter, HTTPException

from app.models.schemas import UmapRunRequest, UmapRunResponse
from app.includes.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/run", response_model=UmapRunResponse)
def umap_run(req: UmapRunRequest):
    run_dir = settings.artifacts_path() / req.run_name
    npz = run_dir / "embeddings.npz"
    if not npz.exists():
        raise HTTPException(
            404,
            f"artifacts/{req.run_name}/embeddings.npz 가 없습니다. "
            f"먼저 /api/tsne/run (또는 임베딩 생성) 을 실행하세요."
        )
    # lazy import inside handler
    from app.services.umap import run_umap, save_outputs_umap

    data = np.load(npz, allow_pickle=True)
    X, y, paths = data["X"], data["y"], data["paths"]
    logger.info(f"[UMAP] Loaded embeddings: X={X.shape}, y={len(y)}")

    Z = run_umap(
        X,
        n_neighbors=req.n_neighbors,
        min_dist=req.min_dist,
        metric=req.metric,
        seed=req.random_state,
    )
    coords_csv, scatter_png = save_outputs_umap(run_dir, X, y, paths, Z)

    return UmapRunResponse(
        run_name=req.run_name,
        total_images=int(X.shape[0]),
        labels=int(len(np.unique(y))),
        coords_csv=f"/artifacts/{req.run_name}/umap_coords.csv",
        scatter_png=f"/artifacts/{req.run_name}/umap_scatter.png",
        embeddings_npz=f"/artifacts/{req.run_name}/embeddings.npz",
    )