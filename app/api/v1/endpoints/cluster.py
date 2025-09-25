# api/v1/endpoints/cluster.py
import logging
from fastapi import APIRouter, HTTPException
from app.models.schemas import ClusterAutoRequest, ClusterAutoResponse
from app.services.clustering import cluster_run_auto
from app.includes.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/auto", response_model=ClusterAutoResponse)
def cluster_auto(req: ClusterAutoRequest):
    try:
        n, counts, csvp, pngp = cluster_run_auto(
            run_name=req.run_name,
            method=req.method,
            k=req.k,
            k_min=req.k_min,
            k_max=req.k_max,
            eps=req.eps,
            min_samples=req.min_samples,
        )
    except FileNotFoundError:
        raise HTTPException(404, f"artifacts/{req.run_name} 가 존재하지 않습니다. 먼저 /api/tsne/run 실행 필요.")
    except Exception as e:
        raise HTTPException(400, f"clustering failed: {e}")

    return ClusterAutoResponse(
        run_name=req.run_name,
        method=req.method,
        n_clusters=n,
        counts={str(k): v for k, v in counts.items()},
        cluster_csv=f"/artifacts/{req.run_name}/tsne_coords_clustered.csv",
        cluster_png=f"/artifacts/{req.run_name}/tsne_scatter_clusters.png",
    )