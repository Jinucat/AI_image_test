import logging
from pathlib import Path
import numpy as np
from fastapi import APIRouter, HTTPException
from app.includes.config import settings
from app.models.schemas import TsneRunRequest, TsneRunResponse
from app.services.dataio import load_from_dirs, load_from_table
from app.services.embeddings import EmbeddingBackend
from app.services.tsne import run_tsne, save_outputs

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/run", response_model=TsneRunResponse)
def tsne_run(req: TsneRunRequest):
    data_root = req.data_root or settings.DATA_ROOT
    table_path = req.table_path
    max_per_class = req.max_per_class if req.max_per_class is not None else settings.MAX_PER_CLASS

    if table_path:
        samples = load_from_table(data_root, table_path)
    else:
        samples = load_from_dirs(data_root, max_per_class=max_per_class if max_per_class > 0 else None)
    if not samples:
        raise HTTPException(400, "이미지를 찾지 못했습니다. data_root/table_path 확인")

    paths, labels = zip(*samples)
    paths = np.array(paths); y = np.array(labels)
    logger.info(f"Total {len(paths)} images, {len(set(labels))} labels")

    # 임베딩
    backend = EmbeddingBackend.create(model_id=req.model_id, device=None)
    X = backend.embed_images(list(paths), batch_size=req.batch_size or settings.BATCH_SIZE)

    # t-SNE
    perplexity = req.perplexity if req.perplexity is not None else settings.PERPLEXITY
    Z = run_tsne(X, perplexity=perplexity, seed=42)

    # 저장
    run_name = req.out_name or "run1"
    run_dir = settings.artifacts_path() / run_name
    coords_csv, scatter_png = save_outputs(run_dir, X, y, paths, Z)

    return TsneRunResponse(
        run_name=run_name,
        total_images=len(paths),
        labels=len(set(labels)),
        coords_csv=f"/artifacts/{run_name}/tsne_coords.csv",
        scatter_png=f"/artifacts/{run_name}/tsne_scatter.png",
        embeddings_npz=f"/artifacts/{run_name}/embeddings.npz"
    )