from __future__ import annotations
import logging
import numpy as np
from fastapi import APIRouter, HTTPException
from app.models.schemas import EmbedRunRequest, EmbedRunResponse
from app.includes.config import settings

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/run", response_model=EmbedRunResponse)
def embed_run(req: EmbedRunRequest):
    if not req.data_root and not req.table_path:
        raise HTTPException(400, "data_root 또는 table_path 중 하나는 필요합니다.")

    from app.services.embeddings import EmbeddingBackend
    from app.services.dataio import load_image_items

    # 1) 로드
    items = load_image_items(
        data_root=req.data_root,
        table_path=req.table_path,
        max_per_class=req.max_per_class,  # None이면 제한 없음
    )
    if not items:
        raise HTTPException(404, "이미지 항목을 찾지 못했습니다.")

    paths = np.array([p for p, _ in items], dtype=object)
    y     = np.array([lab for _, lab in items], dtype=object)

    logger.info(f"[EMBED] total={len(paths)}, labels={len(np.unique(y))}")

    # 2) 임베딩
    backend = EmbeddingBackend.create(model_id=req.model_id, device=None)

    # 🔧 여기만 수정: (X, kept_idx) 반환
    X, kept = backend.embed_paths(paths, batch_size=req.batch_size or 64)

    if kept.size == 0 or X.shape[0] == 0:
        raise HTTPException(400, "임베딩에 성공한 이미지가 없습니다.")

    # 길이 맞추기(깨진 파일/로드 실패 제외)
    paths = paths[kept]
    y     = y[kept]

    # 3) 저장
    run_dir = settings.artifacts_path() / req.out_name
    run_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(run_dir / "embeddings.npz", X=X, y=y, paths=paths)

    return EmbedRunResponse(
        run_name=req.out_name,
        total_images=int(X.shape[0]),
        labels=int(len(np.unique(y))),
        embeddings_npz=f"/artifacts/{req.out_name}/embeddings.npz",
    )