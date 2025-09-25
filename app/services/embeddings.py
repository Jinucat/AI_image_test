# app/services/embeddings.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple, List, Optional

import logging
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

from ..includes.config import settings
from ..utils.images import open_rgb  # 경로→PIL.Image RGB 로더

logger = logging.getLogger(__name__)


def _resolve_device(device: Optional[str]) -> str:
    """
    device == 'auto'면 cuda 가용 시 'cuda' 아니면 'cpu'.
    명시 값이 있으면 그대로 사용.
    """
    if device is None:
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


@dataclass
class EmbeddingBackend:
    model_id: str
    device_str: str
    processor: AutoImageProcessor
    model: AutoModel
    device: torch.device

    # ──────────────────────────────────────────────────────────────────────────
    # 생성자
    # ──────────────────────────────────────────────────────────────────────────
    @classmethod
    def create(cls, model_id: str | None = None, device: str | None = None) -> "EmbeddingBackend":
        mid = model_id or settings.MODEL_ID
        dev = _resolve_device(device or getattr(settings, "DEVICE", "auto"))

        # 이미지 전용 프로세서/모델 (예: SigLIP/CLIP 계열)
        processor = AutoImageProcessor.from_pretrained(mid)
        model = AutoModel.from_pretrained(mid)
        model.eval()
        model.to(dev)

        return cls(
            model_id=mid,
            device_str=dev,
            processor=processor,
            model=model,
            device=torch.device(dev),
        )

    # ──────────────────────────────────────────────────────────────────────────
    # 경로 리스트 → 임베딩
    # ──────────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def embed_images(self, paths: Sequence[str], batch_size: int = 32) -> np.ndarray:
        """
        이미지 '경로' 리스트를 받아 배치 임베딩 후 (N, D) numpy.float32 반환.
        - 내부에서 open_rgb로 이미지를 열어 RGB로 변환.
        """
        feats: List[torch.Tensor] = []

        for i in range(0, len(paths), batch_size):
            batch_paths = paths[i : i + batch_size]
            # 경로 → PIL 이미지
            imgs = []
            for p in batch_paths:
                try:
                    imgs.append(open_rgb(p))
                except Exception as e:
                    logger.warning("[EMBED] skip unreadable image in embed_images: %s (%s)", p, e)

            if not imgs:
                continue

            inputs = self.processor(images=imgs, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(self.device)

            # 모델 타입별 image feature 얻기
            if hasattr(self.model, "get_image_features"):
                f = self.model.get_image_features(pixel_values=pixel_values)
            else:
                out = self.model(pixel_values=pixel_values)
                if hasattr(out, "pooler_output") and out.pooler_output is not None:
                    f = out.pooler_output
                else:
                    f = out.last_hidden_state.mean(dim=1)

            f = torch.nn.functional.normalize(f, dim=1)
            feats.append(f.detach().cpu())

        if not feats:
            return np.zeros((0, 0), dtype=np.float32)

        return torch.cat(feats, dim=0).numpy().astype(np.float32, copy=False)

    # ──────────────────────────────────────────────────────────────────────────
    # 대용량 안전 경로 임베딩 (검증+스킵 포함)
    # ──────────────────────────────────────────────────────────────────────────
    def embed_paths(self, paths, batch_size: int = 64) -> Tuple[np.ndarray, np.ndarray]:
        """
        이미지 '경로' 리스트를 배치로 임베딩.
        반환: (X, kept_idx)
          - X: (N_kept, D) float32
          - kept_idx: 원본 paths에서 성공적으로 임베딩된 항목 인덱스 (int64)
        """
        if isinstance(paths, np.ndarray):
            paths = paths.tolist()

        vecs: List[np.ndarray] = []
        kept: List[int] = []

        n = len(paths)
        for start in range(0, n, batch_size):
            batch_paths = paths[start : start + batch_size]

            # 1) 사전 검증: 깨진 파일/비이미지 스킵 (빠른 verify)
            valid_paths: List[str] = []
            idxs: List[int] = []
            for j, p in enumerate(batch_paths):
                p_str = str(p)
                try:
                    with Image.open(p_str) as im:
                        im.verify()  # 헤더 검증
                except Exception as e:
                    logger.warning("[EMBED] skip unreadable image: %s (%s)", p_str, e)
                    continue
                valid_paths.append(p_str)
                idxs.append(start + j)

            if not valid_paths:
                continue

            # 2) 실제 임베딩: ★ 경로 리스트를 embed_images()에 넘긴다 ★
            V = self.embed_images(valid_paths, batch_size=len(valid_paths))
            if not isinstance(V, np.ndarray):
                V = np.asarray(V, dtype=np.float32)

            if V.size == 0:
                continue

            vecs.append(V.astype(np.float32, copy=False))
            kept.extend(idxs)

        if not vecs:
            return np.zeros((0, 0), dtype=np.float32), np.array([], dtype=np.int64)

        X = np.vstack(vecs).astype(np.float32, copy=False)
        return X, np.array(kept, dtype=np.int64)