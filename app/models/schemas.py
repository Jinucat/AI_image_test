from pydantic import BaseModel, Field
from typing import Optional, Dict, Literal

class TsneRunRequest(BaseModel):
    data_root: Optional[str] = Field(default=None, description="라벨 폴더 구조 루트")
    table_path: Optional[str] = Field(default=None, description="CSV/TSV (columns: path,label)")
    model_id: Optional[str] = None
    batch_size: Optional[int] = None
    perplexity: Optional[float] = None
    max_per_class: Optional[int] = None
    out_name: Optional[str] = Field(default="run1")

class TsneRunResponse(BaseModel):
    run_name: str
    total_images: int
    labels: int
    coords_csv: str
    scatter_png: str
    embeddings_npz: str

# --- 추가: 무감독 군집 색칠 자동화 요청/응답 ---
class ClusterAutoRequest(BaseModel):
    run_name: str = Field(..., description="기존 t-SNE 실행 이름(artifacts/<run_name>)")
    method: Literal["kmeans_auto", "kmeans", "dbscan"] = "kmeans_auto"
    k: Optional[int] = None
    k_min: int = 2
    k_max: int = 12
    eps: float = 0.6
    min_samples: int = 10

class ClusterAutoResponse(BaseModel):
    run_name: str
    method: str
    n_clusters: int
    counts: Dict[str, int]
    cluster_csv: str
    cluster_png: str

from typing import Optional

class UmapRunRequest(BaseModel):
    run_name: str
    n_neighbors: int = 30
    min_dist: float = 0.1
    metric: str = "euclidean"
    random_state: int = 42

class UmapRunResponse(BaseModel):
    run_name: str
    total_images: int
    labels: int
    coords_csv: str
    scatter_png: str
    embeddings_npz: str

class EmbedRunRequest(BaseModel):
    data_root: Optional[str] = Field(default=None, description="라벨 폴더 구조 루트")
    table_path: Optional[str] = Field(default=None, description="CSV/TSV (columns: path,label)")
    model_id: Optional[str] = None
    batch_size: Optional[int] = None
    max_per_class: Optional[int] = None
    out_name: Optional[str] = Field(default="run1")

class EmbedRunResponse(BaseModel):
    run_name: str
    total_images: int
    labels: int
    embeddings_npz: str