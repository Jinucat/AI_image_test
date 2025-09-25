from fastapi import APIRouter
from .endpoints import health, tsne, cluster, umap, embed

api_router = APIRouter(prefix="/api")
api_router.include_router(health.router, tags=["health"])
api_router.include_router(tsne.router,  prefix="/tsne", tags=["tsne"])
api_router.include_router(cluster.router, prefix="/cluster", tags=["cluster"])
api_router.include_router(umap.router,  prefix="/umap", tags=["umap"])
api_router.include_router(embed.router, prefix="/embed", tags=["embed"])