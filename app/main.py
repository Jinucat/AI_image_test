from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from .includes.logging import setup_logging
from .includes.config import settings
from .api.v1.router import api_router

setup_logging()
app = FastAPI(title="Face TSNE API", version="1.0.0")

# 정적 산출물 서빙
app.mount("/artifacts", StaticFiles(directory=settings.artifacts_path()), name="artifacts")

# API Router
app.include_router(api_router)

@app.get("/")
def root():
    return {"service": "face-tsne-api", "artifacts": "/artifacts", "api": "/api"}