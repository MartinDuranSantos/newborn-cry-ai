from fastapi import FastAPI

from app.routers.classify import router
from app.services.classifier import load_model

app = FastAPI(title="Infant Cry AI", description="Clasificador de llanto de bebé en español")

app.include_router(router)


@app.on_event("startup")
def _startup() -> None:
    load_model()


@app.get("/")
def root() -> dict[str, str]:
    return {"mensaje": "Infant Cry AI - Clasificador de llanto de bebé"}