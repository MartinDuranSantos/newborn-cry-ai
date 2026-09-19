from pydantic import BaseModel, Field


class ClassifyResponse(BaseModel):
    categoria: str = Field(..., description="Categoría del llanto en español")
    confianza: float = Field(..., ge=0.0, le=1.0, description="Score de confianza entre 0 y 1")
    advertencia: bool = Field(..., description="True si la confianza es inferior al umbral")
    mensaje: str = Field(..., description="Mensaje descriptivo para el usuario")


class ErrorResponse(BaseModel):
    detail: str
