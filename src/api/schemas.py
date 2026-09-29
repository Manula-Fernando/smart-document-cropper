"""
Pydantic schemas for the Smart Document Cropper REST API.
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class CropRequestBase64(BaseModel):
    image_base64: str = Field(..., description="Base64 encoded image string or data URI")
    safe_margin_percent: float = Field(0.01, description="Outward margin expansion (e.g. 0.01 for 1%)")
    preferred_model: str = Field("auto", description="Model choice: auto, fastvit, lcnet, deeplab, yolo, classical")
    include_annotated_preview: bool = Field(False, description="Include visual preview with corners marked")


class CornerPoint(BaseModel):
    x: float
    y: float


class DocumentDimensions(BaseModel):
    width: int
    height: int


class DocumentClassification(BaseModel):
    category: str
    aspect_ratio: float
    orientation: str
    width_px: int
    height_px: int


class QualityAssessment(BaseModel):
    blur_score: float
    is_blurry: bool
    glare_percentage: float
    has_glare: bool
    mean_brightness: float
    contrast_std: float
    quality_status: str


class CropResponse(BaseModel):
    success: bool
    pipeline_stage: str
    latency_ms: float
    corners_pixel: List[List[float]]
    corners_normalized: List[List[float]]
    original_dimensions: DocumentDimensions
    cropped_dimensions: DocumentDimensions
    geometric_score: float
    coverage_ratio: float
    document_classification: DocumentClassification
    quality_assessment: QualityAssessment
    crop_base64: Optional[str] = None
    annotated_preview_base64: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    version: str
    models_loaded: List[str]
    device: str
    cpu_cores: int
