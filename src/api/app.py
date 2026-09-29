"""
Enterprise FastAPI Application for Smart Document Cropping and Layout Detection.
"""
import os
import io
import time
import multiprocessing
from typing import Optional, List
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form, Query, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse

from .schemas import (
    CropRequestBase64,
    CropResponse,
    HealthResponse,
)
from ..agent.document_cropper_agent import DocumentCropperAgent

app = FastAPI(
    title="Enterprise Smart Document Cropper API",
    description="High-precision, lightweight, CPU-optimized document corner detection and perspective cropping engine for KYC, Invoices, NICs, Passports, and Cheques.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for enterprise frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Agent instance
agent = DocumentCropperAgent()


@app.get("/api/v1/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Healthcheck endpoint for Kubernetes, Docker, and enterprise load balancers."""
    return HealthResponse(
        status="HEALTHY",
        version="1.0.0",
        models_loaded=["FastViT-SA24", "LCNet100", "DeepLabV3-DocSeg", "YOLO11-Doc", "Classical-CV"],
        device="CPU (ONNX Runtime Engine)",
        cpu_cores=multiprocessing.cpu_count()
    )


@app.post("/api/v1/crop", tags=["Cropping"])
async def crop_document_file(
    file: UploadFile = File(..., description="Uploaded document image (JPEG, PNG, WebP)"),
    safe_margin_percent: float = Form(0.01, description="Outward margin expansion (0.01 = 1%)"),
    preferred_model: str = Form("auto", description="auto, fastvit, lcnet, deeplab, yolo, classical"),
    return_image_stream: bool = Form(False, description="If true, returns cropped JPEG image binary directly"),
    include_annotated_preview: bool = Form(False, description="Include preview with corners drawn")
):
    """
    Upload an image file to detect document corners and extract perspective-rectified crop.
    Supports Sri Lankan NICs, Driving Licenses, Passports, Cheques, and Invoices.
    """
    try:
        content = await file.read()
        res = agent.process(
            content,
            safe_margin_percent=safe_margin_percent,
            preferred_model=preferred_model,
            return_base64=not return_image_stream,
            return_annotated_preview=include_annotated_preview
        )

        if return_image_stream:
            # Reconstruct cropped image from raw content
            img = agent._load_image(content)
            corners = np.array(res["corners_pixel"], dtype=np.float32)
            from ..core.transform import perspective_warp
            cropped, _ = perspective_warp(img, corners, safe_margin_percent=safe_margin_percent)
            _, buf = cv2.imencode(".jpg", cropped, [cv2.IMWRITE_JPEG_QUALITY, 95])
            return StreamingResponse(io.BytesIO(buf.tobytes()), media_type="image/jpeg")

        return JSONResponse(content=res)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/crop-json", response_model=CropResponse, tags=["Cropping"])
async def crop_document_json(request: CropRequestBase64):
    """
    JSON API for base64 encoded document image.
    Ideal for server-to-server microservice integrations and mobile app payloads.
    """
    try:
        res = agent.process(
            request.image_base64,
            safe_margin_percent=request.safe_margin_percent,
            preferred_model=request.preferred_model,
            return_base64=True,
            return_annotated_preview=request.include_annotated_preview
        )
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/detect-corners", tags=["Detection"])
async def detect_corners_only(
    file: UploadFile = File(..., description="Uploaded document image")
):
    """
    Returns detected 4 quadrilateral corners and quality assessment without returning warped image data.
    Ultra-low latency (<200ms) for client-side preview rendering.
    """
    try:
        content = await file.read()
        res = agent.process(
            content,
            return_base64=False,
            return_annotated_preview=False
        )
        # Omit any heavy binary payloads
        return {
            "success": res["success"],
            "pipeline_stage": res["pipeline_stage"],
            "latency_ms": res["latency_ms"],
            "corners_pixel": res["corners_pixel"],
            "corners_normalized": res["corners_normalized"],
            "document_classification": res["document_classification"],
            "quality_assessment": res["quality_assessment"],
            "geometric_score": res["geometric_score"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/crop-field", tags=["Semantic Field Cropping"])
async def crop_semantic_field_endpoint(
    file: UploadFile = File(..., description="Uploaded document image"),
    field_query: str = Form(..., description="Target name or label to search and crop (e.g. 'Manula', 'Name', 'NIC')"),
    is_key_value: bool = Form(False, description="If true, finds value to the right/below the field label")
):
    """
    Intelligent Sub-Document Field Cropper:
    Locates a specific field or name inside the document, detects its exact spatial coordinates ('Where'),
    provenance timestamp ('When'), and crops out the isolated high-resolution field image snippet.
    """
    try:
        content = await file.read()
        res = agent.extract_field(
            content,
            query=field_query,
            is_key_value=is_key_value,
            output_dir="eval_results/semantic_crops"
        )
        return JSONResponse(content=res)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

