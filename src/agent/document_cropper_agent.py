"""
DocumentCropperAgent: Reusable, Plug-and-Play Enterprise AI Agent.
Can be embedded into any microservice, orchestration framework (LangChain/LlamaIndex),
REST API, or MCP Server.
"""
from __future__ import annotations

import base64
import io
import time
import logging
from pathlib import Path
from typing import Union, Dict, Any, Optional, List, Tuple

import cv2
import numpy as np
from PIL import Image

from ..core.ensemble import EnterpriseHybridDetector
from ..core.transform import perspective_warp
from ..core.quality import assess_document_quality, classify_document_aspect_ratio
from ..core.field_cropper import SemanticFieldCropper

logger = logging.getLogger("smart_document_cropper.agent")


class DocumentCropperAgent:
    """
    Enterprise Document Cropper Plug-in Agent.
    
    Accepts:
      - File path (str or Path)
      - Raw bytes (bytes)
      - Base64 string (with or without data URI prefix)
      - NumPy ndarray (BGR or RGB)
      - PIL Image
      
    Returns structured processing results with:
      - Cropped image (numpy array, base64, or saved file)
      - Precise 4 quadrilateral coordinates
      - Document classification (NIC, Passport, Cheque, Invoice)
      - Quality scores (blur, glare, resolution)
      - Processing latency
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        """Singleton pattern for efficient model caching in agent pipelines."""
        if cls._instance is None:
            cls._instance = super(DocumentCropperAgent, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, preferred_model: str = "auto"):
        if getattr(self, "_initialized", False):
            return
        
        logger.info("Initializing Enterprise Document Cropper Agent...")
        t0 = time.time()
        self.detector = EnterpriseHybridDetector()
        self.field_cropper = SemanticFieldCropper()
        self.preferred_model = preferred_model
        self._initialized = True
        logger.info(f"Agent ready in {time.time() - t0:.2f}s")

    def _load_image(self, image_input: Union[str, Path, bytes, np.ndarray, Image.Image]) -> np.ndarray:
        """
        Helper to convert varied image input types to standard BGR numpy array.
        Universally supports: .jpg, .jpeg, .jfif, .png, .webp, .bmp, .tiff, .tif, .pjpeg, .pjp, and any PIL/OpenCV format.
        """
        if isinstance(image_input, (str, Path)):
            path_str = str(image_input)
            if path_str.startswith("data:image"):
                # Data URI
                base64_data = path_str.split(",", 1)[1]
                img_bytes = base64.b64decode(base64_data)
                nparr = np.frombuffer(img_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if img is None:
                    try:
                        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                    except Exception:
                        img = None
            else:
                img = None
                # Method A: imdecode from np.fromfile (handles unicode/special Windows paths & JFIF)
                try:
                    img = cv2.imdecode(np.fromfile(path_str, dtype=np.uint8), cv2.IMREAD_COLOR)
                except Exception:
                    pass
                # Method B: standard imread
                if img is None:
                    img = cv2.imread(path_str)
                # Method C: PIL Image fallback (handles all exotic formats: .jfif, .webp, .bmp, .tiff, etc.)
                if img is None:
                    try:
                        pil_img = Image.open(path_str).convert("RGB")
                        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                    except Exception:
                        img = None

            if img is None:
                raise ValueError(f"Could not read image from path: {image_input}")
            return img

        elif isinstance(image_input, bytes):
            nparr = np.frombuffer(image_input, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                try:
                    pil_img = Image.open(io.BytesIO(image_input)).convert("RGB")
                    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                except Exception:
                    img = None
            if img is None:
                raise ValueError("Could not decode image from bytes.")
            return img

        elif isinstance(image_input, np.ndarray):
            return image_input

        elif isinstance(image_input, Image.Image):
            rgb = np.array(image_input.convert("RGB"))
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

    def crop(
        self,
        image_input: Union[str, Path, bytes, np.ndarray, Image.Image],
        safe_margin_percent: float = 0.02,
        auto_orient: bool = True,
        preferred_model: str = "auto"
    ) -> np.ndarray:
        """
        Convenience method: Crops, rectifies perspective, and auto-orients upright in 1 call.
        Returns the rectified BGR NumPy array directly for downstream OCR / LLM ingestion.
        """
        img = self._load_image(image_input)
        corners, _, _ = self.detector.detect(img, preferred_model=preferred_model)
        cropped, _ = perspective_warp(img, corners, safe_margin_percent=safe_margin_percent)
        if auto_orient and cropped is not None and cropped.size > 0:
            from ..core.field_cropper import auto_orient_upright
            cropped, _ = auto_orient_upright(cropped)
        return cropped

    def process(
        self,
        image_input: Union[str, Path, bytes, np.ndarray, Image.Image],
        safe_margin_percent: float = 0.02,
        preferred_model: Optional[str] = None,
        auto_orient: bool = False,
        return_base64: bool = False,
        return_annotated_preview: bool = False,
        output_save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute full end-to-end document cropping and layout verification.
        
        Args:
            image_input: image path, bytes, base64, numpy array, or PIL image.
            safe_margin_percent: outward padding (default 2% = 0.02).
            preferred_model: "auto", "fastvit", "lcnet", "deeplab", "yolo", "classical".
            auto_orient: whether to automatically detect and rotate upright.
            return_base64: whether to encode cropped image into base64 string.
            return_annotated_preview: whether to include an annotated image showing corner points.
            output_save_path: optional file path to save the cropped output.
            
        Returns:
            Structured dictionary with crop, corners, quality, classification, and timing.
        """
        start_time = time.perf_counter()
        img = self._load_image(image_input)
        h, w = img.shape[:2]

        model_mode = preferred_model or self.preferred_model
        
        # 1. Detect 4 corners using Cascaded Hybrid Engine
        corners, stage_used, quad_metrics = self.detector.detect(img, preferred_model=model_mode)

        # 2. Warp document to planar rectangle
        cropped, H_matrix = perspective_warp(img, corners, safe_margin_percent=safe_margin_percent)
        
        # 3. Optional auto-orientation upright
        rotation_deg = 0
        if auto_orient and cropped is not None and cropped.size > 0:
            from ..core.field_cropper import auto_orient_upright
            cropped, rotation_deg = auto_orient_upright(cropped)

        crop_h, crop_w = cropped.shape[:2]

        # 4. Assess quality and document type
        quality = assess_document_quality(cropped)
        doc_layout = classify_document_aspect_ratio(crop_w, crop_h)

        # 5. Optional file saving
        if output_save_path:
            out_p = Path(output_save_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_p), cropped)

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        # Build response
        result: Dict[str, Any] = {
            "success": bool(quad_metrics.get("valid", True)),
            "pipeline_stage": stage_used,
            "latency_ms": float(round(latency_ms, 2)),
            "corners_pixel": [[float(round(pt[0], 2)), float(round(pt[1], 2))] for pt in corners],
            "corners_normalized": [[float(round(pt[0] / w, 4)), float(round(pt[1] / h, 4))] for pt in corners],
            "original_dimensions": {"width": w, "height": h},
            "cropped_dimensions": {"width": crop_w, "height": crop_h},
            "rotation_applied_deg": rotation_deg,
            "geometric_score": quad_metrics.get("score", 0.0),
            "coverage_ratio": quad_metrics.get("coverage", 0.0),
            "document_classification": doc_layout,
            "quality_assessment": quality,
        }

        # Base64 encodings if requested
        if return_base64:
            _, buffer = cv2.imencode(".jpg", cropped, [cv2.IMWRITE_JPEG_QUALITY, 95])
            b64_crop = base64.b64encode(buffer).decode("utf-8")
            result["crop_base64"] = f"data:image/jpeg;base64,{b64_crop}"

        if return_annotated_preview:
            vis = img.copy()
            pts = corners.astype(np.int32)
            cv2.polylines(vis, [pts], isClosed=True, color=(0, 255, 0), thickness=4)
            for i, p in enumerate(pts):
                cv2.circle(vis, tuple(p), 12, (0, 0, 255), -1)
                cv2.putText(vis, str(i + 1), (p[0] + 15, p[1] + 15), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
            
            _, vis_buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 85])
            b64_vis = base64.b64encode(vis_buf).decode("utf-8")
            result["annotated_preview_base64"] = f"data:image/jpeg;base64,{b64_vis}"

        return result

    def extract_field(
        self,
        image_input: Union[str, Path, bytes, np.ndarray, Image.Image],
        query: str,
        is_key_value: bool = False,
        padding_percent: float = 0.15,
        output_dir: Optional[str] = "eval_results/semantic_crops"
    ) -> Dict[str, Any]:
        """
        End-to-End Semantic Field Extraction:
        1. Rectifies the document first so lines are level and planar.
        2. Locates the target entity (e.g. 'Manula' or label 'Name:').
        3. Identifies the exact spatial location ('Where') and temporal provenance ('When').
        4. Micro-crops the isolated target snippet.
        """
        # Step 1: Macro-rectify document
        doc_result = self.process(image_input, return_base64=False)
        img = self._load_image(image_input)
        corners = np.array(doc_result["corners_pixel"], dtype=np.float32)
        rectified_doc, _ = perspective_warp(img, corners)

        # Step 2: Micro-crop the specific field
        matches = self.field_cropper.find_and_crop_field(
            rectified_doc,
            query=query,
            is_key_value=is_key_value,
            padding_percent=padding_percent,
            output_dir=output_dir
        )

        return {
            "document_success": doc_result["success"],
            "document_category": doc_result["document_classification"]["category"],
            "query": query,
            "match_count": len(matches),
            "matches": matches
        }

