"""
Document Quality Assessment and Layout / Type Classification.
Computes blurriness, glare, shadow uniformity, and classifies document format (NIC, Passport, Invoice, Cheque).
"""
from __future__ import annotations

import cv2
import numpy as np
from typing import Dict, Any


def classify_document_aspect_ratio(w: int, h: int) -> Dict[str, Any]:
    """
    Classify standard document layout based on cropped width and height.
    Covers Sri Lankan NICs, Driving Licenses, Passports, Bank Cheques, Invoices.
    """
    if h == 0 or w == 0:
        return {"category": "unknown", "aspect_ratio": 1.0, "orientation": "unknown"}

    aspect = max(w, h) / min(w, h)
    orientation = "landscape" if w >= h else "portrait"

    # Known aspect ratio benchmarks
    # ID Card (CR80 ISO/IEC 7810: 85.60mm x 53.98mm) -> 1.586
    # Passport standard (125mm x 88mm) -> 1.42
    # A4 standard (297mm x 210mm) -> 1.414
    # Cheque standard (e.g. SL / International ~ 180-220mm x 80-90mm) -> 2.1 to 2.6
    
    if 2.0 <= aspect <= 2.9:
        category = "Bank Cheque / Pay Slip"
    elif 1.50 <= aspect <= 1.75:
        category = "National ID (NIC) / Driving License"
    elif 1.35 <= aspect < 1.50:
        category = "Invoice / A4 Form / Passport"
    elif 1.15 <= aspect < 1.35:
        category = "Letter / Receipt"
    else:
        category = "Custom Document / Banner"

    return {
        "category": category,
        "aspect_ratio": float(round(aspect, 3)),
        "orientation": orientation,
        "width_px": int(w),
        "height_px": int(h)
    }


def assess_document_quality(image: np.ndarray) -> Dict[str, Any]:
    """
    Assess cropped document image quality for downstream OCR / KYC verification.
    Computes:
      - Blurriness via Laplacian variance (higher is sharper; >100 is good, <50 is blurry)
      - Glare / Specular highlight percentage (>5% suggests glare covering text)
      - Exposure / Brightness (mean pixel brightness)
      - Contrast (standard deviation of luminance)
    """
    if image is None or image.size == 0:
        return {
            "blur_score": 0.0,
            "is_blurry": True,
            "glare_percentage": 0.0,
            "has_glare": False,
            "quality_status": "POOR"
        }

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

    # 1. Blurriness
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    is_blurry = bool(laplacian_var < 60.0)

    # 2. Glare (pixels with brightness > 250 in high-reflectance regions)
    glare_mask = (gray >= 252)
    glare_pct = (np.count_nonzero(glare_mask) / gray.size) * 100.0
    has_glare = bool(glare_pct > 3.5)

    # 3. Mean brightness and contrast
    mean_brightness = float(np.mean(gray))
    contrast_std = float(np.std(gray))

    # Overall Quality Tier
    if is_blurry and has_glare:
        status = "POOR (Blurry & High Glare)"
    elif is_blurry:
        status = "WARNING (Blur Detected - Re-capture Recommended)"
    elif has_glare:
        status = "WARNING (Glare Detected)"
    elif contrast_std < 25.0:
        status = "WARNING (Low Contrast)"
    else:
        status = "EXCELLENT (Ready for Enterprise OCR & KYC)"

    return {
        "blur_score": float(round(laplacian_var, 2)),
        "is_blurry": is_blurry,
        "glare_percentage": float(round(glare_pct, 2)),
        "has_glare": has_glare,
        "mean_brightness": float(round(mean_brightness, 1)),
        "contrast_std": float(round(contrast_std, 1)),
        "quality_status": status
    }
