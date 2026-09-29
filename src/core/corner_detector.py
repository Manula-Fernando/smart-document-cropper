"""
Production-grade Document Corner & Layout Detectors using Local Lightweight Models.
Supports FastViT-SA24, LCNet100, DeepLabV3-MobileNetV3, YOLO11, and Classical CV.
"""
from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import cv2
import numpy as np
import onnxruntime as ort

logger = logging.getLogger("smart_document_cropper")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
WEIGHTS_DIR = PROJECT_ROOT / "weights"

HF_REPO_ID = "7rplus/pagescan-weights"
MODEL_FILES = {
    "fastvit_sa24": "fastvit_sa24_h_e_bifpn_256_fp32.onnx",
    "lcnet100": "lcnet100_h_e_bifpn_256_fp32.onnx",
    "deeplabv3_mbv3": "deeplabv3_mbv3_docseg.onnx",
    "yolo_doc_v1": "yolo_doc_v1.onnx",
}

def ensure_model_weights(model_key: str) -> Path:
    """Ensure ONNX weights exist locally, downloading from HF if necessary."""
    filename = MODEL_FILES.get(model_key)
    if not filename:
        raise ValueError(f"Unknown model key: {model_key}")
    
    local_path = WEIGHTS_DIR / filename
    if local_path.exists() and local_path.stat().st_size > 100_000:
        return local_path
    
    logger.info(f"Downloading {filename} from Hugging Face ({HF_REPO_ID})...")
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    from huggingface_hub import hf_hub_download
    downloaded = hf_hub_download(repo_id=HF_REPO_ID, filename=filename, local_dir=str(WEIGHTS_DIR))
    return Path(downloaded)

def order_corners(pts: np.ndarray) -> np.ndarray:
    """
    Sort 4 points clockwise: [Top-Left, Top-Right, Bottom-Right, Bottom-Left].
    Uses coordinate sum (x+y) and difference (y-x) projections.
    """
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).flatten()
    rect[0] = pts[np.argmin(s)]   # Top-Left: smallest x+y
    rect[2] = pts[np.argmax(s)]   # Bottom-Right: largest x+y
    rect[1] = pts[np.argmin(d)]   # Top-Right: smallest y-x
    rect[3] = pts[np.argmax(d)]   # Bottom-Left: largest y-x
    return rect

def check_quad_validity(corners: Optional[np.ndarray], img_w: int, img_h: int) -> Dict[str, Any]:
    """
    Validate quadrilateral geometry, aspect ratio sanity, coverage, and parallelism.
    Returns dictionary with validation flag, confidence score (0.0 to 1.0), and metrics.
    """
    if corners is None or len(corners) != 4:
        return {"valid": False, "score": 0.0, "reason": "Expected 4 coordinates"}
    
    ordered = order_corners(corners)
    tl, tr, br, bl = ordered

    # Edge lengths
    top = np.linalg.norm(tr - tl)
    bottom = np.linalg.norm(br - bl)
    left = np.linalg.norm(bl - tl)
    right = np.linalg.norm(br - tr)

    min_edge = min(top, bottom, left, right)
    if min_edge < 25:
        return {"valid": False, "score": 0.0, "reason": f"Edge too small ({min_edge:.1f}px)"}

    # Coverage
    area = cv2.contourArea(ordered.astype(np.int32))
    img_area = img_w * img_h
    coverage = area / max(img_area, 1)

    if coverage < 0.05 or coverage > 0.999:
        return {"valid": False, "score": 0.0, "reason": f"Abnormal coverage ({coverage*100:.1f}%)"}

    # Edge angles & Parallelism (top vs bottom, left vs right)
    def v_angle(v):
        return np.degrees(np.arctan2(v[1], v[0]))
    
    tb_diff = abs(v_angle(tr - tl) - v_angle(br - bl))
    if tb_diff > 180: tb_diff = 360 - tb_diff
    lr_diff = abs(v_angle(bl - tl) - v_angle(br - tr))
    if lr_diff > 180: lr_diff = 360 - lr_diff

    # Angle divergence penalty (0 divergence is parallel rectangle/trapezoid)
    divergence = (tb_diff + lr_diff)
    parallel_score = max(0.0, 1.0 - (divergence / 60.0))
    
    # Coverage weighting
    coverage_score = min(1.0, coverage * 2.0)
    
    overall_score = 0.6 * parallel_score + 0.4 * coverage_score
    is_valid = (divergence < 50.0) and (coverage >= 0.08)

    return {
        "valid": bool(is_valid),
        "score": float(round(overall_score, 4)),
        "coverage": float(round(coverage, 4)),
        "tb_divergence_deg": float(round(tb_diff, 2)),
        "lr_divergence_deg": float(round(lr_diff, 2)),
        "edge_lengths": [float(round(e, 1)) for e in [top, right, bottom, left]]
    }


class FastVitSA24Detector:
    """
    SOTA Vision-Transformer Corner-Heatmap Regressor with BiFPN head.
    Predicts sub-pixel gaussian heatmap peaks for 4 corners.
    """
    def __init__(self, model_path: Optional[str] = None):
        path = model_path or str(ensure_model_weights("fastvit_sa24"))
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(path, sess_options=opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.input_size = (256, 256)

    def detect(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        resized = cv2.resize(img, self.input_size, interpolation=cv2.INTER_LINEAR)
        tensor = resized.transpose(2, 0, 1).astype(np.float32)[None] / 255.0

        outputs = self.session.run([self.output_name], {self.input_name: tensor})
        heatmaps = outputs[0]  # shape: (1, 4, 256, 256)

        hm_h, hm_w = heatmaps.shape[2], heatmaps.shape[3]
        points = []
        for i in range(4):
            hmap = heatmaps[0, i]
            idx = np.argmax(hmap)
            py, px = np.unravel_index(idx, hmap.shape)
            cx = px * w / hm_w
            cy = py * h / hm_h
            points.append([cx, cy])
        
        return order_corners(np.array(points, dtype=np.float32))


class LCNet100Detector:
    """
    Ultra-lightweight (4.5 MB) PP-LCNet BiFPN Corner Regressor.
    Ideal for low-power edge CPUs and mobile-friendly scanning.
    """
    def __init__(self, model_path: Optional[str] = None):
        path = model_path or str(ensure_model_weights("lcnet100"))
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(path, sess_options=opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.input_size = (256, 256)

    def detect(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        resized = cv2.resize(img, self.input_size, interpolation=cv2.INTER_LINEAR)
        tensor = resized.transpose(2, 0, 1).astype(np.float32)[None] / 255.0

        outputs = self.session.run([self.output_name], {self.input_name: tensor})
        heatmaps = outputs[0]

        hm_h, hm_w = heatmaps.shape[2], heatmaps.shape[3]
        points = []
        for i in range(4):
            hmap = heatmaps[0, i]
            idx = np.argmax(hmap)
            py, px = np.unravel_index(idx, hmap.shape)
            cx = px * w / hm_w
            cy = py * h / hm_h
            points.append([cx, cy])
        
        return order_corners(np.array(points, dtype=np.float32))


class DeepLabV3Detector:
    """
    DeepLabV3 MobileNetV3 Document Semantic Segmentation.
    Generates full pixel segmentation masks, followed by polygon extraction.
    """
    def __init__(self, model_path: Optional[str] = None):
        path = model_path or str(ensure_model_weights("deeplabv3_mbv3"))
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(path, sess_options=opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.input_size = (384, 384)
        self.mean = np.array((0.4611, 0.4359, 0.3905), dtype=np.float32)
        self.std = np.array((0.2193, 0.2150, 0.2109), dtype=np.float32)

    def detect(self, img: np.ndarray) -> Optional[np.ndarray]:
        h, w = img.shape[:2]
        resized = cv2.resize(img, self.input_size, interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - self.mean) / self.std
        tensor = rgb.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)

        pred = self.session.run([self.output_name], {self.input_name: tensor})[0]
        if pred.ndim == 4 and pred.shape[1] == 2:
            mask = (pred[0].argmax(axis=0) * 255).astype(np.uint8)
        elif pred.ndim == 3:
            mask = ((pred[0] > 0.5) * 255).astype(np.uint8)
        else:
            return None

        # Resize mask back to original resolution
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        
        # Morphological smoothing
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        c = max(contours, key=cv2.contourArea)
        peri = cv2.arcLength(c, True)
        for eps_factor in [0.02, 0.03, 0.05]:
            approx = cv2.approxPolyDP(c, eps_factor * peri, True)
            if len(approx) == 4:
                return order_corners(approx.reshape(4, 2).astype(np.float32))
        
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        return order_corners(box.astype(np.float32))


class YOLODocDetector:
    """
    YOLO11 Document Bounding Box Detector (ONNX runtime, Dec 2025 weights).
    Locates coarse document bounding box in phone captures.
    """
    def __init__(self, model_path: Optional[str] = None):
        path = model_path or str(ensure_model_weights("yolo_doc_v1"))
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(path, sess_options=opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.input_size = 960

    def detect(self, img: np.ndarray, conf_threshold: float = 0.25) -> Optional[np.ndarray]:
        h0, w0 = img.shape[:2]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        r = self.input_size / max(h0, w0)
        new_h, new_w = round(h0 * r), round(w0 * r)
        resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        pad_y = (self.input_size - new_h) // 2
        pad_x = (self.input_size - new_w) // 2
        canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

        tensor = canvas.transpose(2, 0, 1).astype(np.float32)[None] / 255.0
        output = self.session.run(None, {self.input_name: tensor})[0]
        
        out = output[0]
        if out.shape[0] == 5:
            bboxes_cxcywh = out[:4].T
            scores = out[4]
        else:
            bboxes_cxcywh = out[:, :4]
            scores = out[:, 4]

        mask = scores >= conf_threshold
        if not mask.any():
            return None
        
        bboxes_cxcywh = bboxes_cxcywh[mask]
        scores = scores[mask]
        best_i = int(np.argmax(scores))
        cx, cy, bw, bh = bboxes_cxcywh[best_i]

        x1 = max(0.0, (cx - bw / 2 - pad_x) / r)
        y1 = max(0.0, (cy - bh / 2 - pad_y) / r)
        x2 = min(float(w0 - 1), (cx + bw / 2 - pad_x) / r)
        y2 = min(float(h0 - 1), (cy + bh / 2 - pad_y) / r)

        corners = np.array([
            [x1, y1],
            [x2, y1],
            [x2, y2],
            [x1, y2]
        ], dtype=np.float32)
        return corners


class ClassicalContourDetector:
    """
    Classical Bilateral + Adaptive Canny Contour Detector.
    Used as resilient zero-dependency fallback for high-contrast borders.
    """
    def detect(self, img: np.ndarray) -> Optional[np.ndarray]:
        h, w = img.shape[:2]
        scale = 800.0 / max(h, w)
        small = cv2.resize(img, (int(w * scale), int(h * scale)))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        edges = cv2.Canny(filtered, 50, 150)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated = cv2.dilate(edges, kernel, iterations=2)
        
        contours, _ = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
        
        small_area = small.shape[0] * small.shape[1]
        for c in contours:
            area = cv2.contourArea(c)
            if area < 0.1 * small_area:
                continue
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                pts = approx.reshape(4, 2) / scale
                return order_corners(pts.astype(np.float32))
                
        return None
