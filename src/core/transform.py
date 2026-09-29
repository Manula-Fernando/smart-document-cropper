"""
Geometric transformations, Homography perspective warping, and margin handling.
"""
from __future__ import annotations

import cv2
import numpy as np
from typing import Tuple
from .corner_detector import order_corners


def expand_corners(corners: np.ndarray, margin_percent: float = 0.01) -> np.ndarray:
    """
    Expand detected quadrilateral outward by a percentage margin.
    Useful for ensuring no stamps, seals, or boundary signatures get clipped.
    """
    if margin_percent <= 0:
        return corners
    
    ordered = order_corners(corners)
    centroid = np.mean(ordered, axis=0)
    
    expanded = np.zeros_like(ordered)
    for i, pt in enumerate(ordered):
        direction = pt - centroid
        expanded[i] = pt + direction * margin_percent
        
    return expanded


def perspective_warp(
    image: np.ndarray,
    corners: np.ndarray,
    safe_margin_percent: float = 0.0,
    interpolation: int = cv2.INTER_LANCZOS4
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applies homography perspective transformation to rectify document into a planar rectangular view.
    Preserves document's natural aspect ratio without squishing or stretching.
    
    Args:
        image: Original BGR or RGB image.
        corners: 4 quadrilateral points (x, y).
        safe_margin_percent: optional outward margin expansion (e.g. 0.01 for 1%).
        interpolation: cv2 interpolation flag (INTER_LANCZOS4 for highest text clarity).
        
    Returns:
        (warped_image, homography_matrix)
    """
    if safe_margin_percent > 0:
        corners = expand_corners(corners, safe_margin_percent)
        
    ordered = order_corners(corners)
    tl, tr, br, bl = ordered

    # Compute Euclidean widths
    width_top = np.linalg.norm(tr - tl)
    width_bot = np.linalg.norm(br - bl)
    max_w = int(round(max(width_top, width_bot)))

    # Compute Euclidean heights
    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)
    max_h = int(round(max(height_left, height_right)))

    if max_w <= 20 or max_h <= 20:
        return image, np.eye(3, dtype=np.float32)

    # Destination rectangular coordinates
    dst = np.array([
        [0, 0],
        [max_w - 1, 0],
        [max_w - 1, max_h - 1],
        [0, max_h - 1]
    ], dtype=np.float32)

    # Compute perspective homography matrix
    M = cv2.getPerspectiveTransform(ordered, dst)
    warped = cv2.warpPerspective(image, M, (max_w, max_h), flags=interpolation)

    return warped, M
