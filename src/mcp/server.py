"""
Native Model Context Protocol (MCP) Server for Smart Document Cropping and Layout Extraction.
Allows Claude Desktop, Antigravity, Cursor, OpenAI Agents, and MCP clients to invoke
document cropping and edge detection tools autonomously.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Dict, Any
from mcp.server.fastmcp import FastMCP
from ..agent.document_cropper_agent import DocumentCropperAgent

# Initialize FastMCP Server
mcp = FastMCP("SmartDocumentCropperServer")
agent = DocumentCropperAgent()


@mcp.tool()
def crop_document(
    image_path: str,
    output_path: Optional[str] = None,
    safe_margin_percent: float = 0.01,
    preferred_model: str = "auto"
) -> Dict[str, Any]:
    """
    Detect document boundaries, rectify perspective, crop document, and assess quality.
    
    Args:
        image_path: Absolute or relative local path to the document image file (or base64 data URI).
        output_path: Optional path where cropped rectified image should be saved.
        safe_margin_percent: Outward border margin expansion (default 0.01 = 1%).
        preferred_model: Model selection ('auto', 'fastvit', 'lcnet', 'deeplab', 'yolo', 'classical').
        
    Returns:
        Structured dictionary containing corner coordinates, document layout classification,
        quality scores (blur, glare), execution latency, and saved file path.
    """
    if not output_path and not image_path.startswith("data:image"):
        p = Path(image_path)
        output_path = str(p.parent / f"{p.stem}_cropped{p.suffix}")

    res = agent.process(
        image_path,
        safe_margin_percent=safe_margin_percent,
        preferred_model=preferred_model,
        return_base64=False,
        output_save_path=output_path
    )
    res["saved_to"] = output_path
    return res


@mcp.tool()
def detect_document_corners(
    image_path: str,
    preferred_model: str = "auto"
) -> Dict[str, Any]:
    """
    Detect the four precise quadrilateral corner coordinates of a document without saving an image.
    
    Args:
        image_path: Local file path to the document image.
        preferred_model: Model selection ('auto', 'fastvit', 'lcnet', 'deeplab', 'yolo', 'classical').
        
    Returns:
        4 corner coordinates (pixel and normalized), geometric validity score, and detected layout.
    """
    res = agent.process(
        image_path,
        preferred_model=preferred_model,
        return_base64=False
    )
    return {
        "success": res["success"],
        "pipeline_stage": res["pipeline_stage"],
        "corners_pixel": res["corners_pixel"],
        "corners_normalized": res["corners_normalized"],
        "document_classification": res["document_classification"],
        "geometric_score": res["geometric_score"],
        "latency_ms": res["latency_ms"]
    }


@mcp.tool()
def assess_document_quality(image_path: str) -> Dict[str, Any]:
    """
    Analyze image quality for KYC verification: checks for blurriness, glare highlights,
    resolution, and lighting contrast.
    
    Args:
        image_path: Local file path to the document image.
        
    Returns:
        Quality metrics including blur score (Laplacian variance), glare percentage, and overall status.
    """
    res = agent.process(image_path, return_base64=False)
    return {
        "quality_assessment": res["quality_assessment"],
        "document_classification": res["document_classification"],
        "original_dimensions": res["original_dimensions"]
    }


@mcp.tool()
def crop_semantic_field(
    image_path: str,
    field_query: str,
    is_key_value: bool = False,
    output_dir: str = "eval_results/semantic_crops"
) -> Dict[str, Any]:
    """
    Intelligent Sub-Document Field Cropper:
    Locates a specific field or name (e.g. 'Manula', 'Name', 'NIC', 'Total') inside the document,
    detects its exact spatial coordinates ('Where'), recording timestamp ('When'),
    and crops out the isolated high-resolution field image snippet.
    
    Args:
        image_path: Local path to the document image.
        field_query: Target text or label name to find and crop (e.g. 'Manula', 'Name', 'Total').
        is_key_value: If True, looks for the value to the right of or below the field_query label.
        output_dir: Directory to save the cropped field snippet.
        
    Returns:
        Matches list with exact bounding coordinates ('Where'), extraction timestamp ('When'),
        and saved cropped field image path.
    """
    return agent.extract_field(
        image_path,
        query=field_query,
        is_key_value=is_key_value,
        output_dir=output_dir
    )


if __name__ == "__main__":
    mcp.run()
