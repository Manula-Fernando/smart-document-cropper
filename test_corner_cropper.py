"""
Standalone CLI & Testing Tool: Smart Document Corner Detection & Perspective Cropper.
Demonstrates:
  1. 4-Corner detection with green polygon overlay and corner coordinates.
  2. Perspective rectification into planar document scan.
  3. Automatic upright orientation.
  4. Saves output as '{basename}_detected_corners.jpg' and '{basename}_cropped.jpg' in eval_results/.

Usage:
  python test_corner_cropper.py --input data/03_02_Malshan.jpg
  python test_corner_cropper.py --sample 5
  python test_corner_cropper.py --all
"""
import os
import sys
import argparse
import time
from typing import Union, List, Dict, Any, Optional
from pathlib import Path
import cv2
import numpy as np

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.document_cropper_agent import DocumentCropperAgent
from src.core.transform import perspective_warp
from src.core.field_cropper import auto_orient_upright


VALID_EXTENSIONS = {".jpg", ".jpeg", ".jfif", ".png", ".webp", ".bmp", ".tiff", ".tif", ".pjpeg", ".pjp"}


def process_and_visualize(
    img_path: Union[str, Path],
    agent: DocumentCropperAgent,
    out_dir: Path,
    preferred_model: str = "auto",
    safe_margin: float = 0.01,
    auto_rotate: bool = True
) -> dict:
    img_path = Path(img_path)
    base_name = img_path.stem
    try:
        img = agent._load_image(img_path)
    except Exception as e:
        print(f"❌ Error: Could not read image ({e}): {img_path}")
        return {"success": False, "filename": img_path.name}

    h, w = img.shape[:2]

    # 1. Process via DocumentCropperAgent
    t0 = time.perf_counter()
    res = agent.process(
        img,
        safe_margin_percent=safe_margin,
        preferred_model=preferred_model,
        return_base64=False
    )
    latency_ms = (time.perf_counter() - t0) * 1000.0

    corners = np.array(res["corners_pixel"], dtype=np.float32)

    # 2. Rectify perspective
    cropped, _ = perspective_warp(img, corners, safe_margin_percent=safe_margin)
    
    # 3. Auto-orient upright if requested
    rot_deg = 0
    if auto_rotate and cropped is not None and cropped.size > 0:
        upright, rot_deg = auto_orient_upright(cropped)
    else:
        upright = cropped

    # 4. Create visual overlay image (*_detected_corners.jpg)
    vis = img.copy()
    
    # Adaptive sizing for display overlay
    max_dim = max(h, w)
    line_thickness = max(3, int(max_dim / 400))
    circle_radius = max(8, int(max_dim / 150))
    font_scale = max(0.8, max_dim / 1200.0)
    font_thickness = max(2, int(max_dim / 600))

    pts_int = corners.astype(np.int32)
    
    # Draw green polygon bounding quadrilateral
    cv2.polylines(vis, [pts_int], isClosed=True, color=(0, 255, 0), thickness=line_thickness)
    
    # Corner designations (TL, TR, BR, BL)
    labels = ["1: TL", "2: TR", "3: BR", "4: BL"]
    for i, pt in enumerate(pts_int):
        # Red circle at vertex
        cv2.circle(vis, tuple(pt), circle_radius, (0, 0, 255), -1)
        # White inner core
        cv2.circle(vis, tuple(pt), max(2, int(circle_radius / 3)), (255, 255, 255), -1)
        
        # Text label near corner
        coord_text = f"{labels[i]} ({pt[0]}, {pt[1]})"
        # Offset label so it doesn't overlap corner marker
        tx = pt[0] + 15 if pt[0] < w - 200 else pt[0] - 220
        ty = pt[1] - 15 if pt[1] > 40 else pt[1] + 35
        cv2.putText(vis, coord_text, (int(tx), int(ty)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.7, (0, 0, 0), font_thickness + 2)
        cv2.putText(vis, coord_text, (int(tx), int(ty)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.7, (0, 255, 255), font_thickness)

    # Top info banner
    banner_text = f"Route: {res['pipeline_stage']} | Score: {res['geometric_score']:.2f} | Latency: {res['latency_ms']:.1f}ms"
    # Background rectangle for banner readability
    banner_h = int(60 * (max_dim / 1000.0))
    cv2.rectangle(vis, (0, 0), (w, max(50, banner_h)), (20, 20, 20), -1)
    cv2.putText(vis, banner_text, (20, int(max(35, banner_h * 0.7))),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), font_thickness)

    # 5. Save outputs
    out_dir.mkdir(parents=True, exist_ok=True)
    out_corners_path = out_dir / f"{base_name}_detected_corners.jpg"
    out_cropped_path = out_dir / f"{base_name}_cropped.jpg"

    cv2.imwrite(str(out_corners_path), vis)
    cv2.imwrite(str(out_cropped_path), upright)

    return {
        "success": True,
        "filename": img_path.name,
        "original_size": f"{w}x{h}",
        "cropped_size": f"{upright.shape[1]}x{upright.shape[0]}",
        "stage": res["pipeline_stage"],
        "geometric_score": res["geometric_score"],
        "category": res["document_classification"]["category"],
        "rotation_deg": rot_deg,
        "latency_ms": res["latency_ms"],
        "corners": res["corners_pixel"],
        "corners_image": str(out_corners_path),
        "cropped_image": str(out_cropped_path)
    }


def main():
    parser = argparse.ArgumentParser(description="Test Smart Document Corner Detector & Cropper")
    parser.add_argument("--input", "-i", type=str, default=None, help="Path to single document image or folder")
    parser.add_argument("--sample", "-s", type=int, default=None, help="Test N sample images from data/ directory")
    parser.add_argument("--all", "-a", action="store_true", help="Test all images in data/ directory")
    parser.add_argument("--model", "-m", type=str, default="auto", choices=["auto", "fastvit", "lcnet", "deeplab", "yolo", "classical"], help="Model strategy")
    parser.add_argument("--margin", type=float, default=0.01, help="Safe border expansion margin (default 0.01 = 1%)")
    parser.add_argument("--out", "-o", type=str, default="eval_results", help="Output directory for generated images")
    args = parser.parse_args()

    out_dir = PROJECT_ROOT / args.out
    data_dir = PROJECT_ROOT / "data"

    print("=" * 70)
    print("🚀 SMART DOCUMENT CROPPER - CORNER DETECTION TEST HARNESS")
    print("=" * 70)
    print(f"Model Selection:    {args.model}")
    print(f"Safe Margin:        {args.margin * 100:.1f}%")
    print(f"Output Directory:   {out_dir}")
    print("-" * 70)

    agent = DocumentCropperAgent()

    # Determine files to process
    files_to_process = []
    if args.input:
        in_p = Path(args.input)
        if in_p.is_file():
            files_to_process = [in_p]
        elif in_p.is_dir():
            files_to_process = sorted([p for p in in_p.glob("*") if p.suffix.lower() in VALID_EXTENSIONS])
        else:
            print(f"❌ Input not found: {args.input}")
            sys.exit(1)
    elif args.all:
        files_to_process = sorted([p for p in data_dir.glob("*") if p.suffix.lower() in VALID_EXTENSIONS])
    elif args.sample:
        all_f = sorted([p for p in data_dir.glob("*") if p.suffix.lower() in VALID_EXTENSIONS])
        files_to_process = all_f[:args.sample]
    else:
        # Default: test 03_02_Malshan.jpg and 02_05_Senuri.jpg
        default_files = [data_dir / "03_02_Malshan.jpg", data_dir / "02_05_Senuri.jpg", data_dir / "01_05_Senuri.jpg"]
        files_to_process = [f for f in default_files if f.exists()]
        if not files_to_process:
            files_to_process = sorted([p for p in data_dir.glob("*") if p.suffix.lower() in VALID_EXTENSIONS])[:3]

    print(f"Found {len(files_to_process)} document(s) to process.\n")

    results = []
    for idx, fpath in enumerate(files_to_process, 1):
        print(f"[{idx}/{len(files_to_process)}] Processing: {fpath.name} ...")
        res = process_and_visualize(
            fpath,
            agent=agent,
            out_dir=out_dir,
            preferred_model=args.model,
            safe_margin=args.margin,
            auto_rotate=True
        )
        if res.get("success"):
            print(f"   ✅ Stage: {res['stage']} | Score: {res['geometric_score']*100:.1f}% | Latency: {res['latency_ms']:.1f}ms")
            print(f"   🖼️ Saved Corner Overlay: {res['corners_image']}")
            print(f"   ✂️ Saved Cropped Doc:    {res['cropped_image']}")
            results.append(res)
        else:
            print(f"   ❌ Failed to detect corners on {fpath.name}")

    print("\n" + "=" * 70)
    print("📊 CORNER DETECTION SUMMARY:")
    print("=" * 70)
    print(f"{'Filename':<28} | {'Stage':<18} | {'Score':<7} | {'Latency':<9} | {'Cropped Size':<12}")
    print("-" * 70)
    for r in results:
        print(f"{r['filename']:<28} | {r['stage']:<18} | {r['geometric_score']*100:>5.1f}% | {r['latency_ms']:>6.1f}ms | {r['cropped_size']:<12}")
    print("=" * 70)
    print(f"All images saved to: {out_dir.resolve()}\n")


if __name__ == "__main__":
    main()
