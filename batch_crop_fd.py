"""
Batch Cropper and Orientation Tool for fd_tst_imgs (anchors and positives).
Crops document perspective and rotates upright to natural reading orientation.
Outputs to:
  - fd_tst_imgs/fd_tst_imgs/anchors_cropped
  - fd_tst_imgs/fd_tst_imgs/positives_cropped
"""
import sys
import time
from pathlib import Path
import cv2
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.document_cropper_agent import DocumentCropperAgent


def process_folder(folder_path: Path, out_dir: Path, agent: DocumentCropperAgent, label: str):
    images = sorted(list(folder_path.glob("*.jpg")))
    total = len(images)
    print(f"\n=======================================================")
    print(f"🚀 Processing {total} {label} images from: {folder_path.name}")
    print(f"📁 Destination: {out_dir}")
    print(f"=======================================================")

    out_dir.mkdir(parents=True, exist_ok=True)
    success_count = 0
    t0_folder = time.perf_counter()

    for idx, img_p in enumerate(images, 1):
        clean_filename = f"{img_p.stem.strip()}.jpg"
        save_path = out_dir / clean_filename
        t_start = time.perf_counter()

        try:
            # Crop + Rectify + Auto-Orient Upright in 1 call
            cropped = agent.crop(img_p, safe_margin_percent=0.02, auto_orient=True)
            if cropped is not None and cropped.size > 0:
                cv2.imwrite(str(save_path), cropped)
                elapsed_ms = (time.perf_counter() - t_start) * 1000.0
                h, w = cropped.shape[:2]
                print(f"[{idx:3d}/{total}] ✅ {clean_filename:<25} -> {w}x{h} px ({elapsed_ms:5.0f}ms)")
                success_count += 1
            else:
                print(f"[{idx:3d}/{total}] ⚠️ Empty crop for {img_p.name}")
        except Exception as e:
            print(f"[{idx:3d}/{total}] ❌ Failed {img_p.name}: {e}")

    total_sec = time.perf_counter() - t0_folder
    print(f"\n✨ Finished {label}: {success_count}/{total} images cropped successfully in {total_sec:.1f}s (avg {total_sec/total*1000:.0f}ms/img)")
    return success_count, total


def main():
    base_fd = Path("fd_tst_imgs/fd_tst_imgs").resolve()
    anchors_dir = base_fd / "anchors"
    positives_dir = base_fd / "positives"

    anchors_cropped_dir = base_fd / "anchors_cropped"
    positives_cropped_dir = base_fd / "positives_cropped"

    if not anchors_dir.exists() or not positives_dir.exists():
        print(f"❌ Error: Required input directories not found in {base_fd}")
        sys.exit(1)

    print("=" * 65)
    print("🏢 ENTERPRISE SMART DOCUMENT CROPPER - BATCH PROCESSING")
    print("=" * 65)
    print("Initializing Cascaded Hybrid AI Ensemble & OCR Orientation...")
    agent = DocumentCropperAgent()
    print("Agent ready.")

    # 1. Process Anchors
    s_a, t_a = process_folder(anchors_dir, anchors_cropped_dir, agent, label="Anchors")

    # 2. Process Positives
    s_p, t_p = process_folder(positives_dir, positives_cropped_dir, agent, label="Positives")

    print("\n" + "=" * 65)
    print("📊 BATCH PROCESSING COMPLETE SUMMARY")
    print("=" * 65)
    print(f"Anchors Cropped:   {s_a}/{t_a} -> {anchors_cropped_dir}")
    print(f"Positives Cropped: {s_p}/{t_p} -> {positives_cropped_dir}")
    print("=" * 65)


if __name__ == "__main__":
    main()

