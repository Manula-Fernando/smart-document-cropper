"""
Enterprise Smart Document Cropper & Orientation Engine: Python SDK Example.

Shows how to:
1. Crop, rectify perspective, and auto-orient any document in 1 line.
2. Obtain structured document classification, quality assessment, and coordinates.
3. Extract semantic fields (e.g. Total, Patient Name, Dates).
"""
import sys
from pathlib import Path
import cv2

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.document_cropper_agent import DocumentCropperAgent


def main():
    print("🚀 Initializing Enterprise Document Cropper Agent...")
    cropper = DocumentCropperAgent()

    # Sample input (replace with your image file path, bytes, or numpy array)
    sample_path = "sample_document.jpg"
    
    if not Path(sample_path).exists():
        print(f"ℹ️ Sample file '{sample_path}' not found. Showing code example:")
        print("-" * 60)
        print("""
# 1. One-line Cropping + Auto-Orientation
from src.agent.document_cropper_agent import DocumentCropperAgent
cropper = DocumentCropperAgent()

# Returns rectified BGR numpy array directly
cropped_img = cropper.crop("my_document.jpg", safe_margin_percent=0.02, auto_orient=True)
cv2.imwrite("output_cropped.jpg", cropped_img)

# 2. Full Enterprise Metadata Pipeline
result = cropper.process("my_document.jpg", auto_orient=True, return_base64=True)
print("Pipeline Stage:", result["pipeline_stage"])
print("Corners:", result["corners_pixel"])
print("Rotation applied:", result["rotation_applied_deg"])
print("Quality:", result["quality_assessment"])
print("Classification:", result["document_classification"])

# 3. Semantic Field Extraction (KIE)
field_data = cropper.extract_field("my_document.jpg", query="Total", is_key_value=True)
print("Detected Field:", field_data)
        """)
        print("-" * 60)
        return

    # 1. One-line cropping and auto-orientation
    print(f"\n1. Processing '{sample_path}'...")
    cropped = cropper.crop(sample_path, safe_margin_percent=0.02, auto_orient=True)
    out_path = "output_rectified.jpg"
    cv2.imwrite(out_path, cropped)
    print(f"   ✅ Saved cropped & upright document to '{out_path}' ({cropped.shape[1]}x{cropped.shape[0]} px)")

    # 2. Detailed processing with metadata
    res = cropper.process(sample_path, auto_orient=True)
    print(f"   ✅ Stage: {res['pipeline_stage']}")
    print(f"   ✅ Geometric Score: {res['geometric_score']*100:.1f}%")
    print(f"   ✅ Rotation Applied: {res.get('rotation_applied_deg', 0)}°")
    print(f"   ✅ Classification: {res['document_classification']['category']}")


if __name__ == "__main__":
    main()
