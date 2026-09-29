import cv2
import numpy as np
from src.core.field_cropper import SemanticFieldCropper
from src.agent.document_cropper_agent import DocumentCropperAgent
from src.core.transform import perspective_warp

agent = DocumentCropperAgent()
img_path = 'data/01_05_Senuri.jpg'
doc_res = agent.process(img_path, return_base64=False)
corners = np.array(doc_res['corners_pixel'], dtype=np.float32)

img = cv2.imread(img_path)
rectified, _ = perspective_warp(img, corners)

fc = SemanticFieldCropper()
queries = ['pharmacy name', 'telephone', 'address', 'handwritten items']

print("==================================================")
print("PROMPT-DRIVEN FIELD CROPPING RESULTS ON 01_05_Senuri.jpg")
print("==================================================")

for q in queries:
    res = fc.query_and_crop(rectified, query=q)
    print(f"\n[QUERY]: '{q}'")
    print(f"  -> Matched Concept: {res.get('matched_concept')}")
    print(f"  -> Extracted Text:  {res.get('text_extracted')}")
    print(f"  -> Where (BBox):    {res.get('where_spatial', {}).get('bbox_pixel')}")
    print(f"  -> Dimensions:      {res.get('cropped_dimensions')}")
    print(f"  -> Saved Crop:      {res.get('cropped_artifact_path')}")
