import cv2
import pytesseract
import os
import json
from datetime import datetime

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Test on rectified crop
img_path = 'eval_results/01_02_Senuri_cropped.jpg'
img = cv2.imread(img_path)

data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)

print(f"Total tokens detected: {len(data['text'])}")

targets = ['SENADHEERA', '16467', 'MEEGODA', 'MEDICAL']
os.makedirs('eval_results/semantic_crops', exist_ok=True)

results = []
for i in range(len(data['text'])):
    word = data['text'][i].strip()
    if not word:
        continue
    for t in targets:
        if t.lower() in word.lower():
            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            conf = float(data['conf'][i])
            
            # Crop with safe margin
            pad_x = int(w * 0.1) + 8
            pad_y = int(h * 0.1) + 6
            y1 = max(0, y - pad_y)
            y2 = min(img.shape[0], y + h + pad_y)
            x1 = max(0, x - pad_x)
            x2 = min(img.shape[1], x + w + pad_x)
            
            field_crop = img[y1:y2, x1:x2]
            crop_filename = f"{t.lower()}_crop.jpg"
            out_path = os.path.join('eval_results/semantic_crops', crop_filename)
            cv2.imwrite(out_path, field_crop)
            
            record = {
                "target_query": t,
                "detected_text": word,
                "confidence": conf,
                "where_spatial": {
                    "bounding_box": [x, y, w, h],
                    "crop_coords": [x1, y1, x2, y2],
                    "document_dimensions": [img.shape[1], img.shape[0]]
                },
                "when_temporal": {
                    "timestamp_iso": datetime.now().isoformat(),
                    "source_image": img_path
                },
                "cropped_artifact": out_path
            }
            results.append(record)
            print(f"-> Found '{t}' -> text: '{word}', bbox: [x={x}, y={y}, w={w}, h={h}], saved: {out_path}")

print("\nSummary Results JSON:")
print(json.dumps(results, indent=2))
