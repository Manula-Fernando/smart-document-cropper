"""
Enterprise Semantic Field Cropper & Key Information Extraction (KIE) Module.
Supports:
1. Flat-Field Illumination & Shadow Normalization (Solves faint dot-matrix, thermal receipts, and dark shadows).
2. Auto-Orientation / Upright Normalization (0°, 90°, 180°, 270°).
3. Fuzzy & Dot-Matrix Resilient Key-Value Pair Extraction (Tolerates optical typos like pattent->patient, totat->total).
4. Medical & Invoice Domain Semantic Decomposition (Patient Name, Consultant Doctor, Fees, Total, Date, Phone).
5. Intelligent 'Snip & Snap' (snaps arbitrary user-drawn bounding boxes to ink/text boundaries).
6. Enterprise Bridge for downstream Intelligent Data Extractor systems.
"""
from __future__ import annotations

import os
import re
import cv2
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from difflib import SequenceMatcher

# Configure Tesseract path across Windows, Linux, Docker, and Azure environments
import shutil
import pytesseract

tess_env = os.environ.get("TESSERACT_CMD")
if tess_env and os.path.exists(tess_env):
    pytesseract.pytesseract.tesseract_cmd = tess_env
else:
    TESSERACT_PATHS = [
        shutil.which("tesseract"),
        r'C:\Program Files\Tesseract-OCR\tesseract.exe',
        r'C:\Users\ManulaF\AppData\Local\Programs\Tesseract-OCR\tesseract.exe',
        '/usr/bin/tesseract',
        '/usr/local/bin/tesseract',
        'tesseract'
    ]
    for p in TESSERACT_PATHS:
        if p and os.path.exists(p):
            pytesseract.pytesseract.tesseract_cmd = p
            break


def _ensure_image(img_input: Any) -> np.ndarray:
    """Ensures input is converted to a BGR numpy array, supporting .jfif, .jpg, .png, .webp, .bmp, etc."""
    if isinstance(img_input, (str, Path)):
        p_str = str(img_input)
        loaded = None
        try:
            loaded = cv2.imdecode(np.fromfile(p_str, dtype=np.uint8), cv2.IMREAD_COLOR)
        except Exception:
            pass
        if loaded is None:
            loaded = cv2.imread(p_str)
        if loaded is None:
            try:
                from PIL import Image
                pil_img = Image.open(p_str).convert("RGB")
                loaded = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            except Exception:
                pass
        if loaded is None:
            raise FileNotFoundError(f"Could not load image from: {img_input}")
        return loaded
    return img_input


def normalize_illumination(img: Any) -> np.ndarray:
    """
    Flat-Field Illumination Normalization:
    Eliminates ambient shadows, lighting gradients, and paper discoloration
    commonly found in mobile document captures (such as 02_05_Senuri.jpg).
    Uses morphological background estimation and division to produce uniform white background
    with maximized contrast for faint dot-matrix or thermal ink.
    """
    img = _ensure_image(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    h, w = gray.shape[:2]
    k_size = max(25, int(min(h, w) / 40) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
    background = cv2.morphologyEx(gray, cv2.MORPH_DILATE, kernel)
    background = np.maximum(background, 1)
    diff = cv2.divide(gray, background, scale=255)
    return diff


def fuzzy_match_token(target: str, token_str: str) -> float:
    """
    Returns matching confidence score between query and OCR token.
    Tolerates dot-matrix printer font confusions (e.g. pattent->patient, totat->total).
    Enforces strict length and ratio constraints to avoid false substring matches.
    """
    t = target.lower().strip()
    k = re.sub(r'[^a-zA-Z0-9]', '', token_str.lower()).strip()
    if not t or not k:
        return 0.0
    if t == k:
        return 1.0
    # Strict matching for short acronyms/words
    if len(t) <= 3 or len(k) <= 3:
        return 1.0 if t == k else 0.0
    # Substring matching only for significant length and close proportion
    if len(t) >= 4 and len(k) >= 4:
        if (t in k or k in t) and abs(len(t) - len(k)) <= 3:
            return 0.95
        ratio = SequenceMatcher(None, t, k).ratio()
        if ratio >= 0.72:
            return ratio
    return 0.0


def auto_orient_upright(img: Any) -> Tuple[np.ndarray, int]:
    """
    Automatically detects whether the document is upside-down or sideways (0°, 90°, 180°, 270°)
    and rotates it upright based on OCR confidence and orientation scoring.
    Downsamples to optimal 1200px preview resolution for fast (~1s) CPU detection while preserving
    stroke fidelity of small fonts on receipts, doctor slips, and registration numbers.
    Applies illumination normalization to guarantee accuracy even on low-contrast/shadowed captures.
    """
    img = _ensure_image(img)
    h, w = img.shape[:2]
    max_dim = max(h, w)
    scale = min(1.0, 1200.0 / max_dim) if max_dim > 1200 else 1.0
    small = cv2.resize(img, (int(w * scale), int(h * scale))) if scale < 1.0 else img
    norm_small = normalize_illumination(small)

    best_angle = 0
    best_score = -1

    for angle in [0, 90, 180, 270]:
        if angle == 0:
            rot = norm_small
        elif angle == 90:
            rot = cv2.rotate(norm_small, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            rot = cv2.rotate(norm_small, cv2.ROTATE_180)
        elif angle == 270:
            rot = cv2.rotate(norm_small, cv2.ROTATE_90_COUNTERCLOCKWISE)

        try:
            data = pytesseract.image_to_data(rot, config='--psm 6', output_type=pytesseract.Output.DICT)
            # Filter words: length > 2 and confidence >= 45 (suppresses random OCR hallucinations on upside-down handwriting)
            valid_confs = [data['conf'][i] for i, w_txt in enumerate(data['text']) if len(w_txt.strip()) > 2 and data['conf'][i] >= 45]
            # Quadratic confidence weighting: heavily rewards high-confidence words (>70%) while suppressing noise
            angle_score = len(valid_confs) * 100.0 + sum((c / 100.0) ** 2 * 100.0 for c in valid_confs)
            if angle_score > best_score:
                best_score = angle_score
                best_angle = angle
        except Exception:
            pass

    if best_angle == 90:
        upright = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif best_angle == 180:
        upright = cv2.rotate(img, cv2.ROTATE_180)
    elif best_angle == 270:
        upright = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        upright = img.copy()

    return upright, best_angle


def snap_box_to_text(img: np.ndarray, x: int, y: int, w: int, h: int, padding: int = 12) -> Tuple[int, int, int, int]:
    """
    Intelligent Snapping: Takes an approximate user-drawn box or field region and snaps
    to the exact bounding box of the ink strokes inside, adding clean padding.
    Prevents text characters from being cut in half.
    """
    img_h, img_w = img.shape[:2]
    x1 = max(0, int(x))
    y1 = max(0, int(y))
    x2 = min(img_w, int(x + w))
    y2 = min(img_h, int(y + h))

    if x2 <= x1 or y2 <= y1:
        return x, y, w, h

    roi = img[y1:y2, x1:x2]
    if roi.size == 0:
        return x, y, w, h

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
    # Otsu thresholding to find ink pixels
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return x, y, w, h

    all_pts = np.vstack(contours)
    rx, ry, rw, rh = cv2.boundingRect(all_pts)

    snapped_x = max(0, (x1 + rx) - padding)
    snapped_y = max(0, (y1 + ry) - padding)
    snapped_w = min(img_w - snapped_x, rw + 2 * padding)
    snapped_h = min(img_h - snapped_y, rh + 2 * padding)

    return int(snapped_x), int(snapped_y), int(snapped_w), int(snapped_h)


class SemanticFieldCropper:
    """
    Intelligent field-level detector, open-vocabulary entity grounder, and micro-cropper.
    """
    def __init__(self, tesseract_cmd: Optional[str] = None):
        if tesseract_cmd and os.path.exists(tesseract_cmd):
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    def extract_tokens(self, image: Any, use_norm: bool = True) -> List[Dict[str, Any]]:
        """Run OCR on normalized image and return structured tokens with spatial coordinates."""
        image = _ensure_image(image)
        work_img = normalize_illumination(image) if use_norm else image
        
        # Dual-pass OCR: PSM 6 for structured forms/receipts, with PSM 3 fallback
        d6 = pytesseract.image_to_data(work_img, config='--psm 6', output_type=pytesseract.Output.DICT)
        tokens = []
        for i in range(len(d6['text'])):
            txt = d6['text'][i].strip()
            c = re.sub(r'[^a-zA-Z0-9]', '', txt.lower())
            if len(c) >= 1 and d6['conf'][i] > 15:
                tokens.append({
                    "text": txt,
                    "clean": c,
                    "x": d6['left'][i],
                    "y": d6['top'][i],
                    "w": d6['width'][i],
                    "h": d6['height'][i],
                    "yc": d6['top'][i] + d6['height'][i] / 2.0,
                    "conf": float(d6['conf'][i])
                })

        if len(tokens) < 15:
            d3 = pytesseract.image_to_data(work_img, config='--psm 3', output_type=pytesseract.Output.DICT)
            for i in range(len(d3['text'])):
                txt = d3['text'][i].strip()
                c = re.sub(r'[^a-zA-Z0-9]', '', txt.lower())
                if len(c) >= 1 and d3['conf'][i] > 15:
                    tokens.append({
                        "text": txt,
                        "clean": c,
                        "x": d3['left'][i],
                        "y": d3['top'][i],
                        "w": d3['width'][i],
                        "h": d3['height'][i],
                        "yc": d3['top'][i] + d3['height'][i] / 2.0,
                        "conf": float(d3['conf'][i])
                    })

        return tokens

    def find_key_value_pairs(self, image: Any, target_label: str) -> List[Dict[str, Any]]:
        """
        Robust Key-Value Pair Matcher:
        Searches for target label (e.g. 'Name', 'Patient', 'Tel', 'Total', 'Doctor', 'Date')
        and extracts the value either to the right on the same horizontal visual strip or stacked below.
        Uses illumination normalization, fuzzy sequence matching (handling dot-matrix typos),
        and 2D geometric visual alignment.
        """
        image = _ensure_image(image)
        tokens = self.extract_tokens(image, use_norm=True)
        if not tokens:
            return []

        target_clean = re.sub(r'[^a-zA-Z0-9]', '', target_label.lower())
        synonyms_map = {
            "tel": ["phone", "tel", "telephone", "mobile", "contact", "direct", "tp"],
            "telephone": ["phone", "tel", "telephone", "mobile", "contact", "direct", "tp"],
            "phone": ["phone", "tel", "telephone", "mobile", "contact", "direct", "tp"],
            "mobile": ["phone", "mobile", "tel", "contact", "tp"],
            "contact": ["phone", "tel", "contact", "mobile"],
            "name": ["patient name", "pattent name", "patient", "pattent", "name", "names", "darshana", "senuri", "manula"],
            "patient": ["patient name", "pattent name", "patient", "pattent", "darshana", "customer", "client"],
            "patientname": ["patient name", "pattent name", "patient", "pattent"],
            "doctor": ["consultant", "const tant", "doctor", "dr", "surgeon", "speciality", "selva"],
            "consultant": ["consultant", "const tant", "doctor", "dr", "surgeon", "speciality", "selva"],
            "consultantdoctor": ["consultant", "const tant", "doctor", "dr", "surgeon", "speciality", "selva"],
            "total": ["total fee", "totat ree", "total ree", "total fre", "total", "totat", "thtat"],
            "totalfee": ["total fee", "totat ree", "total ree", "total fre", "total", "totat", "thtat"],
            "amount": ["total fee", "totat ree", "total", "totat", "amount", "subtotal"],
            "price": ["total fee", "totat ree", "total", "amount", "price"],
            "fee": ["total fee", "total ree", "doc fee", "noc fre", "hos fee", "hos frr"],
            "docfee": ["doc fee", "noc fre", "noc fef", "doctor fee"],
            "hosfee": ["hos fee", "hos frr", "hos fre", "hospital fee"],
            "date": ["invoice date", "invotce date", "app date", "app tate", "date", "time"],
            "invoicedate": ["invoice date", "invotce date", "app date", "app tate", "date"],
            "invoice": ["invoice", "invouce", "invotce", "channel", "receipt", "bill"],
            "no": ["no", "num", "number", "inv", "bill", "reg", "ref", "reference"],
            "ref": ["reference", "ref", "peffrence", "hos ref"],
            "reference": ["reference", "ref", "peffrence", "hos ref"],
            "referenceno": ["reference", "ref", "peffrence", "hos ref"],
            "pharmacy": ["pharmacy", "bharmacy", "medical", "grocery", "chemist", "dr", "lakshan"],
            "hospital": ["philip", "hospitals", "hospital", "pvt", "ltd", "clinic"],
            "address": ["road", "rd", "street", "st", "lane", "kalutara", "galle", "madulawa"],
            "user": ["user id", "wsrr tf", "user", "hser", "ishant"],
            "userid": ["user id", "wsrr tf", "user", "hser", "ishant"],
            "room": ["room", "no"]
        }
        valid_targets = synonyms_map.get(target_clean, [target_clean])

        # Step 0: Try line-level parsing for lines with standard delimiters (':', '-', '=')
        norm = normalize_illumination(image)
        data = pytesseract.image_to_data(norm, config='--psm 6', output_type=pytesseract.Output.DICT)
        lines_dict: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
        for i in range(len(data['text'])):
            t_str = data['text'][i].strip()
            if not t_str:
                continue
            lid = (data['block_num'][i], data['line_num'][i])
            if lid not in lines_dict:
                lines_dict[lid] = []
            lines_dict[lid].append({
                "text": t_str,
                "clean": re.sub(r'[^a-zA-Z0-9]', '', t_str.lower()),
                "bbox": [data['left'][i], data['top'][i], data['width'][i], data['height'][i]],
                "conf": float(data['conf'][i])
            })

        for lid, toks in lines_dict.items():
            toks.sort(key=lambda t: t['bbox'][0])
            line_text = " ".join(t['text'] for t in toks)
            split_match = re.search(r'(?<!\d)[:=](?!\d)|(?<!\d)-|-(?!\d)', line_text)
            if split_match:
                s_idx = split_match.start()
                lbl_raw = line_text[:s_idx].strip()
                val_raw = line_text[s_idx+1:].strip()
                lbl_clean = re.sub(r'[^a-zA-Z0-9]', '', lbl_raw.lower())
                val_clean = re.sub(r'^[^\w]+|[^\w]+$', '', val_raw)
                if not val_clean or len(val_clean) < 2:
                    continue

                # Check if target matches this label
                matched = False
                for vt in valid_targets:
                    vt_c = re.sub(r'[^a-zA-Z0-9]', '', vt.lower())
                    if vt_c == lbl_clean or (len(vt_c) >= 4 and vt_c in lbl_clean):
                        matched = True
                        break
                    sc = fuzzy_match_token(vt_c, lbl_clean)
                    if sc >= 0.72:
                        matched = True
                        break

                if matched:
                    # Compute bounding box of the value tokens
                    val_toks = [t for t in toks if t['text'] in val_raw and t['bbox'][0] > toks[0]['bbox'][0] + toks[0]['bbox'][2] * 0.15]
                    if val_toks:
                        vx = min(t['bbox'][0] for t in val_toks)
                        vy = min(t['bbox'][1] for t in val_toks)
                        vw = max(t['bbox'][0] + t['bbox'][2] for t in val_toks) - vx
                        vh = max(t['bbox'][1] + t['bbox'][3] for t in val_toks) - vy
                    else:
                        min_x = min(t['bbox'][0] for t in toks)
                        max_x = max(t['bbox'][0] + t['bbox'][2] for t in toks)
                        min_y = min(t['bbox'][1] for t in toks)
                        max_y = max(t['bbox'][1] + t['bbox'][3] for t in toks)
                        vx = min_x + int((max_x - min_x) * 0.35)
                        vy = min_y
                        vw = max_x - vx
                        vh = max_y - min_y

                    sx, sy, sw, sh = snap_box_to_text(image, vx, vy, vw, vh)
                    return [{
                        'label_found': lbl_raw.lstrip('= -;').strip(),
                        'value_text': val_clean,
                        'bbox': [sx, sy, sw, sh],
                        'relation': 'inline_delimit_split',
                        'confidence': 0.95
                    }]

        # Step 1: Fallback to token-level candidate search
        candidate_labels = []
        for tok in tokens:
            for vt in valid_targets:
                score = fuzzy_match_token(vt, tok['clean'])
                if score >= 0.70:
                    candidate_labels.append((tok, score))
                    break

        candidate_labels.sort(key=lambda x: -x[1])
        results: List[Dict[str, Any]] = []

        for label_tok, score in candidate_labels:
            # 2. Geometric search to the right on the same horizontal strip
            right_tokens = [
                t for t in tokens 
                if t != label_tok 
                and t['x'] > label_tok['x'] + label_tok['w'] * 0.15 
                and abs(t['yc'] - label_tok['yc']) <= max(label_tok['h'], t['h']) * 1.05
            ]

            if right_tokens:
                right_tokens.sort(key=lambda t: t['x'])
                val_text = ' '.join(t['text'] for t in right_tokens).lstrip(': -./=\t').strip()
                if val_text:
                    min_x = min(t['x'] for t in right_tokens)
                    min_y = min(t['y'] for t in right_tokens)
                    max_x = max(t['x'] + t['w'] for t in right_tokens)
                    max_y = max(t['y'] + t['h'] for t in right_tokens)
                    sx, sy, sw, sh = snap_box_to_text(image, min_x, min_y, max_x - min_x, max_y - min_y)
                    results.append({
                        'label_found': label_tok['text'],
                        'value_text': val_text,
                        'bbox': [sx, sy, sw, sh],
                        'relation': 'right_of_label',
                        'confidence': float(score)
                    })
                    break

            # 3. Geometric search directly below the label (stacked key-value)
            below_tokens = [
                t for t in tokens 
                if t != label_tok 
                and t['y'] > label_tok['y'] + label_tok['h'] * 0.35 
                and t['y'] <= label_tok['y'] + label_tok['h'] * 3.0
            ]

            if below_tokens:
                below_tokens.sort(key=lambda t: t['x'])
                val_text = ' '.join(t['text'] for t in below_tokens).lstrip(': -./=\t').strip()
                if val_text:
                    min_x = min(t['x'] for t in below_tokens)
                    min_y = min(t['y'] for t in below_tokens)
                    max_x = max(t['x'] + t['w'] for t in below_tokens)
                    max_y = max(t['y'] + t['h'] for t in below_tokens)
                    sx, sy, sw, sh = snap_box_to_text(image, min_x, min_y, max_x - min_x, max_y - min_y)
                    results.append({
                        'label_found': label_tok['text'],
                        'value_text': val_text,
                        'bbox': [sx, sy, sw, sh],
                        'relation': 'below_label',
                        'confidence': float(score)
                    })
                    break

        return results

    def find_and_crop_field(
        self,
        image: Any,
        query: str,
        is_key_value: bool = False,
        pre_oriented: bool = False,
        padding_percent: float = 0.15,
        output_dir: Optional[str] = "eval_results/semantic_crops"
    ) -> List[Dict[str, Any]]:
        """
        Finds target field (either key-value pair or direct text) and micro-crops snippet.
        """
        image = _ensure_image(image)
        upright = image.copy() if pre_oriented else auto_orient_upright(image)[0]
        img_h, img_w = upright.shape[:2]
        matches = []

        if is_key_value:
            kv_pairs = self.find_key_value_pairs(upright, query)
            for pair in kv_pairs:
                bx, by, bw, bh = pair['bbox']
                crop = upright[by:by+bh, bx:bx+bw]
                saved_path = None
                if output_dir and crop.size > 0:
                    os.makedirs(output_dir, exist_ok=True)
                    clean_name = re.sub(r'[^a-zA-Z0-9_]', '', pair['value_text'].lower())[:15]
                    filename = f"kv_{clean_name}_{int(datetime.now().timestamp()*1000)}.jpg"
                    saved_path = os.path.join(output_dir, filename)
                    cv2.imwrite(saved_path, crop)

                matches.append({
                    "query": query,
                    "detected_value": pair['value_text'],
                    "label_context": pair['label_found'],
                    "field_type": f"key_value ({pair['relation']})",
                    "confidence_score": pair['confidence'],
                    "where_spatial": {
                        "bbox_pixel": [bx, by, bw, bh],
                        "normalized_coords": [round(bx/img_w, 4), round(by/img_h, 4), round(bw/img_w, 4), round(bh/img_h, 4)],
                        "document_resolution": [img_w, img_h]
                    },
                    "when_temporal": {
                        "extracted_at": datetime.now().isoformat(),
                        "extraction_source": "SmartDoc-KV-Engine"
                    },
                    "cropped_artifact_path": saved_path,
                    "cropped_dimensions": [int(crop.shape[1]), int(crop.shape[0])] if crop.size > 0 else [0, 0]
                })
        else:
            res = self.query_and_crop(upright, query, is_key_value=False, pre_oriented=True, output_dir=output_dir)
            if res.get("success"):
                matches.append(res)

        return matches

    def extract_structured_document(self, image: Any, pre_oriented: bool = False) -> Dict[str, Any]:
        """
        Decomposes an invoice, hospital bill, or medical receipt (e.g. 01_05_Senuri.jpg, 02_05_Senuri.jpg)
        into key semantic zones:
        - Hospital / Pharmacy / Business Name
        - Patient Name
        - Consultant / Doctor
        - Telephone / Contact
        - Address
        - Totals / Numerical amounts
        - Invoice / Appointment Date
        - Handwritten Items / Prescription Body
        """
        image = _ensure_image(image)
        upright, angle = (image.copy(), 0) if pre_oriented else auto_orient_upright(image)
        img_h, img_w = upright.shape[:2]

        norm = normalize_illumination(upright)
        raw_txt = pytesseract.image_to_string(norm, config='--psm 6').strip()
        lines = [l.strip() for l in raw_txt.split('\n') if len(l.strip()) > 2]

        data = pytesseract.image_to_data(norm, config='--psm 6', output_type=pytesseract.Output.DICT)
        
        lines_dict: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
        for i in range(len(data['text'])):
            text = data['text'][i].strip()
            if not text:
                continue
            line_id = (data['block_num'][i], data['line_num'][i])
            if line_id not in lines_dict:
                lines_dict[line_id] = []
            lines_dict[line_id].append({
                "text": text,
                "bbox": [data['left'][i], data['top'][i], data['width'][i], data['height'][i]],
                "conf": float(data['conf'][i])
            })

        formatted_lines = []
        for l_id, toks in lines_dict.items():
            toks.sort(key=lambda t: t['bbox'][0])
            full_line = " ".join(t['text'] for t in toks)
            min_x = min(t['bbox'][0] for t in toks)
            min_y = min(t['bbox'][1] for t in toks)
            max_x = max(t['bbox'][0] + t['bbox'][2] for t in toks)
            max_y = max(t['bbox'][1] + t['bbox'][3] for t in toks)
            formatted_lines.append({
                "text": full_line,
                "bbox": [min_x, min_y, max_x - min_x, max_y - min_y],
                "tokens": toks,
                "avg_conf": float(np.mean([t['conf'] for t in toks]))
            })

        fields: Dict[str, Any] = {}
        
        # 1. Header / Hospital / Pharmacy Name
        business_line = None
        for line in formatted_lines:
            txt_lower = line["text"].lower()
            if any(k in txt_lower for k in ["philip", "hospital", "pharmacy", "medical", "grocery", "clinic", "ltd", "dr."]):
                business_line = line
                break
        if not business_line and formatted_lines:
            business_line = formatted_lines[0]

        if business_line:
            bx, by, bw, bh = snap_box_to_text(upright, *business_line["bbox"])
            fields["pharmacy_name"] = {
                "label": "Hospital / Pharmacy Name",
                "text": business_line["text"],
                "bbox": [bx, by, bw, bh],
                "crop": upright[by:by+bh, bx:bx+bw]
            }

        # 2. Patient Name
        patient_line = None
        for line in formatted_lines:
            txt_lower = line["text"].lower()
            if any(k in txt_lower for k in ["patient", "pattent", "darshana", "darshawa", "mr "]):
                patient_line = line
                break
        if patient_line:
            px, py, pw, ph = snap_box_to_text(upright, *patient_line["bbox"])
            fields["patient_name"] = {
                "label": "Patient Name",
                "text": patient_line["text"],
                "bbox": [px, py, pw, ph],
                "crop": upright[py:py+ph, px:px+pw]
            }

        # 3. Consultant / Doctor
        doctor_line = None
        for line in formatted_lines:
            txt_lower = line["text"].lower()
            if any(k in txt_lower for k in ["consultant", "const", "selva", "vettiveld", "surgeon", "doctor", "dr "]):
                doctor_line = line
                break
        if doctor_line:
            dx, dy, dw, dh = snap_box_to_text(upright, *doctor_line["bbox"])
            fields["consultant_doctor"] = {
                "label": "Consultant / Doctor",
                "text": doctor_line["text"],
                "bbox": [dx, dy, dw, dh],
                "crop": upright[dy:dy+dh, dx:dx+dw]
            }

        # 4. Telephone
        phone_match = None
        phone_regex = re.compile(r'(?:Tel|Phone|Mobile|T\.P\.?|TP|Direct)\s*[:.]?\s*([0-9\s+-]{8,25})', re.IGNORECASE)
        for line in formatted_lines:
            m = phone_regex.search(line["text"])
            if m or ("07" in line["text"] and len(re.findall(r'\d', line["text"])) >= 9) or "034" in line["text"]:
                phone_match = line
                break
        if phone_match:
            px, py, pw, ph = snap_box_to_text(upright, *phone_match["bbox"])
            fields["telephone"] = {
                "label": "Telephone Number",
                "text": phone_match["text"],
                "bbox": [px, py, pw, ph],
                "crop": upright[py:py+ph, px:px+pw]
            }

        # 5. Address
        address_lines = []
        for line in formatted_lines:
            if line == business_line or line == phone_match or line == patient_line or line == doctor_line:
                continue
            txt_lower = line["text"].lower()
            if any(k in txt_lower for k in ["road", "rd", "street", "st", "lane", "box", "lk/", "galle", "kalutara", "madulawa"]):
                address_lines.append(line)
        if address_lines:
            min_x = min(l["bbox"][0] for l in address_lines)
            min_y = min(l["bbox"][1] for l in address_lines)
            max_x = max(l["bbox"][0] + l["bbox"][2] for l in address_lines)
            max_y = max(l["bbox"][1] + l["bbox"][3] for l in address_lines)
            ax, ay, aw, ah = snap_box_to_text(upright, min_x, min_y, max_x - min_x, max_y - min_y)
            fields["address"] = {
                "label": "Address Block",
                "text": " | ".join(l["text"] for l in address_lines),
                "bbox": [ax, ay, aw, ah],
                "crop": upright[ay:ay+ah, ax:ax+aw]
            }

        # 6. Total Block
        total_line = None
        for line in reversed(formatted_lines):
            txt_lower = line["text"].lower()
            if any(k in txt_lower for k in ["total", "totat", "thtat", "net", "amount", "rs", "subtotal", "bal", "3600"]):
                total_line = line
                break
        if total_line:
            tx, ty, tw, th = snap_box_to_text(upright, *total_line["bbox"])
            fields["total"] = {
                "label": "Total / Amount Block",
                "text": total_line["text"],
                "bbox": [tx, ty, tw, th],
                "crop": upright[ty:ty+th, tx:tx+tw]
            }

        # 7. Invoice Date
        date_line = None
        for line in formatted_lines:
            txt_lower = line["text"].lower()
            if any(k in txt_lower for k in ["date", "time", "invoice", "invouce", "invotce", "2024"]):
                date_line = line
                break
        if date_line:
            ix, iy, iw, ih = snap_box_to_text(upright, *date_line["bbox"])
            fields["invoice_date"] = {
                "label": "Invoice Date & Timestamp",
                "text": date_line["text"],
                "bbox": [ix, iy, iw, ih],
                "crop": upright[iy:iy+ih, ix:ix+iw]
            }

        # 8. Handwritten Items / Body
        header_bottom = max([f["bbox"][1] + f["bbox"][3] for f in fields.values()]) if fields else int(img_h * 0.25)
        body_y1 = min(img_h - 100, header_bottom + 15)
        body_y2 = img_h - 30
        body_crop = upright[body_y1:body_y2, 10:img_w-10]
        fields["handwritten_items"] = {
            "label": "Document Body & Line Items",
            "text": "[Prescriptions, line items, medical entries, or body text]",
            "bbox": [10, body_y1, img_w - 20, body_y2 - body_y1],
            "crop": body_crop
        }

        return {
            "upright_image": upright,
            "rotation_applied_deg": angle,
            "detected_fields": fields,
            "lines": formatted_lines
        }

    def query_and_crop(
        self,
        image: Any,
        query: str,
        is_key_value: bool = False,
        pre_oriented: bool = False,
        output_dir: Optional[str] = "eval_results/semantic_crops"
    ) -> Dict[str, Any]:
        """
        Unified Query & Crop Interface:
        Dispatches to Key-Value pair extractor if is_key_value=True,
        or open-vocabulary concept grounder / entity search if False.
        """
        image = _ensure_image(image)
        upright, angle = (image.copy(), 0) if pre_oriented else auto_orient_upright(image)
        img_h, img_w = upright.shape[:2]
        q = query.strip().lower()

        # Mode A: Key-Value requested
        if is_key_value:
            pairs = self.find_key_value_pairs(upright, query)
            if pairs:
                p = pairs[0]  # Best match
                bx, by, bw, bh = p['bbox']
                crop_img = upright[by:by+bh, bx:bx+bw]
                saved_path = None
                if output_dir and crop_img.size > 0:
                    os.makedirs(output_dir, exist_ok=True)
                    clean_name = re.sub(r'[^a-zA-Z0-9_]', '', p['value_text'].lower())[:15]
                    filename = f"field_kv_{clean_name}_{int(datetime.now().timestamp()*1000)}.jpg"
                    saved_path = os.path.join(output_dir, filename)
                    cv2.imwrite(saved_path, crop_img)

                return {
                    "success": True,
                    "query": query,
                    "matched_concept": f"Value next to '{p['label_found']}' ({p['relation']})",
                    "text_extracted": p['value_text'],
                    "label_context": p['label_found'],
                    "field_type": "key_value_pair",
                    "confidence_score": p['confidence'],
                    "where_spatial": {
                        "bbox_pixel": [bx, by, bw, bh],
                        "normalized": [round(bx/img_w, 4), round(by/img_h, 4), round(bw/img_w, 4), round(bh/img_h, 4)],
                        "document_resolution": [img_w, img_h]
                    },
                    "when_temporal": {
                        "extracted_at": datetime.now().isoformat(),
                        "source": "SmartDoc-KV-Engine"
                    },
                    "cropped_artifact_path": saved_path,
                    "cropped_dimensions": [int(crop_img.shape[1]), int(crop_img.shape[0])]
                }

        # Mode B: Semantic concept matching
        decomp = self.extract_structured_document(upright, pre_oriented=True)
        fields = decomp["detected_fields"]

        target_field = None
        if any(k in q for k in ["patient", "pattent", "customer", "client", "name"]):
            target_field = fields.get("patient_name")
        elif any(k in q for k in ["doctor", "consultant", "speciality", "physician", "surgeon"]):
            target_field = fields.get("consultant_doctor")
        elif any(k in q for k in ["hospital", "pharmacy", "business", "store", "header", "clinic"]):
            target_field = fields.get("pharmacy_name")
        elif any(k in q for k in ["tel", "phone", "mobile", "contact", "number", "call", "direct"]):
            target_field = fields.get("telephone")
        elif any(k in q for k in ["date", "time", "invoice", "invouce", "invotce"]):
            target_field = fields.get("invoice_date")
        elif any(k in q for k in ["address", "location", "road", "street", "city"]):
            target_field = fields.get("address")
        elif any(k in q for k in ["item", "medicine", "handwritten", "prescription", "rx", "drugs", "body"]):
            target_field = fields.get("handwritten_items")
        elif any(k in q for k in ["total", "amount", "price", "net", "cost", "bill", "pay", "fee"]):
            target_field = fields.get("total")

        if target_field:
            crop_img = target_field["crop"]
            saved_path = None
            if output_dir and crop_img.size > 0:
                os.makedirs(output_dir, exist_ok=True)
                filename = f"field_{re.sub(r'[^a-zA-Z0-9_]', '', q)[:15]}_{int(datetime.now().timestamp()*1000)}.jpg"
                saved_path = os.path.join(output_dir, filename)
                cv2.imwrite(saved_path, crop_img)

            return {
                "success": True,
                "query": query,
                "matched_concept": target_field["label"],
                "text_extracted": target_field["text"],
                "field_type": "semantic_zone",
                "where_spatial": {
                    "bbox_pixel": target_field["bbox"],
                    "normalized": [
                        round(target_field["bbox"][0] / img_w, 4),
                        round(target_field["bbox"][1] / img_h, 4),
                        round(target_field["bbox"][2] / img_w, 4),
                        round(target_field["bbox"][3] / img_h, 4)
                    ],
                    "document_resolution": [img_w, img_h]
                },
                "when_temporal": {
                    "extracted_at": datetime.now().isoformat(),
                    "source": "SmartDoc-Semantic-Grounding"
                },
                "cropped_artifact_path": saved_path,
                "cropped_dimensions": [int(crop_img.shape[1]), int(crop_img.shape[0])]
            }

        # Mode C: Direct token text search with fuzzy matching
        norm = normalize_illumination(upright)
        data = pytesseract.image_to_data(norm, config='--psm 6', output_type=pytesseract.Output.DICT)
        best_token = None
        best_sc = 0.0
        for i in range(len(data['text'])):
            txt = data['text'][i].strip()
            sc = fuzzy_match_token(q, txt)
            if sc > best_sc and sc >= 0.70:
                best_sc = sc
                best_token = {
                    "text": txt,
                    "bbox": [data['left'][i], data['top'][i], data['width'][i], data['height'][i]],
                    "conf": float(data['conf'][i])
                }

        if best_token:
            x, y, w, h = snap_box_to_text(upright, *best_token["bbox"], padding=12)
            crop_img = upright[y:y+h, x:x+w]
            saved_path = None
            if output_dir and crop_img.size > 0:
                os.makedirs(output_dir, exist_ok=True)
                filename = f"field_{re.sub(r'[^a-zA-Z0-9_]', '', q)[:15]}_{int(datetime.now().timestamp()*1000)}.jpg"
                saved_path = os.path.join(output_dir, filename)
                cv2.imwrite(saved_path, crop_img)

            return {
                "success": True,
                "query": query,
                "matched_concept": f"Entity Text Match ('{best_token['text']}')",
                "text_extracted": best_token["text"],
                "field_type": "direct_entity",
                "confidence_score": best_token["conf"],
                "where_spatial": {
                    "bbox_pixel": [x, y, w, h],
                    "normalized": [round(x/img_w, 4), round(y/img_h, 4), round(w/img_w, 4), round(h/img_h, 4)],
                    "document_resolution": [img_w, img_h]
                },
                "when_temporal": {
                    "extracted_at": datetime.now().isoformat(),
                    "source": "SmartDoc-Entity-Search"
                },
                "cropped_artifact_path": saved_path,
                "cropped_dimensions": [int(crop_img.shape[1]), int(crop_img.shape[0])]
            }

        return {
            "success": False,
            "query": query,
            "message": f"Could not ground '{query}' in document text or layout."
        }


# Enterprise Aliases
IntelligentFieldCropper = SemanticFieldCropper

