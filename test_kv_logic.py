import cv2, pytesseract, re
import numpy as np

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def extract_key_value_pairs(img, target_label):
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    
    # Form line groups
    lines = {}
    for i in range(len(data['text'])):
        txt = data['text'][i].strip()
        if not txt or data['conf'][i] < 20: continue
        line_key = (data['block_num'][i], data['par_num'][i], data['line_num'][i])
        if line_key not in lines:
            lines[line_key] = []
        lines[line_key].append({
            'text': txt,
            'bbox': (data['left'][i], data['top'][i], data['width'][i], data['height'][i]),
            'conf': data['conf'][i]
        })
        
    target_clean = re.sub(r'[^a-zA-Z0-9]', '', target_label.lower())
    
    results = []
    # Sort lines by top coordinate
    sorted_lines = sorted(lines.items(), key=lambda item: min(t['bbox'][1] for t in item[1]))

    for idx_line, (line_key, tokens) in enumerate(sorted_lines):
        tokens.sort(key=lambda t: t['bbox'][0])
        line_text = ' '.join(t['text'] for t in tokens)
        
        # Check tokens in this line
        for idx_tok, tok in enumerate(tokens):
            clean_tok = re.sub(r'[^a-zA-Z0-9]', '', tok['text'].lower())
            if target_clean in clean_tok or clean_tok in target_clean:
                # 1. Look for value to the right on the same line
                value_tokens = tokens[idx_tok+1:]
                if value_tokens:
                    val_text = ' '.join(t['text'] for t in value_tokens).lstrip(': -.')
                    min_x = min(t['bbox'][0] for t in value_tokens)
                    min_y = min(t['bbox'][1] for t in value_tokens)
                    max_x = max(t['bbox'][0] + t['bbox'][2] for t in value_tokens)
                    max_y = max(t['bbox'][1] + t['bbox'][3] for t in value_tokens)
                    results.append({
                        'label_found': tok['text'],
                        'value_text': val_text,
                        'value_bbox': [min_x, min_y, max_x - min_x, max_y - min_y],
                        'relation': 'right_of_label'
                    })
                else:
                    # 2. If no tokens to the right, check line immediately below (stacked key-value)
                    if idx_line + 1 < len(sorted_lines):
                        next_line_tokens = sorted_lines[idx_line + 1][1]
                        val_text = ' '.join(t['text'] for t in next_line_tokens).lstrip(': -.')
                        min_x = min(t['bbox'][0] for t in next_line_tokens)
                        min_y = min(t['bbox'][1] for t in next_line_tokens)
                        max_x = max(t['bbox'][0] + t['bbox'][2] for t in next_line_tokens)
                        max_y = max(t['bbox'][1] + t['bbox'][3] for t in next_line_tokens)
                        results.append({
                            'label_found': tok['text'],
                            'value_text': val_text,
                            'value_bbox': [min_x, min_y, max_x - min_x, max_y - min_y],
                            'relation': 'below_label'
                        })
    return results

# Test 1
img1 = cv2.imread('eval_results/01_02_Senuri_cropped.jpg')
res1 = extract_key_value_pairs(img1, 'Reg')
print('Results for target "Reg":', res1)

res2 = extract_key_value_pairs(img1, 'No')
print('Results for target "No":', res2)

# Test 2 on 01_05_Senuri
img2 = cv2.imread('eval_results/01_05_Senuri_upright.jpg')
res3 = extract_key_value_pairs(img2, 'Tel')
print('Results for target "Tel":', res3)
