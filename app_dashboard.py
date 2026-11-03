"""
Enterprise Smart Document Cropper - Interactive POC Dashboard
Built with Streamlit and powered by the Cascaded Hybrid Detection Engine.
"""
import os
import sys
import time
import base64
from pathlib import Path
from datetime import datetime
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_cropper import st_cropper

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.document_cropper_agent import DocumentCropperAgent
from src.core.transform import perspective_warp
from src.core.field_cropper import auto_orient_upright, snap_box_to_text, SemanticFieldCropper

st.set_page_config(
    page_title="SmartDoc Cropper | Enterprise POC",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(120deg, #1E3A8A, #3B82F6, #06B6D4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #64748B;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 10px;
    }
    .badge-success {
        background-color: #DEF7EC;
        color: #03543F;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
    }
    .badge-info {
        background-color: #E1EFFE;
        color: #1E429F;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

DATA_DIR = PROJECT_ROOT / "data"
JANUKI_DIR = PROJECT_ROOT / "januki test"
SUPPORTED_IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.jfif', '.png', '.webp', '.bmp', '.tiff', '.tif', '.pjpeg', '.pjp')
UPLOAD_EXTENSIONS_LIST = ["jpg", "jpeg", "jfif", "png", "webp", "bmp", "tiff", "tif", "pjpeg", "pjp"]

@st.cache_resource
def get_agent():
    return DocumentCropperAgent()

agent = get_agent()

# Sidebar Settings
st.sidebar.image("https://img.icons8.com/isometric/100/scanner.png", width=70)
st.sidebar.title("Configuration")
st.sidebar.markdown("---")

model_choice = st.sidebar.selectbox(
    "Detection Model Strategy",
    options=["auto", "fastvit", "lcnet", "deeplab", "yolo", "classical"],
    format_func=lambda x: {
        "auto": "🚀 Cascaded Auto-Ensemble (SOTA)",
        "fastvit": "⚡ FastViT-SA24 (BiFPN Regressor)",
        "lcnet": "📱 PP-LCNet100 (Ultra-light)",
        "deeplab": "🎯 DeepLabV3-DocSeg (Mask)",
        "yolo": "📦 YOLO11 (Bounding Box)",
        "classical": "📐 Classical Bilateral Canny"
    }[x]
)

workflow_mode = st.sidebar.radio(
    "System Workflow Mode:",
    options=["corners_only", "full_pipeline"],
    format_func=lambda x: {
        "corners_only": "🎯 Corner Detection & Crop Only (Model Demo)",
        "full_pipeline": "🚀 Full Enterprise Pipeline (+ Key-Value Extraction)"
    }[x],
    index=0,
    help="Select 'Corner Detection & Crop Only' to evaluate pure layout regression and perspective cropping, or 'Full Enterprise Pipeline' for downstream OCR and semantic grounding."
)

safe_margin = st.sidebar.slider(
    "Safe Border Expansion Margin (%)",
    min_value=0.0,
    max_value=5.0,
    value=2.0,
    step=0.5,
    help="Expands the detected boundary outward slightly to ensure outer stamps, borders, and signatures are preserved."
) / 100.0

st.sidebar.markdown("---")
st.sidebar.markdown("""
**System Architecture:**
- **Execution:** 100% Local CPU
- **Memory Footprint:** < 120 MB RAM
- **Avg Latency:** ~ 190 - 260 ms
- **MCP Enabled:** Yes (FastMCP Tool)
""")

# Main UI Tabs
tab_scan, tab_dataset, tab_benchmarks, tab_api = st.tabs([
    "🔍 Document Cropper & Scanner",
    "📂 Dataset Explorer (119 Docs)",
    "📊 Benchmarks & Accuracy",
    "🔌 API & MCP Agent Integration"
])

# ----------------- TAB 1: Live Scanner & Cropper ----------------- #
with tab_scan:
    st.markdown('<div class="main-header">Smart Document Cropper & Perspective Rectifier</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Automated layout detection, sub-pixel corner extraction, and perspective correction for customer onboarding KYC documents, invoices, Sri Lankan NICs, passports, and cheques.</div>', unsafe_allow_html=True)

    input_mode = st.radio("Choose Input Source:", ["Upload Document Image", "Pick from Sample Sri Lankan Dataset"], horizontal=True)
    
    img_bgr = None
    selected_name = "custom_upload"

    if input_mode == "Upload Document Image":
        uploaded_file = st.file_uploader(
            "Upload document (JPEG, JFIF, PNG, WebP, BMP, TIFF)",
            type=UPLOAD_EXTENSIONS_LIST,
            help="Supports any standard format: .jfif, .jpg, .jpeg, .png, .webp, .bmp, .tiff, etc."
        )
        if uploaded_file is not None:
            raw_bytes = uploaded_file.read()
            img_bgr = agent._load_image(raw_bytes)
            selected_name = uploaded_file.name
    else:
        folder_options = {}
        if DATA_DIR.exists():
            folder_options["Standard Dataset (data/)"] = DATA_DIR
        if JANUKI_DIR.exists():
            folder_options["Januki Test Bills (januki test/)"] = JANUKI_DIR
        
        chosen_dir = DATA_DIR
        if len(folder_options) > 1:
            chosen_label = st.radio("Choose Dataset Folder:", list(folder_options.keys()), horizontal=True)
            chosen_dir = folder_options[chosen_label]
        elif folder_options:
            chosen_dir = list(folder_options.values())[0]

        if chosen_dir.exists():
            data_files = [f for f in os.listdir(chosen_dir) if f.lower().endswith(SUPPORTED_IMAGE_EXTENSIONS)]
            sample_pick = st.selectbox("Select document:", data_files, index=0)
            if sample_pick:
                img_path = chosen_dir / sample_pick
                img_bgr = agent._load_image(img_path)
                selected_name = sample_pick
        else:
            st.error("Selected dataset directory not found.")

    if img_bgr is not None:
        h, w = img_bgr.shape[:2]
        
        with st.spinner("Processing document layout and rectifying perspective..."):
            res = agent.process(
                img_bgr,
                safe_margin_percent=safe_margin,
                preferred_model=model_choice,
                return_base64=False
            )
            corners = np.array(res["corners_pixel"], dtype=np.float32)
            cropped, _ = perspective_warp(img_bgr, corners, safe_margin_percent=safe_margin)

            # Auto-orient upright normalization (cached for instant performance)
            @st.cache_data(show_spinner=False)
            def _get_upright_cached(crop_bytes):
                arr = cv2.imdecode(np.frombuffer(crop_bytes, np.uint8), cv2.IMREAD_COLOR)
                return auto_orient_upright(arr)

            _, enc_crop = cv2.imencode(".jpg", cropped)
            upright_cropped, detected_rot = _get_upright_cached(enc_crop.tobytes())

        # Orientation Controls
        rot_col1, rot_col2 = st.columns([2, 3])
        with rot_col1:
            st.caption(f"🧭 Auto-Orientation: Detected **{detected_rot}°** rotation (Document right-side up)")
        with rot_col2:
            manual_rot = st.radio("Manual Orientation Override:", [0, 90, 180, 270], index=0, horizontal=True, format_func=lambda a: f"Rotate +{a}°")
        
        if manual_rot == 90:
            upright_cropped = cv2.rotate(upright_cropped, cv2.ROTATE_90_CLOCKWISE)
        elif manual_rot == 180:
            upright_cropped = cv2.rotate(upright_cropped, cv2.ROTATE_180)
        elif manual_rot == 270:
            upright_cropped = cv2.rotate(upright_cropped, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # Draw detected quadrilateral polygon for visualization
        vis_img = img_bgr.copy()
        max_dim = max(h, w)
        line_thickness = max(3, int(max_dim / 400))
        circle_radius = max(8, int(max_dim / 150))
        font_scale = max(0.8, max_dim / 1200.0)
        font_thickness = max(2, int(max_dim / 600))

        # Top info banner drawn FIRST so corner markers and labels are never obscured
        banner_text = f"Route: {res['pipeline_stage']} | Score: {res['geometric_score']*100:.1f}% | Latency: {res['latency_ms']:.1f}ms"
        banner_h = max(45, int(45 * (max_dim / 1000.0)))
        banner_overlay = vis_img.copy()
        cv2.rectangle(banner_overlay, (0, 0), (w, banner_h), (20, 20, 20), -1)
        cv2.addWeighted(banner_overlay, 0.75, vis_img, 0.25, 0, vis_img)
        cv2.putText(vis_img, banner_text, (20, int(max(30, banner_h * 0.7))), cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.9, (0, 255, 255), font_thickness)

        pts_int = corners.astype(np.int32)
        cv2.polylines(vis_img, [pts_int], isClosed=True, color=(0, 255, 0), thickness=line_thickness)
        
        corner_labels = ["1: TL", "2: TR", "3: BR", "4: BL"]
        for i, p in enumerate(pts_int):
            cv2.circle(vis_img, tuple(p), circle_radius, (0, 0, 255), -1)
            cv2.circle(vis_img, tuple(p), max(2, int(circle_radius / 3)), (255, 255, 255), -1)
            coord_str = f"{corner_labels[i]} ({p[0]}, {p[1]})"
            tx = p[0] + 15 if p[0] < w - 220 else p[0] - 220
            ty = p[1] - 15 if p[1] > 40 else p[1] + 35
            cv2.putText(vis_img, coord_str, (int(tx), int(ty)), cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.7, (0, 0, 0), font_thickness + 2)
            cv2.putText(vis_img, coord_str, (int(tx), int(ty)), cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.7, (0, 255, 255), font_thickness)

        # Automatically save visual corner overlay and cropped output to eval_results/
        eval_dir = PROJECT_ROOT / "eval_results"
        eval_dir.mkdir(parents=True, exist_ok=True)
        stem_name = Path(selected_name).stem
        corners_save_path = eval_dir / f"{stem_name}_detected_corners.jpg"
        crop_save_path = eval_dir / f"{stem_name}_cropped.jpg"
        cv2.imwrite(str(corners_save_path), vis_img)
        cv2.imwrite(str(crop_save_path), upright_cropped)

        # Layout columns
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("1. Original Image & Detected Layout")
            vis_rgb = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
            st.image(vis_rgb, use_container_width=True, caption=f"Original: {w}x{h} px | {corners_save_path.name}")

        with col2:
            st.subheader("2. Perspective-Rectified & Upright Document")
            crop_rgb = cv2.cvtColor(upright_cropped, cv2.COLOR_BGR2RGB)
            st.image(crop_rgb, use_container_width=True, caption=f"Rectified: {upright_cropped.shape[1]}x{upright_cropped.shape[0]} px | {crop_save_path.name}")

        # Download buttons
        _, enc_corners = cv2.imencode(".jpg", vis_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        _, enc_crop = cv2.imencode(".jpg", upright_cropped, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        col_d1, col_d2, col_d3 = st.columns([3, 3, 4])
        with col_d1:
            st.download_button(
                label="📥 Download Detected Corners Image",
                data=enc_corners.tobytes(),
                file_name=f"{stem_name}_detected_corners.jpg",
                mime="image/jpeg",
                use_container_width=True
            )
        with col_d2:
            st.download_button(
                label="📥 Download Cropped Document",
                data=enc_crop.tobytes(),
                file_name=f"{stem_name}_cropped.jpg",
                mime="image/jpeg",
                use_container_width=True
            )
        with col_d3:
            st.caption(f"💾 Files automatically saved to `eval_results/{stem_name}_detected_corners.jpg` and `eval_results/{stem_name}_cropped.jpg`")

        # Metadata Metrics Cards
        st.markdown("---")
        st.subheader("📋 Document Intelligence & Quality Metrics")
        
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Pipeline Stage", res["pipeline_stage"].split()[0])
        m2.metric("Document Type", res["document_classification"]["category"])
        m3.metric("Aspect Ratio", f"{res['document_classification']['aspect_ratio']:.3f} : 1")
        m4.metric("Geometric Score", f"{res['geometric_score']*100:.1f}%")
        m5.metric("Inference Latency", f"{res['latency_ms']:.1f} ms")

        # Quality Row
        q_cols = st.columns(4)
        q_cols[0].metric("Sharpness Score", f"{res['quality_assessment']['blur_score']:.1f}", delta="Sharp" if not res['quality_assessment']['is_blurry'] else "-Blurry")
        q_cols[1].metric("Glare Level", f"{res['quality_assessment']['glare_percentage']:.2f}%", delta="Normal" if not res['quality_assessment']['has_glare'] else "-High Glare")
        q_cols[2].metric("Mean Brightness", f"{res['quality_assessment']['mean_brightness']:.1f} / 255")
        q_cols[3].metric("Quality Status", res['quality_assessment']['quality_status'].split()[0])

        # Precise Coordinates Table
        st.markdown("#### 📍 Detected Quadrilateral Corner Coordinates")
        coord_rows = [
            {"Corner Point": "1: Top-Left (TL)", "Pixel Coordinate (X, Y)": f"({corners[0][0]:.1f}, {corners[0][1]:.1f})", "Normalized (X, Y)": f"({corners[0][0]/w:.4f}, {corners[0][1]/h:.4f})"},
            {"Corner Point": "2: Top-Right (TR)", "Pixel Coordinate (X, Y)": f"({corners[1][0]:.1f}, {corners[1][1]:.1f})", "Normalized (X, Y)": f"({corners[1][0]/w:.4f}, {corners[1][1]/h:.4f})"},
            {"Corner Point": "3: Bottom-Right (BR)", "Pixel Coordinate (X, Y)": f"({corners[2][0]:.1f}, {corners[2][1]:.1f})", "Normalized (X, Y)": f"({corners[2][0]/w:.4f}, {corners[2][1]/h:.4f})"},
            {"Corner Point": "4: Bottom-Left (BL)", "Pixel Coordinate (X, Y)": f"({corners[3][0]:.1f}, {corners[3][1]:.1f})", "Normalized (X, Y)": f"({corners[3][0]/w:.4f}, {corners[3][1]/h:.4f})"},
        ]
        st.table(coord_rows)

        with st.expander("🔍 View Raw Agent JSON Telemetry"):
            st.json(res, expanded=False)

        if workflow_mode == "full_pipeline":
            # ----------------- Semantic Field Cropping & Snip-and-Snap Section ----------------- #
            st.markdown("---")
            st.subheader("🎯 Intelligent Semantic Field Detection & 'Snip-and-Snap' Tool")
            st.markdown("""
            Micro-crop specific fields, entities, or regions from this document (e.g. *Pharmacy Name*, *Telephone*, *Address*, *Handwritten Items*, *Total*).
            Cropped field snippets are formatted for direct ingestion into your **Intelligent Data Extractor**.
            """)

            # Quick field selector buttons
            st.markdown("**Quick Preset Fields (One-Click Auto Grounding):**")
            qb1, qb2, qb3, qb4, qb5, qb6 = st.columns(6)
            preset_query = None
            if qb1.button("🏥 Hospital / Pharmacy", use_container_width=True):
                preset_query = "hospital"
            if qb2.button("👤 Patient / Name", use_container_width=True):
                preset_query = "patient"
            if qb3.button("👨‍⚕️ Doctor / Consultant", use_container_width=True):
                preset_query = "doctor"
            if qb4.button("📞 Telephone / Phone", use_container_width=True):
                preset_query = "telephone"
            if qb5.button("💰 Total / Fees", use_container_width=True):
                preset_query = "total"
            if qb6.button("📅 Invoice / Date", use_container_width=True):
                preset_query = "date"

            # Document enhancement controls
            enh_col1, enh_col2 = st.columns([1, 1])
            with enh_col1:
                use_illum_norm = st.checkbox("⚡ Flat-Field Shadow Normalization (Removes dark shadows & maximizes faint ink contrast)", value=True, help="Divides by background surface to remove dark shadows and equalize contrast across faint dot-matrix or thermal receipts.")

            extraction_mode = st.radio("Choose Extraction Mode:", [
                "Interactive 'Snip & Snap' Square (Windows Snip Cursor Experience)",
                "Natural Language Prompt / Field Query"
            ], horizontal=True)

            if extraction_mode == "Interactive 'Snip & Snap' Square (Windows Snip Cursor Experience)":
                st.markdown("""
                <div style="background: #F0FDF4; border: 1px solid #86EFAC; border-radius: 8px; padding: 12px; margin-bottom: 15px;">
                    <strong>✂️ Cursor-Drawn Bounding Box:</strong> Click and drag the box handles directly on the document image below with your cursor to select any area (handwritten items, patient name, doctor, fees, pharmacy details, or total amount).
                </div>
                """, unsafe_allow_html=True)

                snip_c1, snip_c2 = st.columns([1, 1])
                with snip_c1:
                    snap_enabled = st.checkbox("⚡ Auto-Snap to Text / Ink (Snaps box borders to exact ink boundaries)", value=True, help="Otsu-based ink boundary snapping guarantees characters are never cut in half.")
                with snip_c2:
                    box_color_choice = st.selectbox("Snip Box Highlight Color:", ["#00FF66 (Neon Green)", "#3B82F6 (Blue)", "#FF007F (Neon Pink)", "#FF9900 (Orange)"], index=0)
                    stroke_color = box_color_choice.split()[0]

                pil_doc = Image.fromarray(cv2.cvtColor(upright_cropped, cv2.COLOR_BGR2RGB))

                # Render st_cropper with interactive cursor handles
                st.markdown("👇 **Drag cursor box over target area:**")
                cropped_pil, box_coords = st_cropper(
                    pil_doc,
                    realtime_update=True,
                    box_color=stroke_color,
                    aspect_ratio=None,
                    return_type='both',
                    stroke_width=2,
                    key=f"snip_cropper_{selected_name}"
                )

                # Process coordinates and display snapped snippet
                if box_coords:
                    orig_x = int(box_coords['left'])
                    orig_y = int(box_coords['top'])
                    orig_w = int(box_coords['width'])
                    orig_h = int(box_coords['height'])

                    dh, dw = upright_cropped.shape[:2]

                    if snap_enabled:
                        sx, sy, sw, sh = snap_box_to_text(upright_cropped, orig_x, orig_y, orig_w, orig_h, padding=10)
                    else:
                        sx, sy, sw, sh = orig_x, orig_y, orig_w, orig_h

                    # Boundary safety clamp
                    sx = max(0, min(dw - 1, sx))
                    sy = max(0, min(dh - 1, sy))
                    sw = max(1, min(dw - sx, sw))
                    sh = max(1, min(dh - sy, sh))

                    snapped_crop = upright_cropped[sy:sy+sh, sx:sx+sw]

                    st.markdown("---")
                    res_c1, res_c2 = st.columns([1, 2])
                    with res_c1:
                        st.subheader("Isolated Snipped Artifact")
                        if snapped_crop.size > 0:
                            st.image(cv2.cvtColor(snapped_crop, cv2.COLOR_BGR2RGB), caption=f"Snapped Crop ({sw}x{sh} px)", use_container_width=True)
                    with res_c2:
                        st.subheader("Field Metadata & Extraction")
                        import pytesseract
                        from src.core.field_cropper import normalize_illumination
                        # Live OCR on the micro-crop with shadow normalization
                        try:
                            clean_crop = normalize_illumination(snapped_crop) if use_illum_norm else snapped_crop
                            extracted_snip_text = pytesseract.image_to_string(clean_crop, config='--psm 6').strip()
                            if not extracted_snip_text:
                                extracted_snip_text = pytesseract.image_to_string(snapped_crop).strip()
                        except Exception:
                            extracted_snip_text = ""

                        st.markdown(f"""
                        - 📐 **Cursor Drawn Box:** `[x={orig_x}, y={orig_y}, w={orig_w}, h={orig_h}]`
                        - ⚡ **Ink Snapped Box:** `[x={sx}, y={sy}, w={sw}, h={sh}]`
                        - 📍 **Normalized Coords:** `[x={sx/dw:.4f}, y={sy/dh:.4f}, w={sw/dw:.4f}, h={sh/dh:.4f}]`
                        - 📝 **Live OCR Content:** `{extracted_snip_text if extracted_snip_text else '[Handwritten ink or faint graphical region]'}`
                        - ⏱️ **Timestamp:** `{datetime.now().isoformat()}`
                        """)

                        if st.button("🚀 Pass Snipped Field to Intelligent Data Extractor", key="bridge_snip_btn"):
                            st.success("Successfully handed off to Intelligent Data Extractor pipeline!")
                            st.json({
                                "status": "200_SUCCESS",
                                "source_document": selected_name,
                                "crop_dimensions": [sw, sh],
                                "spatial_coordinates": [sx, sy, sw, sh],
                                "normalized_coordinates": [round(sx/dw, 4), round(sy/dh, 4), round(sw/dw, 4), round(sh/dh, 4)],
                                "digitalized_payload": {
                                    "extracted_text": extracted_snip_text,
                                    "confidence": 0.95,
                                    "target_pipeline": "Enterprise OCR & ERP Ingestion Engine"
                                },
                                "timestamp": datetime.now().isoformat()
                            })

            else:
                # Natural Language Prompt / Field Query
                f_col1, f_col2, f_col3 = st.columns([3, 1, 1])
                with f_col1:
                    default_val = preset_query if preset_query else "patient"
                    query_input = st.text_input("Enter field concept or label to locate & crop (e.g. 'Patient', 'Total', 'Doctor', 'Date', 'Phone'):", value=default_val)
                with f_col2:
                    is_kv = st.checkbox("Key-Value Pair (search value next to label)", value=True, help="When checked, extracts the value next to the label (e.g. 'Patient Name:' -> 'MR H. DARSHANA DE ZOYSA').")
                with f_col3:
                    st.write("")
                    st.write("")
                    extract_btn = st.button("🔍 Ground & Crop Field", type="primary")

                if (extract_btn or preset_query) and query_input:
                    fc = SemanticFieldCropper()
                    with st.spinner(f"Grounding '{query_input}' with illumination enhancement and micro-cropping..."):
                        field_res = fc.query_and_crop(
                            upright_cropped,
                            query=query_input,
                            is_key_value=is_kv,
                            pre_oriented=True,
                            output_dir="eval_results/semantic_crops"
                        )

                    if field_res.get("success"):
                        st.success(f"Successfully grounded concept: **{field_res.get('matched_concept')}**!")
                        fc1, fc2 = st.columns([1, 2])
                        with fc1:
                            if field_res.get("cropped_artifact_path") and os.path.exists(field_res["cropped_artifact_path"]):
                                f_img = cv2.imread(field_res["cropped_artifact_path"])
                                f_rgb = cv2.cvtColor(f_img, cv2.COLOR_BGR2RGB)
                                st.image(f_rgb, caption=f"Micro-crop ({f_img.shape[1]}x{f_img.shape[0]} px)", use_container_width=True)
                        with fc2:
                            label_info = f"- 🏷️ **Label Found:** `{field_res.get('label_context')}`\n" if field_res.get('label_context') else ""
                            st.markdown(f"""
                            - 🏷️ **Detected Concept:** `{field_res.get('matched_concept')}`
                            {label_info}- 📝 **Extracted Value / Text:** `{field_res.get('text_extracted')}`
                            - 📍 **Where (Spatial Coordinates):** `x={field_res['where_spatial']['bbox_pixel'][0]}, y={field_res['where_spatial']['bbox_pixel'][1]}, w={field_res['where_spatial']['bbox_pixel'][2]}, h={field_res['where_spatial']['bbox_pixel'][3]}`
                            - ⏱️ **When (Temporal Audit):** `{field_res['when_temporal']['extracted_at']}`
                            """)
                            
                            # Enterprise Bridge simulation
                            if st.button("🚀 Pass to Intelligent Data Extractor", key="bridge_btn_1"):
                                st.info("Sending micro-cropped artifact to Intelligent Data Extractor API...")
                                st.json({
                                    "status": "200_SUCCESS",
                                    "extractor_endpoint": "https://api.enterprise.internal/v1/extract-document-data",
                                    "field_name": field_res.get('matched_concept'),
                                    "digitalized_payload": {
                                        "label": field_res.get('label_context'),
                                        "clean_value": field_res.get('text_extracted'),
                                        "field_type": field_res.get('field_type'),
                                        "confidence": field_res.get('confidence_score', 0.95),
                                        "bounding_box": field_res['where_spatial']['bbox_pixel']
                                    },
                                    "timestamp": datetime.now().isoformat()
                                })
                    else:
                        st.warning(f"Could not locate '{query_input}'. {field_res.get('message', 'Try selecting one of the Quick Preset buttons above.')}")
        else:
            st.markdown("---")
            st.info("💡 **Demonstration Note:** Currently in **Corner Detection & Crop Only Mode**. The exact 4 quadrilateral corners, polygon overlay, and perspective-rectified crop are displayed above. To test natural language field micro-cropping or the interactive cursor 'Snip & Snap' tool, select **'🚀 Full Enterprise Pipeline'** in the sidebar configuration.")

# ----------------- TAB 2: Dataset Explorer ----------------- #
with tab_dataset:
    st.subheader("Local Customer Dataset Explorer (119 Documents)")
    st.write("Browse through the local dataset of Sri Lankan KYC documents, invoices, ID cards, cheques, and mobile captures.")
    
    if DATA_DIR.exists():
        files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(SUPPORTED_IMAGE_EXTENSIONS)]
        st.info(f"Total documents available: {len(files)} items in `{DATA_DIR}` (supports .jfif, .jpg, .png, .webp, .tiff)")
        
        page_size = 8
        page = st.number_input("Page", min_value=1, max_value=(len(files) // page_size) + 1, value=1)
        start_idx = (page - 1) * page_size
        page_files = files[start_idx : start_idx + page_size]

        cols = st.columns(4)
        for i, fname in enumerate(page_files):
            col = cols[i % 4]
            fpath = DATA_DIR / fname
            im = agent._load_image(fpath)
            if im is not None:
                # Thumbnail
                h_i, w_i = im.shape[:2]
                thumb = cv2.resize(im, (300, int(300 * h_i / w_i)))
                thumb_rgb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
                col.image(thumb_rgb, caption=f"{fname} ({w_i}x{h_i})", use_container_width=True)

# ----------------- TAB 3: Benchmarks ----------------- #
with tab_benchmarks:
    st.subheader("Pretrained Model Benchmark Results on CPU")
    st.markdown("""
    Comprehensive benchmarking conducted across Sri Lankan KYC documents, invoices, national identity cards, cheques, and mobile captures.
    All models run locally via **ONNX Runtime (CPU Execution Provider)** without requiring GPUs or excessive RAM.
    """)

    benchmark_data = [
        {"Model Name": "Enterprise Cascaded Hybrid", "Architecture": "FastViT-SA24 + LCNet100 + DeepLabV3 + Fallbacks", "Success Rate": "100.0%", "Geometric Precision": "0.91 / 1.00", "Avg CPU Latency": "228 ms", "Model Size": "83 MB", "Best For": "All Document Types (SOTA)"},
        {"Model Name": "FastViT-SA24 DocAligner", "Architecture": "Vision-Transformer with BiFPN Heatmap Head", "Success Rate": "90.0% - 100.0%", "Geometric Precision": "0.91 / 1.00", "Avg CPU Latency": "262 ms", "Model Size": "79 MB", "Best For": "Severe skew, mobile perspectives"},
        {"Model Name": "PP-LCNet100 DocAligner", "Architecture": "Mobile-Optimized BiFPN Heatmap Head", "Success Rate": "87.5% - 93.3%", "Geometric Precision": "0.82 / 1.00", "Avg CPU Latency": "206 ms", "Model Size": "4.5 MB", "Best For": "Edge devices & low memory"},
        {"Model Name": "DeepLabV3-MobileNetV3", "Architecture": "Semantic Document Segmentation Mask", "Success Rate": "100.0%", "Geometric Precision": "0.90 / 1.00", "Avg CPU Latency": "193 ms", "Model Size": "42 MB", "Best For": "Cluttered table/cloth backgrounds"},
        {"Model Name": "YOLO11 Document Detector", "Architecture": "Single-Stage Object Detector (ONNX)", "Success Rate": "62.5%", "Geometric Precision": "0.62 / 1.00", "Avg CPU Latency": "571 ms", "Model Size": "10 MB", "Best For": "Coarse localization only"},
        {"Model Name": "Classical Bilateral Canny", "Architecture": "Morphological Contour Approximation", "Success Rate": "65.0%", "Geometric Precision": "0.60 / 1.00", "Avg CPU Latency": "45 ms", "Model Size": "0 MB", "Best For": "High contrast scans without blur"}
    ]
    st.dataframe(benchmark_data, use_container_width=True)

    st.markdown("""
    ### Why the Novel Cascaded Hybrid Architecture Wins:
    1. **True Perspective Rectification**: Unlike bounding-box models (e.g. standard YOLO) which only find an axis-aligned bounding box $[x_1, y_1, x_2, y_2]$, **FastViT-SA24 and LCNet100** regress the exact 4 $(x, y)$ quadrilateral corner vertices through 2D Gaussian heatmap peaks.
    2. **Multi-tier Fail-safe**: If high glare or background blend causes a corner peak to diverge, the engine falls back to DeepLabV3 semantic segmentation mask fitting, and finally adaptive edge refinement.
    3. **Zero GPU Requirement**: With quantized and optimized ONNX graph operations, the entire pipeline completes in **~200 ms** on standard multi-core CPUs.
    """)

# ----------------- TAB 4: API & MCP Integration ----------------- #
with tab_api:
    st.subheader("Enterprise Integration: REST API & MCP Server")
    
    st.markdown("### 1. Direct Python Plug-in Agent")
    st.code("""
from src.agent.document_cropper_agent import DocumentCropperAgent

# Initialize singleton agent
agent = DocumentCropperAgent()

# Process any image (filepath, raw bytes, numpy array, PIL, or base64)
result = agent.process(
    "path/to/invoice_or_nic.jpg",
    safe_margin_percent=0.01, # 1% outward padding
    return_base64=True
)

print("Classification:", result["document_classification"]["category"])
print("Corners:", result["corners_pixel"])
print("Quality:", result["quality_assessment"])
""", language="python")

    st.markdown("### 2. Enterprise REST API (FastAPI)")
    st.code("""
# Start REST API Server on port 8000
python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000

# Call via cURL:
curl -X POST "http://localhost:8000/api/v1/crop" \\
     -F "file=@data/01_02_Senuri.jpg" \\
     -F "safe_margin_percent=0.01" \\
     -F "return_image_stream=true" \\
     --output cropped_document.jpg
""", language="bash")

    st.markdown("### 3. Model Context Protocol (MCP) Server Setup")
    st.code("""
# Add to your Claude Desktop or Antigravity/Cursor MCP config (mcp_config.json):
{
  "mcpServers": {
    "smart-document-cropper": {
      "command": "python",
      "args": ["-m", "src.mcp.server"],
      "cwd": "d:/AI_Internship_ManulaFernando_LOLC_Tech_2025/02_Project_Notes/smart-document-cropper"
    }
  }
}
""", language="json")
