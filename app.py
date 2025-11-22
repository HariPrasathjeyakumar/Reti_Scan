# app.py — Retina DR Classifier (Royal Blue, Full Diagnostic Suite)
# Requirements: see requirements.txt provided after this file.
import streamlit as st
import numpy as np
from PIL import Image, ImageFilter, ImageDraw, ImageFont
import requests, io, os, time, base64
import onnxruntime as ort
import plotly.graph_objects as go
import streamlit.components.v1 as components
from fpdf import FPDF

# ---------------------------
# Page config + Theme (Royal Blue)
# ---------------------------
st.set_page_config(page_title="Retina DR Classifier — Pro", layout="wide", initial_sidebar_state="expanded")

PRIMARY = "#246AF2"
DARK = "#0D1A2D"
ACCENT = "#4BA3FF"
BG = "#F6F9FC"
CARD_BG = "#FFFFFF"

st.markdown(f"""
<style>
body {{ background: {BG}; }}
header .css-1v3fvcr {{ background: linear-gradient(90deg, {PRIMARY}, {ACCENT}); }}
.section-title {{ font-size:22px; font-weight:700; color:{DARK}; margin-bottom:6px; }}
.kpi {{ background:{CARD_BG}; border-radius:12px; padding:16px; box-shadow: 0 6px 18px rgba(12,20,40,0.06); }}
.small-muted {{ color:#6b7280; font-size:13px; }}
.btn-primary {{ background:{PRIMARY}; color:white; border-radius:8px; padding:8px 14px; }}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Utilities: Download from Google Drive
# ---------------------------
def download_from_drive(file_id, dest):
    if os.path.exists(dest):
        return
    URL = "https://drive.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={'id': file_id}, stream=True)
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            response = session.get(URL, params={'id': file_id, 'confirm': value}, stream=True)
            break
    with open(dest, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)

# ---------------------------
# Load ONNX model (cached)
# ---------------------------
@st.cache_resource(show_spinner=False)
def load_onnx(model_file="ddr_model.onnx"):
    if not os.path.exists(model_file):
        # <--- Insert your ONNX file ID here (already uploaded to Drive)
        FILE_ID = "1lzIEhnZhpRzMhiZqOY-vlwRr8avHstGm"
        download_from_drive(FILE_ID, model_file)
    sess = ort.InferenceSession(model_file, providers=['CPUExecutionProvider'])
    return sess

session = load_onnx()

# input/output names
INPUT_NAME = session.get_inputs()[0].name
OUTPUT_NAME = session.get_outputs()[0].name

# ---------------------------
# Helpers: mapping & preprocessing
# ---------------------------
IMG_SIZE = (300, 300)

def preprocess_pil(img: Image.Image):
    img = img.resize(IMG_SIZE)
    arr = np.array(img).astype(np.float32)
    # EfficientNet preprocessing: scale to [-1,1]
    arr = (arr / 127.5) - 1.0
    arr = np.expand_dims(arr, axis=0).astype(np.float32)
    return arr

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum(axis=-1, keepdims=True)

def map_stage_from_pred(pred_class):
    # model classes 0..4 -> clinical mapping:
    if pred_class == 0:
        return "No DR"
    elif pred_class == 1:
        return "Mild DR"
    elif pred_class == 2:
        return "Moderate DR"
    else:
        return "Severe DR"

# Colors for stages
STAGE_META = {
    "No DR":     {"color": "#2ecc71", "priority": 0},
    "Mild DR":   {"color": "#f1c40f", "priority": 1},
    "Moderate DR":{"color": "#e67e22", "priority": 2},
    "Severe DR": {"color": "#e74c3c", "priority": 3},
}

# ---------------------------
# Occlusion sensitivity (Grad-CAM alternative)
# ---------------------------
@st.cache_data(show_spinner=False)
def occlusion_heatmap(pil_img, baseline=0.0, patch_size=40, stride=20, downsample=1):
    """
    occlusion sensitivity map: slide a gray patch and record drop in predicted prob.
    Returns heatmap (H x W) resized to original image size.
    """
    orig = pil_img.convert("RGB")
    w, h = orig.size
    # work on reduced resolution to speed up
    work_w = (w // downsample)
    work_h = (h // downsample)
    img_small = orig.resize((work_w, work_h))
    base_arr = preprocess_pil(img_small)
    preds_base = session.run(None, {INPUT_NAME: base_arr})[0][0]
    proba_base = softmax(preds_base)
    top_idx = int(np.argmax(proba_base))
    top_base = proba_base[top_idx]

    # heatmap grid
    heatmap = np.zeros((work_h, work_w), dtype=np.float32)
    counts = np.zeros_like(heatmap)

    patch_w = max(1, patch_size // downsample)
    patch_h = max(1, patch_size // downsample)
    step_x = max(1, stride // downsample)
    step_y = max(1, stride // downsample)

    arr_full = np.array(img_small).astype(np.float32)
    for y in range(0, work_h, step_y):
        for x in range(0, work_w, step_x):
            img_copy = arr_full.copy()
            x1 = x
            y1 = y
            x2 = min(work_w, x1 + patch_w)
            y2 = min(work_h, y1 + patch_h)
            # apply gray patch (mean color)
            img_copy[y1:y2, x1:x2, :] = baseline * 255.0
            im_patch = Image.fromarray(img_copy.astype(np.uint8))
            arr_patch = preprocess_pil(im_patch)
            preds_patch = session.run(None, {INPUT_NAME: arr_patch})[0][0]
            proba_patch = softmax(preds_patch)[top_idx]
            # record drop in probability
            drop = top_base - proba_patch
            # accumulate on heatmap (fill region)
            heatmap[y1:y2, x1:x2] += drop
            counts[y1:y2, x1:x2] += 1

    # normalize by counts
    counts[counts == 0] = 1
    heatmap = heatmap / counts
    # resize heatmap to original image size
    heatmap_img = Image.fromarray(np.uint8(255 * (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)))
    heatmap_img = heatmap_img.resize((w, h)).convert("L")
    return heatmap_img, top_idx, top_base

# ---------------------------
# PDF report creation
# ---------------------------
def create_pdf_report(pil_img, stage, conf, probs, classes_labels, heatmap_img=None, patient_info=None):
    """
    Create a simple PDF report combining the image, classification and heatmap.
    Returns bytes of the PDF.
    """
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.add_page()
    pdf.set_font("Arial", size=14)
    pdf.set_text_color(12, 20, 40)
    pdf.cell(0, 8, "Retina Diabetic Retinopathy Report", ln=True, align="C")
    pdf.ln(6)
    # Patient info
    if patient_info:
        pdf.set_font("Arial", size=11)
        for k, v in patient_info.items():
            pdf.cell(0, 6, f"{k}: {v}", ln=True)
        pdf.ln(4)

    # Add prediction
    pdf.set_font("Arial", size=12, style="B")
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 6, f"Predicted Stage: {stage}", ln=True)
    pdf.set_font("Arial", size=11)
    pdf.cell(0, 6, f"Confidence: {conf*100:.2f}%", ln=True)
    pdf.ln(6)

    # Add image
    # convert pil to temporary file
    img_bytes = io.BytesIO()
    pil_img.save(img_bytes, format="JPEG")
    img_bytes.seek(0)
    tmp_path = "/tmp/report_img.jpg"
    with open(tmp_path, "wb") as f:
        f.write(img_bytes.read())
    pdf.image(tmp_path, x=15, w=90)
    # Add heatmap thumbnail if present
    if heatmap_img is not None:
        hm_bytes = io.BytesIO()
        heatmap_img.convert("RGB").save(hm_bytes, format="JPEG")
        hm_bytes.seek(0)
        tmp_hm = "/tmp/report_heat.jpg"
        with open(tmp_hm, "wb") as fh:
            fh.write(hm_bytes.read())
        pdf.image(tmp_hm, x=110, w=90)

    pdf.ln(8)
    pdf.set_font("Arial", size=11)
    pdf.multi_cell(0, 6, "Class probabilities:")
    for idx, (lab, p) in enumerate(zip(classes_labels, probs)):
        pdf.cell(0, 6, f"  {lab}: {p*100:.2f}%", ln=True)

    out = pdf.output(dest='S').encode('latin-1')
    return out

# ---------------------------
# Sidebar — patient info and history
# ---------------------------
st.sidebar.title("Patient")
name = st.sidebar.text_input("Patient name")
age = st.sidebar.number_input("Age", min_value=0, max_value=120, value=50)
notes = st.sidebar.text_area("Notes (optional)", height=80)

st.sidebar.markdown("---")
if 'history' not in st.session_state:
    st.session_state.history = []

if st.sidebar.button("Clear History"):
    st.session_state.history = []

# ---------------------------
# Main layout
# ---------------------------
left_col, right_col = st.columns([2.2, 1])

with left_col:
    st.markdown("<div class='section-title'>Upload & Analyze Retina Image</div>", unsafe_allow_html=True)
    uploaded = st.file_uploader("Choose retinal fundus image (jpg/png)", type=["jpg","jpeg","png"])
    if uploaded:
        pil = Image.open(uploaded).convert("RGB")
        st.image(pil, width=540, caption="Uploaded image")
        # analyze button
        if st.button("Run Analysis", key="an1"):
            start = time.time()
            with st.spinner("Running model inference..."):
                arr = preprocess_pil(pil)
                logits = session.run(None, {INPUT_NAME: arr})[0][0]
                probs = softmax(logits)
                top_idx = int(np.argmax(probs))
                top_prob = float(probs[top_idx])
                stage = map_stage_from_pred(top_idx)

            # KPIs
            st.markdown("---")
            k1, k2, k3 = st.columns(3)
            k1.markdown(f"<div class='kpi'><div style='color:{PRIMARY};font-weight:700'>Prediction</div><div style='font-size:20px;margin-top:6px'>{stage}</div></div>", unsafe_allow_html=True)
            k2.markdown(f"<div class='kpi'><div style='color:{PRIMARY};font-weight:700'>Confidence</div><div style='font-size:20px;margin-top:6px'>{top_prob*100:.2f}%</div></div>", unsafe_allow_html=True)
            # show top class
            classes = ["Grade 0","Grade 1","Grade 2","Grade 3","Grade 4"]
            k3.markdown(f"<div class='kpi'><div style='color:{PRIMARY};font-weight:700'>Top Class</div><div style='font-size:20px;margin-top:6px'>{classes[top_idx]}</div></div>", unsafe_allow_html=True)

            # store history
            entry = {
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "name": name or "Unknown",
                "age": age,
                "stage": stage,
                "confidence": top_prob
            }
            st.session_state.history.insert(0, entry)

            # Probability bar chart (plotly)
            fig = go.Figure([go.Bar(x=classes, y=(probs*100).tolist(), marker_color=[PRIMARY]*len(classes))])
            fig.update_layout(title_text="Class probabilities (%)", yaxis=dict(range=[0,100]), template="simple_white", height=320)
            st.plotly_chart(fig, use_container_width=True)

            # Animated confidence meter
            prog = st.progress(0)
            for i in range(0, int(top_prob*100)+1):
                prog.progress(i)
                time.sleep(0.005)
            prog.empty()

            # Occlusion heatmap (may take a few seconds)
            with st.spinner("Computing occlusion sensitivity map (this may take ~10–40s)..."):
                hm, top_idx2, base_prob = occlusion_heatmap(pil, baseline=0.5, patch_size=64, stride=32, downsample=2)
                # overlay heatmap on image
                overlay = pil.copy().convert("RGBA")
                hm_color = hm.convert("L").resize(pil.size)
                # create color map red
                cmap = np.array([ [255,0,0,int(v)] for v in np.array(hm_color).flatten() ])  # not used directly
                # create red heat mask
                red_mask = Image.new("RGBA", pil.size, (255,0,0,0))
                red_pixels = hm_color.point(lambda p: int(p*0.6))  # alpha
                red_mask.putalpha(red_pixels)
                overlay = Image.alpha_composite(overlay, red_mask)
                # show side-by-side
                c1, c2 = st.columns([1,1])
                with c1:
                    st.image(pil, caption="Original", use_column_width=True)
                with c2:
                    st.image(overlay, caption="Occlusion heatmap (red areas reduce top-prob)", use_column_width=True)

            # Create PDF report and provide download button
            pdf_bytes = create_pdf_report(pil, stage, top_prob, probs, classes, heatmap_img=hm, patient_info={"Name": name or "Unknown", "Age": age, "Notes": notes})
            b64 = base64.b64encode(pdf_bytes).decode()
            href = f'<a href="data:application/octet-stream;base64,{b64}" download="retina_report_{int(time.time())}.pdf">📥 Download PDF report</a>'
            st.markdown(href, unsafe_allow_html=True)

            st.success(f"Analysis complete — predicted stage: {stage} ({top_prob*100:.2f}%) — time: {time.time()-start:.1f}s")

with right_col:
    st.markdown("<div class='section-title'>Recent Analyses</div>", unsafe_allow_html=True)
    if st.session_state.history:
        for h in st.session_state.history[:8]:
            st.markdown(f"**{h['time']}** — {h['name']} (age {h['age']})  —  **{h['stage']}**  — {h['confidence']*100:.1f}%")
    else:
        st.info("No analyses yet. Upload an image and click Run Analysis.")

    st.markdown("---")
    st.markdown("<div class='section-title'>About</div>", unsafe_allow_html=True)
    st.markdown("""
    **Retina DR Classifier — Pro**  
    Royal Blue theme • ONNX runtime • Occlusion sensitivity heatmap  
    Use this tool for screening support only — not a definitive diagnosis.
    """)
