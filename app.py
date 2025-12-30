import streamlit as st
import subprocess
from PIL import Image
import numpy as np
import io, os, time, json
from collections import Counter

# ---------------------------------------------------
# Page Setup
# ---------------------------------------------------
st.set_page_config(
    page_title="PPE Compliance Intelligence",
    page_icon="🦺",
    layout="wide"
)

# ---------------------------------------------------
# Custom Styling
# ---------------------------------------------------
st.markdown("""
<style>
html, body, [class*="css"] { font-family: "Segoe UI", sans-serif; }
.stApp {
    background-image:
        radial-gradient(circle at top, rgba(37,99,235,0.35), rgba(15,23,42,0.98) 55%),
        url("https://images.pexels.com/photos/546819/pexels-photo-546819.jpeg?auto=compress&cs=tinysrgb&w=1600");
    background-size: cover;
    background-attachment: fixed;
}
.card { background: rgba(15,23,42,0.92); padding:18px; border-radius:16px; margin-bottom:18px; }
.card-soft { background: rgba(15,23,42,0.80); padding:16px; border-radius:14px; margin-bottom:12px; }
.section-title { font-size:20px; color:#93c5fd; font-weight:600; }
.subtext { font-size:13px; color:#cbd5f5; }
.kpi-label { font-size:13px; color:#f1f5f9; }
.kpi-value { font-size:24px; font-weight:600; }
.badge { padding:6px 10px; border-radius:999px; background:#0f172a; border:1px solid #1d4ed8; margin:3px; display:inline-block; }
.chip-ok { background:rgba(22,163,74,0.18); border:1px solid #22c55e; padding:5px 10px; border-radius:999px; }
.chip-risk { background:rgba(220,38,38,0.18); border:1px solid #f87171; padding:5px 10px; border-radius:999px; }
.footer-text { text-align:center; font-size:13px; margin-top:24px; color:#e5e7eb; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# Hero
# ---------------------------------------------------
st.markdown("""
<div style="text-align:center">
<h1 style="color:#e5e7eb;">PPE Compliance Intelligence Dashboard</h1>
<p style="color:#f1f5f9;">
AI that scans construction sites for helmets, vests, gloves & safety shoes
</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# Paths & constants
# ---------------------------------------------------
MODEL_FILENAME = "PPE.pt"
RUNS_DIR = "runs/detect"

CLASS_NAMES = {
    0:'gloves',1:'hardhat',2:'no-gloves',3:'no-hardhat',
    4:'no-vest',5:'person',6:'shoes',7:'vest'
}

if not os.path.exists(MODEL_FILENAME):
    st.error("❌ PPE.pt not found in app directory")
    st.stop()

# ---------------------------------------------------
# YOLO CLI runner (FIXED – headless safe)
# ---------------------------------------------------
def run_yolo_cli(image_path):
    if os.path.exists(RUNS_DIR):
        subprocess.run(["rm", "-rf", RUNS_DIR])

    # 🔑 Critical: force headless OpenCV
    env = os.environ.copy()
    env["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
    env["QT_QPA_PLATFORM"] = "offscreen"
    env["DISPLAY"] = ""
    env["CUDA_VISIBLE_DEVICES"] = ""  # CPU-only, safest on Streamlit Cloud

    cmd = [
        "yolo", "predict",
        f"model={MODEL_FILENAME}",
        f"source={image_path}",
        "conf=0.4",
        "imgsz=640",
        "save=True",
        "save_txt=True",
        "save_conf=True"
    ]

    subprocess.run(cmd, env=env, check=True)

    pred_dir = os.path.join(RUNS_DIR, "predict")

    img = next(
        os.path.join(pred_dir, f)
        for f in os.listdir(pred_dir)
        if f.lower().endswith((".jpg", ".png"))
    )

    lbl_dir = os.path.join(pred_dir, "labels")
    lbl = os.path.join(lbl_dir, os.listdir(lbl_dir)[0])

    return img, lbl

def parse_labels(label_file, img_w, img_h):
    detections = []
    with open(label_file) as f:
        for line in f:
            c, x, y, w, h, conf = map(float, line.split())
            x1 = (x - w/2) * img_w
            y1 = (y - h/2) * img_h
            x2 = (x + w/2) * img_w
            y2 = (y + h/2) * img_h
            detections.append({
                "class_id": int(c),
                "class_name": CLASS_NAMES.get(int(c), str(c)),
                "conf": conf,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2
            })
    return detections

# ---------------------------------------------------
# Layout
# ---------------------------------------------------
left_col, right_col = st.columns([1.4, 1.1])

# ---------------- LEFT ----------------
with left_col:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload site image", type=["jpg", "jpeg", "png"])
    st.markdown("</div>", unsafe_allow_html=True)

    detections_list = []
    annotated = None
    infer_time = None

    if uploaded:
        input_img = Image.open(uploaded).convert("RGB")
        st.image(input_img, use_container_width=True)

        temp_img = "temp.jpg"
        input_img.save(temp_img)

        with st.spinner("Running PPE detection…"):
            t0 = time.time()
            out_img, out_lbl = run_yolo_cli(temp_img)
            infer_time = time.time() - t0

        annotated = Image.open(out_img)
        st.image(annotated, caption=f"Processed in {infer_time:.2f}s", use_container_width=True)

        detections_list = parse_labels(out_lbl, input_img.width, input_img.height)

# ---------------- RIGHT ----------------
with right_col:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>📊 Safety Snapshot</div>", unsafe_allow_html=True)

    if not detections_list:
        st.markdown("<i>No detections yet</i>", unsafe_allow_html=True)
    else:
        counts = Counter(d["class_id"] for d in detections_list)
        persons = counts.get(5, 0)
        helmets = counts.get(1, 0)
        vests = counts.get(7, 0)
        shoes = counts.get(6, 0)
        gloves = counts.get(0, 0)
        no_flags = counts.get(3, 0) + counts.get(4, 0) + counts.get(2, 0)

        compliant = helmets + vests + shoes + gloves
        denom = max(1, compliant + no_flags)
        score = int(100 * compliant / denom)

        k1, k2, k3 = st.columns(3)
        k1.metric("Workers", persons)
        k2.metric("PPE Issues", no_flags)
        k3.metric("Compliance", f"{score}%")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            "<span class='chip-ok'>Compliant</span>" if no_flags == 0
            else "<span class='chip-risk'>Issues Detected</span>",
            unsafe_allow_html=True
        )

        st.markdown("<br><div class='section-title'>Detected Items</div>", unsafe_allow_html=True)
        for cid, cnt in counts.items():
            st.markdown(
                f"<span class='badge'>{CLASS_NAMES[cid]}: {cnt}</span>",
                unsafe_allow_html=True
            )

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------
# Webcam (same pipeline)
# ---------------------------------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>🎥 Quick Webcam Test</div>", unsafe_allow_html=True)
cam = st.camera_input("Capture frame")

if cam:
    cam_img = Image.open(cam).convert("RGB")
    cam_img.save("cam.jpg")
    out_img, _ = run_yolo_cli("cam.jpg")
    st.image(Image.open(out_img), use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------
# Footer
# ---------------------------------------------------
st.markdown("""
<div class="footer-text">
Built by Sharan · B.Tech AI & Data Science<br>
Production-style demo: detection → analytics → exports
</div>
""", unsafe_allow_html=True)
