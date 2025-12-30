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
html, body, [class*="css"] {
    font-family: "Segoe UI", sans-serif;
}
.stApp {
    background-image:
        radial-gradient(circle at top, rgba(37,99,235,0.35), rgba(15,23,42,0.98) 55%),
        url("https://images.pexels.com/photos/546819/pexels-photo-546819.jpeg?auto=compress&cs=tinysrgb&w=1600");
    background-size: cover;
    background-attachment: fixed;
}
.card { background: rgba(15,23,42,0.92); padding: 18px; border-radius: 16px; margin-bottom: 18px; }
.card-soft { background: rgba(15,23,42,0.8); padding: 16px; border-radius: 14px; margin-bottom: 12px; }
.section-title { font-size: 20px; font-weight: 600; color: #93c5fd; }
.subtext { font-size: 13px; color: #cbd5f5; }
.kpi-label { font-size: 13px; color: #f1f5f9; }
.kpi-value { font-size: 24px; font-weight: 600; }
.badge { padding: 6px 10px; border-radius: 999px; background: rgba(15,23,42,0.8); border: 1px solid #1d4ed8; margin: 3px; display: inline-block; }
.chip-ok { background: rgba(22,163,74,0.18); border: 1px solid #22c55e; color: #bbf7d0; padding: 5px 10px; border-radius: 999px; }
.chip-risk { background: rgba(220,38,38,0.18); border: 1px solid #f87171; color: #fecaca; padding: 5px 10px; border-radius: 999px; }
.footer-text { text-align: center; font-size: 13px; margin-top: 24px; color: #e5e7eb; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# Hero
# ---------------------------------------------------
st.markdown("""
<div style="text-align:center;">
<h1 style="color:#e5e7eb;">PPE Compliance Intelligence Dashboard</h1>
<p style="color:#f1f5f9;">AI that scans construction sites for PPE compliance.</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# Model config (CLI)
# ---------------------------------------------------
MODEL_FILENAME = "PPE.pt"
RUNS_DIR = "runs/detect"

if not os.path.exists(MODEL_FILENAME):
    st.error("❌ PPE.pt not found in project root")
    st.stop()

def run_yolo_cli(image_path):
    subprocess.run(["rm", "-rf", RUNS_DIR])
    cmd = [
        "yolo", "predict",
        f"model={MODEL_FILENAME}",
        f"source={image_path}",
        "conf=0.4",
        "save=True",
        "save_txt=True"
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    exp_dir = next(
        (os.path.join(RUNS_DIR, d) for d in os.listdir(RUNS_DIR)),
        None
    )
    return exp_dir

def load_results(exp_dir, image_shape):
    img_path = next(p for p in os.listdir(exp_dir) if p.endswith(".jpg"))
    label_path = os.path.join(exp_dir, "labels", img_path.replace(".jpg", ".txt"))

    detections = []
    if os.path.exists(label_path):
        with open(label_path) as f:
            for line in f:
                cid, xc, yc, w, h, conf = map(float, line.split())
                detections.append({
                    "class_id": int(cid),
                    "conf": conf
                })

    annotated = Image.open(os.path.join(exp_dir, img_path))
    return annotated, detections

CLASS_NAMES = {
    0: 'gloves', 1: 'hardhat', 2: 'no-gloves', 3: 'no-hardhat',
    4: 'no-vest', 5: 'person', 6: 'shoes', 7: 'vest'
}

# ---------------------------------------------------
# Layout
# ---------------------------------------------------
left_col, right_col = st.columns([1.4, 1.1])

# ---------------- LEFT ----------------
with left_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload site image", type=["jpg","png","jpeg"])
    st.markdown('</div>', unsafe_allow_html=True)

    annotated = None
    detections_list = []

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        img.save("input.jpg")
        st.image(img, caption="Input Image", use_container_width=True)

        with st.spinner("Running PPE detection…"):
            t0 = time.time()
            exp_dir = run_yolo_cli("input.jpg")
            annotated, detections_list = load_results(exp_dir, img.size)
            infer_time = time.time() - t0

        st.image(annotated, caption=f"Processed in {infer_time:.2f}s", use_container_width=True)

# ---------------- RIGHT ----------------
with right_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📊 Safety Snapshot</div>', unsafe_allow_html=True)

    if not detections_list:
        st.markdown("<i>No detections yet.</i>", unsafe_allow_html=True)
    else:
        counts = Counter([d["class_id"] for d in detections_list])

        def count(name):
            return sum(v for k,v in counts.items() if CLASS_NAMES.get(k)==name)

        persons = count("person")
        helmets = count("hardhat")
        vests = count("vest")
        gloves = count("gloves")
        shoes = count("shoes")
        risks = count("no-hardhat")+count("no-vest")+count("no-gloves")

        compliant = helmets+vests+gloves+shoes
        denom = max(1, compliant+risks)
        score = int(100*compliant/denom)

        k1,k2,k3 = st.columns(3)
        k1.markdown(f"<div class='kpi-value'>{persons}</div><div class='kpi-label'>Workers</div>", unsafe_allow_html=True)
        k2.markdown(f"<div class='kpi-value'>{risks}</div><div class='kpi-label'>PPE Issues</div>", unsafe_allow_html=True)
        k3.markdown(f"<div class='kpi-value'>{score}%</div><div class='kpi-label'>Compliance</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            "<span class='chip-ok'>Compliant</span>" if risks==0 else
            "<span class='chip-risk'>Issues detected</span>",
            unsafe_allow_html=True
        )

        st.markdown("<br><br><div class='section-title'>Detected Items</div>", unsafe_allow_html=True)
        for cid,cnt in counts.items():
            st.markdown(f"<span class='badge'>{CLASS_NAMES.get(cid)}: {cnt}</span>", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------
# Webcam
# ---------------------------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
cam = st.camera_input("Webcam (optional)")
if cam:
    cam_img = Image.open(cam).convert("RGB")
    cam_img.save("cam.jpg")
    st.image(cam_img, use_container_width=True)

    with st.spinner("Analyzing webcam frame…"):
        exp = run_yolo_cli("cam.jpg")
        cam_annotated, _ = load_results(exp, cam_img.size)
    st.image(cam_annotated, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------
# Footer
# ---------------------------------------------------
st.markdown("""
<div class="footer-text">
Built by Sharan · B.Tech AI & Data Science<br>
Production-style demo: detection → analytics → exportable insights
</div>
""", unsafe_allow_html=True)
