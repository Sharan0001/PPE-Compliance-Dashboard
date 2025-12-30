import streamlit as st
import subprocess
from PIL import Image
import numpy as np
import io, os, time, json
from collections import Counter
import shutil
import uuid

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
.card-soft { background: rgba(15,23,42,0.80); padding: 16px; border-radius: 14px; margin-bottom: 12px; }
.section-title { font-size: 20px; font-weight: 600; color: #93c5fd; }
.subtext { font-size: 13px; color: #cbd5f5; }
.kpi-label { font-size: 13px; color: #f1f5f9; }
.kpi-value { font-size: 24px; font-weight: 600; }
.badge { padding: 6px 10px; border-radius: 999px; background: rgba(15,23,42,0.8); border: 1px solid #1d4ed8; margin: 3px; font-size: 13px; display: inline-block; }
.chip-ok { background: rgba(22,163,74,0.18); border: 1px solid rgba(34,197,94,0.9); color: #bbf7d0; padding: 5px 10px; border-radius: 999px; }
.chip-risk { background: rgba(220,38,38,0.18); border: 1px solid rgba(248,113,113,0.95); color: #fecaca; padding: 5px 10px; border-radius: 999px; }
.footer-text { text-align:center; font-size:13px; color:#e5e7eb; margin-top:24px; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# Hero
# ---------------------------------------------------
st.markdown("""
<div style="text-align:center;">
<h1 style="color:#e5e7eb;">PPE Compliance Intelligence Dashboard</h1>
<p style="color:#f1f5f9;">AI-powered PPE detection for construction safety audits</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# Model config
# ---------------------------------------------------
MODEL_FILENAME = "PPE.pt"
RUNS_DIR = "runs/detect"

if not os.path.exists(MODEL_FILENAME):
    st.error("❌ Model file `PPE.pt` not found.")
    st.stop()

# ---------------------------------------------------
# Helper: Run YOLO via CLI
# ---------------------------------------------------
def run_yolo_cli(image_path):
    run_id = f"predict_{uuid.uuid4().hex[:8]}"
    out_dir = os.path.join(RUNS_DIR, run_id)

    cmd = [
        "yolo",
        "predict",
        f"model={MODEL_FILENAME}",
        f"source={image_path}",
        "conf=0.40",
        "imgsz=640",
        f"project={RUNS_DIR}",
        f"name={run_id}",
        "save=True",
        "save_txt=True",
        "save_conf=True"
    ]

    subprocess.run(cmd, check=True)

    img_files = [f for f in os.listdir(out_dir) if f.lower().endswith((".jpg", ".png"))]
    txt_dir = os.path.join(out_dir, "labels")

    return out_dir, img_files[0], txt_dir

# ---------------------------------------------------
# Layout
# ---------------------------------------------------
left_col, right_col = st.columns([1.4, 1.1])

# ---------------- LEFT ----------------
with left_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📤 Upload Site Image</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("", type=["jpg", "jpeg", "png"])
    st.markdown('</div>', unsafe_allow_html=True)

    detections_list = []
    annotated = None
    infer_time = None

    if uploaded:
        tmp_img = f"tmp_{uuid.uuid4().hex}.jpg"
        Image.open(uploaded).convert("RGB").save(tmp_img)

        t0 = time.time()
        out_dir, out_img, txt_dir = run_yolo_cli(tmp_img)
        infer_time = time.time() - t0

        annotated = Image.open(os.path.join(out_dir, out_img))
        st.image(annotated, caption=f"Detections ({infer_time:.2f}s)", use_container_width=True)

        # Parse labels
        for txt in os.listdir(txt_dir):
            with open(os.path.join(txt_dir, txt)) as f:
                for line in f:
                    cls, *vals = line.split()
                    detections_list.append({
                        "class_id": int(cls),
                        "conf": float(vals[-1])
                    })

        os.remove(tmp_img)

# ---------------- RIGHT ----------------
with right_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📊 Safety Snapshot</div>', unsafe_allow_html=True)

    if not detections_list:
        st.markdown("<i>No detections yet.</i>", unsafe_allow_html=True)
    else:
        counts = Counter([d["class_id"] for d in detections_list])

        CLASS_MAP = {
            0: "gloves", 1: "hardhat", 2: "no-gloves",
            3: "no-hardhat", 4: "no-vest",
            5: "person", 6: "shoes", 7: "vest"
        }

        persons = counts.get(5, 0)
        no_flags = counts.get(3, 0) + counts.get(4, 0) + counts.get(2, 0)
        compliant = counts.get(1, 0) + counts.get(7, 0) + counts.get(6, 0) + counts.get(0, 0)

        score = 100 if persons == 0 else int(100 * compliant / max(1, compliant + no_flags))

        k1, k2, k3 = st.columns(3)
        k1.markdown(f"<div class='kpi-value'>{persons}</div><div class='kpi-label'>Workers</div>", unsafe_allow_html=True)
        k2.markdown(f"<div class='kpi-value'>{no_flags}</div><div class='kpi-label'>Issues</div>", unsafe_allow_html=True)
        k3.markdown(f"<div class='kpi-value'>{score}%</div><div class='kpi-label'>Compliance</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            "<span class='chip-ok'>Compliant</span>" if no_flags == 0
            else "<span class='chip-risk'>Issues detected</span>",
            unsafe_allow_html=True
        )

        st.markdown("<br><br>", unsafe_allow_html=True)
        for cid, cnt in counts.items():
            st.markdown(f"<span class='badge'>{CLASS_MAP.get(cid, cid)}: {cnt}</span>", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------
# Footer
# ---------------------------------------------------
st.markdown("""
<div class="footer-text">
Built by Sharan · B.Tech AI & Data Science<br>
YOLO inference via CLI for Streamlit-Cloud stability
</div>
""", unsafe_allow_html=True)
