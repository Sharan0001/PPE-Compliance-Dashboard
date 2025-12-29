import streamlit as st
from ultralytics import YOLO
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
# Custom Styling (with background image)
# ---------------------------------------------------
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: "Segoe UI", sans-serif;
}

/* Full app background with tech image + dark overlay */
.stApp {
    background-image:
        radial-gradient(circle at top, rgba(37,99,235,0.35), rgba(15,23,42,0.98) 55%),
        url("https://images.pexels.com/photos/546819/pexels-photo-546819.jpeg?auto=compress&cs=tinysrgb&w=1600");
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
    background-position: center center;
}

/* Card container */
.card {
    background: rgba(15,23,42,0.92);
    padding: 18px 20px;
    border-radius: 16px;
    border: 1px solid rgba(148,163,184,0.35);
    box-shadow: 0 10px 35px rgba(15,23,42,0.9);
    margin-bottom: 18px;
}

/* Soft card */
.card-soft {
    background: rgba(15,23,42,0.80);
    padding: 16px 18px;
    border-radius: 14px;
    border: 1px solid rgba(30,64,175,0.6);
    margin-bottom: 12px;
}

/* Section titles */
.section-title {
    font-size: 20px;
    font-weight: 600;
    color: #93c5fd;
    margin-bottom: 4px;
}

/* Subtext */
.subtext {
    font-size: 13px;
    color: #cbd5f5;
}

/* KPIs */
.kpi-label {
    font-size: 13px;
    color: #f1f5f9;
}
.kpi-value {
    font-size: 24px;
    font-weight: 600;
}

/* Badge */
.badge {
    display: inline-block;
    padding: 6px 10px;
    border-radius: 999px;
    background: rgba(15,23,42,0.8);
    border: 1px solid #1d4ed8;
    margin: 3px 4px;
    font-size: 13px;
}

/* Status chips */
.chip-ok {
    padding: 5px 10px;
    border-radius: 999px;
    background: rgba(22,163,74,0.18);
    border: 1px solid rgba(34,197,94,0.9);
    color: #bbf7d0;
    font-size: 13px;
}
.chip-risk {
    padding: 5px 10px;
    border-radius: 999px;
    background: rgba(220,38,38,0.18);
    border: 1px solid rgba(248,113,113,0.95);
    color: #fecaca;
    font-size: 13px;
}

/* Footer */
.footer-text {
    text-align: center;
    font-size: 13px;
    margin-top: 24px;
    color: #e5e7eb;
    text-shadow: 0 0 8px rgba(15,23,42,0.9);
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# Hero Section
# ---------------------------------------------------
st.markdown(
    """
    <div style="text-align:center; padding-top:10px; padding-bottom:5px;">
        <h1 style="color:#e5e7eb; margin-bottom:4px;">PPE Compliance Intelligence Dashboard</h1>
        <p style="color:#f1f5f9; font-size:15px;">
            AI that scans construction sites for helmets, vests, gloves & safety shoes — built for safety audits, risk reporting and on-site monitoring.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------
# Model Loading
# ---------------------------------------------------
MODEL_FILENAME = "PPE.pt"

@st.cache_resource
def load_yolo_model():
    model = YOLO(MODEL_FILENAME)
    return model

if not os.path.exists(MODEL_FILENAME):
    st.error("❌ Model file `PPE.pt` not found. Place it in the same folder as `app.py`.")
    st.stop()

model = load_yolo_model()

# ---------------------------------------------------
# Top 3 info cards (product & business oriented)
# ---------------------------------------------------
info_col1, info_col2, info_col3 = st.columns(3)
with info_col1:
    st.markdown(
        """
        <div class="card-soft">
            <div class="section-title">🎯 Problem</div>
            <div class="subtext">
                Manual PPE checks are slow and prone to human error.
                This demo shows how AI can continuously monitor safety compliance from site images or CCTV frames.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with info_col2:
    st.markdown(
        """
        <div class="card-soft">
            <div class="section-title">🏗️ Use Case</div>
            <div class="subtext">
                HSE teams upload site photos, instantly see missing PPE items,
                and export visual evidence for reports or incident investigations.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with info_col3:
    st.markdown(
        """
        <div class="card-soft">
            <div class="section-title">📈 Impact</div>
            <div class="subtext">
                Reduce inspection time, improve audit coverage, and get consistent
                safety insights that can be pushed to dashboards or mobile apps.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------
# Main layout: left (image), right (analytics)
# ---------------------------------------------------
left_col, right_col = st.columns([1.4, 1.1])

# -------------------- LEFT: Image Upload & Results --------------------
with left_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📤 Upload Site Image</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtext">Use a construction image where workers are visible. The model will highlight detected PPE items.</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader("", type=["jpg", "jpeg", "png"])
    st.markdown('</div>', unsafe_allow_html=True)

    annotated = None
    detections_list = []
    infer_time = None
    class_names = {}

    if uploaded:
        input_image = Image.open(uploaded).convert("RGB")
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🖼️ Input</div>', unsafe_allow_html=True)
        st.image(input_image, caption="Uploaded Image", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Run inference
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🤖 Detection Results</div>', unsafe_allow_html=True)
        with st.spinner("Running PPE detection…"):
            t0 = time.time()
            results = model.predict(
                source=np.asarray(input_image),
                conf=0.40,
                max_det=40,
                imgsz=640
            )
            infer_time = time.time() - t0

        # Annotated image
        plotted = results[0].plot()
        annotated = Image.fromarray(plotted)
        st.image(
            annotated,
            caption=f"Detections (processed in {infer_time:.2f}s)",
            use_container_width=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

        # Extract detections
        boxes = results[0].boxes
        try:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            cls_ids = boxes.cls.cpu().numpy().astype(int)
        except Exception:
            xyxy, confs, cls_ids = np.array([]), np.array([]), np.array([])

        try:
            class_names = model.model.names
        except Exception:
            class_names = {
                0: 'gloves',
                1: 'hardhat',
                2: 'no-gloves',
                3: 'no-hardhat',
                4: 'no-vest',
                5: 'person',
                6: 'shoes',
                7: 'vest'
            }

        for (b, c, cid) in zip(xyxy, confs, cls_ids):
            detections_list.append({
                "x1": float(b[0]), "y1": float(b[1]),
                "x2": float(b[2]), "y2": float(b[3]),
                "conf": float(c),
                "class_id": int(cid),
                "class_name": class_names.get(int(cid), str(cid))
            })

# -------------------- RIGHT: Business View & Insights --------------------
with right_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📊 Safety Snapshot</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtext">High-level metrics that a safety or operations manager would care about.</div>', unsafe_allow_html=True)

    if not detections_list:
        st.markdown("<br><i>No detections yet. Upload an image to see analytics.</i>", unsafe_allow_html=True)
    else:
        # Aggregate counts
        counts_by_id = Counter([d["class_id"] for d in detections_list])

        # Try resolving names
        def cname(cid):
            if isinstance(class_names, dict):
                return class_names.get(cid, str(cid))
            if isinstance(class_names, list) and cid < len(class_names):
                return class_names[cid]
            return str(cid)

        # Map some semantic roles
        persons = counts_by_id.get(
            next((k for k,v in class_names.items() if v == "person"), 5),
            0
        ) if isinstance(class_names, dict) else counts_by_id.get(5, 0)

        # lookups
        def count_by_name(name, default_id):
            if isinstance(class_names, dict):
                match_ids = [cid for cid, n in class_names.items() if n == name]
                if match_ids:
                    return sum(counts_by_id.get(cid, 0) for cid in match_ids)
                return counts_by_id.get(default_id, 0)
            else:
                # assume fixed mapping order if list
                mapping = {
                    "gloves": 0, "hardhat": 1, "no-gloves": 2, "no-hardhat": 3,
                    "no-vest": 4, "person": 5, "shoes": 6, "vest": 7
                }
                cid = mapping.get(name, default_id)
                return counts_by_id.get(cid, 0)

        helmets = count_by_name("hardhat", 1)
        vests = count_by_name("vest", 7)
        shoes = count_by_name("shoes", 6)
        no_helmet = count_by_name("no-hardhat", 3)
        no_vest = count_by_name("no-vest", 4)
        no_gloves = count_by_name("no-gloves", 2)
        gloves = count_by_name("gloves", 0)

        non_compliance_flags = no_helmet + no_vest + no_gloves

        # Compliance score (simple heuristic)
        if persons == 0:
            compliance_score = 100
        else:
            # treat PPE items + vs missing as basic signal
            compliant_signals = helmets + vests + shoes + gloves
            risk_signals = non_compliance_flags
            denom = max(1, compliant_signals + risk_signals)
            compliance_score = max(0, min(100, int(100 * compliant_signals / denom)))

        # KPI row
        k1, k2, k3 = st.columns(3)
        with k1:
            st.markdown('<div class="kpi-label">Workers detected</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="kpi-value">{persons}</div>', unsafe_allow_html=True)
        with k2:
            st.markdown('<div class="kpi-label">PPE issues flagged</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="kpi-value">{non_compliance_flags}</div>', unsafe_allow_html=True)
        with k3:
            st.markdown('<div class="kpi-label">Compliance score</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="kpi-value">{compliance_score}%</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Status chips
        if non_compliance_flags == 0:
            st.markdown("<span class='chip-ok'>Site looks compliant based on this frame</span>", unsafe_allow_html=True)
        else:
            st.markdown("<span class='chip-risk'>Potential PPE issues detected in this frame</span>", unsafe_allow_html=True)

        # Detailed breakdown badges
        st.markdown("<br><br><div class='section-title'>Detected Items</div>", unsafe_allow_html=True)
        for cid, cnt in counts_by_id.items():
            st.markdown(
                f"<span class='badge'>{cname(cid)}: {cnt}</span>",
                unsafe_allow_html=True
            )

        # Risk & Action panel
        st.markdown("<br><div class='section-title'>Risk & Suggested Actions</div>", unsafe_allow_html=True)
        st.markdown("<div class='subtext'>Turn detections into actions a safety manager can take.</div>", unsafe_allow_html=True)
        st.write("- ⚠️ **Helmet issues**: " +
                 ("none observed in this frame." if no_helmet == 0 else f"{no_helmet} worker(s) flagged without helmet."))
        st.write("- ⚠️ **Vest issues**: " +
                 ("none observed." if no_vest == 0 else f"{no_vest} worker(s) without high-visibility vest."))
        st.write("- ⚠️ **Gloves issues**: " +
                 ("none observed." if no_gloves == 0 else f"{no_gloves} worker(s) without gloves."))
        st.write("- ✅ Consider integrating this pipeline with CCTV feeds or site-capture apps to automatically log non-compliant frames.")

    st.markdown('</div>', unsafe_allow_html=True)

    # Downloads card
    if annotated is not None and detections_list:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📥 Export</div>', unsafe_allow_html=True)
        st.markdown('<div class="subtext">Use these exports in reports, presentations or audit documentation.</div>', unsafe_allow_html=True)

        buf = io.BytesIO()
        annotated.save(buf, format="JPEG")
        st.download_button(
            "Download annotated image",
            data=buf.getvalue(),
            file_name="ppe_detection.jpg",
            mime="image/jpeg"
        )

        st.download_button(
            "Download raw detection JSON",
            data=json.dumps(detections_list, indent=2),
            file_name="detections.json",
            mime="application/json"
        )

        st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------
# 🎥 Webcam Demo (NEW FEATURE)
# ---------------------------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">🎥 Quick Webcam Test</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtext">Capture a frame from your webcam to simulate on-site monitoring from a fixed camera.</div>',
    unsafe_allow_html=True,
)

cam_image = st.camera_input("Use webcam (optional)")

if cam_image is not None:
    cam_pil = Image.open(cam_image).convert("RGB")
    st.image(cam_pil, caption="Webcam Frame", use_container_width=True)

    with st.spinner("Analyzing webcam frame…"):
        t0_cam = time.time()
        cam_results = model.predict(
            source=np.asarray(cam_pil),
            conf=0.40,
            max_det=40,
            imgsz=640
        )
        cam_infer_time = time.time() - t0_cam

    cam_plotted = cam_results[0].plot()
    cam_annotated = Image.fromarray(cam_plotted)

    st.image(
        cam_annotated,
        caption=f"Webcam detections (processed in {cam_infer_time:.2f}s)",
        use_container_width=True
    )

    # -------------------- Webcam Compliance Analytics --------------------
    # Extract detections
    cam_boxes = cam_results[0].boxes
    cam_detections = []
    try:
        cam_xyxy = cam_boxes.xyxy.cpu().numpy()
        cam_confs = cam_boxes.conf.cpu().numpy()
        cam_cls_ids = cam_boxes.cls.cpu().numpy().astype(int)
    except Exception:
        cam_xyxy, cam_confs, cam_cls_ids = np.array([]), np.array([]), np.array([])

    # Class names fallback
    try:
        cam_class_names = model.model.names
    except Exception:
        cam_class_names = {
            0: 'gloves',
            1: 'hardhat',
            2: 'no-gloves',
            3: 'no-hardhat',
            4: 'no-vest',
            5: 'person',
            6: 'shoes',
            7: 'vest'
        }

    for (b, c, cid) in zip(cam_xyxy, cam_confs, cam_cls_ids):
        cam_detections.append({
            "class_id": int(cid),
            "class_name": cam_class_names.get(int(cid), str(cid)),
            "conf": float(c)
        })

    # Compliance summary card
    st.markdown("<br><div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>📊 Webcam Safety Snapshot</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtext'>Live analytics computed from the current webcam frame.</div>", unsafe_allow_html=True)

    if not cam_detections:
        st.markdown("<br><i>No PPE detections found in this webcam frame.</i>", unsafe_allow_html=True)
    else:
        from collections import Counter

        counts_by_id = Counter([d["class_id"] for d in cam_detections])

        # Utility
        def cname(cid):
            return cam_class_names.get(cid, str(cid))

        # Count helpers
        def count(name, fallback):
            match = [cid for cid,n in cam_class_names.items() if n == name]
            if match:
                return sum(counts_by_id.get(cid, 0) for cid in match)
            return counts_by_id.get(fallback, 0)

        persons = count("person", 5)
        helmets = count("hardhat", 1)
        vests = count("vest", 7)
        shoes = count("shoes", 6)
        no_helmet = count("no-hardhat", 3)
        no_vest = count("no-vest", 4)
        no_gloves = count("no-gloves", 2)
        gloves = count("gloves", 0)

        non_compliance_flags = no_helmet + no_vest + no_gloves

        # Compliance score
        if persons == 0:
            compliance_score = 100
        else:
            compliant_sig = helmets + vests + shoes + gloves
            risk_sig = non_compliance_flags
            denom = max(1, compliant_sig + risk_sig)
            compliance_score = max(0, min(100, int(100 * compliant_sig / denom)))

        # KPIs
        k1, k2, k3 = st.columns(3)
        with k1:
            st.markdown('<div class="kpi-label">Workers detected</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="kpi-value">{persons}</div>', unsafe_allow_html=True)
        with k2:
            st.markdown('<div class="kpi-label">PPE issues flagged</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="kpi-value">{non_compliance_flags}</div>', unsafe_allow_html=True)
        with k3:
            st.markdown('<div class="kpi-label">Compliance score</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="kpi-value">{compliance_score}%</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Status chip
        if non_compliance_flags == 0:
            st.markdown("<span class='chip-ok'>Webcam frame looks compliant</span>", unsafe_allow_html=True)
        else:
            st.markdown("<span class='chip-risk'>PPE issues detected in webcam frame</span>", unsafe_allow_html=True)

        # Detailed counts
        st.markdown("<br><br><div class='section-title'>Detected Items</div>", unsafe_allow_html=True)
        for cid, cnt in counts_by_id.items():
            st.markdown(
                f"<span class='badge'>{cname(cid)}: {cnt}</span>",
                unsafe_allow_html=True
            )

        # Actions
        st.markdown("<br><div class='section-title'>Risk & Suggested Actions</div>", unsafe_allow_html=True)
        st.markdown("<div class='subtext'>Real-time recommendations derived from webcam analysis.</div>", unsafe_allow_html=True)

        st.write("- ⚠️ **Helmet issues**: " +
                 ("none observed." if no_helmet == 0 else f"{no_helmet} worker(s) without helmet."))
        st.write("- ⚠️ **Vest issues**: " +
                 ("none observed." if no_vest == 0 else f"{no_vest} worker(s) without high-visibility vest."))
        st.write("- ⚠️ **Gloves issues**: " +
                 ("none observed." if no_gloves == 0 else f"{no_gloves} worker(s) without gloves."))
        st.write("- 🔄 Capture another frame for updated risk assessment.")

    st.markdown("</div>", unsafe_allow_html=True)


st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------
# Footer (portfolio focused)
# ---------------------------------------------------
st.markdown("""
<div class="footer-text">
Built by Sharan · B.Tech AI & Data Science<br>
Designed as a production-style demo: detection model → analytics dashboard → exportable insights.
</div>
""", unsafe_allow_html=True)
