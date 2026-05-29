import streamlit as st
import onnxruntime as ort
import numpy as np
from PIL import Image
import json
import time
import os
import gdown

# ── Download Model if Not Present ────────────────────────────
MODEL_PATH = "cropguard_model.onnx"

if not os.path.exists(MODEL_PATH):
    with st.spinner("🔄 Downloading AI model... This may take a minute..."):
        url = "https://drive.google.com/uc?id=1S38XjoWaH7twd4mBIpLQTx_0A8d1e3ZC"
        gdown.download(url, MODEL_PATH, quiet=False)

st.set_page_config(
    page_title="CropGuard — Crop Disease Detection",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

if "show_about" not in st.session_state:
    st.session_state.show_about = False
if "result" not in st.session_state:
    st.session_state.result = None

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Playfair+Display:wght@600;700;800&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html, body, .stApp { font-family: 'Inter', sans-serif; background: #ffffff; color: #1a1a2e; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 100% !important; }
[data-testid="stVerticalBlock"] { gap: 0 !important; }
section[data-testid="stSidebar"] { display: none !important; }
.stAlert { display: none !important; }
.stFileUploader > div { background: transparent !important; border: none !important; padding: 0 !important; }
.stFileUploader label { display: none !important; }
[data-testid="stFileUploaderFileName"] { display: none !important; }
div[class*="uploadedFile"] { display: none !important; }
[data-testid="stHorizontalBlock"] { gap: 0 !important; }
[data-testid="stColumn"] { background: transparent !important; padding: 0 !important; }
.stButton { margin: 0 !important; }

/* ── STYLE REAL STREAMLIT UPLOADER ── */
[data-testid="stFileUploader"] {
    background: #f0fdf4 !important;
    border: 2px dashed #86efac !important;
    border-radius: 12px !important;
    padding: 0.5rem !important;
}
[data-testid="stFileUploader"] label {
    display: block !important;
    font-size: 0.88rem !important;
    font-weight: 600 !important;
    color: #166534 !important;
    margin-bottom: 0.5rem !important;
    padding-left: 0.5rem !important;
}
[data-testid="stFileUploaderDropzone"] {
    background: #f0fdf4 !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 1.25rem !important;
    text-align: center !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] div {
    color: #166534 !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] span {
    color: #6b7280 !important;
    font-size: 0.78rem !important;
}
[data-testid="stFileUploaderDropzone"] button {
    background: #166534 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.45rem 1.25rem !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    cursor: pointer !important;
    margin-top: 0.5rem !important;
}

/* ── ANNOUNCEMENT BAR ── */
.top-bar {
    background: #166534;
    padding: 1.42rem 0;
    text-align: center;
    font-size: 0.78rem;
    color: rgba(255,255,255,0.88);
    font-weight: 500;
    letter-spacing: 0.3px;
}
.top-bar span { color: #86efac; font-weight: 700; }

/* ── NAVBAR — single full width bar ── */
.navbar {
    background: #ffffff;
    border-bottom: 1px solid #e5e7eb;
    padding: 0 3rem;
    height: 66px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    width: 100%;
    position: relative;
    z-index: 99;
}
.nav-brand { display: flex; align-items: center; gap: 12px; }
.nav-logo-box {
    width: 40px; height: 40px;
    background: #166534;
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.1rem;
    flex-shrink: 0;
}
.nav-name {
    font-family: 'Playfair Display', serif;
    font-size: 1.3rem;
    font-weight: 700;
    color: #166534;
}
.nav-tag {
    font-size: 0.68rem;
    color: #6b7280;
    font-weight: 400;
    letter-spacing: 0.3px;
    margin-top: 1px;
}
.nav-about-btn {
    background: #f0fdf4;
    border: 1.5px solid #86efac;
    color: #166534;
    padding: 0.42rem 1.1rem;
    border-radius: 8px;
    font-size: 0.83rem;
    font-weight: 700;
    cursor: pointer;
    font-family: 'Inter', sans-serif;
    white-space: nowrap;
}
.nav-about-btn:hover { background: #dcfce7; }

/* ── ABOUT PANEL ── */
.about-panel {
    background: #f0fdf4;
    border-bottom: 2px solid #bbf7d0;
    padding: 1.75rem 3rem;
    display: grid;
    grid-template-columns: 1.4fr 1fr 1fr 1fr;
    gap: 2rem;
}
.ap-label {
    font-size: 0.63rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: #166534;
    margin-bottom: 0.55rem;
}
.ap-text { font-size: 0.85rem; color: #374151; line-height: 1.8; }
.ap-text strong { color: #166534; }
.ap-num { font-family: 'Playfair Display', serif; font-size: 1.5rem; font-weight: 700; color: #166534; }
.ap-lbl { font-size: 0.75rem; color: #6b7280; margin-top: 0.1rem; }

/* ── HERO ── */
.hero {
    background: linear-gradient(150deg, #d1fae5 0%, #86efac 50%, #4ade80 100%);
    padding: 3rem 3rem 2.5rem;
    display: grid;
    grid-template-columns: 1fr 420px;
    gap: 3rem;
    align-items: center;
    border-bottom: 1px solid #16a34a;
}
.hero-eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #166534;
    color: #fff;
    padding: 0.28rem 0.9rem;
    border-radius: 50px;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 1rem;
}
.hero-h1 {
    font-family: 'Playfair Display', serif;
    font-size: 2.8rem;
    font-weight: 800;
    color: #052e16;
    line-height: 1.15;
    margin-bottom: 0.9rem;
    letter-spacing: -0.3px;
}
.hero-h1 em {
    color: #14532d;
    font-style: normal;
}
.hero-p {
    font-size: 0.97rem;
    color: #14532d;
    line-height: 1.8;
    margin-bottom: 1.5rem;
    max-width: 500px;
}
.hero-chips { display: flex; gap: 0.6rem; flex-wrap: wrap; }
.hero-chip {
    background: rgba(255,255,255,0.75);
    border: 1.5px solid rgba(255,255,255,0.9);
    color: #166534;
    padding: 0.32rem 0.85rem;
    border-radius: 50px;
    font-size: 0.78rem;
    font-weight: 600;
    box-shadow: 0 1px 3px rgba(22,101,52,0.1);
}
.hero-right {
    background: #ffffff;
    border-radius: 18px;
    padding: 1.75rem;
    box-shadow: 0 4px 20px rgba(22,101,52,0.1);
    border: 1px solid #d1fae5;
}
.hero-right-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #14532d;
    margin-bottom: 1.1rem;
    padding-bottom: 0.7rem;
    border-bottom: 1px solid #e5e7eb;
}
.hero-step {
    display: flex;
    align-items: flex-start;
    gap: 0.85rem;
    margin-bottom: 0.9rem;
}
.hero-step-n {
    background: #166534; color: #fff;
    min-width: 26px; height: 26px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.72rem; font-weight: 700;
    flex-shrink: 0; margin-top: 1px;
}
.hero-step-t { font-size: 0.88rem; color: #374151; line-height: 1.55; }
.hero-step-t strong { color: #166634; }

/* ── CONTENT AREA ── */
.content-area { padding: 2rem 3rem; background: #fafafa; }

/* ── WHITE CARDS ── */
.white-card {
    background: #ffffff;
    border-radius: 16px;
    padding: 1.6rem;
    border: 1px solid #e5e7eb;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}
.card-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.05rem;
    font-weight: 700;
    color: #14532d;
    margin-bottom: 1rem;
    padding-bottom: 0.65rem;
    border-bottom: 1px solid #f3f4f6;
}

/* ── UPLOAD DROP ZONE ── */
.drop-zone {
    border: 2px dashed #86efac;
    border-radius: 12px;
    padding: 1.75rem 1rem;
    text-align: center;
    background: #f0fdf4;
    cursor: pointer;
    transition: all 0.2s;
    margin-bottom: 0;
}
.drop-zone:hover { border-color: #16a34a; background: #dcfce7; }
.dz-icon { font-size: 2.2rem; margin-bottom: 0.45rem; }
.dz-title { font-weight: 700; color: #166534; font-size: 0.92rem; margin-bottom: 0.2rem; }
.dz-hint { color: #9ca3af; font-size: 0.78rem; }

/* ── IMAGE PREVIEW ── */
.img-preview {
    margin-top: 0.85rem;
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid #e5e7eb;
    background: #f9fafb;
    text-align: center;
    padding: 0.5rem;
}
.img-meta {
    margin-top: 0.5rem;
    background: #f0fdf4;
    border: 1px solid #d1fae5;
    border-radius: 7px;
    padding: 0.42rem 0.85rem;
    font-size: 0.75rem;
    color: #166534;
    font-weight: 600;
}

/* ── TIP ROWS ── */
.tip-row {
    display: flex;
    gap: 0.75rem;
    padding: 0.65rem 0;
    border-bottom: 1px solid #f3f4f6;
    align-items: flex-start;
}
.tip-row:last-child { border-bottom: none; padding-bottom: 0; }
.tip-icon-box {
    width: 30px; height: 30px;
    background: #f0fdf4;
    border-radius: 7px;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.85rem; flex-shrink: 0;
}
.tip-txt { font-size: 0.85rem; color: #4b5563; line-height: 1.5; }
.tip-txt strong { color: #166534; }

/* ── ANALYSE BUTTON ── */
.stButton > button {
    font-family: 'Inter', sans-serif !important;
    font-weight: 700 !important;
    border-radius: 10px !important;
    transition: all 0.2s !important;
    width: 100% !important;
    background: #166534 !important;
    border: none !important;
    color: #ffffff !important;
    font-size: 1rem !important;
    padding: 0.82rem 1rem !important;
    margin-top: 0.85rem !important;
    box-shadow: 0 4px 12px rgba(22,101,52,0.25) !important;
}
.stButton > button:hover {
    background: #15803d !important;
    box-shadow: 0 6px 18px rgba(22,101,52,0.35) !important;
    transform: translateY(-1px) !important;
}

/* ── RESULTS ── */
.result-area { padding: 2rem 3rem; background: #ffffff; }
.result-card {
    background: #fff;
    border-radius: 16px;
    border: 1px solid #e5e7eb;
    box-shadow: 0 4px 14px rgba(0,0,0,0.05);
    overflow: hidden;
}
.result-card-head {
    background: #f9fafb;
    padding: 0.85rem 1.5rem;
    border-bottom: 1px solid #e5e7eb;
    font-size: 0.72rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #374151;
}
.result-card-body { padding: 1.5rem; }
.res-healthy {
    background: linear-gradient(135deg,#f0fdf4,#dcfce7);
    border: 1.5px solid #86efac;
    border-radius: 12px;
    padding: 1.2rem;
    display: flex; align-items: center; gap: 1rem;
    margin-bottom: 1.1rem;
}
.res-disease {
    background: linear-gradient(135deg,#fef2f2,#fee2e2);
    border: 1.5px solid #fca5a5;
    border-radius: 12px;
    padding: 1.2rem;
    display: flex; align-items: center; gap: 1rem;
    margin-bottom: 1.1rem;
}
.res-icon { font-size: 2.4rem; flex-shrink: 0; }
.res-pill-h {
    display: inline-block;
    background: #166534; color: #fff;
    font-size: 0.58rem; font-weight: 700;
    letter-spacing: 1.5px; text-transform: uppercase;
    padding: 0.17rem 0.65rem; border-radius: 50px;
    margin-bottom: 0.28rem;
}
.res-pill-d {
    display: inline-block;
    background: #dc2626; color: #fff;
    font-size: 0.58rem; font-weight: 700;
    letter-spacing: 1.5px; text-transform: uppercase;
    padding: 0.17rem 0.65rem; border-radius: 50px;
    margin-bottom: 0.28rem;
}
.res-name {
    font-family: 'Playfair Display', serif;
    font-size: 1.25rem; font-weight: 700;
    color: #14532d; line-height: 1.3;
}
.conf-row { display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.38rem; }
.conf-lbl { font-size: 0.74rem; color: #374151; font-weight: 700; }
.conf-val { font-size: 1rem; font-weight: 700; color: #166534; }
.conf-bg { background: #f3f4f6; border-radius: 50px; height: 9px; overflow: hidden; margin-bottom: 1.1rem; }
.conf-bar { background: linear-gradient(90deg,#166534,#22c55e); height: 100%; border-radius: 50px; }
.remedy-box {
    background: #f0fdf4; border: 1px solid #bbf7d0;
    border-radius: 11px; padding: 1.1rem 1.35rem; margin-bottom: 1rem;
}
.remedy-lbl {
    font-size: 0.68rem; font-weight: 800;
    text-transform: uppercase; letter-spacing: 1.5px;
    color: #166534; margin-bottom: 0.5rem;
}
.remedy-body { font-size: 1rem; color: #1a2e1a; line-height: 1.75; font-weight: 500; }
.note-box {
    background: #fafafa; border: 1px solid #f3f4f6;
    border-radius: 10px; padding: 0.9rem 1.1rem;
}
.note-lbl { font-size: 0.68rem; font-weight: 800; text-transform: uppercase; letter-spacing: 1.5px; color: #374151; margin-bottom: 0.45rem; }
.note-text { font-size: 0.82rem; color: #374151; line-height: 1.65; }
.again-box {
    background: #166534; border-radius: 10px;
    padding: 0.85rem 1.1rem;
    display: flex; align-items: center; gap: 0.8rem;
    margin-top: 1rem;
}
.again-t { font-weight: 700; color: #fff; font-size: 0.88rem; margin-bottom: 0.1rem; }
.again-s { color: rgba(255,255,255,0.5); font-size: 0.76rem; }

/* ── AWAITING ── */
.awaiting {
    background: #fff;
    border: 1.5px dashed #d1fae5;
    border-radius: 16px;
    padding: 3rem 1.5rem;
    text-align: center;
}
.aw-icon { font-size: 2.8rem; opacity: 0.2; margin-bottom: 0.7rem; }
.aw-title { font-family: 'Playfair Display', serif; font-size: 1rem; color: #9ca3af; margin-bottom: 0.35rem; }
.aw-sub { color: #d1d5db; font-size: 0.82rem; line-height: 1.6; }

/* ── CROPS ── */
.crops-area { padding: 2.5rem 3rem; background: #f0fdf4; border-top: 1px solid #d1fae5; }
.crops-head { font-family: 'Playfair Display', serif; font-size: 1.3rem; font-weight: 700; color: #14532d; margin-bottom: 0.35rem; }
.crops-sub { font-size: 0.85rem; color: #6b7280; margin-bottom: 1.25rem; }
.crops-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 0.7rem;
}
.crop-tile {
    background: #fff;
    border: 1px solid #d1fae5;
    border-radius: 12px;
    padding: 0.8rem 0.5rem;
    text-align: center;
    font-size: 0.8rem;
    font-weight: 600;
    color: #166534;
    transition: all 0.2s;
    cursor: default;
}
.crop-tile:hover { background: #dcfce7; border-color: #86efac; transform: translateY(-2px); }
.crop-tile-icon { font-size: 1.35rem; display: block; margin-bottom: 0.3rem; }

/* ── FOOTER ── */
.footer-bar {
    background: #14532d;
    padding: 1.75rem 3rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 0.75rem;
}
.footer-brand { font-family: 'Playfair Display', serif; font-size: 1.1rem; font-weight: 700; color: #fff; margin-bottom: 0.2rem; }
.footer-sub { color: rgba(255,255,255,0.45); font-size: 0.76rem; }
.footer-r { color: rgba(255,255,255,0.3); font-size: 0.73rem; text-align: right; line-height: 1.7; }
</style>
""", unsafe_allow_html=True)

# ── Load Model ─────────────────────────────────────────────────
@st.cache_resource
def load_resources():
    session = ort.InferenceSession(MODEL_PATH)
    with open("class_names.json") as f:
        class_names = json.load(f)
    with open("remedies.json") as f:
        remedies = json.load(f)
    return session, class_names, remedies

session, class_names, remedies = load_resources()

# ── ANNOUNCEMENT BAR ────────────────────────────────────────────
st.markdown("""
<div class="top-bar">
    🌾 CropGuard uses AI to detect <span>42 crop diseases</span> across 15 crop types — Free to use
</div>
""", unsafe_allow_html=True)

# ── NAVBAR ── About button overlaid via negative margin ─────────
nav_l, nav_r = st.columns([5, 1])
with nav_l:
    st.markdown("""
    <div style="background:#ffffff;border-bottom:1px solid #e5e7eb;
                padding:0 3rem;height:66px;display:flex;align-items:center;
                box-shadow:0 1px 4px rgba(0,0,0,0.06);">
        <div style="display:flex;align-items:center;gap:12px;">
            <div style="width:40px;height:40px;background:#166534;border-radius:10px;
                        display:flex;align-items:center;justify-content:center;font-size:1.1rem;">
                🌿</div>
            <div>
                <div style="font-family:'Playfair Display',serif;font-size:1.3rem;
                            font-weight:700;color:#166534;line-height:1.2;">CropGuard</div>
                <div style="font-size:0.68rem;color:#6b7280;font-weight:400;">
                    Plant Disease Detection</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
with nav_r:
    st.markdown("""
    <div style="background:#ffffff;border-bottom:1px solid #e5e7eb;height:66px;
                display:flex;align-items:center;justify-content:flex-end;
                padding-right:3rem;box-shadow:0 1px 4px rgba(0,0,0,0.06);">
    </div>""", unsafe_allow_html=True)
    if st.button("ℹ️  About"):
        st.session_state.show_about = not st.session_state.show_about

st.markdown("""
<style>
div[data-testid="stHorizontalBlock"]:nth-of-type(2) {
    gap: 0 !important; margin: 0 !important; padding: 0 !important;
}
div[data-testid="stHorizontalBlock"]:nth-of-type(2) > div {
    padding: 0 !important;
}
div[data-testid="stHorizontalBlock"]:nth-of-type(2) > div:last-child {
    display: flex !important;
    align-items: center !important;
    justify-content: flex-end !important;
    padding-right: 3rem !important;
    background: #ffffff !important;
    border-bottom: 1px solid #e5e7eb !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06) !important;
    height: 66px !important;
}
div[data-testid="stHorizontalBlock"]:nth-of-type(2) .stButton > button {
    background: #f0fdf4 !important;
    border: 1.5px solid #86efac !important;
    color: #166534 !important;
    font-size: 0.83rem !important;
    padding: 0.42rem 1.1rem !important;
    border-radius: 8px !important;
    margin-top: 0 !important;
    width: auto !important;
    box-shadow: none !important;
    font-weight: 700 !important;
}
div[data-testid="stHorizontalBlock"]:nth-of-type(2) .stButton > button:hover {
    background: #dcfce7 !important;
    transform: none !important;
    box-shadow: none !important;
}
</style>
""", unsafe_allow_html=True)

# ── ABOUT PANEL ─────────────────────────────────────────────────
if st.session_state.show_about:
    st.markdown("""
    <div class="about-panel">
        <div>
            <div class="ap-label">About CropGuard</div>
            <div class="ap-text">
                CropGuard is an AI-powered crop disease detection system built to help
                farmers identify plant diseases early using a photograph of a leaf.
                It provides instant disease identification and treatment recommendations.
            </div>
        </div>
        <div>
            <div class="ap-label">Developer</div>
            <div class="ap-text">
                <strong>Arun Krishnan G.</strong><br>
                Certified Specialist in<br>Machine Learning and AI<br><br>
                ICTAK Academy, Kerala, India
            </div>
        </div>
        <div>
            <div class="ap-label">Technology</div>
            <div class="ap-text">
                Model: MobileNetV2<br>
                Technique: Transfer Learning<br>
                Framework: TensorFlow / Keras<br>
                Dataset: PlantVillage + Rice<br>
                Interface: Streamlit
            </div>
        </div>
        <div>
            <div class="ap-label">Model Performance</div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.85rem;margin-top:0.15rem;">
                <div><div class="ap-num">94.75%</div><div class="ap-lbl">Accuracy</div></div>
                <div><div class="ap-num">42</div><div class="ap-lbl">Disease Classes</div></div>
                <div><div class="ap-num">89K+</div><div class="ap-lbl">Training Images</div></div>
                <div><div class="ap-num">15</div><div class="ap-lbl">Crop Types</div></div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── HERO ────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div>
        <div class="hero-eyebrow">🤖 AI Powered Plant Health</div>
        <div class="hero-h1">
            Protect Your Crops<br>with <em>Smart Detection</em>
        </div>
        <p class="hero-p">
            Take a photo of your plant leaf and let our AI identify diseases
            instantly. Get clear treatment advice in seconds — no agricultural
            expert needed.
        </p>
        <div class="hero-chips">
            <span class="hero-chip">🦠 42 Disease Types</span>
            <span class="hero-chip">🌾 15 Crop Varieties</span>
            <span class="hero-chip">⚡ Instant Results</span>
            <span class="hero-chip">💊 Treatment Advice</span>
        </div>
    </div>
    <div class="hero-right">
        <div class="hero-right-title">📋 How It Works</div>
        <div class="hero-step">
            <div class="hero-step-n">1</div>
            <div class="hero-step-t"><strong>Photograph</strong> the affected leaf clearly in natural daylight</div>
        </div>
        <div class="hero-step">
            <div class="hero-step-n">2</div>
            <div class="hero-step-t"><strong>Upload</strong> the image using the upload section below</div>
        </div>
        <div class="hero-step">
            <div class="hero-step-n">3</div>
            <div class="hero-step-t"><strong>Analyse</strong> — our AI processes the image in seconds</div>
        </div>
        <div class="hero-step">
            <div class="hero-step-n">4</div>
            <div class="hero-step-t"><strong>Read</strong> the diagnosis and follow the treatment advice</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── UPLOAD + TIPS ────────────────────────────────────────────────
st.markdown('<div class="content-area">', unsafe_allow_html=True)

up_col, tip_col = st.columns([1, 1], gap="large")

with up_col:
    st.markdown("""
    <div class="white-card">
        <div class="card-title">📤 Upload Leaf Image</div>
        <div style="font-size:0.85rem;color:#4b5563;margin-bottom:0.75rem;line-height:1.5;">
            Select a clear photo of your plant leaf.<br>
            Click <strong style="color:#166534;">Browse files</strong> below or drag and drop your image directly.
        </div>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Choose leaf image — JPG, JPEG or PNG",
        type=["jpg","jpeg","png"],
        label_visibility="visible"
    )

    # Fix 2: Clear result when image is removed
    if not uploaded_file:
        st.session_state.result = None

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        disp  = image.copy()
        disp.thumbnail((360, 260), Image.LANCZOS)
        st.markdown('<div class="img-preview">', unsafe_allow_html=True)
        st.image(disp, width=disp.width)
        st.markdown('</div>', unsafe_allow_html=True)
        kb = len(uploaded_file.getvalue())/1024
        st.markdown(f"""
        <div class="img-meta">
            📷 &nbsp;{uploaded_file.name[:38]} &nbsp;·&nbsp; {kb:.1f} KB &nbsp;·&nbsp; {image.width}×{image.height}px
        </div>""", unsafe_allow_html=True)
        # Analyse button right below image in same column
        analyse = st.button("🔍  Analyse This Leaf", use_container_width=True)
    else:
        analyse = False

with tip_col:
    st.markdown("""
    <div class="white-card">
        <div class="card-title">💡 Tips for a Good Photo</div>
        <div class="tip-row">
            <div class="tip-icon-box">☀️</div>
            <div class="tip-txt"><strong>Use natural daylight</strong> — take photos outdoors or near a window</div>
        </div>
        <div class="tip-row">
            <div class="tip-icon-box">🎯</div>
            <div class="tip-txt"><strong>Keep it in focus</strong> — ensure the leaf is sharp and clearly visible</div>
        </div>
        <div class="tip-row">
            <div class="tip-icon-box">🍃</div>
            <div class="tip-txt"><strong>One leaf at a time</strong> — a single leaf in the frame gives the best result</div>
        </div>
        <div class="tip-row">
            <div class="tip-icon-box">🚫</div>
            <div class="tip-txt"><strong>Avoid shadows</strong> — shadows or reflections can reduce accuracy</div>
        </div>
        <div class="tip-row">
            <div class="tip-icon-box">📐</div>
            <div class="tip-txt"><strong>Fill the frame</strong> — the leaf should take up most of the photo</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ── RUN PREDICTION ───────────────────────────────────────────────
if uploaded_file and analyse:
    with st.spinner("🔬 Analysing your leaf image..."):
        time.sleep(0.5)
        img_r = image.resize((224, 224))
        arr = np.array(img_r).astype(np.float32) / 255.0
        arr = np.expand_dims(arr, axis=0)
        input_name = session.get_inputs()[0].name
        preds  = session.run(None, {input_name: arr})
        output = preds[0][0]
        idx    = np.argmax(output)
        conf   = output[idx] * 100
        dname  = class_names[idx]
        rem   = remedies.get(dname, "Please consult your local agricultural expert.")
        ddisp = dname.replace("___"," — ").replace("_"," ")
        ok    = "healthy" in dname.lower()
        st.session_state.result = (ok, ddisp, conf, rem)

# ── RESULTS ─────────────────────────────────────────────────────
if st.session_state.result:
    ok, ddisp, conf, rem = st.session_state.result
    icon     = "✅" if ok else "⚠️"
    res_cls  = "res-healthy" if ok else "res-disease"
    pill_cls = "res-pill-h" if ok else "res-pill-d"
    pill_txt = "Plant is Healthy" if ok else "Disease Detected"

    st.markdown('<div class="result-area">', unsafe_allow_html=True)
    st.markdown('<div style="font-family:Playfair Display,serif;font-size:1.3rem;font-weight:700;color:#14532d;margin-bottom:1.25rem;">📊 Detection Results</div>', unsafe_allow_html=True)

    r1, r2 = st.columns([1, 1], gap="large")

    with r1:
        st.markdown(f"""
        <div class="result-card">
            <div class="result-card-head">🔬 AI Diagnosis</div>
            <div class="result-card-body">
                <div class="{res_cls}">
                    <div class="res-icon">{icon}</div>
                    <div>
                        <div class="{pill_cls}">{pill_txt}</div>
                        <div class="res-name">{ddisp}</div>
                    </div>
                </div>
                <div class="conf-row">
                    <span class="conf-lbl">Model Confidence</span>
                    <span class="conf-val">{conf:.1f}%</span>
                </div>
                <div class="conf-bg">
                    <div class="conf-bar" style="width:{conf:.1f}%"></div>
                </div>
                <div class="again-box">
                    <div style="font-size:1.5rem;flex-shrink:0;">🔄</div>
                    <div>
                        <div class="again-t">Check another leaf?</div>
                        <div class="again-s">Remove the current image and upload a new one above</div>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with r2:
        st.markdown(f"""
        <div class="result-card">
            <div class="result-card-head">💊 Recommended Treatment</div>
            <div class="result-card-body">
                <div class="remedy-box">
                    <div class="remedy-lbl">🌿 What to Do</div>
                    <div class="remedy-body">{rem}</div>
                </div>
                <div class="note-box">
                    <div class="note-lbl">⚠️ Important Note</div>
                    <div class="note-text">
                        This is an AI-based prediction. For severe cases or when in doubt,
                        please consult a qualified agricultural expert or your local
                        agricultural extension officer for professional advice.
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

elif uploaded_file and not st.session_state.result:
    st.markdown("""
    <div style="padding:1.5rem 3rem 0;">
        <div class="awaiting">
            <div class="aw-icon">🔬</div>
            <div class="aw-title">Ready to Analyse</div>
            <div class="aw-sub">Click <strong>Analyse This Leaf</strong> above<br>to detect diseases in your image</div>
        </div>
    </div>""", unsafe_allow_html=True)

# ── SUPPORTED CROPS ─────────────────────────────────────────────
st.markdown("""
<div class="crops-area">
    <div class="crops-head">🌾 Supported Crops</div>
    <div class="crops-sub">CropGuard detects diseases across these 15 crop varieties</div>
    <div class="crops-grid">
        <div class="crop-tile"><span class="crop-tile-icon">🍎</span>Apple</div>
        <div class="crop-tile"><span class="crop-tile-icon">🫐</span>Blueberry</div>
        <div class="crop-tile"><span class="crop-tile-icon">🍒</span>Cherry</div>
        <div class="crop-tile"><span class="crop-tile-icon">🌽</span>Corn</div>
        <div class="crop-tile"><span class="crop-tile-icon">🍇</span>Grape</div>
        <div class="crop-tile"><span class="crop-tile-icon">🍊</span>Orange</div>
        <div class="crop-tile"><span class="crop-tile-icon">🍑</span>Peach</div>
        <div class="crop-tile"><span class="crop-tile-icon">🫑</span>Bell Pepper</div>
        <div class="crop-tile"><span class="crop-tile-icon">🥔</span>Potato</div>
        <div class="crop-tile"><span class="crop-tile-icon">🍓</span>Raspberry</div>
        <div class="crop-tile"><span class="crop-tile-icon">🌾</span>Rice</div>
        <div class="crop-tile"><span class="crop-tile-icon">🫘</span>Soybean</div>
        <div class="crop-tile"><span class="crop-tile-icon">🎃</span>Squash</div>
        <div class="crop-tile"><span class="crop-tile-icon">🍓</span>Strawberry</div>
        <div class="crop-tile"><span class="crop-tile-icon">🍅</span>Tomato</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── FOOTER ───────────────────────────────────────────────────────
st.markdown("""
<div class="footer-bar">
    <div>
        <div class="footer-brand">🌿 CropGuard</div>
        <div class="footer-sub">AI-Powered Crop Disease Detection System</div>
        <div class="footer-sub">Developed by Arun Krishnan G. &nbsp;·&nbsp; ICTAK Academy, Kerala, India</div>
    </div>
    <div class="footer-r">
        Powered by MobileNetV2 + Transfer Learning<br>
        TensorFlow / Keras &nbsp;·&nbsp; Streamlit
    </div>
</div>
""", unsafe_allow_html=True)
