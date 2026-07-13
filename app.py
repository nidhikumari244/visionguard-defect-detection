
import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
from sklearn.neighbors import NearestNeighbors
import timm
from torchvision import transforms
from transformers import CLIPModel, CLIPProcessor
import gdown
import os

st.set_page_config(
    page_title="VisionGuard",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main { background-color: #0f172a; }
    .stApp { background-color: #0f172a; color: #f1f5f9; }
    .title {
        font-size: 3em; font-weight: 900; text-align: center;
        background: linear-gradient(90deg, #6366f1, #06b6d4);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .subtitle {
        text-align: center; color: #94a3b8;
        font-size: 1.1em; margin-bottom: 30px;
    }
    .metric-card {
        background: #1e293b; border-radius: 12px;
        padding: 16px; text-align: center;
        border: 1px solid #334155;
    }
    .metric-value {
        font-size: 2em; font-weight: 800;
        color: #6366f1;
    }
    .metric-label { color: #94a3b8; font-size: 0.85em; }
    .result-anomaly {
        background: rgba(239,68,68,0.1);
        border: 1px solid rgba(239,68,68,0.3);
        border-radius: 12px; padding: 20px;
    }
    .result-normal {
        background: rgba(16,185,129,0.1);
        border: 1px solid rgba(16,185,129,0.3);
        border-radius: 12px; padding: 20px;
    }
</style>
""", unsafe_allow_html=True)

FILE_IDS = {
    "bottle": {
        "mb":   "12jPbpYSJYqVtGf_Je7fqIb9H8vg8DGe7",
        "mean": "1J7qANJ9NwlWLvA85NIsJTHj8nLHDIce1",
        "std":  "1JrwP7vAHQDS2GFbxp_bz6jxNDL-bt1Ix"
    },
    "cable": {
        "mb":   "1Czk1LYB8H6yCeYoePeuKLSnVfT-yfTGU",
        "mean": "1XtAdXrMOgUStd3LNxPbUTqIJlm29fiWu",
        "std":  "1zOGtL04MKJ13TyHpRT9KoRn4Bolb1Yuc"
    },
    "carpet": {
        "mb":   "14_Wft_NEUiL--sz53KWMx1nU1JpR8sKx",
        "mean": "1SU4hpqcPtuBtXI2u_enHvRU5r0lWfIXw",
        "std":  "1svqnt9U4wyKNFRZdzKcLRq71FhK6R038"
    },
    "wood": {
        "mb":   "1X7qjdWLQrhct_USbLg33-Y50iruTgM7a",
        "mean": "1XBQIu1PFy5qOIxlxVvnCY8TVjnlk-K2I",
        "std":  "1YE-BSpF0fnzj_3W3MTZ0ZZq-jyYvvE96"
    },
    "screw": {
        "mb":   "1G26CdNH0pys-BZIPwBRcklDolQZRn2TJ",
        "mean": "1AEY6QJwWKqbDdy0C7GEggtIgdV_3QNsv",
        "std":  "1oT-md07TqB8Vxjuyvn5qgoTdNy5YkSLP"
    }
}

THRESHOLDS = {
    "bottle":55.59,"cable":48.27,
    "carpet":124.20,"wood":87.90,"screw":66.95
}

CANDIDATES = {
    "bottle":["a chip or crack on glass","a broken piece","liquid contamination","a scratch","normal clean bottle"],
    "cable":["a bent wire","a cut cable","a missing wire","insulation damage","normal cable"],
    "carpet":["a color stain","a cut or tear","a hole in fabric","thread damage","normal carpet"],
    "wood":["a scratch on wood","a hole in wood","a stain on wood","wood damage","normal wood"],
    "screw":["damaged thread","a scratch on screw","deformed head","contamination","normal screw"]
}

@st.cache_resource
def load_models():
    backbone = timm.create_model(
        "wide_resnet50_2", pretrained=True,
        features_only=True, out_indices=(1,2))
    backbone.eval()
    transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model.eval()
    return backbone, transform, clip_model, clip_processor

@st.cache_resource
def load_memory_bank(category):
    os.makedirs("memory_banks", exist_ok=True)
    ids = FILE_IDS[category]
    mb_path   = f"memory_banks/{category}_mb.npy"
    mean_path = f"memory_banks/{category}_mean.npy"
    std_path  = f"memory_banks/{category}_std.npy"
    if not os.path.exists(mb_path):
        gdown.download(id=ids["mb"],   output=mb_path,   quiet=False)
        gdown.download(id=ids["mean"], output=mean_path, quiet=False)
        gdown.download(id=ids["std"],  output=std_path,  quiet=False)
    mb   = np.load(mb_path)
    mean = np.load(mean_path)
    std  = np.load(std_path)
    nn   = NearestNeighbors(n_neighbors=1, algorithm="brute").fit(mb)
    return nn, mean, std

def get_features(img, backbone, transform):
    t = transform(img).unsqueeze(0)
    with torch.no_grad():
        f = backbone(t)
    f0,f1 = f[0],f[1]
    f1u = torch.nn.functional.interpolate(
        f1,size=f0.shape[2:],mode="bilinear",align_corners=False)
    c = torch.cat([f0,f1u],dim=1)
    b,ch,h,w = c.shape
    return c.permute(0,2,3,1).reshape(
        -1,ch).numpy().astype(np.float32),(h,w)

def make_heatmap(img, amap):
    inp = np.array(img.resize((224,224)))
    am  = cv2.resize(amap,(224,224))
    amn = (am-am.min())/(am.max()-am.min()+1e-8)
    hm  = cv2.applyColorMap(
        (amn*255).astype(np.uint8),cv2.COLORMAP_JET)
    hm  = cv2.cvtColor(hm,cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(inp,0.6,hm,0.4,0)

# Header
st.markdown('<div class="title">🔍 VisionGuard</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-Powered Industrial Defect Detection | Unsupervised Anomaly Detection + CLIP Zero-Shot Description</div>', unsafe_allow_html=True)

# Tech badges
st.markdown("""
<div style="text-align:center;margin-bottom:24px">
    <span style="background:#e0e7ff;color:#3730a3;padding:4px 12px;border-radius:20px;font-size:0.8em;font-weight:600;margin:3px">PyTorch</span>
    <span style="background:#cffafe;color:#0e7490;padding:4px 12px;border-radius:20px;font-size:0.8em;font-weight:600;margin:3px">Wide ResNet-50</span>
    <span style="background:#fef3c7;color:#92400e;padding:4px 12px;border-radius:20px;font-size:0.8em;font-weight:600;margin:3px">CLIP ViT-B/32</span>
    <span style="background:#dcfce7;color:#166534;padding:4px 12px;border-radius:20px;font-size:0.8em;font-weight:600;margin:3px">MVTec AD</span>
</div>
""", unsafe_allow_html=True)

# Stats
col1,col2,col3,col4 = st.columns(4)
with col1:
    st.markdown('<div class="metric-card"><div class="metric-value">5</div><div class="metric-label">Categories</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="metric-card"><div class="metric-value">0.97+</div><div class="metric-label">Best Pixel AUROC</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="metric-card"><div class="metric-value">0</div><div class="metric-label">Defect Labels Needed</div></div>', unsafe_allow_html=True)
with col4:
    st.markdown('<div class="metric-card"><div class="metric-value">768</div><div class="metric-label">Feature Dimensions</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Load models
with st.spinner("Loading AI models..."):
    backbone, transform, clip_model, clip_processor = load_models()

# Input section
st.markdown("### 🔬 Product Inspection")
col_left, col_right = st.columns(2)

with col_left:
    uploaded = st.file_uploader(
        "Upload Product Image",
        type=["png","jpg","jpeg"])
    category = st.selectbox(
        "Product Category",
        ["bottle","cable","carpet","wood","screw"],
        format_func=lambda x: {
            "bottle":"🍾 Bottle — cracks, chips, contamination",
            "cable":"🔌 Cable — bent wires, cuts, missing parts",
            "carpet":"🟫 Carpet — stains, cuts, holes",
            "wood":"🪵 Wood — scratches, holes, stains",
            "screw":"🔩 Screw — thread damage, deformation"
        }[x])
    run_btn = st.button(
        "🔍 Run VisionGuard Inspection",
        type="primary", use_container_width=True)

with col_right:
    if uploaded:
        st.image(uploaded, caption="Uploaded Image", use_column_width=True)

# Run inspection
if run_btn and uploaded:
    with st.spinner(f"Loading {category} memory bank..."):
        nn, mean, std = load_memory_bank(category)

    with st.spinner("Running AI inspection..."):
        img = Image.open(uploaded).convert("RGB")
        patches,(h,w) = get_features(img, backbone, transform)
        patches_norm = (patches-mean)/std
        distances,_ = nn.kneighbors(patches_norm)
        amap = distances.reshape(h,w)
        score = float(distances.max())
        overlay = make_heatmap(img, amap)

        # CLIP
        inp = np.array(img.resize((224,224)))
        am  = cv2.resize(amap,(224,224))
        cy,cx = np.unravel_index(np.argmax(am),am.shape)
        pad=40
        crop = inp[max(0,cy-pad):min(224,cy+pad),
                   max(0,cx-pad):min(224,cx+pad)]
        cands = CANDIDATES[category]
        ins = clip_processor(
            text=cands,images=Image.fromarray(crop),
            return_tensors="pt",padding=True)
        with torch.no_grad():
            out = clip_model(**ins)
            probs = out.logits_per_image.softmax(dim=1)[0]
        top2 = probs.topk(2)
        idx = top2.indices.numpy()
        pvals = top2.values.numpy()
        desc = cands[idx[0]]
        conf = round(float(pvals[0])*100,1)

    thresh = THRESHOLDS[category]
    is_anomaly = score > thresh
    decision = "ANOMALY DETECTED" if is_anomaly else "NORMAL"
    conf_level = "HIGH" if score>thresh*1.5 else "MEDIUM" if score>thresh else "LOW"

    st.markdown("---")
    st.markdown("### 📊 Inspection Results")

    # Decision
    if is_anomaly:
        st.markdown(f'''
        <div class="result-anomaly">
            <h2 style="color:#ef4444;margin:0">⚠️ ANOMALY DETECTED</h2>
            <p style="color:#fca5a5;margin:8px 0 0 0">
                Confidence: {conf_level} | Score: {score:.4f} | Threshold: {thresh:.2f}
            </p>
        </div>''', unsafe_allow_html=True)
    else:
        st.markdown(f'''
        <div class="result-normal">
            <h2 style="color:#10b981;margin:0">✅ NORMAL PRODUCT</h2>
            <p style="color:#6ee7b7;margin:8px 0 0 0">
                Score: {score:.4f} | Threshold: {thresh:.2f}
            </p>
        </div>''', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Images
    c1,c2 = st.columns(2)
    with c1:
        st.image(overlay, caption="🌡️ Anomaly Heatmap", use_column_width=True)
        st.caption("Red = anomalous region · Blue = normal region")
    with c2:
        st.image(img.resize((224,224)), caption="📷 Original", use_column_width=True)

    # CLIP
    st.markdown("### 🧠 CLIP Zero-Shot Description")
    st.info(f"**{desc}** ({conf}% confidence)")
    for i,(c,p) in enumerate(zip(cands, probs.numpy())):
        st.progress(float(p), text=f"{c}: {p*100:.1f}%")

    # Metrics
    st.markdown("### 📈 Benchmark Results")
    st.dataframe({
        "Category":["🍾 Bottle","🔌 Cable","🟫 Carpet","🪵 Wood","🔩 Screw"],
        "Image AUROC":[0.929,0.778,0.939,0.983,0.544],
        "Pixel AUROC":[0.954,0.915,0.958,0.923,0.970],
        "F1 Score":[0.870,0.706,0.912,0.959,0.254]
    }, hide_index=True, use_container_width=True)

elif run_btn and not uploaded:
    st.warning("Please upload an image first.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#94a3b8;font-size:0.85em">
    <strong style="color:#f1f5f9">VisionGuard</strong> ·
    B.Tech CSE Internship Project ·
    ML & Deep Learning for Image Processing<br>
    PyTorch · Wide ResNet-50 · CLIP ViT-B/32 · MVTec AD Benchmark
</div>
""", unsafe_allow_html=True)
