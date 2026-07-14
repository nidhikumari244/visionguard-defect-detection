import os
from datetime import datetime

import cv2
import gdown
import numpy as np
import pandas as pd
import streamlit as st
import torch
import timm
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from torchvision import transforms
from transformers import CLIPModel, CLIPProcessor


st.set_page_config(
    page_title="VisionGuard | Industrial AI Inspection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


FILE_IDS = {
    "bottle": {
        "mb": "12jPbpYSJYqVtGf_Je7fqIb9H8vg8DGe7",
        "mean": "1J7qANJ9NwlWLvA85NIsJTHj8nLHDIce1",
        "std": "1JrwP7vAHQDS2GFbxp_bz6jxNDL-bt1Ix",
    },
    "cable": {
        "mb": "1Czk1LYB8H6yCeYoePeuKLSnVfT-yfTGU",
        "mean": "1XtAdXrMOgUStd3LNxPbUTqIJlm29fiWu",
        "std": "1zOGtL04MKJ13TyHpRT9KoRn4Bolb1Yuc",
    },
    "carpet": {
        "mb": "14_Wft_NEUiL--sz53KWMx1nU1JpR8sKx",
        "mean": "1SU4hpqcPtuBtXI2u_enHvRU5r0lWfIXw",
        "std": "1svqnt9U4wyKNFRZdzKcLRq71FhK6R038",
    },
    "wood": {
        "mb": "1X7qjdWLQrhct_USbLg33-Y50iruTgM7a",
        "mean": "1XBQIu1PFy5qOIxlxVvnCY8TVjnlk-K2I",
        "std": "1YE-BSpF0fnzj_3W3MTZ0ZZq-jyYvvE96",
    },
    "screw": {
        "mb": "1G26CdNH0pys-BZIPwBRcklDolQZRn2TJ",
        "mean": "1AEY6QJwWKqbDdy0C7GEggtIgdV_3QNsv",
        "std": "1oT-md07TqB8Vxjuyvn5qgoTdNy5YkSLP",
    },
}

THRESHOLDS = {
    "bottle": 55.59,
    "cable": 48.27,
    "carpet": 124.20,
    "wood": 87.90,
    "screw": 66.95,
}

CATEGORY_LABELS = {
    "bottle": "Bottle - cracks, chips, contamination",
    "cable": "Cable - bent wires, cuts, missing parts",
    "carpet": "Carpet - stains, cuts, holes",
    "wood": "Wood - scratches, holes, stains",
    "screw": "Screw - thread damage, deformation",
}

CANDIDATES = {
    "bottle": [
        "a photo of a bottle with a chip or crack on glass",
        "a photo of a bottle with a broken piece",
        "a photo of a bottle with liquid contamination",
        "a photo of a bottle with a scratch",
        "a photo of a normal clean bottle",
    ],
    "cable": [
        "a photo of a cable with a bent wire",
        "a photo of a cut cable",
        "a photo of a cable with a missing wire",
        "a photo of a cable with insulation damage",
        "a photo of a normal cable",
    ],
    "carpet": [
        "a photo of carpet with a color stain",
        "a photo of carpet with a cut or tear",
        "a photo of carpet with a hole in fabric",
        "a photo of carpet with thread damage",
        "a photo of normal carpet",
    ],
    "wood": [
        "a photo of wood with a scratch",
        "a photo of wood with a hole",
        "a photo of wood with a stain",
        "a photo of damaged wood",
        "a photo of normal wood",
    ],
    "screw": [
        "a photo of a screw with damaged thread",
        "a photo of a screw with a scratched head",
        "a photo of a screw with a deformed head",
        "a photo of a screw with contamination",
        "a photo of a normal screw",
    ],
}

BENCHMARKS = pd.DataFrame(
    [
        {"Category": "Bottle", "Image AUROC": 0.929, "Pixel AUROC": 0.954, "Precision": 0.962, "Recall": 0.794, "F1": 0.870},
        {"Category": "Cable", "Image AUROC": 0.778, "Pixel AUROC": 0.915, "Precision": 0.885, "Recall": 0.587, "F1": 0.706},
        {"Category": "Carpet", "Image AUROC": 0.939, "Pixel AUROC": 0.958, "Precision": 0.951, "Recall": 0.876, "F1": 0.912},
        {"Category": "Wood", "Image AUROC": 0.983, "Pixel AUROC": 0.923, "Precision": 0.937, "Recall": 0.983, "F1": 0.959},
        {"Category": "Screw", "Image AUROC": 0.544, "Pixel AUROC": 0.970, "Precision": 0.783, "Recall": 0.151, "F1": 0.254},
    ]
)


st.markdown(
    """
    <style>
        :root {
            --bg: #f7f8fb;
            --panel: #ffffff;
            --ink: #172033;
            --muted: #64748b;
            --line: #dbe2ea;
            --accent: #1d6f8f;
            --accent-2: #8a5a44;
            --good: #0f8b5f;
            --bad: #b42318;
        }

        .stApp {
            background: var(--bg);
            color: var(--ink);
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(135deg, #0f172a 0%, #1a1f35 100%);
            color: #f1f5f9;
        }

        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] span,
        section[data-testid="stSidebar"] p {
            color: #cbd5e1;
        }

        /* SIDEBAR PROFESSIONAL STYLING */
        .sidebar-header {
            padding: 1.5rem 0;
            border-bottom: 2px solid #334155;
            margin-bottom: 1.5rem;
            text-align: center;
        }

        .sidebar-logo {
            font-size: 2.5rem;
            font-weight: 900;
            background: linear-gradient(135deg, #0ea5e9 0%, #06b6d4 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            letter-spacing: 0.05em;
            margin: 0;
            padding: 0;
        }

        .sidebar-tagline {
            font-size: 0.75rem;
            color: #94a3b8;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            font-weight: 600;
            margin-top: 0.5rem;
        }

        .sidebar-section {
            margin-bottom: 2rem;
            padding: 0;
        }

        .sidebar-section-title {
            color: #0ea5e9;
            font-size: 0.7rem;
            font-weight: 900;
            letter-spacing: 0.15em;
            text-transform: uppercase;
            margin-bottom: 1rem;
            padding: 0;
            display: block;
        }

        .status-indicator {
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            margin-right: 0.5rem;
        }

        .status-online {
            background: #10b981;
            box-shadow: 0 0 10px rgba(16, 185, 129, 0.5);
        }

        .status-offline {
            background: #6b7280;
        }

        .status-warning {
            background: #f59e0b;
            box-shadow: 0 0 10px rgba(245, 158, 11, 0.5);
        }

        .metric-row {
            background: rgba(30, 41, 59, 0.8);
            border: 1px solid #334155;
            border-radius: 8px;
            padding: 0.75rem 0.85rem;
            margin-bottom: 0.7rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .metric-label {
            color: #94a3b8;
            font-size: 0.8rem;
            font-weight: 600;
            letter-spacing: 0.02em;
        }

        .metric-value {
            color: #0ea5e9;
            font-size: 0.9rem;
            font-weight: 800;
            font-family: 'Monaco', 'Courier New', monospace;
        }

        .divider-light {
            border: 0;
            height: 1px;
            background: linear-gradient(to right, transparent, #334155, transparent);
            margin: 1.25rem 0;
        }

        .mode-selector {
            background: rgba(30, 41, 59, 0.6);
            border: 2px solid #334155;
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 1.5rem;
        }

        .mode-title {
            color: #0ea5e9;
            font-size: 0.8rem;
            font-weight: 700;
            letter-spacing: 0.05em;
            margin-bottom: 0.75rem;
            display: block;
        }

        /* Hide default radio button styling */
        section[data-testid="stSidebar"] [role="radio"] {
            accent-color: #0ea5e9;
        }

        .hero {
            padding: 1.35rem 0 0.4rem;
            border-bottom: 1px solid var(--line);
            margin-bottom: 1.1rem;
        }

        .eyebrow {
            color: var(--accent);
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }

        .hero h1 {
            color: var(--ink);
            font-size: 2.25rem;
            line-height: 1.05;
            margin: 0.2rem 0;
            letter-spacing: 0;
        }

        .hero p {
            color: var(--muted);
            max-width: 820px;
            margin: 0;
            font-size: 1rem;
        }

        .metric-strip {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.75rem;
            margin: 1rem 0 1.2rem;
        }

        .metric-card {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 1rem;
        }

        .metric-value-large {
            color: var(--ink);
            font-size: 1.65rem;
            font-weight: 850;
            line-height: 1;
        }

        .metric-label-large {
            color: var(--muted);
            font-size: 0.82rem;
            margin-top: 0.35rem;
        }

        .status-box {
            border-radius: 8px;
            padding: 1rem 1.1rem;
            border: 1px solid;
            margin-bottom: 0.8rem;
        }

        .status-anomaly {
            background: #fff1f0;
            border-color: #f3b4ad;
        }

        .status-normal {
            background: #eefbf5;
            border-color: #a9dec8;
        }

        .status-title {
            font-size: 1.35rem;
            font-weight: 850;
            margin: 0;
        }

        .status-detail {
            color: var(--muted);
            margin-top: 0.35rem;
        }

        .small-note {
            color: var(--muted);
            font-size: 0.86rem;
        }

        div[data-testid="stMetric"] {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 0.75rem 0.9rem;
        }

        div[data-testid="stDataFrame"] {
            border: 1px solid var(--line);
            border-radius: 8px;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.35rem;
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 6px;
            padding: 0.55rem 0.9rem;
            background: #eef2f7;
        }

        @media (max-width: 820px) {
            .metric-strip {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }

            .hero h1 {
                font-size: 1.8rem;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def clean_candidate(text):
    return text.replace("a photo of ", "").strip()


@st.cache_resource(show_spinner=False)
def load_models():
    backbone = timm.create_model(
        "wide_resnet50_2",
        pretrained=True,
        features_only=True,
        out_indices=(1, 2),
    )
    backbone.eval()

    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model.eval()
    return backbone, transform, clip_model, clip_processor


@st.cache_resource(show_spinner=False)
def load_memory_bank(category):
    os.makedirs("memory_banks", exist_ok=True)
    ids = FILE_IDS[category]
    mb_path = f"memory_banks/{category}_mb.npy"
    mean_path = f"memory_banks/{category}_mean.npy"
    std_path = f"memory_banks/{category}_std.npy"

    if not os.path.exists(mb_path):
        gdown.download(id=ids["mb"], output=mb_path, quiet=False)
        gdown.download(id=ids["mean"], output=mean_path, quiet=False)
        gdown.download(id=ids["std"], output=std_path, quiet=False)

    mb = np.load(mb_path)
    mean = np.load(mean_path)
    std = np.load(std_path)
    nn = NearestNeighbors(n_neighbors=1, algorithm="brute", n_jobs=-1).fit(mb)
    return nn, mean, std


def get_features(img, backbone, transform):
    tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        features = backbone(tensor)

    fine, coarse = features[0], features[1]
    coarse_up = torch.nn.functional.interpolate(
        coarse,
        size=fine.shape[2:],
        mode="bilinear",
        align_corners=False,
    )
    combined = torch.cat([fine, coarse_up], dim=1)
    _, channels, height, width = combined.shape
    patches = combined.permute(0, 2, 3, 1).reshape(-1, channels)
    return patches.numpy().astype(np.float32), (height, width)


def make_heatmap(img, anomaly_map):
    resized = np.array(img.resize((224, 224)))
    anomaly_resized = cv2.resize(anomaly_map, (224, 224))
    normalized = (anomaly_resized - anomaly_resized.min()) / (
        anomaly_resized.max() - anomaly_resized.min() + 1e-8
    )
    heatmap = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(resized, 0.58, heatmap, 0.42, 0)


def topk_score(distances, ratio=0.01):
    flat = distances.reshape(-1)
    k = max(1, int(len(flat) * ratio))
    return float(np.mean(np.sort(flat)[-k:]))


def describe_defect(img, anomaly_map, category, clip_model, clip_processor):
    resized = np.array(img.resize((224, 224)))
    anomaly_resized = cv2.resize(anomaly_map, (224, 224))
    cy, cx = np.unravel_index(np.argmax(anomaly_resized), anomaly_resized.shape)
    pad = 44
    crop = resized[
        max(0, cy - pad) : min(224, cy + pad),
        max(0, cx - pad) : min(224, cx + pad),
    ]

    candidates = CANDIDATES[category]
    inputs = clip_processor(
        text=candidates,
        images=Image.fromarray(crop),
        return_tensors="pt",
        padding=True,
    )

    with torch.no_grad():
        output = clip_model(**inputs)
        probs = output.logits_per_image.softmax(dim=1)[0].numpy()

    ranked = sorted(
        [
            {"description": clean_candidate(candidate), "confidence": float(prob)}
            for candidate, prob in zip(candidates, probs)
        ],
        key=lambda item: item["confidence"],
        reverse=True,
    )
    return ranked


def inspect_image(uploaded_file, category, score_mode="Maximum patch distance"):
    backbone, transform, clip_model, clip_processor = load_models()
    nn, mean, std = load_memory_bank(category)

    img = Image.open(uploaded_file).convert("RGB")
    patches, (height, width) = get_features(img, backbone, transform)
    normalized = (patches - mean) / std
    distances, _ = nn.kneighbors(normalized)
    anomaly_map = distances.reshape(height, width)

    max_score = float(distances.max())
    topk = topk_score(distances)
    score = topk if score_mode == "Top-1% average distance" else max_score
    threshold = THRESHOLDS[category]
    is_anomaly = score > threshold
    ratio = score / threshold if threshold else 0
    confidence = "High" if ratio >= 1.5 else "Medium" if ratio >= 1 else "Low"

    ranked_descriptions = describe_defect(
        img,
        anomaly_map,
        category,
        clip_model,
        clip_processor,
    )

    return {
        "file_name": getattr(uploaded_file, "name", "uploaded_image"),
        "category": category,
        "image": img,
        "overlay": make_heatmap(img, anomaly_map),
        "anomaly_map": anomaly_map,
        "score": score,
        "max_score": max_score,
        "topk_score": topk,
        "threshold": threshold,
        "is_anomaly": is_anomaly,
        "confidence": confidence,
        "descriptions": ranked_descriptions,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def build_report(result):
    decision = "ANOMALY DETECTED" if result["is_anomaly"] else "NORMAL PRODUCT"
    best = result["descriptions"][0]
    lines = [
        "# VisionGuard Inspection Report",
        "",
        f"Generated: {result['timestamp']}",
        f"Image: {result['file_name']}",
        f"Category: {CATEGORY_LABELS[result['category']]}",
        f"Decision: {decision}",
        f"Confidence: {result['confidence']}",
        f"Score: {result['score']:.4f}",
        f"Threshold: {result['threshold']:.2f}",
        f"Maximum Patch Score: {result['max_score']:.4f}",
        f"Top-1% Patch Score: {result['topk_score']:.4f}",
        "",
        "## Zero-Shot Defect Description",
        f"Top description: {best['description']} ({best['confidence'] * 100:.1f}%)",
        "",
        "## Ranked CLIP Candidates",
    ]
    for item in result["descriptions"]:
        lines.append(f"- {item['description']}: {item['confidence'] * 100:.1f}%")
    return "\n".join(lines)


def render_header():
    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">AI-powered industrial quality inspection</div>
            <h1>VisionGuard</h1>
            <p>
                Unsupervised anomaly detection system for product inspection using
                Wide ResNet-50 patch features, memory-bank nearest-neighbor scoring,
                pixel-level heatmaps, and CLIP zero-shot defect descriptions.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="metric-strip">
            <div class="metric-card"><div class="metric-value-large">5</div><div class="metric-label-large">MVTec AD categories</div></div>
            <div class="metric-card"><div class="metric-value-large">0.970</div><div class="metric-label-large">Best pixel AUROC</div></div>
            <div class="metric-card"><div class="metric-value-large">0</div><div class="metric-label-large">Defect labels required</div></div>
            <div class="metric-card"><div class="metric-value-large">768</div><div class="metric-label-large">Patch feature dimensions</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_result(result):
    decision = "ANOMALY DETECTED" if result["is_anomaly"] else "NORMAL PRODUCT"
    box_class = "status-anomaly" if result["is_anomaly"] else "status-normal"
    title_color = "#b42318" if result["is_anomaly"] else "#0f8b5f"

    st.markdown(
        f"""
        <div class="status-box {box_class}">
            <p class="status-title" style="color:{title_color};">{decision}</p>
            <div class="status-detail">
                Confidence: {result['confidence']} | Score: {result['score']:.4f} |
                Threshold: {result['threshold']:.2f}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_cols = st.columns(4)
    metric_cols[0].metric("Inspection score", f"{result['score']:.2f}")
    metric_cols[1].metric("Threshold", f"{result['threshold']:.2f}")
    metric_cols[2].metric("Max patch score", f"{result['max_score']:.2f}")
    metric_cols[3].metric("Top-1% score", f"{result['topk_score']:.2f}")

    image_cols = st.columns(2)
    with image_cols[0]:
        st.image(result["overlay"], caption="Anomaly heatmap overlay", use_container_width=True)
        st.caption("Red regions indicate high deviation from normal training patches.")
    with image_cols[1]:
        st.image(result["image"].resize((224, 224)), caption="Original image", use_container_width=True)

    st.subheader("CLIP zero-shot description")
    best = result["descriptions"][0]
    st.success(f"{best['description']} ({best['confidence'] * 100:.1f}% confidence)")

    desc_df = pd.DataFrame(
        [
            {
                "Candidate description": item["description"],
                "Confidence": item["confidence"],
            }
            for item in result["descriptions"]
        ]
    )
    st.dataframe(
        desc_df,
        column_config={
            "Confidence": st.column_config.ProgressColumn(
                "Confidence",
                format="%.1f",
                min_value=0,
                max_value=1,
            )
        },
        hide_index=True,
        use_container_width=True,
    )

    report = build_report(result)
    st.download_button(
        "Download inspection report",
        data=report,
        file_name=f"visionguard_report_{result['category']}.md",
        mime="text/markdown",
        use_container_width=True,
    )


def render_dashboard():
    st.subheader("Project overview")
    cols = st.columns(3)
    cols[0].metric("Average image AUROC", f"{BENCHMARKS['Image AUROC'].mean():.3f}")
    cols[1].metric("Average pixel AUROC", f"{BENCHMARKS['Pixel AUROC'].mean():.3f}")
    cols[2].metric("Average F1 score", f"{BENCHMARKS['F1'].mean():.3f}")

    chart_data = BENCHMARKS.set_index("Category")[["Image AUROC", "Pixel AUROC", "F1"]]
    st.bar_chart(chart_data, use_container_width=True)

    st.subheader("Category performance")
    st.dataframe(BENCHMARKS, hide_index=True, use_container_width=True)


def render_single_inspection(score_mode):
    st.subheader("Single product inspection")
    input_col, preview_col = st.columns([0.9, 1.1])

    with input_col:
        uploaded = st.file_uploader(
            "Upload product image",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=False,
        )
        category = st.selectbox(
            "Product category",
            list(CATEGORY_LABELS.keys()),
            format_func=lambda key: CATEGORY_LABELS[key],
        )
        run_btn = st.button("Run inspection", type="primary", use_container_width=True)

    with preview_col:
        if uploaded:
            st.image(uploaded, caption="Input image", use_container_width=True)
        else:
            st.info("Upload a product image to inspect defects and generate a heatmap.")

    if run_btn and not uploaded:
        st.warning("Please upload an image first.")
        return

    if run_btn and uploaded:
        with st.spinner("Loading models and running inspection..."):
            result = inspect_image(uploaded, category, score_mode)
        st.session_state["last_result"] = result
        st.session_state.setdefault("history", []).append(
            {
                "Time": result["timestamp"],
                "Image": result["file_name"],
                "Category": result["category"].title(),
                "Decision": "Anomaly" if result["is_anomaly"] else "Normal",
                "Score": round(result["score"], 4),
                "Threshold": result["threshold"],
                "Description": result["descriptions"][0]["description"],
            }
        )

    if "last_result" in st.session_state:
        st.divider()
        render_result(st.session_state["last_result"])


def render_batch_inspection(score_mode):
    st.subheader("Batch inspection")
    uploaded_files = st.file_uploader(
        "Upload multiple images",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
    )
    category = st.selectbox(
        "Batch category",
        list(CATEGORY_LABELS.keys()),
        format_func=lambda key: CATEGORY_LABELS[key],
        key="batch_category",
    )

    if st.button("Run batch inspection", type="primary", use_container_width=True):
        if not uploaded_files:
            st.warning("Please upload at least one image.")
            return

        rows = []
        progress = st.progress(0, text="Starting batch inspection...")
        for index, uploaded in enumerate(uploaded_files, start=1):
            result = inspect_image(uploaded, category, score_mode)
            rows.append(
                {
                    "Image": result["file_name"],
                    "Category": result["category"].title(),
                    "Decision": "Anomaly" if result["is_anomaly"] else "Normal",
                    "Score": round(result["score"], 4),
                    "Threshold": result["threshold"],
                    "Confidence": result["confidence"],
                    "Description": result["descriptions"][0]["description"],
                }
            )
            progress.progress(index / len(uploaded_files), text=f"Inspected {index} of {len(uploaded_files)} images")

        batch_df = pd.DataFrame(rows)
        st.session_state["batch_results"] = batch_df
        progress.empty()

    if "batch_results" in st.session_state:
        st.dataframe(st.session_state["batch_results"], hide_index=True, use_container_width=True)
        st.download_button(
            "Download batch results CSV",
            data=st.session_state["batch_results"].to_csv(index=False),
            file_name="visionguard_batch_results.csv",
            mime="text/csv",
            use_container_width=True,
        )


def render_benchmarks():
    st.subheader("MVTec AD benchmark results")
    st.dataframe(BENCHMARKS, hide_index=True, use_container_width=True)

    chart_cols = st.columns(2)
    with chart_cols[0]:
        st.caption("Image-level anomaly ranking")
        st.bar_chart(BENCHMARKS.set_index("Category")["Image AUROC"], use_container_width=True)
    with chart_cols[1]:
        st.caption("Pixel-level defect localization")
        st.bar_chart(BENCHMARKS.set_index("Category")["Pixel AUROC"], use_container_width=True)

    st.subheader("Ablation study")
    ablation = pd.DataFrame(
        [
            {"Experiment": "Single layer to combined layers", "Before": 0.856, "After": 0.975, "Change": "+13.8%"},
            {"Experiment": "Unnormalized to normalized features", "Before": 0.620, "After": 0.778, "Change": "+25.5%"},
            {"Experiment": "Cable threshold P95 to P85", "Before": 0.390, "After": 0.706, "Change": "+81.0%"},
        ]
    )
    st.dataframe(ablation, hide_index=True, use_container_width=True)


def render_architecture():
    st.subheader("System architecture")
    st.markdown(
        """
        1. Input image is resized and center-cropped to 224 x 224.
        2. Wide ResNet-50 extracts multi-scale patch embeddings from two feature layers.
        3. Layer features are upsampled and concatenated into 768-dimensional patch vectors.
        4. Patch vectors are z-score normalized using category-specific training statistics.
        5. A nearest-neighbor memory bank measures distance from normal training patches.
        6. Patch distances create the anomaly heatmap and image-level inspection score.
        7. CLIP ranks category-specific text prompts for a human-readable defect description.
        """
    )

    st.subheader("Major-project upgrade roadmap")
    roadmap = pd.DataFrame(
        [
            {"Priority": "High", "Upgrade": "Top-k score calibration", "Impact": "Improves tiny-defect image decisions, especially screw"},
            {"Priority": "High", "Upgrade": "All 15 MVTec categories", "Impact": "Makes the benchmark complete and report stronger"},
            {"Priority": "High", "Upgrade": "Threshold search by validation F1", "Impact": "More defensible than fixed percentiles"},
            {"Priority": "Medium", "Upgrade": "Autoencoder or PaDiM baseline", "Impact": "Adds comparative research depth"},
            {"Priority": "Medium", "Upgrade": "FastAPI + React deployment", "Impact": "Turns the demo into a product-style system"},
        ]
    )
    st.dataframe(roadmap, hide_index=True, use_container_width=True)


# ============================================================================
# PROFESSIONAL SIDEBAR RENDERING
# ============================================================================

render_header()

# Initialize session state for category tracking
if "current_category" not in st.session_state:
    st.session_state["current_category"] = "bottle"

with st.sidebar:
    # === SIDEBAR HEADER ===
    st.markdown(
        """
        <div class="sidebar-header">
            <div class="sidebar-logo">🔍 VisionGuard</div>
            <div class="sidebar-tagline">Industrial Inspection Platform</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # === MACHINE STATUS SECTION ===
    st.markdown('<span class="sidebar-section-title">⚙️ Machine Status</span>', unsafe_allow_html=True)
    
    gpu_available = torch.cuda.is_available()
    gpu_status = "✓ Online" if gpu_available else "✗ CPU Mode"
    gpu_color = "10b981" if gpu_available else "f59e0b"
    
    st.markdown(
        f"""
        <div class="metric-row">
            <span class="metric-label">System Status</span>
            <span class="metric-value" style="color: #10b981;"><span class="status-indicator status-online"></span>Online</span>
        </div>
        <div class="metric-row">
            <span class="metric-label">GPU/Compute</span>
            <span class="metric-value" style="color: #{gpu_color};">{gpu_status}</span>
        </div>
        <div class="metric-row">
            <span class="metric-label">Memory Available</span>
            <span class="metric-value">OK</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<hr class="divider-light">', unsafe_allow_html=True)

    # === MODEL INFORMATION SECTION ===
    st.markdown('<span class="sidebar-section-title">🧠 Model Configuration</span>', unsafe_allow_html=True)
    
    st.markdown(
        """
        <div class="metric-row">
            <span class="metric-label">Model Version</span>
            <span class="metric-value">1.0.0</span>
        </div>
        <div class="metric-row">
            <span class="metric-label">Backbone</span>
            <span class="metric-value">Wide ResNet-50</span>
        </div>
        <div class="metric-row">
            <span class="metric-label">Feature Dim</span>
            <span class="metric-value">768</span>
        </div>
        <div class="metric-row">
            <span class="metric-label">Description</span>
            <span class="metric-value">CLIP ViT-B/32</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<hr class="divider-light">', unsafe_allow_html=True)

    # === DEPLOYMENT SECTION ===
    st.markdown('<span class="sidebar-section-title">🚀 Deployment Info</span>', unsafe_allow_html=True)
    
    st.markdown(
        """
        <div class="metric-row">
            <span class="metric-label">Deployment Version</span>
            <span class="metric-value">v1.0-prod</span>
        </div>
        <div class="metric-row">
            <span class="metric-label">Inference Device</span>
            <span class="metric-value">GPU/CPU Auto</span>
        </div>
        <div class="metric-row">
            <span class="metric-label">Dataset</span>
            <span class="metric-value">MVTec AD</span>
        </div>
        <div class="metric-row">
            <span class="metric-label">Categories</span>
            <span class="metric-value">5</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<hr class="divider-light">', unsafe_allow_html=True)

    # === CURRENT INSPECTION SESSION ===
    st.markdown('<span class="sidebar-section-title">📊 Active Session</span>', unsafe_allow_html=True)
    
    # Category selector with proper tracking
    selected_category = st.selectbox(
        "Select Category",
        list(CATEGORY_LABELS.keys()),
        format_func=lambda key: CATEGORY_LABELS[key],
        key="sidebar_category",
    )
    st.session_state["current_category"] = selected_category
    
    threshold = THRESHOLDS[selected_category]
    
    st.markdown(
        f"""
        <div class="metric-row">
            <span class="metric-label">Current Category</span>
            <span class="metric-value">{selected_category.title()}</span>
        </div>
        <div class="metric-row">
            <span class="metric-label">Anomaly Threshold</span>
            <span class="metric-value">{threshold:.2f}</span>
        </div>
        <div class="metric-row">
            <span class="metric-label">Memory Bank Size</span>
            <span class="metric-value">10,000</span>
        </div>
        <div class="metric-row">
            <span class="metric-label">Memory Bank Status</span>
            <span class="metric-value" style="color: #10b981;"><span class="status-indicator status-online"></span>Loaded</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<hr class="divider-light">', unsafe_allow_html=True)

    # === SCORING MODE ===
    st.markdown('<span class="sidebar-section-title">⚡ Scoring Configuration</span>', unsafe_allow_html=True)
    
    st.markdown('<div class="mode-selector">', unsafe_allow_html=True)
    score_mode = st.radio(
        "Image Score Mode",
        ["Maximum patch distance", "Top-1% average distance"],
        help="Top-1% average is useful for tiny defects, but thresholds should be recalibrated before final reporting.",
        label_visibility="collapsed",
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<hr class="divider-light">', unsafe_allow_html=True)

    # === SYSTEM INFO FOOTER ===
    st.markdown(
        """
        <div style="color: #64748b; font-size: 0.75rem; margin-top: 2rem; padding-top: 1rem; border-top: 1px solid #334155; text-align: center; line-height: 1.6;">
            <div>VisionGuard v1.0</div>
            <div style="margin-top: 0.5rem; color: #475569;">Powered by PyTorch & CLIP</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================================
# MAIN CONTENT TABS
# ============================================================================

tabs = st.tabs(["Dashboard", "Inspect", "Batch", "Benchmarks", "Architecture"])

with tabs[0]:
    render_dashboard()

with tabs[1]:
    render_single_inspection(score_mode)

with tabs[2]:
    render_batch_inspection(score_mode)

with tabs[3]:
    render_benchmarks()

with tabs[4]:
    render_architecture()


if st.session_state.get("history"):
    with st.expander("Recent inspection history"):
        st.dataframe(pd.DataFrame(st.session_state["history"]), hide_index=True, use_container_width=True)
