"""Streamlit app: Alzheimer's stage classification from brain MRI (Group 8).
Place this file in the PROJECT ROOT (next to src/, models/, dataset/).
Run locally:  streamlit run streamlit_app.py
Academic demo only - NOT a medical device."""
import sys, time
from pathlib import Path

import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from evaluate_final import load_checkpoint, extract_state, make_model, IMG_SIZE  # reuse project code

CKPT = {"CNN": ROOT / "models/cnn_model.pth", "ResNet18": ROOT / "models/resnet18_model.pth"}
DEFAULT_CLASSES = ["Mild Demented", "Moderate Demented", "Non Demented", "Very Mild Demented"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource(show_spinner="Loading model...")
def get_model(kind):
    ckpt = load_checkpoint(CKPT[kind], device)
    state, meta = extract_state(ckpt)
    classes = list(meta.get("classes") or DEFAULT_CLASSES)
    model = make_model(kind, len(classes))
    model.load_state_dict(state)
    model.to(device).eval()
    return model, classes, bool(meta.get("normalize", False))


def predict(image, kind):
    model, classes, norm = get_model(kind)
    ops = [transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor()]
    if norm:
        ops.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    x = transforms.Compose(ops)(image.convert("RGB")).unsqueeze(0).to(device)
    t = time.perf_counter()
    with torch.no_grad():
        probs = torch.softmax(model(x), 1)[0].cpu()
    ms = (time.perf_counter() - t) * 1000
    return {c: float(p) for c, p in zip(classes, probs)}, ms


st.set_page_config(page_title="Alzheimer's MRI Classifier", page_icon="🧠", layout="centered")
st.title("🧠 Alzheimer's Stage Classification from Brain MRI")
st.caption("Group 8 | Academic demo only - NOT a medical device, not clinically validated.")

kind = st.radio("Model", list(CKPT), index=1, horizontal=True)

samples = sorted((ROOT / "dataset/sample_mri").glob("*.png"))
source = st.radio("Image source", ["Upload", "Sample image"], horizontal=True)

image = None
if source == "Upload":
    f = st.file_uploader("Upload a brain MRI slice", type=["png", "jpg", "jpeg"])
    if f:
        image = Image.open(f)
elif samples:
    choice = st.selectbox("Choose a sample", samples, format_func=lambda p: p.name)
    image = Image.open(choice)

if image is not None:
    st.image(image, caption="Input image", use_container_width=True)
    if st.button("Classify", type="primary"):
        probs, ms = predict(image, kind)
        best = max(probs, key=probs.get)
        st.subheader(f"Prediction: {best} ({probs[best]:.1%})")
        st.bar_chart(probs)
        st.caption(f"Model: {kind} | Device: {device} | Inference: {ms:.1f} ms")
