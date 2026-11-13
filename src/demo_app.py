"""Prototype demo: upload a brain MRI slice -> predicted stage + class probabilities (CNN or ResNet18).
Run (Colab/local):  pip install gradio ;  python src/demo_app.py
Command-line alternative:  python src/demo_app.py --image path/to/img.png --model ResNet18
Academic demo only - NOT a medical device."""
import argparse, time
from pathlib import Path
import torch
from PIL import Image
from torchvision import transforms
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_final import load_checkpoint, extract_state, make_model, IMG_SIZE

ROOT = Path(__file__).resolve().parents[1]
CKPT = {"CNN": ROOT / "models/cnn_model.pth", "ResNet18": ROOT / "models/resnet18_model.pth"}
DEFAULT_CLASSES = ["Mild Demented", "Moderate Demented", "Non Demented", "Very Mild Demented"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_cache = {}

def get_model(kind):
    if kind not in _cache:
        ckpt = load_checkpoint(CKPT[kind], device)
        state, meta = extract_state(ckpt)
        classes = list(meta.get("classes") or DEFAULT_CLASSES)
        m = make_model(kind, len(classes)); m.load_state_dict(state); m.to(device).eval()
        _cache[kind] = (m, classes, bool(meta.get("normalize", False)))
    return _cache[kind]

def predict(image, kind="ResNet18"):
    model, classes, norm = get_model(kind)
    ops = [transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor()]
    if norm: ops.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    x = transforms.Compose(ops)(image.convert("RGB")).unsqueeze(0).to(device)
    t = time.perf_counter()
    with torch.no_grad(): p = torch.softmax(model(x), 1)[0].cpu()
    if device.type == "cuda": torch.cuda.synchronize()
    ms = (time.perf_counter() - t) * 1000
    return {c: float(v) for c, v in zip(classes, p)}, ms

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--image"); ap.add_argument("--model", default="ResNet18", choices=list(CKPT))
    a = ap.parse_args()
    if a.image:
        probs, ms = predict(Image.open(a.image), a.model)
        for c, v in sorted(probs.items(), key=lambda kv: -kv[1]): print(f"{c:22s} {v:.3f}")
        print(f"device={device}  inference={ms:.2f} ms"); return
    import gradio as gr
    def run(img, kind):
        if img is None: return None, "Upload an MRI image."
        probs, ms = predict(img, kind)
        return probs, f"Model: {kind} | Device: {device} | Inference: {ms:.2f} ms\nAcademic demo only - not for medical use."
    gr.Interface(run, [gr.Image(type="pil", label="Brain MRI slice"), gr.Radio(list(CKPT), value="ResNet18", label="Model")],
                 [gr.Label(num_top_classes=4, label="Predicted stage"), gr.Textbox(label="Info")],
                 title="Alzheimer's Stage Classification from Brain MRI (Group 8)",
                 examples=[[str(p), "ResNet18"] for p in sorted((ROOT / "dataset/sample_mri").glob("*.png"))[:4]]).launch(share=True)

if __name__ == "__main__": main()
