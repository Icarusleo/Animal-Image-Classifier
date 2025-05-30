import gradio as gr
import shutil
import os
import zipfile
from PIL import Image
import torch
import timm
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, accuracy_score
from pathlib import Path

MODEL_PATH = "best_deit_model.pth"
with open("class_names.txt", "r") as f:
    CLASS_NAMES = [line.strip() for line in f.readlines()]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = timm.create_model("deit_base_patch16_224", pretrained=False, num_classes=len(CLASS_NAMES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval().to(device)

def predict_single(img: Image.Image):
    os.makedirs("uploaded_single", exist_ok=True)
    img_path = os.path.join("uploaded_single", "input.jpg")
    img.save(img_path)

    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred_idx = torch.argmax(probs).item()
        confidence = probs[0][pred_idx].item()

    return f"Görsel kaydedildi: {img_path}\n\n🧠 Tahmin: {CLASS_NAMES[pred_idx]}\n🔢 Güven: {confidence*100:.2f}%"

def predict_batch(files):
    results = []
    for f in files:
        img = Image.open(f.name).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            pred = torch.argmax(probs, 1).item()
            label = CLASS_NAMES[pred]
            confidence = probs[0][pred].item()
            results.append(f"{os.path.basename(f.name)} → {label} ({confidence*100:.2f}%)")

    return "\n".join(results)

def evaluate_uploaded_test(files):
    test_dir = "uploaded_test"

    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    os.makedirs(test_dir, exist_ok=True)

    for file in files:
        if zipfile.is_zipfile(file.name):
            with zipfile.ZipFile(file.name, 'r') as zip_ref:
                zip_ref.extractall(test_dir)
        else:
            default_dir = os.path.join(test_dir, "default")
            os.makedirs(default_dir, exist_ok=True)
            shutil.copy(file.name, os.path.join(default_dir, os.path.basename(file.name)))

    image_paths = list(Path(test_dir).rglob("*.jpg"))
    all_preds, all_labels = [], []

    for img_path in image_paths:
        img = Image.open(img_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            pred = torch.argmax(probs, 1).item()

        true_label_name = img_path.parent.name
        if true_label_name in CLASS_NAMES:
            true_label_index = CLASS_NAMES.index(true_label_name)
            all_preds.append(pred)
            all_labels.append(true_label_index)
        else:
            print(f"⚠️ {true_label_name} sınıfı class_names.txt içinde yok. Atlandı.")

    acc = accuracy_score(all_labels, all_preds)
    used_classes = sorted(set(all_labels + all_preds))
    used_class_names = [CLASS_NAMES[i] for i in used_classes]

    report = classification_report(
        all_labels, all_preds,
        labels=used_classes,
        target_names=used_class_names
    )

    return f"✅ Accuracy: {acc*100:.2f}%\n\n📊 Classification Report:\n{report}"


single_tab = gr.Interface(
    fn=predict_single,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="🖼️ Tekil Görsel Sınıflandırma"
)

batch_tab = gr.Interface(
    fn=predict_batch,
    inputs=gr.File(file_types=[".jpg", ".png"], file_count="multiple"),
    outputs="text",
    title="📁 Toplu Görsel Sınıflandırma"
)

folder_tab = gr.Interface(
    fn=evaluate_uploaded_test,
    inputs=gr.File(label="📦 Zip veya klasör içeriği", file_types=[".rar",".zip", ".jpg", ".png"], file_count="multiple"),
    outputs="text",
    title="📂 Etiketli Klasör Testi (Accuracy + F1)"
)

demo = gr.TabbedInterface([single_tab, batch_tab, folder_tab],
                          tab_names=["Tek Görsel", "Çoklu Görsel", "Test Seti (.zip)"])
demo.launch()
