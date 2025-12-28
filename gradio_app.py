import gradio as gr
import torch
import torchvision.transforms as transforms
from PIL import Image
import os

from src.model import get_model

# --------------------
# MODEL YÜKLEME
# --------------------
MODEL_PATH = "models/realwaste_resnet18.pth"
CLASS_NAMES = [
    "Cardboard", "Food Organics", "Glass",
    "Metal", "Paper", "Plastic",
    "Textile Trash", "Vegetation", "Wood"
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = get_model(num_classes=len(CLASS_NAMES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# --------------------
# TRANSFORM
# --------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# --------------------
# TAHMİN FONKSİYONU
# --------------------
def predict(image):
    if image is None:
        return "Lütfen bir görüntü yükleyin."

    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)[0]
        pred_idx = torch.argmax(probs).item()

    result = f"""
    Tahmin Edilen Sınıf: {CLASS_NAMES[pred_idx]}

    Olasılık Değeri: {probs[pred_idx]*100:.2f} %
    """
    return result

# --------------------
# ÖRNEK GÖRSELLER
# --------------------
examples = [
    "demo_images/carton.jpg",
    "demo_images/organic.jpg",
    "demo_images/plastic.jpg"
]

# --------------------
# CSS (RENK + BOYUT DÜZENİ)
# --------------------
custom_css = """
body {
    background-color: #f4f6f9;
}

#title {
    text-align: center;
    font-size: 28px;
    font-weight: 700;
    color: #1f2937;
}

#subtitle {
    text-align: center;
    font-size: 16px;
    color: #4b5563;
    margin-bottom: 20px;
}

.gr-button {
    background-color: #2563eb !important;
    color: white !important;
    border-radius: 8px;
}

.gr-box {
    border-radius: 12px;
}
"""

# --------------------
# ARAYÜZ
# --------------------
with gr.Blocks() as demo:

    gr.Markdown(
        "<div id='title'>Atık Görüntülerinde Derin Öğrenme ile Sınıflandırma</div>"
    )
    gr.Markdown(
        "<div id='subtitle'>RealWaste veri seti ile eğitilmiş ResNet18 modeli</div>"
    )

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(
                label="Atık Görüntüsü",
                type="pil",
                height=250
            )
            predict_btn = gr.Button("Tahmin Et")

        with gr.Column(scale=1):
            output_text = gr.Textbox(
                label="Model Çıktısı",
                lines=6
            )

    gr.Examples(
        examples=examples,
        inputs=image_input,
        label="Hazır Örnekler (Tıklayarak Deneyin)"
    )

    predict_btn.click(
        fn=predict,
        inputs=image_input,
        outputs=output_text
    )

# --------------------
# ÇALIŞTIR
# --------------------
demo.launch(
    css=custom_css,
    theme=gr.themes.Base(),
    share=True
)







