import gradio as gr
import torch
import torchvision.transforms as transforms
from torchvision import models
from torch import nn
from PIL import Image

# ======================
# MODEL AYARLARI
# ======================
class_names = [
    "Cardboard",
    "Food Organics",
    "Glass",
    "Metal",
    "Miscellaneous Trash",
    "Paper",
    "Plastic",
    "Textile Trash",
    "Vegetation"
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(torch.load("models/realwaste_resnet18.pth", map_location=device))
model.to(device)
model.eval()

# ======================
# IMAGE TRANSFORM
# ======================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ======================
# TAHMIN FONKSIYONU
# ======================
def predict(image):
    if image is None:
        return "Lütfen bir görüntü yükleyin."

    image = Image.fromarray(image)
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()

    return f"""
Tahmin Edilen Sınıf: {class_names[pred]}
Güven Oranı: %{confidence*100:.2f}
"""

# ======================
# CUSTOM CSS (RENK BURADA)
# ======================
custom_css = """
body {
    background-color: #f4f6f8;
}

#header {
    background-color: #2c3e50;
    color: white;
    padding: 20px;
    border-radius: 12px;
    margin-bottom: 20px;
}

.card {
    background-color: white;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.08);
}

footer {
    color: #555;
}
"""

# ======================
# GRADIO ARAYUZ
# ======================
with gr.Blocks(css=custom_css, theme=gr.themes.Base()) as demo:

    gr.HTML("""
    <div id="header">
        <h2>Atık Görüntülerinde Derin Öğrenme ile Sınıflandırma</h2>
        <p>RealWaste veri seti ile eğitilmiş ResNet18 modeli</p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.HTML("<div class='card'>")
            image_input = gr.Image(
                label="Atık Görüntüsü Yükle",
                type="numpy",
                height=220
            )
            predict_btn = gr.Button("Tahmin Et")
            gr.HTML("</div>")

        with gr.Column(scale=1):
            gr.HTML("<div class='card'>")
            output = gr.Textbox(
                label="Model Çıktısı",
                lines=5
            )
            gr.HTML("</div>")

    gr.Markdown("### Hazır Örnek Görseller")

    gr.Examples(
        examples=[
            "demo_images/cardboard.jpg",
            "demo_images/organic.jpg",
            "demo_images/plastic.jpg"
        ],
        inputs=image_input
    )

    predict_btn.click(
        fn=predict,
        inputs=image_input,
        outputs=output
    )

    gr.Markdown("""
    ---
    **Model:** ResNet18 (Transfer Learning)  
    **Veri Seti:** RealWaste  
    **Sınıf Sayısı:** 9  
    """)

demo.launch()






