import gradio as gr
import torch
import torchvision.transforms as transforms
from torchvision import models
from torch import nn
from PIL import Image
import numpy as np
import plotly.graph_objects as go

# ===============================
# MODEL ve SINIFLAR
# ===============================
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
model.load_state_dict(
    torch.load("models/realwaste_resnet18.pth", map_location=device)
)
model.to(device)
model.eval()

# ===============================
# GÖRÜNTÜ DÖNÜŞÜMÜ
# ===============================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# ===============================
# TAHMİN FONKSİYONU
# ===============================
def predict(image):
    if image is None:
        return None, "Lütfen bir görüntü yükleyin."

    image_pil = Image.fromarray(image)
    image_tensor = transform(image_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()

    confidences = dict(zip(class_names, probs))
    predicted_class = max(confidences, key=confidences.get)

    # Grafik
    fig = go.Figure(
        data=[
            go.Bar(
                x=list(confidences.values()),
                y=list(confidences.keys()),
                orientation="h"
            )
        ]
    )
    fig.update_layout(
        title="Sınıflara Göre Güven Skorları",
        xaxis_title="Olasılık",
        yaxis_title="Sınıf",
        height=350
    )

    result_text = f"""
### Tahmin Sonucu

**Tahmin Edilen Sınıf:** {predicted_class}

**Güven Oranı:** {confidences[predicted_class]:.2%}
"""

    return fig, result_text

# ===============================
# CSS (SADE + ŞIK)
# ===============================
custom_css = """
body {
    background-color: #f4f6f8;
}

.header {
    background: #2c3e50;
    padding: 18px;
    border-radius: 10px;
    text-align: center;
    color: white;
    margin-bottom: 20px;
}

.section-box {
    background: white;
    padding: 16px;
    border-radius: 10px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.06);
}
"""

# ===============================
# GRADIO ARAYÜZÜ
# ===============================
with gr.Blocks(css=custom_css, theme=gr.themes.Base()) as demo:

    gr.HTML("""
    <div class="header">
        <h2>Atık Görüntülerinde Derin Öğrenme ile Sınıflandırma</h2>
        <p>RealWaste veri seti ile eğitilmiş ResNet18 modeli</p>
    </div>
    """)

    with gr.Row():

        # SOL PANEL
        with gr.Column(scale=1):
            with gr.Group(elem_classes="section-box"):
                gr.Markdown("### Görüntü Yükleme")

                image_input = gr.Image(
                    label="Atık Görüntüsü",
                    type="numpy",
                    height=230
                )

                predict_btn = gr.Button("Tahmin Et")

                gr.Markdown("### Hazır Örnekler")
                gr.Markdown(
                    "Aşağıdaki örnek görsellerden birine tıklayarak otomatik yükleyebilirsiniz."
                )

        # SAĞ PANEL
        with gr.Column(scale=1):
            with gr.Group(elem_classes="section-box"):
                gr.Markdown("### Model Çıktısı")

                plot_output = gr.Plot()
                result_output = gr.Markdown()

    predict_btn.click(
        fn=predict,
        inputs=image_input,
        outputs=[plot_output, result_output]
    )

    # OTOMATİK ÖRNEK GÖRSELLER
    gr.Examples(
        examples=[
            "demo_images/cardboard.jpg",
            "demo_images/organic.jpg",
            "demo_images/plastic.jpg"
        ],
        inputs=image_input,
        label="Örnek Görseller"
    )

    gr.Markdown("""
---
**Model:** ResNet18 (Transfer Learning)  
**Veri Seti:** RealWaste  
**Sınıf Sayısı:** 9  
""")

# ===============================
# ÇALIŞTIR
# ===============================
if __name__ == "__main__":
    demo.launch(share=True)




