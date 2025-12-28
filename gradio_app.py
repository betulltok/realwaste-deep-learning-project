import gradio as gr
import torch
import torchvision.transforms as transforms
from torchvision import models
from torch import nn
from PIL import Image

# ===============================
# SINIF İSİMLERİ
# ===============================
class_names = [
    'Cardboard',
    'Food Organics',
    'Glass',
    'Metal',
    'Miscellaneous Trash',
    'Paper',
    'Plastic',
    'Textile Trash',
    'Vegetation'
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===============================
# MODEL YÜKLE
# ===============================
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(torch.load("models/realwaste_model.pth", map_location=device))
model.to(device)
model.eval()

# ===============================
# TRANSFORM
# ===============================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ===============================
# TAHMİN FONKSİYONU
# ===============================
def predict(image):
    image = Image.fromarray(image)
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        probs = torch.softmax(output, dim=1)[0]
        conf, pred = torch.max(probs, 0)

    predicted_class = class_names[pred.item()]
    confidence = conf.item() * 100

    return predicted_class, f"{confidence:.2f}%"

# ===============================
# GRADIO ARAYÜZ
# ===============================
with gr.Blocks() as demo:

    gr.Markdown(
        """
        # Atik Siniflandirma Sistemi
        Derin ogrenme tabanli goruntu analizi ile atik turu tahmini.
        """
    )

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(
                type="numpy",
                label="Atik Goruntusu Yukle"
            )
            analyze_btn = gr.Button("Analiz Et")

        with gr.Column():
            class_output = gr.Textbox(
                label="Tahmin Edilen Sinif",
                interactive=False
            )
            conf_output = gr.Textbox(
                label="Guven Orani",
                interactive=False
            )

    analyze_btn.click(
        fn=predict,
        inputs=image_input,
        outputs=[class_output, conf_output]
    )

    gr.Markdown(

