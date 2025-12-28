import torch
import gradio as gr
from torchvision import models, transforms
from torch import nn
from PIL import Image

# -----------------------
# AYARLAR
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# -----------------------
# MODEL YÜKLEME
# -----------------------
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(
    torch.load("models/realwaste_resnet18.pth", map_location=device)
)
model.to(device)
model.eval()

# -----------------------
# TRANSFORM
# -----------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -----------------------
# TAHMİN FONKSİYONU
# -----------------------
def predict(image):
    image = Image.fromarray(image).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)[0]

    idx = torch.argmax(probs).item()
    confidence = probs[idx].item()

    return f"Tahmin: {class_names[idx]}\nGüven: %{confidence*100:.2f}"

# -----------------------
# GRADIO ARAYÜZÜ
# -----------------------
with gr.Blocks() as demo:

    gr.Markdown("## Atık Sınıflandırma Sistemi")
    gr.Markdown(
        "Bir atık görseli yükleyin veya aşağıdaki örneklerden birini seçin."
    )

    image_input = gr.Image(type="numpy", label="Girdi Görseli")
    output_text = gr.Textbox(label="Model Çıktısı")

    predict_button = gr.Button("Analiz Et")

    gr.Examples(
        examples=[
            ["demo_images/cardboard.jpeg"],
            ["demo_images/organic.jpeg"],
            ["demo_images/plastic.jpeg"]
        ],
        inputs=image_input,
        label="Örnek Görseller"
    )

    predict_button.click(
        fn=predict,
        inputs=image_input,
        outputs=output_text
    )

# -----------------------
# ÇALIŞTIR
# -----------------------
if __name__ == "__main__":
    demo.launch(share=True)



