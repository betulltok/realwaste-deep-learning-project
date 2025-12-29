import gradio as gr
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import os

# --------------------
# MODEL TANIMI
# --------------------
def get_model(num_classes):
    """ResNet18 modeli oluştur"""
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

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
        return "Lütfen bir görüntü yükleyin.", None
    
    # PIL Image'e çevir
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        pred_idx = torch.argmax(probs).item()
    
    # Tüm sınıflar için olasılıklar
    confidence_dict = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}
    
    result_text = f"**Tahmin:** {CLASS_NAMES[pred_idx]}\n**Güven:** %{probs[pred_idx]*100:.1f}"
    
    return result_text, confidence_dict

# --------------------
# ÖRNEK GÖRSELLER (varsa)
# --------------------
example_images = []
if os.path.exists("demo_images"):
    for img_name in ["carton.jpg", "organic.jpg", "plastic.jpg"]:
        img_path = os.path.join("demo_images", img_name)
        if os.path.exists(img_path):
            example_images.append(img_path)

# Eğer örnek görseller yoksa boş liste kullan
if not example_images:
    example_images = None

# --------------------
# ARAYÜZ
# --------------------
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    
    gr.Markdown("# Atık Sınıflandırma Sistemi")
    gr.Markdown("ResNet18 modeli ile atık türü tahmini")
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(
                label="Görüntü Yükleyin",
                type="pil"
            )
            predict_btn = gr.Button("Tahmin Et", variant="primary")
            
            if example_images:
                gr.Examples(
                    examples=example_images,
                    inputs=image_input,
                    label="Örnek Görseller"
                )
        
        with gr.Column():
            output_text = gr.Textbox(
                label="Sonuç",
                lines=3
            )
            output_chart = gr.Label(
                label="Olasılık Dağılımı",
                num_top_classes=9
            )
    
    predict_btn.click(
        fn=predict,
        inputs=image_input,
        outputs=[output_text, output_chart]
    )

# --------------------
# ÇALIŞTIR
# --------------------
if __name__ == "__main__":
    demo.launch(share=True)







