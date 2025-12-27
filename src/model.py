# src/model.py

import torch.nn as nn
from torchvision import models

def get_model(num_classes):
    """
    ResNet18 tabanlı sınıflandırma modeli
    """
    model = models.resnet18(weights="IMAGENET1K_V1")

    # Önceden öğrenilmiş katmanları dondur
    for param in model.parameters():
        param.requires_grad = False

    # Son katmanı değiştir
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model
