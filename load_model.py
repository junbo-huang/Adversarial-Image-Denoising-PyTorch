import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from models.resnet import MyResNet18

def load_resnet(arch="resnet18", num_classes=10, device="cpu", pretrained=True, grayscale=False):
    if arch == "resnet18":
        model = MyResNet18(grayscale=grayscale)
    else:
        raise ValueError("Model not supported.")
    
    # load pretrained weights
    if pretrained:
        model.load_state_dict(resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).state_dict())

    # change output size
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # move to device
    model = model.to(device)

    return model

if __name__ == "__main__":
    model = load_resnet()
    print(model)