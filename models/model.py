from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn

def get_model(num_classes):
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
