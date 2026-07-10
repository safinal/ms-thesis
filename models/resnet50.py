import torch
from torch import nn
import torchvision

class ResNet50(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT if pretrained else None)
        d = self.model.fc.in_features
        self.model.fc = nn.Linear(d, 2)

    def forward (self, X):
        X = self.model(X)
        return X
    
    def __str__(self):
        return 'ResNet50 :)'