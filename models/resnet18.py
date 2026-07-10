import torch
from torch import nn
import torchvision

class ResNet18(nn.Module):
    def __init__(self):
        super().__init__()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = torchvision.models.resnet18(weights=None)
        d = self.model.fc.in_features
        self.model.fc = nn.Linear(d, 2)

    def forward (self, X):
        X = self.model(X)
        return X
    
    def __str__(self):
        return 'ResNet18 :)'