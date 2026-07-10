import torch 
from torch import nn
from .resnet50 import ResNet50

class StandardScaler:
    def __init__(self, mean=None, std=None, epsilon=1e-7):
        self.mean = mean
        self.std = std
        self.epsilon = epsilon

    def fit(self, values):
        dims = list(range(values.dim() - 1))
        self.mean = torch.mean(values, dim=dims)
        self.std = torch.std(values, dim=dims)

    def transform(self, values):
        return (values - self.mean) / (self.std + self.epsilon)

    def fit_transform(self, values):
        self.fit(values)
        return self.transform(values)
    
class StdResNet50(nn.Module):
    def __init__(self, num_classes, pretrained_path=None):
        super(StdResNet50, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = ResNet50().to(self.device)
        if pretrained_path:
            model.load_state_dict(torch.load(pretrained_path, map_location=self.device))
        self.model = nn.Sequential(*list(model.model.children())[:-1])
        self.scaler = StandardScaler()
        self.fc = model.model.fc

    def forward(self, x):
        x = self.model(x)
        x = torch.flatten(x, 1)
        x = self.scaler.fit_transform(x)
        x = self.fc(x)
        return x