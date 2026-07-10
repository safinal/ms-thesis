import torch.nn as nn
import torch
class FC(nn.Module):
    def __init__(self, num_features, num_classes):
        super(FC, self).__init__()
        self.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        x = self.fc(x)
        return x

def get_fc(device, path, num_features=768, num_classes=2):
    model = FC(num_features, num_classes).to(device)
    model.device = device

    sd = torch.load(path)
    try:
        sd = sd['classifier']
    except:
        pass
    new_sd = {}

    for n, value in sd.items():
        if n.startswith('fc'):
            new_sd[n] = value

    model.load_state_dict(new_sd)

    return model