import torch


class CNN(torch.nn.Module):
    def __init__(self, num_classes=5):
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, 5)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(6, 16, 5)
        self.fc1 = torch.nn.Linear(16 * 5 * 5, 120)  # 16 * 5 * 5
        self.fc2 = torch.nn.Linear(120, 84)  # Activations layer
        self.fc = torch.nn.Linear(84, num_classes)
        self.relu_1 = torch.nn.ReLU()
        self.relu_2 = torch.nn.ReLU()
        
        self.activation_layer = torch.nn.ReLU

    def forward(self, x):
        # Doing this way because only want to save activations
        # for fc linear layers - see later
        x = self.pool(torch.nn.functional.relu(self.conv1(x)))
        x = self.pool(torch.nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = self.relu_1(self.fc1(x))
        x = self.relu_2(self.fc2(x))
        x = self.fc(x)
        return x