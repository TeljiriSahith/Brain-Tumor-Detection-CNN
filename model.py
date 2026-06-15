import torch
import torch.nn as nn

class CNN(nn.Module):

    def __init__(self):
        super().__init__()

        # Convolution Layers
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=3
        )

        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3
        )

        # Activation Function
        self.relu = nn.ReLU()

        # Pooling Layer
        self.pool = nn.MaxPool2d(2)

        # Flatten Layer
        self.flatten = nn.Flatten()

        # Fully Connected Layers
        self.fc1 = nn.Linear(28800, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):

        # Conv Block 1
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        # Conv Block 2
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        # Flatten
        x = self.flatten(x)

        # Fully Connected Layers
        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)

        return x