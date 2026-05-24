import torch
import torch.nn as nn


class VisualSentry(nn.Module):
    def __init__(self, num_classes=47):
        super(VisualSentry, self).__init__()

        # Block 1: Input (B, 1, 64, 64) -> Output (B, 16, 64, 64)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # Downsamples to (B, 16, 32, 32)

        # Block 2: Input (B, 16, 32, 32) -> Output (B, 32, 32, 32)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # Downsamples to (B, 32, 16, 16)

        # Fully Connected Head
        # 32 channels * 16 * 16 spatial pixels = 8192 flattened inputs
        self.fc1 = nn.Linear(32 * 16 * 16, 128)
        self.relu3 = nn.ReLU()
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x, return_features=False):
        # Localized spatial feature mapping
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        feature_maps = self.pool2(x)

        # Flatten spatial maps for the dense layers
        flattened = feature_maps.view(feature_maps.size(0), -1)

        # Bottleneck style representation
        latent_vector = self.relu3(self.fc1(flattened))

        if return_features:
            return latent_vector  # 128-neuron stylometric fingerprint
        return self.classifier(latent_vector)