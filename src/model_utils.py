import torch
from torchvision import models


class LectinClassifier(torch.nn.Module):
    def __init__(self, num_classes=4):
        super(LectinClassifier, self).__init__()
        # Use a lightweight but powerful ResNet18
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        num_ftrs = self.backbone.fc.in_features

        # Remove original fully connected layer
        self.backbone.fc = torch.nn.Identity()

        # Head 1: Border Intensity
        self.border_head = torch.nn.Sequential(
            torch.nn.Linear(num_ftrs, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(512, num_classes),
        )

        # Head 2: Inner Intensity
        self.inner_head = torch.nn.Sequential(
            torch.nn.Linear(num_ftrs, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(512, num_classes),
        )

    def forward(self, x):
        features = self.backbone(x)
        border_out = self.border_head(features)
        inner_out = self.inner_head(features)
        return border_out, inner_out
