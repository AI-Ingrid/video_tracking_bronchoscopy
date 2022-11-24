import torchvision
from torch import nn


class SegmentDetNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = torchvision.models.resnet18(pretrained=True, progress=True)
        self.model.fc = nn.Linear(512, 128)
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )
        # Only train the last layers of the network
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.fc.parameters():
            param.requires_grad = True
        for param in self.model.layer4.parameters():
            param.requires_grad = True
        for param in self.classifier.parameters():
            param.requires_grad = True

    def forward(self, x):
        x = self.model(x)
        x = self.classifier(x)
        return x
