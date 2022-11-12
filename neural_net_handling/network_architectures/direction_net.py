import torchvision
from torch import nn


class DirectionDetNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.resnet18(progress=True)
        # Output 2 meaning up or down
        self.model.fc = nn.Linear(256, 2)

        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.fc.parameters():
            param.requires_grad = True
        for param in self.model.layer4.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.model(x)
