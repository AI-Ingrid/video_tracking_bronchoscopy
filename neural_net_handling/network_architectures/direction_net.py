import torchvision
from torch import nn


class DirectionDetNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.first_layer = nn.Conv2d(
            in_channels=15,
            out_channels=3,
            kernel_size=3,
            stride=1,
            padding=1)
        self.model = torchvision.models.resnet18(pretrained=True, progress=True)
        # Output 2 meaning forward or backward in the airways
        self.model.fc = nn.Linear(512, 2)
        # Softmax (ikke nødvendig for binær output) ? Sigmoid

        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.fc.parameters():
            param.requires_grad = True
        for param in self.model.layer4.parameters():
            param.requires_grad = True

    def forward(self, x):
        x = self.first_layer(x)
        return self.model(x)
