import torchvision
from torch import nn


class SegmentDetNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = torchvision.models.resnet18(pretrained=True, progress=True)
        self.model.fc = nn.Linear(512, num_classes)
        #self.last_layer = nn.Softmax(dim=num_classes)

        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.fc.parameters():
            param.requires_grad = True
        for param in self.model.layer4.parameters():
            param.requires_grad = True

    def forward(self, x):
        x = self.model(x)
        #x = self.last_layer(x)
        return x
