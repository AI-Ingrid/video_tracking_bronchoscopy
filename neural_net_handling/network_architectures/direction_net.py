import torchvision
from torch import nn
import torch
from parameters import batch_size, frame_dimension


class TimeDistributed(nn.Module):
    def __init__(self, module):
        super(TimeDistributed, self).__init__()
        self.module = module

    def forward(self, X):
        X = X.transpose(2, 3)
        X = X.transpose(2, 4)

        # Reshape 5 dim input into 4 dim input by merging columns batch_size and time_steps
        # In: [batch_dim, time_steps, height, width, channels]
        # Out: [batch_dim * time_steps, height, width, channels]
        org_shape = tuple(X.shape)
        X_reshaped = X.reshape((torch.prod(torch.tensor(org_shape[:2])),) + org_shape[2:])
        output = self.module(X_reshaped.float())

        # Reshape back to 5 dim again
        output_reshaped = output.reshape(org_shape[:2] + (output.shape[-1],))
        return output_reshaped


class DirectionDetNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Feature extractor
        self.feature_extractor = torchvision.models.resnet18(pretrained=True)

        # Reshaping 5D to 4D
        self.time_distributed = TimeDistributed(self.feature_extractor)

        # Recurrent Neural Network
        self.RNN = nn.Sequential(
            nn.LSTM(1000, 128, 1, batch_first=True),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Softmax(dim=1)
        )

        # Handle training for certain layers
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        for param in self.feature_extractor.layer3.parameters():
            param.requires_grad = True
        for param in self.feature_extractor.layer4.parameters():
            param.requires_grad = True
        for param in self.feature_extractor.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        x = self.time_distributed(x)
        x = self.LSTM(x)[0]
        x = self.RNN(x)
        return x
