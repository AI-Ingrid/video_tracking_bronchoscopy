import torch
from typing import OrderedDict, Tuple, List
from torch import nn
import torchvision.models as models
import torchvision.ops as ops


class Layer(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=2, padding=0),
            nn.ReLU(),
        )


class ResNet(torch.nn.Module):
    """
    This is a basic backbone for SSD.
    The feature extractor outputs a list of 6 feature maps, with the sizes:
    """

    def __init__(self,
                 output_channels: List[int],
                 image_channels: int,
                 output_feature_sizes: List[Tuple[int]]):
        super().__init__()
        self.out_channels = output_channels
        self.output_feature_shape = output_feature_sizes

        # Get pretrained Retina Network
        self.model = models.resnet18(pretrained=True)

        # Create two more layers
        self.layer5 = Layer(512, 256)
        self.layer6 = Layer(256, 256)

        self.fpn = ops.FeaturePyramidNetwork([64, 128, 256, 512, 256, 256], 256)

    def forward_first_layer(self, model, image):
        """Executing forward pass for the zeroth Retina Net layer"""
        x = model.conv1(image)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)
        return x

    def forward(self, x):
        """
        Performing forward pass for a layer at the time and saving every output in an array.
        The forward functiom should output features with shape:
            [shape(-1, 256, 32, 256),
            shape(-1, 512, 16, 128),
            shape(-1, 1024, 8, 64),
            shape(-1, 2048, 4, 32),
            shape(-1, 2048, 2, 16),
            shape(-1, 2048, 1, 8)]
        When done, the array of outputs is passed into the FPN and the outputs from FPN is returned
        """
        out_features = []
        features_dict = OrderedDict()

        # Layer 0
        x = self.forward_first_layer(self.model, x)

        # Layer 1
        features_dict['feat0'] = self.model.layer1(x)

        # Layer 2
        features_dict['feat1'] = self.model.layer2(features_dict['feat0'])

        # Layer 3
        features_dict['feat2'] = self.model.layer3(features_dict['feat1'])

        # Layer 4
        features_dict['feat3'] = self.model.layer4(features_dict['feat2'])

        # Layer 5
        features_dict['feat4'] = self.layer5(features_dict['feat3'])

        # Layer 6
        features_dict['feat5'] = self.layer6(features_dict['feat4'])

        # Forward to FPN
        out_features = self.fpn(features_dict)

        # Test
        for idx, feature in enumerate(out_features.values()):
            out_channel = self.out_channels[idx]
            h, w = self.output_feature_shape[idx]
            expected_shape = (out_channel, h, w)
            assert feature.shape[1:] == expected_shape, \
                f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"
        assert len(out_features) == len(self.output_feature_shape), \
            f"Expected that the length of the outputted features to be: {len(self.output_feature_shape)}, but it was: {len(out_features)}"
        return tuple(out_features.values())