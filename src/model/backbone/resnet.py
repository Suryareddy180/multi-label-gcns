import torch.nn as nn
from torchvision import models

class ResNet(nn.Module):
    def _init_(self, num_layers: int = 50, weights: str = None, progress: bool = True):
        super(ResNet, self)._init_()

        assert num_layers in [34, 50, 101], "Only ResNet-34, 50, and 101 are supported."

        # Select appropriate model without pretrained weights (weights=None)
        if num_layers == 34:
            self.resnet = models.resnet34(weights=None, progress=progress)
        elif num_layers == 50:
            self.resnet = models.resnet50(weights=None, progress=progress)
        else:
            self.resnet = models.resnet101(weights=None, progress=progress)

    def get_backbone(self):
        # Removes last two layers (avgpool and fc)
        return nn.Sequential(*list(self.resnet.children())[:-2])

    def forward(self, x):
        return self.resnet(x)