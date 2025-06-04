import torch.nn as nn
from torchvision import models


import torch.nn as nn
from torchvision import models


class ResNet(nn.Module):
    def __init__(
        self,
        num_layers: int = 50,
        pretrained: bool = False,  # Explicit control over pretraining
        progress: bool = True
    ):
        super(ResNet, self).__init__()
        assert num_layers in [34, 50, 101]

        # Define the model based on num_layers and pretrained setting
        if num_layers == 34:
            self.resnet = models.resnet34(weights=None if not pretrained else models.ResNet34_Weights.DEFAULT, progress=progress)
        elif num_layers == 50:
            self.resnet = models.resnet50(weights=None if not pretrained else models.ResNet50_Weights.DEFAULT, progress=progress)
        else:
            self.resnet = models.resnet101(weights=None if not pretrained else models.ResNet101_Weights.DEFAULT, progress=progress)

    def get_backbone(self):
        return nn.Sequential(*list(self.resnet.children())[:-2])

    def forward(self, x):
        return self.resnet(x)