import os
import torch
import torch.nn as nn
import tqdm
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from torch.utils import data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from DivisiveNormalization import DivisiveNorm

class Divisive(nn.Module):
    """
    AlexNet with Divisive normalization Norm
    """
    def __init__(self, num_classes=1000):
        super().__init__()
        # input size should be : (b x 3 x 227 x 227)
        # The image in the original paper states that width and height are 224 pixels, but
        # the dimensions after first convolution layer do not lead to 55 x 55.
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4),  # (b x 96 x 54 x 54)
            nn.ReLU(),
            DivisiveNorm(lamb=[20.], alpha=[1.], beta=[1.], k=[1.]),
            nn.MaxPool2d(kernel_size=3, stride=2),  # (b x 96 x 27 x 27)
            nn.Conv2d(96, 256, 5, padding=2),  # (b x 256 x 27 x 27)
            nn.ReLU(),
            DivisiveNorm(lamb=[20.], alpha=[1.], beta=[1.], k=[1.]),
            nn.MaxPool2d(kernel_size=3, stride=2),  # (b x 256 x 13 x 13)
            nn.Conv2d(256, 384, 3, padding=1),  # (b x 384 x 13 x 13)
            nn.ReLU(),
            DivisiveNorm(lamb=[20.], alpha=[1.], beta=[1.5], k=[1.]),
            nn.Conv2d(384, 384, 3, padding=1),  # (b x 384 x 13 x 13)
            nn.ReLU(),
            DivisiveNorm(lamb=[20.], alpha=[1.], beta=[1.], k=[1.]),
            nn.Conv2d(384, 256, 3, padding=1),  # (b x 256 x 13 x 13)
            nn.ReLU(),
            DivisiveNorm(lamb=[20.], alpha=[1.], beta=[1.], k=[1.]),
            nn.MaxPool2d(kernel_size=3, stride=2),  # (b x 256 x 6 x 6)
        )
        # classifier is just a name for linear layers
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=False),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=False),
            nn.Linear(4096, num_classes),
        )

    def init_bias(self):
        for layer in self.features:
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, mean=0, std=0.01)
                nn.init.constant_(layer.bias, 0)
            if isinstance(layer, DivisiveNorm):
                print("INSIDE INSTANCE OF ")
                nn.init.constant_(layer.size, 5)
                nn.init.constant_(layer.alpha, [1.])
                nn.init.constant_(layer.beta, .75)
                nn.init.constant_(layer.k, 1.)
        # original paper = 1 for Conv2d layers 2nd, 4th, and 5th conv layers
        nn.init.constant_(self.features[3].bias, 1)
        nn.init.constant_(self.features[8].bias, 1)
        nn.init.constant_(self.features[10].bias, 1)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
def alexnet(pretrained=False, progress=True, **kwargs):
    r"""
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = AlexNet(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['alexnet'],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model
