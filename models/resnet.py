import torch
import numpy as np
from models.resnet18 import resnet18
from layers.virtual_radar import VirtualRadar
'''
pytorch resnet 18 model with VirtualRadar layer to convert graph NTU data to
spectrograms(images)
'''


class Model(torch.nn.Module):
    def __init__(self,
                 num_classes=60,
                 num_filters=64,
                 image_size=256,
                 device='cuda:0'):
        super(Model, self).__init__()
        self.base_model = resnet18(num_classes=num_classes,
                                   num_filters=num_filters)
        self.virtual_radar = VirtualRadar(wavelength=5e-4, device=device)
        self.image_size = image_size

    def forward(self, x):
        x = self.virtual_radar(x)
        x = x.unsqueeze(dim=1)
        x = torch.nn.functional.interpolate(x, self.image_size)
        x = self.base_model(x)
        return x
