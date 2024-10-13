# Modified from
# https://github.com/pytorch/vision/blob/release/0.8.0/torchvision/models/resnet.py

import torch
import torch.nn as nn
from tools import quantize
from .resnet import ResNet,  BasicBlock, BasicBlock50


__all__ = ['Qresnet18', 'Qresnet34', 'Qresnet50']

class QResNet(ResNet):

    def __init__(self,
                 codebook,
                 block, layers,
                 all_layers=True,
                 quantize=True,
                 device_number=1,
                 num_classes=100,
                 **kwargs):
        super(QResNet, self).__init__(block, layers, num_classes=num_classes)

        self.codebook = codebook
        self.all_layers = all_layers
        self.quantization = quantize
        self.device_number = device_number

        if 'show_scale' in kwargs:
            self.show_scale = kwargs["show_scale"]

        if 'stopping_temperature' in kwargs:
            self.stopping_temperature = kwargs['stopping_temperature']

        if 'search_range' in kwargs:
            self.search_range = kwargs['search_range']

        if 'waiting_steps' in kwargs:
            self.waiting_steps = kwargs['waiting_steps']

    @quantize
    def eval(self):

        return self.train(False)

def _resnet(codebook, quantize, all_layers, device_number, block, layers, **kwargs):
    model = QResNet(codebook, block, layers, all_layers, quantize, device_number,
                    **kwargs)

    return model


def Qresnet18(codebook, quantize, all_layers, device_number, **kwargs):

    return _resnet(codebook, quantize, all_layers, device_number,
                   BasicBlock, [2, 2, 2, 2], **kwargs)

def Qresnet34(codebook, quantize, all_layers, device_number, **kwargs):

    return _resnet(codebook, quantize, all_layers, device_number,
                   BasicBlock, [3, 4, 6, 3], **kwargs)


def Qresnet50(codebook, quantize, all_layers, device_number, **kwargs):

    return _resnet(codebook, quantize, all_layers, device_number,
                   BasicBlock50, [3, 4, 6, 3], **kwargs)
