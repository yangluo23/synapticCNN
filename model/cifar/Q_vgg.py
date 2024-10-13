'''
Modified from https://github.com/pytorch/vision.git
'''
import math
import torch.nn as nn

from tools import quantize
from .vgg import VGG, cfg

__all__ = [
     'Qvgg11','Qvgg13',  'Qvgg16','Qvgg32',  'Qvgg64'
]


class QVGG(VGG):
    '''
    VGG model
    '''

    def __init__(self,
                 codebook,
                 features,
                 all_layers=True,
                 quantize=True,
                 device_number=1,
                 **kwargs):
        super(QVGG, self).__init__(features)

        self.codebook = codebook
        self.all_layers = all_layers
        self.quantization = quantize
        self.device_number = device_number

        if 'show_scale' in kwargs:
            self.show_scale = kwargs["show_scale"]

    @quantize
    def eval(self):

        return self.train(False)

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def Qvgg6(codebook, all_layers, quantize, device_number, **kwargs):
    """VGG 6-layer model (configuration "T")"""
    return QVGG(codebook=codebook, features=make_layers(cfg['S']),
                all_layers=all_layers, quantize=quantize, device_number=device_number, **kwargs)


def Qvgg8(codebook, all_layers, quantize, device_number, **kwargs):
    """VGG 8-layer model (configuration "T")"""
    return QVGG(codebook=codebook, features=make_layers(cfg['T']),
                all_layers=all_layers, quantize=quantize, device_number=device_number, **kwargs)

def Qvgg11(codebook, all_layers, quantize, device_number, **kwargs):
    """VGG 11-layer model (configuration "A")"""
    return QVGG(codebook=codebook, features=make_layers(cfg['A']),
                all_layers=all_layers, quantize=quantize, device_number=device_number,**kwargs)

def Qvgg13(codebook, all_layers, quantize, device_number, **kwargs):
    """VGG 13-layer model (configuration "B")"""
    return QVGG(codebook=codebook, features=make_layers(cfg['B']),
                all_layers=all_layers, quantize=quantize, device_number=device_number, **kwargs)

def Qvgg16(codebook, all_layers, quantize, device_number, **kwargs):
    """VGG 16-layer model (configuration "D")"""
    return QVGG(codebook=codebook, features=make_layers(cfg['D']),
                all_layers=all_layers, quantize=quantize, device_number=device_number, **kwargs)