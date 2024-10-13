'''
Modified from https://github.com/pytorch/vision.git
'''
import math

import torch.nn as nn

__all__ = [
    'VGG', 'vgg6', 'vgg8', 'vgg11', 'vgg13', 'vgg16',
]


class VGG(nn.Module):
    '''
    VGG model
    '''
    def __init__(self, features):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 10),
        )
         # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


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


cfg = {
    'S': [32, 'M', 'M', 512, 'M', 'M', 512, 'M'],
    'T': [64, 'M', 64, 'M', 128, 'M', 256, 'M', 512, 'M'],
    'A': [64, 'M', 64, 'M', 64, 128, 'M', 128, 256, 'M', 256, 512, 'M'],
    'B': [64, 64, 'M', 64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 256, 512, 'M'],
    'D': [64, 64, 'M', 64, 64, 'M', 128, 128, 128, 'M', 256, 256, 256, 'M', 256, 256, 512, 'M'],
}


def vgg6():
    """VGG 11-layer model (configuration "A")"""
    return VGG(make_layers(cfg['S']))

def vgg8():
    """VGG 11-layer model (configuration "A")"""
    return VGG(make_layers(cfg['T']))

def vgg11():
    """VGG 11-layer model (configuration "A")"""
    return VGG(make_layers(cfg['A']))

def vgg13():
    """VGG 13-layer model (configuration "B")"""
    return VGG(make_layers(cfg['B']))

def vgg16():
    """VGG 16-layer model (configuration "D")"""
    return VGG(make_layers(cfg['D']))


if __name__ == '__main__':
    import torch
    model = vgg16()
    input = torch.randn((1,3,32,32))
    n_feature = 9
    model(input)






