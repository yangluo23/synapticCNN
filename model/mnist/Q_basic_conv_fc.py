from tools import quantize

from .basic_conv_fc import BasicCNN, BasicCNN5, BasicCNN7

class QBasicCNN(BasicCNN):

    def __init__(self,
                 codebook,
                 all_layers=True,
                 quantize=True,
                 device_number=1,
                 init="",
                 **kwargs):
        super(QBasicCNN, self).__init__(init=init)

        self.codebook = codebook
        self.all_layers = all_layers
        self.quantization = quantize
        self.device_number = device_number
        self.kwargs = kwargs

        if 'show_scale' in kwargs:
            self.show_scale = kwargs["show_scale"]

    @quantize
    def eval(self):

        return self.train(False)


class QBasicCNN5(BasicCNN5):

    def __init__(self,
                 codebook,
                 all_layers=True,
                 quantize=True,
                 device_number=1,
                 init="",
                 **kwargs):
        super(QBasicCNN5, self).__init__(init=init)

        self.codebook = codebook
        self.all_layers = all_layers
        self.quantization = quantize
        self.device_number = device_number
        self.kwargs = kwargs

        if 'show_scale' in kwargs:
            self.show_scale = kwargs["show_scale"]

    @quantize
    def eval(self):
        return self.train(False)

class QBasicCNN7(BasicCNN7):

    def __init__(self,
                 codebook,
                 all_layers=True,
                 quantize=True,
                 device_number=1,
                 init="",
                 **kwargs):
        super(QBasicCNN7, self).__init__(init=init)

        self.codebook = codebook
        self.all_layers = all_layers
        self.quantization = quantize
        self.device_number = device_number
        self.kwargs = kwargs

        if 'show_scale' in kwargs:
            self.show_scale = kwargs["show_scale"]

    @quantize
    def eval(self):
        return self.train(False)