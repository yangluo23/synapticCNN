import torch
import torch.nn as nn

class BasicCNN(nn.Module):
    def __init__(self, init=""):
        super(BasicCNN, self).__init__()
        self.conv1 = nn.Sequential(         # input shape (1, 28, 28)
            nn.MaxPool2d(kernel_size=2),
            # 3x3x1x1000
            nn.Conv2d(
                in_channels=1,
                out_channels=1000,
                kernel_size=3,
                stride=1,
                padding=1,
            ),                              # output shape (1000, 14, 14)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),    # output shape (1000, 7, 7)
        )
        # 1x1x1000x9
        self.conv2 = nn.Sequential(         # input shape (1000, 7, 7)
            nn.Conv2d(1000, 9, 1, 2, 6),    # output shape (9, 10, 10)
            nn.ReLU(),  # activation
        )
        # 9x10x10
        self.out = nn.Linear(9 * 10 * 10, 10)

        # init
        self.init = init
        if self.init:  # default uniform
            self.__weight_init()

    def __weight_init(self):
        for m in self.modules():
            if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):

                if self.init == "kaiming:":
                    torch.nn.init.kaiming_uniform_(m.weight)
                    torch.nn.init.kaiming_uniform_(m.bias)
                if self.init == "xavier":
                    torch.nn.init.xavier_uniform_(m.weight)
                    torch.nn.init.zeros_(m.bias)


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output


class BasicCNN5(BasicCNN):
    def __init__(self, init=""):
        super(BasicCNN5, self).__init__()

        self.conv1 = nn.Sequential(  # input shape (1, 28, 28)
            nn.MaxPool2d(kernel_size=2),
            # 3x3x1x1000
            nn.Conv2d(
                in_channels=1,
                out_channels=1000,
                kernel_size=3,
                stride=1,
                padding=1,
            ),  # output shape (1000, 14, 14)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # output shape (1000, 7, 7)
        )
        # 1x1x1000x9
        self.conv2 = nn.Sequential(  # input shape (1000, 7, 7)
            nn.Conv2d(1000, 9, 1, 2, 6),  # output shape (9, 10, 10)
            nn.ReLU(),  # activation
        )

        # 1x1x9x1000
        self.conv3 = nn.Sequential(  # input shape (9, 10, 10)
            nn.Conv2d(9, 1000, 1, 1),  # output shape (9, 10, 10)
            nn.ReLU(),  # activation
        )

        # 1x1x1000x9
        self.conv4 = nn.Sequential(  # input shape (9, 10, 10)
            nn.Conv2d(1000, 9, 1, 1),  # output shape (9, 10, 10)
            nn.ReLU(),  # activation
        )

        # 9x10x10
        self.out = nn.Linear(9 * 10 * 10, 10)

        # init
        self.init = init
        if self.init:  # default uniform
            self.__weight_init()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output

class BasicCNN7(BasicCNN):
    def __init__(self, init=""):
        super(BasicCNN7, self).__init__()

        self.conv1 = nn.Sequential(  # input shape (1, 28, 28)
            nn.MaxPool2d(kernel_size=2),
            # 3x3x1x1000
            nn.Conv2d(
                in_channels=1,
                out_channels=1000,
                kernel_size=3,
                stride=1,
                padding=1,
            ),  # output shape (1000, 14, 14)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # output shape (1000, 7, 7)
        )
        # 1x1x1000x9
        self.conv2 = nn.Sequential(  # input shape (1000, 7, 7)
            nn.Conv2d(1000, 9, 1, 2, 6),  # output shape (9, 10, 10)
            nn.ReLU(),  # activation
        )

        # 1x1x9x1000
        self.conv3 = nn.Sequential(  # input shape (9, 10, 10)
            nn.Conv2d(9, 1000, 1, 1),  # output shape (9, 10, 10)
            nn.ReLU(),  # activation
        )

        # 1x1x1000x9
        self.conv4 = nn.Sequential(  # input shape (9, 10, 10)
            nn.Conv2d(1000, 9, 1, 1),  # output shape (9, 10, 10)
            nn.ReLU(),  # activation
        )

        # 1x1x9x1000
        self.conv5 = nn.Sequential(  # input shape (9, 10, 10)
            nn.Conv2d(9, 1000, 1, 1),  # output shape (9, 10, 10)
            nn.ReLU(),  # activation
        )

        # 1x1x1000x9
        self.conv6 = nn.Sequential(  # input shape (9, 10, 10)
            nn.Conv2d(1000, 9, 1, 1),  # output shape (9, 10, 10)
            nn.ReLU(),  # activation
        )

        # 9x10x10
        self.out = nn.Linear(9 * 10 * 10, 10)

        # init
        self.init = init
        if self.init:  # default uniform
            self.__weight_init()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output
