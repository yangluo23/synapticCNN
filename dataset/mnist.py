import os

import torch
import torchvision
from torchvision import transforms

def prepare_mnist_dataloader(num_workers=8, train_batch_size=128, eval_batch_size=256):

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    # Mnist digits dataset
    DOWNLOAD_MNIST =False
    if not (os.path.exists('.dataset//mnist/')) or not os.listdir('./dataset/mnist/'):
        DOWNLOAD_MNIST = True

    train_set = torchvision.datasets.MNIST(root="./dataset/mnist/", train=True,
                                           download=DOWNLOAD_MNIST,
                                           transform=train_transform)

    test_set = torchvision.datasets.MNIST(root="./dataset/mnist/", train=False,
                                          download=DOWNLOAD_MNIST,
                                          transform=test_transform)
    #
    train_sampler = torch.utils.data.RandomSampler(train_set)
    test_sampler = torch.utils.data.SequentialSampler(test_set)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set, batch_size=train_batch_size,
        sampler=train_sampler, num_workers=num_workers)

    test_loader = torch.utils.data.DataLoader(
        dataset=test_set, batch_size=eval_batch_size,
        sampler=test_sampler, num_workers=num_workers)

    return train_loader, test_loader