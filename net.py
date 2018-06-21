import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from enum import Enum
from numpy.random import normal


# Define a transformation
Transform = {
    # Normal transform for CIFAR10
    'NORMAL_TRANSFORM': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]),
    # Transform for MNIST using MNIST mean and standard deviation
    'MNIST_TRANSFORM': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
}

Hyperparameters = {
    'CIFAR10_PARAMS': {
        'batch_size': 128,
        'test_batch_size': 100,
        'epochs': 10,
        'learning_rate': 0.01,
        'momentum': 0.9
    },
    'MNIST_PARAMS': {
        'batch_size': 64,
        'test_batch_size': 1000,
        'epochs': 5,
        'learning_rate': 0.01,
        'momentum': 0.5
    }
}


class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class CIFARNet(nn.Module):
    def __init__(self):
        super(CIFARNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Dataset(Enum):
        # MNIST Dataset - handwritten numerals
        MNIST = {
        'trainset': torchvision.datasets.MNIST(
            root='../data', train=True, download=True,
            transform=Transform['MNIST_TRANSFORM']),
        'testset': torchvision.datasets.MNIST(
            root='../data', train=False, download=True,
            transform=Transform['MNIST_TRANSFORM']),
        'parser': Hyperparameters['MNIST_PARAMS'],
        'net': MNISTNet
        }
        #CIFAR10 Dataset - classified pictures
        CIFAR10 = {
        'trainset': torchvision.datasets.CIFAR10(
            root='../data', train=True, download=True,
            transform=Transform['NORMAL_TRANSFORM']),
        'testset': torchvision.datasets.CIFAR10(
            root='../data', train=False, download=True,
            transform=Transform['NORMAL_TRANSFORM']),
        'parser': Hyperparameters['CIFAR10_PARAMS'],
        'net': CIFARNet
        }


def save_gradients(model):
    grad_map = {}
    mean_map = {}
    var_map = {}

    for p in model.parameters():
        if p.grad is not None:
            grad_map[p] = p.grad.data.clone()
            mean_map[p] = p.grad.mean(0).data.clone()
            var_map[p] = p.grad.var(0).data.clone()

    return grad_map, mean_map, var_map


def load_gradients(model, gradient_map):
    mean_map = {}
    var_map = {}

    for p in model.parameters():
        if p.grad is not None:
            p.grad.data.copy_(gradient_map[p])
            mean_map[p] = p.grad.mean(0).data.clone()
            var_map[p] = p.grad.var(0).data.clone()
    return mean_map, var_map


def update_grad(model, grad, old_mean, old_var, new_mean, new_var):
    for p in model.parameters():
        delta_grad = (grad[p] - old_mean[p])
        dvar = new_var[p] / old_var[p].clamp(min=1e-10)
        grad[p] = (delta_grad * dvar) + new_mean[p]

    return grad


def update_meanvar(model, rmean, rvar, new_mean, new_var):
    if rmean == {}:
        for p in model.parameters():
            rmean[p] = 0.1 * new_mean[p]
            rvar[p] = 0.1 * new_var[p]
    for p in model.parameters():
        rmean[p] = 0.9 * rmean[p] + 0.1 * new_mean[p]
        rvar[p] = 0.9 * rvar[p] + 0.1 * new_var[p]

    return rmean, rvar
