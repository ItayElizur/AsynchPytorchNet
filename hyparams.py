import argparse

parser = argparse.ArgumentParser(description='Dataset parser')
#
parser.add_argument('dataset', nargs=1, choices=['CIFAR10', 'MNIST'])
# Input batch size for training
parser.add_argument('-bs', '--batch-size', type=int, default=-1, metavar='N')
# Input batch size for testing
parser.add_argument('-tbs', '--test-batch-size', type=int, default=-1, metavar='N')
# Number of epochs to train
parser.add_argument('-e', '--epochs', type=int, default=-1, metavar='N')
# Learning rate
parser.add_argument('-lr', '--learning-rate', type=float, default=-1, metavar='LR')
# SGD momentum
parser.add_argument('-m', '--momentum', type=float, default=-1, metavar='M')
# Disables CUDA training
parser.add_argument('-nc', '--no-cuda', action='store_true')
# Enables Nesterov training (applies momentum before gradient)
parser.add_argument('-n', '--nesterov', action='store_false')
# How many batches to wait before logging training status
parser.add_argument('-li', '--log-interval', type=int, default=10, metavar='N')
# What correction to use
parser.add_argument('-gc', '--gradient-correction', choices=['none', 'worker', 'master'], default='none')
# How many workers
parser.add_argument('-nw', '--num-workers', type=int, default=1, metavar='N')
# Minimum delay of workers
parser.add_argument('-mind', '--min-delay', type=int, default=20, metavar='N')
# Mean delay of workers
parser.add_argument('-meand', '--mean-delay', type=int, default=5, metavar='N')
# Deviation of delay of workers
parser.add_argument('-s', '--sigma', type=int, default=2, metavar='N')
# Distribution space of work for worker
parser.add_argument('-ud', '--uni-dist', type=int, default=3, metavar='N')


def getMNISTParser():
    return {
        'batch_size': 64,
        'test_batch_size': 1000,
        'epochs': 5,
        'learning_rate': 0.01,
        'momentum': 0.5
    }


def getCIFAR10Parser():
    return {
        'batch_size': 128,
        'test_batch_size': 100,
        'epochs': 10,
        'learning_rate': 0.01,
        'momentum': 0.9
    }
