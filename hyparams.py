import argparse

parser = argparse.ArgumentParser(description='Dataset parser')
# Working dataset
parser.add_argument('dataset', nargs=1, choices=['CIFAR10', 'MNIST'],
                    help='Working dataset')
# Training batch size
parser.add_argument('-bs', '--batch-size', type=int, default=-1, metavar='N',
                    help='Training batch size')
# Testing batch size
parser.add_argument('-tbs', '--test-batch-size', type=int, default=-1, metavar='N',
                    help='Testing batch size')
# Number of epochs to train
parser.add_argument('-e', '--epochs', type=int, default=-1, metavar='N',
                    help='Number of epochs to train')
# Learning rate
parser.add_argument('-lr', '--learning-rate', type=float, default=-1, metavar='LR',
                    help='Learning rate')
# Momentum
parser.add_argument('-m', '--momentum', type=float, default=-1, metavar='M',
                    help='Momentum')
# Disables CUDA training
parser.add_argument('-nc', '--no-cuda', action='store_true',
                    help='Disables CUDA training')
# Disables nesterov training (applies momentum before gradient)
parser.add_argument('-nn', '--no-nesterov', action='store_true',
                    help='Disables nesterov training')
# Logging interval between batches
parser.add_argument('-li', '--log-interval', type=int, default=10, metavar='N',
                    help='Logging interval between batches')
# Correction type to use
parser.add_argument('-gc', '--gradient-correction', choices=['none', 'worker', 'master'], default='none',
                    help='Correction type to use. Possible options: master, worker, none')
# Number of workers
parser.add_argument('-nw', '--num-workers', type=int, default=1, metavar='N',
                    help='Number of workers')
# Minimum number of iterations for batch
parser.add_argument('-mind', '--min-delay', type=int, default=20, metavar='N',
                    help='Minimum number of iterations for batch')
# Mean number of iterations for batch
parser.add_argument('-meand', '--mean-delay', type=int, default=5, metavar='N',
                    help='Mean number of iterations for batch')
# Standard Deviation of number of iterations for batch
parser.add_argument('-s', '--sigma', type=int, default=2, metavar='N',
                    help='Standard Deviation of number of iterations for batch')
# Distribution space of batch for worker
parser.add_argument('-ud', '--uni-dist', type=int, default=3, metavar='N',
                    help='Distribution space of batch for worker')



