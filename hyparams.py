import argparse


def milestones(arg):
    return [int(x) for x in arg.split(',')]

def weight_decay(arg):
    return int(arg.split('e')[0])*10**int(arg.split('e')[1])

parser = argparse.ArgumentParser(description='Dataset parser')
# Working dataset
parser.add_argument('dataset', nargs=1, choices=['CIFAR10', 'MNIST'],
                    help='Working dataset')
# Training batch size
parser.add_argument('-bs', '--batch-size', type=int, default=-1, metavar='BATCH SIZE',
                    help='Training batch size')
# Testing batch size
parser.add_argument('-tbs', '--tbatch-size', type=int, default=-1, metavar='TEST BATCH SIZE',
                    help='Testing batch size')
# Number of epochs to train
parser.add_argument('-e', '--epochs', type=int, default=-1, metavar='EPOCHS',
                    help='Number of epochs to train')
# Learning rate
parser.add_argument('-lr', '--learning-rate', type=float, default=-1, metavar='LEARNING RATE',
                    help='Learning rate')
# Momentum
parser.add_argument('-m', '--momentum', type=float, default=-1, metavar='MOMENTUM',
                    help='Momentum')
# Disables nesterov training (applies momentum before gradient)
parser.add_argument('-nn', '--no-nesterov', action='store_true',
                    help='Disables nesterov training')
# Weight decay (L2 regularization) size
parser.add_argument('-wd', '--weight-decay', type=weight_decay, default=0, metavar='WEIGHT DECAY',
                    help='Weight decay (L2 regularization) size')
# Disables CUDA training
parser.add_argument('-nc', '--no-cuda', action='store_true',
                    help='Disables CUDA training')
# Learning rate decay milestones
parser.add_argument('-lrm', '--lr-milestones', type=milestones, metavar='LR MILESTONES',
                    help='Learning rate decay milestones (Insert comma seperated)')
# Learning rate decay multiple
parser.add_argument('-g', '--gamma', type=float, default=1, metavar='LR GAMMA',
                    help='Learning rate decay multiple')
# Logging interval between batches
parser.add_argument('-li', '--log-interval', type=int, default=10, metavar='LOG INTERVAL',
                    help='Logging interval between batches')
# Correction type to use
parser.add_argument('-gc', '--gradient-correction', metavar='GRADIENT CORRECTION',
                    choices=['none', 'worker', 'master'], default='none',
                    help='Correction type to use. Possible options: master, worker, none')
# Number of workers
parser.add_argument('-nw', '--num-workers', type=int, default=1, metavar='NUM WORKERS',
                    help='Number of workers')
# Minimum number of iterations for batch
parser.add_argument('-mind', '--min-delay', type=int, default=20, metavar='MIN DELAY',
                    help='Minimum number of iterations for batch')
# Mean number of iterations for batch
parser.add_argument('-meand', '--mean-delay', type=int, default=5, metavar='MEAN DELAY',
                    help='Mean number of iterations for batch')
# Standard Deviation of number of iterations for batch
parser.add_argument('-s', '--sigma', type=int, default=2, metavar='SIGMA',
                    help='Standard Deviation of number of iterations for batch')
# Distribution space of batch for worker
parser.add_argument('-ud', '--uni-dist', type=int, default=3, metavar='UNIFORM DISTRIBUTION',
                    help='Distribution space of batch for worker')
