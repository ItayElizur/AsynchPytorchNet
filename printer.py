import matplotlib.pyplot as plt
import os


class Printer():
    def __init__(self, args, trainlen):
        self.epoch_size = int(trainlen / args.batch_size)
        path = '../logs/'
        if not os.path.exists(path):
            os.mkdir(path)
        dset = args.dataset
        wnum = '_' + str(args.num_workers)
        update = '_' + args.gradient_correction
        self.file = path + dset + wnum + update + '.log'

    def train_print(self, epoch, batch, loss):
        plt.scatter(epoch * self.epoch_size + batch, loss)

        message = 'Train Epoch: {}\t[Completed Batches: {}]\tLoss: {:.6f}'
        message = message.format(epoch, batch, loss)
        with open(self.file, 'a') as writer:
            writer.write(message + '\n')
            print(message)

    def test_print(self, loss, correct, size):
        message = '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'
        message = message.format(loss, correct, size, 100. * correct / size)
        with open(self.file, 'a') as writer:
            writer.write(message)
            print(message)

    def show_loss(self):
        plt.show()
