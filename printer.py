import os


class Printer():
    def __init__(self, args, trainlen):
        self.args = args
        self.epoch_size = int(trainlen / self.args.batch_size)
        path = './logs/'
        if not os.path.exists(path):
            os.mkdir(path)
        dset = self.args.dataset
        wnum = '_' + str(self.args.num_workers)
        update = '_' + self.args.gradient_correction
        numlog = ' (1).log'
        self.file = path + dset + wnum + update + numlog
        while (os.path.isfile(self.file)):
            self.file = self.file[:-6] + str(int(self.file[-6])+1) + self.file[-5:]

    def train_print(self, epoch, batch, loss):
        if batch == 0:
            return

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

    def args_print(self):
        args_map = vars(self.args)
        for arg in args_map:
            print(arg, ": ", args_map[arg], end='\t')
        print()


#    def helpPrint(self, model, mean, var):
#       count = 0
#       with open(self.file, 'a') as writer:
#        for p in model.parameters():
#            count += 1
#            if p.grad is not None:
#                msg = "Batch layer: {} ---- mean: (mean: {}, var: {}) and var: (mean: {}, var:{})"
#                msg = msg.format(count, mean[p].mean(), mean[p].var(), var[p].mean(), var[p].var())
#                writer.write(msg + '\n')
