import os
import shutil
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import net
from torch.utils.data import DataLoader
from worker import Worker
from collections import deque
import hyparams


class Master():
    def __init__(self, args):
        # Define dataset and parameters
        args.dataset = args.dataset[0]
        dataset = net.Dataset[args.dataset]
        vars(args).update({k: dataset.value['parser'][k] for (k, v) in vars(args).items() if v == -1})
        self.args = args

        self.args.cuda = not self.args.no_cuda and torch.cuda.is_available()
        kwargs = {'num_workers': 1, 'pin_memory': True} if self.args.cuda else {}

        # Download training and testing set
        self.trainset = dataset.value['trainset']
        testset = dataset.value['testset']
        self.testloader = torch.utils.data.DataLoader(
            testset, batch_size=self.args.test_batch_size, shuffle=True, **kwargs)

        # Initialize model
        self.model = dataset.value['net']()
        if self.args.cuda:
            self.model.cuda()
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.args.learning_rate,
                                   momentum=self.args.momentum, nesterov=self.args.nesterov)

        # Initialize Workers
        self.workers_list = self.__assignWorker()
        path = '../logs/'
        if not os.path.exists(path):
            os.mkdir(path)
        dset = self.args.dataset
        wnum = '_' + str(self.args.num_workers)
        update = '_' + self.args.gradient_correction
        self.file = path + dset + wnum + update + '.log'

    def __assignWorker(self):
        workers_list = []
        for worker_num in range(self.args.num_workers):
            worker_delay = net.randomDelay(self.args)
            worker = Worker(self.args, self.trainset, self.optimizer, worker_delay, self.args.uni_dist)
            workers_list.append(worker)

        return workers_list

    def __init_params(self):
        self.rmean = {}
        self.rvar = {}
        self.max_delay = self.args.min_delay + 2 * self.args.mean_delay

    def __process_gradient(self, worker, work, batch_num):
        old_delay, gradient, loss = work
        if self.args.gradient_correction == 'worker' and old_delay > 0.8 * self.max_delay:
            return worker.update(self.model, batch_num)

        mean, var = net.load_gradients(self.model, gradient)
        if self.args.gradient_correction == 'master' and old_delay > 0.8 * self.max_delay:
            gradient = net.update_grad(self.model, gradient, mean, var, self.rmean, self.rvar)
        else:
            self.rmean, self.rvar = net.update_meanvar(self.model, self.rmean, self.rvar, mean, var)
        self.optimizer.step()
        self.optimizer.zero_grad()
        # Calculate new gradient and delay
        return worker.train(self.model, batch_num)

    def train(self):
        self.__init_params()
        delay_arr = deque([[] for i in range(self.max_delay + self.args.uni_dist + 1)])
        batch_num = 0
        self.model.train()
        for i, worker in enumerate(self.workers_list):
            result = worker.train(self.model, batch_num)
            delay_arr[result[0]] += [(worker, result)]
            batch_num += 1

        for epoch in range(1, self.args.epochs + 1):
            latest_loss = 0
            while batch_num < int(len(self.trainset) / self.args.batch_size):
                delay_arr.append([])
                for worker, work in delay_arr.popleft():
                    latest_loss = work[2]
                    new_result = self.__process_gradient(worker, work, batch_num)
                    delay_arr[new_result[0]] += [(worker, new_result)]
                    batch_num += 1
                    # print results
                    if batch_num % self.args.log_interval == 0:
                        self.__train_print(epoch, batch_num, latest_loss)

            # Every epoch test results
            self.test()
            batch_num = 0

    def test(self):
        self.model.eval()
        testloss = 0
        correct = 0
        for data, target in self.testloader:
            if self.args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=True), Variable(target)
            output = self.model(data)
            # sum up batch loss
            testloss += nn.functional.cross_entropy(output, target, size_average=False).data[0]
            # get the index of the max log-probability
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()

        testloss /= len(self.testloader.dataset)
        self.__test_print(testloss, correct, len(self.testloader.dataset))

    def __train_print(self, epoch, batch, loss):
        message = 'Train Epoch: {}\t[Completed Batches: {}]\tLoss: {:.6f}'
        message = message.format(epoch, batch, loss)
        with open(self.file, 'a') as writer:
            writer.write(message + '\n')
            print(message)

    def __test_print(self, loss, correct, size):
        message = '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'
        message = message.format(loss, correct, size, 100. * correct / size)
        with open(self.file, 'a') as writer:
            writer.write(message)
            print(message)


if __name__ == "__main__":
    args = hyparams.parser.parse_args()
    master = Master(args)
    master.train()
