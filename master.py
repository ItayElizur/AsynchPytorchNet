import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import net
from torch.utils.data import DataLoader
from worker import Worker
from printer import Printer
from collections import deque
import hyparams
from numpy.random import normal


class Master():
    def __init__(self, args):
        # Define dataset and parameters
        args.dataset = args.dataset[0]
        dataset = net.Dataset[args.dataset]
        vars(args).update({k: dataset.value['parser'][k] for (k, v) in vars(args).items() if v == -1})
        self.args = args
        self.args.nesterov = not self.args.no_nesterov
        self.args.cuda = not self.args.no_cuda and torch.cuda.is_available()
        kwargs = {'num_workers': 1, 'pin_memory': True} if self.args.cuda else {}

        # Download training and testing set
        trainset = dataset.value['trainset']
        self.trainloader = DataLoader(
            trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
        testset = dataset.value['testset']
        self.testloader = torch.utils.data.DataLoader(
            testset, batch_size=self.args.tbatch_size, shuffle=True, **kwargs)

        # Initialize model
        self.model = dataset.value['net']()
        if self.args.cuda:
            self.model.cuda()
        self.optimizer = optim.SGD(
            self.model.parameters(), lr=self.args.learning_rate,
            momentum=self.args.momentum, nesterov=self.args.nesterov,
            weight_decay=self.args.weight_decay)
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=self.args.lr_milestones,
            gamma=self.args.gamma)

        # Initialize Workers and Printer
        self.workers_list = self.__assignWorker()
        self.printer = Printer(self.args, len(trainset))
        self.printer.args_print()
        print('DONE Initialize')

    def __assignWorker(self):
        workers_list = []
        for worker_num in range(self.args.num_workers):
            # Calculate mean delay of worker
            delay = int(normal(args.mean_delay, args.sigma))
            delay = min(delay, 2 * args.mean_delay)
            worker_delay = args.min_delay + max(0, delay)

            worker = Worker(self.args, self.optimizer, worker_delay, self.args.uni_dist)
            workers_list.append(worker)

        return workers_list

    def __init_params(self):
        self.rmean = {}
        self.rvar = {}
        self.max_delay = self.args.min_delay + 2 * self.args.mean_delay

    def __process_gradient(self, worker, work, batch):
        old_delay, gradient, loss = work
        if (old_delay == 0):
            return 1, worker.train(self.model, batch)
        if self.args.gradient_correction == 'worker' and old_delay > 0.8 * self.max_delay:
            return 0, worker.update(self.model)

        mean, var = net.load_gradients(self.model, gradient)
        if self.args.gradient_correction == 'master' and old_delay > 0.8 * self.max_delay:
            gradient = net.update_grad(self.model, gradient, mean, var, self.rmean, self.rvar)
        else:
            self.rmean, self.rvar = net.update_meanvar(self.model, self.rmean, self.rvar, mean, var)
        self.optimizer.step()
        # Calculate new gradient and delay
        return 1, worker.train(self.model, batch)

    def __process_gradient2(self, epoch, worker, work, batch):
        old_delay, gradient, mean, var, loss = work
        if (old_delay == 0):
            return 1, worker.train2(self.model, batch, self.rmean, self.rvar)

        nmean, nvar = net.load_gradients(self.model, gradient)
        if self.args.gradient_correction == 'master' and old_delay > 0.8 * self.max_delay and epoch > 1:
            gradient = net.update_grad(self.model, gradient, mean, var, self.rmean, self.rvar)
        else:
            self.rmean, self.rvar = net.update_meanvar(self.model, self.rmean, self.rvar, nmean, nvar)
        self.optimizer.step()
        # Calculate new gradient and delay
        return 1, worker.train2(self.model, batch, self.rmean, self.rvar)

    def train(self):
        self.__init_params()
        delay_arr = deque([deque([]) for i in range(self.max_delay + self.args.uni_dist + 1)])
        delay_arr[0] += [(worker, (0, None, 0)) for worker in self.workers_list]

        for epoch in range(1, self.args.epochs + 1):
            self.model.train()
            self.scheduler.step()
            latest_loss = 0
            for batch_num, batch in enumerate(self.trainloader):
                trained_batch = 0
                while trained_batch == 0:
                    while delay_arr[0] == deque([]):
                        delay_arr.rotate(-1)

                    worker, work = delay_arr[0].popleft()
                    latest_loss = work[2]
                    trained_batch, new_result = self.__process_gradient(worker, work, batch)
                    delay_arr[new_result[0]] += [(worker, new_result)]
                # print results
                if batch_num % self.args.log_interval == 0:
                    self.printer.train_print(epoch, batch_num, latest_loss)

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
            data, target = Variable(data), Variable(target)
            output = self.model(data)
            # sum up batch loss
            testloss += nn.functional.cross_entropy(output, target, size_average=False).data.item()
            # get the index of the max log-probability
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()

        testloss /= len(self.testloader.dataset)
        self.printer.test_print(testloss, correct, len(self.testloader.dataset))


if __name__ == "__main__":
    args = hyparams.parser.parse_args()
    master = Master(args)
    master.train()
