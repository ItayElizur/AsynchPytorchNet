from torch.autograd import Variable
from torch import nn
from torch.utils.data import DataLoader
from numpy import random
import itertools
import net
from math import isnan

class Worker():
    def __init__(self, args, trainset, optimizer, avg_delay, delay_sigma):
        self.args = args
        kwargs = {'num_workers': 1, 'pin_memory': True} if self.args.cuda else {}
        self.trainloader = DataLoader(trainset, batch_size=args.batch_size,
                                      **kwargs)
        self.updateloader = DataLoader(trainset, batch_size=int(args.batch_size / 2),
                                      **kwargs)
        self.optimizer = optimizer
        self.min_delay = avg_delay - delay_sigma
        self.max_delay = avg_delay + delay_sigma

    def train(self, model, index):
        self.optimizer.zero_grad()

        # Run Network over one batch and calculate loss
        data, target = next(itertools.islice(self.trainloader, index, index + 1))
        # Move to GPU if possible
        if self.args.cuda:
            data, target = data.cuda(), target.cuda()

        # Turn data and targer into Variables for later gradient calc
        data, target = Variable(data), Variable(target)

        # Calculate loss and gradients
        output = model(data)
        loss = nn.functional.cross_entropy(output, target)  # nll_loss(output, target)
        loss.backward()

        # Save gradients mean and variace per layer, then send back
        self.grad, self.mean, self.var = net.save_gradients(model)
        delay = random.randint(self.min_delay, self.max_delay)
        return (delay, self.grad, loss.data[0])

    def update(self, model, index):
        self.optimizer.zero_grad()

        # Run Network over data batch and calculate loss
        data, target = next(itertools.islice(self.updateloader, index, index + 1))
        # Move to GPU if possible
        if self.args.cuda:
            data, target = data.cuda(), target.cuda()

        # Turn data and targer into Variables for later gradient calc
        data, target = Variable(data), Variable(target)

        # Calculate loss and gradients
        output = model(data)
        loss = nn.functional.cross_entropy(output, target) # nll_loss(output, target)
        loss.backward()

        # Send gradients and delay time
        _, mean, var = net.save_gradients(model)
        grad = net.update_grad(model, self.grad, self.mean, self.var, mean, var)
        delay = random.randint(self.min_delay / 2, self.max_delay / 2)
        return (delay, grad, loss.data[0])


#def helpPrint(model, mean, var):
#    count = 0
#    with open('../logs/CIFAR10_1.log', 'a') as writer:
#        for p in model.parameters():
#            count += 1
#            if p.grad is not None:
#                msg = "Batch layer: {} ---- mean: (mean: {}, var: {}) and var: (mean: {}, var:{})"
#                msg = msg.format(count, mean[p].mean(), mean[p].var(), var[p].mean(), var[p].var())
#                writer.write(msg + '\n')
