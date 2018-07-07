from torch.autograd import Variable
from torch import nn
from torch.utils.data import DataLoader
from numpy import random
import net

class Worker():
    def __init__(self, args, optimizer, avg_delay, delay_sigma):
        self.args = args
        self.optimizer = optimizer
        self.min_delay = avg_delay - delay_sigma
        self.max_delay = avg_delay + delay_sigma

    def train(self, model, batch):
        # Run Network over one batch and calculate loss
        self.batch = batch
        data, target = batch
        # Move to GPU if possible
        if self.args.cuda:
            data, target = data.cuda(), target.cuda()

        # Turn data and targer into Variables for later gradient calc
        data, target = Variable(data), Variable(target)

        # Calculate loss and gradients
        output = model(data)
        loss = nn.functional.cross_entropy(output, target)  # nll_loss(output, target)
        self.optimizer.zero_grad()
        loss.backward()

        # Save gradients mean and variace per layer, then send back
        self.grad, self.mean, self.var = net.save_gradients(model)
        delay = random.randint(self.min_delay, self.max_delay)
        return (delay, self.grad, loss.data.item())

    def train2(self, model, batch, rmean, rvar):
        # Run Network over one batch and calculate loss
        self.batch = batch
        data, target = batch
        # Move to GPU if possible
        if self.args.cuda:
            data, target = data.cuda(), target.cuda()

        # Turn data and targer into Variables for later gradient calc
        data, target = Variable(data), Variable(target)

        # Calculate loss and gradients
        output = model(data)
        loss = nn.functional.cross_entropy(output, target)  # nll_loss(output, target)
        self.optimizer.zero_grad()
        loss.backward()

        # Save gradients mean and variace per layer, then send back
        self.grad, _, _ = net.save_gradients(model)
        delay = random.randint(self.min_delay, self.max_delay)
        return (delay, self.grad, rmean, rvar, loss.data.item())

    def update(self, model):
        # Run Network over one batch and calculate loss
        data, target = self.batch
        data, _ = data.split(int(self.args.batch_size/2))
        target, _ = target.split(int(self.args.batch_size/2))
        # Move to GPU if possible
        if self.args.cuda:
            data, target = data.cuda(), target.cuda()

        # Turn data and targer into Variables for later gradient calc
        data, target = Variable(data), Variable(target)

        # Calculate loss and gradients
        output = model(data)
        loss = nn.functional.cross_entropy(output, target)  # nll_loss(output, target)
        self.optimizer.zero_grad()
        loss.backward()

        # Send gradients and delay time
        _, mean, var = net.save_gradients(model)
        grad = net.update_grad2(model, self.grad, self.mean, self.var, mean, var)
        delay = random.randint(self.min_delay / 2, self.max_delay / 2)
        return (delay, grad, loss.data.item())
