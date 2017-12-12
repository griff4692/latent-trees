import torch
from torch.autograd import Variable
import numpy as np
from random import random
import abc, six
from abc import ABCMeta
import math
from utils import cudify
import torch.nn.functional as F

def create_stack(args, max_size):
    if args.continuous_stack:
        return ContinuousStack(args, max_size)
    else:
        return DefaultStack(args)

@six.add_metaclass(ABCMeta)
class BaseStack:
    @abc.abstractmethod
    def add(self, state, valence, id=0):
        pass

    @abc.abstractmethod
    def pop(self, valence):
        pass

    @abc.abstractmethod
    def peek(self):
        pass

    @abc.abstractmethod
    def peek_two(self):
        pass

class DefaultStack(BaseStack):
    def __init__(self, args):
        self.args = args
        self.states = []
        self.dim = args.hidden_size
        self.zero_state = (cudify(self.args, Variable(torch.zeros(1, self.dim), requires_grad=False)),
                cudify(self.args, Variable(torch.zeros(1, self.dim), requires_grad=False)))

    def add(self, state, valence, id=0):
        self.states.append(state)

    def pop(self, valence):
        try:
            self.states.pop()
            return True
        except IndexError:
            return False

    def peek(self):
        if self.size() == 0:
            return self.zero_state

        top = self.states[-1]
        return top

    def peek_two(self):
        if self.size() == 0:
            return self.zero_state, self.zero_state
        if self.size() == 1:
            return self.states[-1], self.zero_state

        return self.states[-1], self.states[-2]

    def size(self):
        return len(self.states)

class ContinuousStack(BaseStack):
    def __init__(self, args, max_size):
        self.args = args
        self.dim = self.args.hidden_size
        self.max_size = max_size
        self.valences = None # cudify(self.args, Variable(torch.zeros(max_size, 1), requires_grad=False))
        self.cum_valences = None # cudify(self.args, Variable(torch.zeros(max_size, 1), requires_grad=False))
        self.h = None # cudify(self.args, Variable(torch.zeros(max_size, self.dim), requires_grad=False))
        self.c = None # cudify(self.args, Variable(torch.zeros(max_size, self.dim), requires_grad=False))
        self.stack_p = 0
        self.num_pop = 0
        self.zero_state = Variable(torch.FloatTensor(1, self.dim).zero_(), requires_grad=False)

    def one_valence(self):
        return cudify(self.args, Variable(torch.FloatTensor([[1.0]]), requires_grad=False))

    def zero_valence(self):
        return cudify(self.args, Variable(torch.FloatTensor([[0.0]]), requires_grad=False))

    def add(self, state, valence, id=0):
        hs, cs = state

        valence = valence.unsqueeze(1)
        if self.size() == 0:
            self.valences = valence
            self.h = hs
            self.c = cs
            self.cum_valences = self.zero_valence()
        else:
            self.valences = torch.cat([self.valences, valence], dim=0)
            self.h = torch.cat([self.h, hs], dim=0)
            self.c = torch.cat([self.c, cs], dim=0)
            self.cum_valences = self.cum_valences + valence
            self.cum_valences = torch.cat([self.cum_valences, self.zero_valence()], dim=0)

        self.stack_p += 1


    def peek(self):
        if self.size() == 0:
            return self.zero_state, self.zero_state

        mask = F.relu(1.0 - self.cum_valences) # ReLU
        x = torch.cat([self.valences, mask], dim=1)

        x_min, _ = torch.min(x, dim=1)
        read_mask = x_min.unsqueeze(1)
        h = torch.mul(read_mask, self.h.clone()).sum(dim=0, keepdim=True)
        c = torch.mul(read_mask, self.c.clone()).sum(dim=0, keepdim=True)

        return h, c

    def print_stack_state(self):
        print("Pushes=%d.  Pops=%d" % (self.stack_p, self.num_pop))
        print("Stack pointer is at index %d" % self.stack_p)
        print("\nValences\n")
        print(self.valences[:self.stack_p + 1].data.numpy())
        print("\nCumValences\n")
        print(self.cum_valences[:self.stack_p + 1].data.numpy())
        print("\n\n")

    def peek_two(self):
        if self.size() == 0:
            return (self.zero_state, self.zero_state), (self.zero_state, self.zero_state)

        h1, c1 = self.peek()

        valence = self.one_valence()
        temp_valences = F.relu(self.valences - F.relu(valence - self.cum_valences))
        temp_cum_valences = F.relu(self.cum_valences - valence)

        temp_mask = F.relu(1.0 - temp_cum_valences)
        min_val, _ = torch.min(torch.cat([temp_valences, temp_mask], dim=1), dim=1)
        temp_read_mask = min_val.unsqueeze(1)

        h2 = torch.mul(temp_read_mask, self.h).sum(dim=0, keepdim=True)
        c2 = torch.mul(temp_read_mask, self.c).sum(dim=0, keepdim=True)
        return (h1, c1), (h2, c2)

    def size(self):
        return self.stack_p

    def pop(self, valence):
        if self.size() == 0:
            return True
        self.num_pop += 1

        plus = torch.cat([valence, F.relu(self.valences.sum() - 1.0)], dim=0)
        valence, _ = torch.min(plus, dim=0)

        thresh = F.relu(valence - self.cum_valences.clone())
        self.valences = F.relu(self.valences - thresh)
        self.cum_valences = F.relu(self.cum_valences - valence)
        return True

# register all subclasses to base class
BaseStack.register(DefaultStack)
BaseStack.register(ContinuousStack)

if __name__=='__main__':
    def rand_vec(dim):
        return np.random.rand(dim,)

    dim = 1
    s = Stack(dim)

    vec = rand_vec(dim)
    print("Adding %.2f with strength %.2f" % (vec, 0.5))
    s.add(vec, 0.5)
    print("Read is %.2f" % s.peek()[0])

    vec = rand_vec(dim)
    print("Adding %.2f with strength %.2f" % (vec, 0.5))
    s.add(vec, 0.5)
    print("Read is %.2f" % s.peek()[0])

    print("Popping 0.8")
    s.pop(0.8)
    print("Read is %.2f" % s.peek()[0])

    vec = rand_vec(dim)
    print("Adding %.2f with strength %.2f" % (vec, 0.9))
    s.add(vec, 0.9)
    print("Read is %.2f" % s.peek()[0])

    print("Popping 0.5")
    s.pop(0.5)
    print("Read is %.2f" % s.peek()[0])
