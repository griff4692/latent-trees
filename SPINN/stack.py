import torch
from torch.autograd import Variable
import numpy as np
from random import random
import abc, six
from abc import ABCMeta


def create_stack(dim, use_continuous=False):
    if use_continuous:
        return ContinuousStack(dim)
    else:
        return DefaultStack(dim)

@six.add_metaclass(ABCMeta)
class BaseStack:

    def __init_(self):
        pass

    @abc.abstractmethod
    def add(self, state, valence, id=0):
        pass

    @abc.abstractmethod
    def pop(self, valence):
        pass

    @abc.abstractmethod
    def peek(self):
        pass

class DefaultStack(BaseStack):
    # Figure out Thin Stack.
    # Seems like an overhead at this stage.
    def __init__(self, dim):
        self.states = []
        self.dim = dim

    def add(self, state, valence, id=0):
        self.states.append(state)

    def pop(self, valence):
        self.states.pop()

    def peek(self):
        top = self.states[-1]
        return top

    def size(self):
        return len(self.states)

class ContinuousStack(BaseStack):
    def __init__(self, dim):
        self.dim = dim
        self.valences = None
        self.hs = None
        self.cs = None

    def add(self, state, valence, id=0):
        if self.valences is None:
            self.valences = valence
            self.hs, self.cs = state
        else:
            self.valences = torch.cat([self.valences, valence], 0)
            self.hs = torch.cat([self.hs, state[0]], 0)
            self.cs = torch.cat([self.cs, state[1]], 0)


    def reduce(self, flavor, mass_remaining):
        size = self.size()

        if size == 0:
            print ("Warning!  Empty stack...")
            return None

        if flavor == 'peek':
            # don't overwrite
            read_mask = Variable(torch.FloatTensor(size,).zero_())

        # top of the stack
        idx = size - 1
        while mass_remaining.data[0] > 0.0 and idx >=0:
            mass_coeff = torch.min(self.valences[idx],
                mass_remaining)

            if flavor == 'peek':
                read_mask[idx] = mass_coeff
            else:
                self.valences[idx] = self.valences[idx] - mass_coeff

            mass_remaining -= mass_coeff
            idx -= 1

        if flavor == 'peek':
            read_mask = read_mask.view(size, 1)
            reduced_hs = torch.mul(read_mask, self.hs).sum(0, keepdim=True)
            reduced_cs = torch.mul(read_mask, self.cs).sum(0, keepdim=True)

            return reduced_hs, reduced_hs


    def peek(self):
        valence = Variable(torch.FloatTensor([1.0]))
        return self.reduce('peek', valence)

    def size(self):
        if self.valences is None:
            return 0

        return self.valences.size()[0]

    def pop(self, valence):
        self.reduce('pop', valence)


# register all subclasses to base class
BaseStack.register(DefaultStack)
BaseStack.register(ContinuousStack)

if __name__=='__main__':
    def rand_vec(dim):
        return np.random.rand(dim,)

    dim = 1
    s = Stack(dim)

    vec = rand_vec(dim)
    print ("Adding %.2f with strength %.2f" % (vec, 0.5))
    s.add(vec, 0.5)
    print ("Read is %.2f" % s.peek()[0])

    vec = rand_vec(dim)
    print ("Adding %.2f with strength %.2f" % (vec, 0.5))
    s.add(vec, 0.5)
    print ("Read is %.2f" % s.peek()[0])

    print ("Popping 0.8")
    s.pop(0.8)
    print ("Read is %.2f" % s.peek()[0])

    vec = rand_vec(dim)
    print ("Adding %.2f with strength %.2f" % (vec, 0.9))
    s.add(vec, 0.9)
    print ("Read is %.2f" % s.peek()[0])

    print ("Popping 0.5")
    s.pop(0.5)
    print ("Read is %.2f" % s.peek()[0])
