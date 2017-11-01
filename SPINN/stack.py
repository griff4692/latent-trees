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
    def __init__(self, dim):
        self.zero_state = (
            Variable(torch.zeros(1, dim), requires_grad=False),
            Variable(torch.zeros(1, dim), requires_grad=False)
        )

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
    # TODO Figure out Thin Stack.
    def __init__(self, dim):
        super(DefaultStack, self).__init__(dim)
        self.states = []
        self.dim = dim

    def add(self, state, valence, id=0):
        self.states.append(state)

    def pop(self, valence):
        self.states.pop()

    def peek(self):
        if self.size() == 0:
            return zero_state

        top = self.states[-1]
        return top

    def peek_two(self):
        if self.size() == 0:
            return self.zero_state, self.zero_state
        if self.size() == 1:
            return self.states[-1], self.zero_state

        return self.states[-1], self.states[-2]

        second_top = self.states[-2]
        return second_top

    def size(self):
        return len(self.states)

class ContinuousStack(BaseStack):
    def __init__(self, dim):
        super(ContinuousStack, self).__init__(dim)
        self.dim = dim
        self.valences = None
        self.hs = None
        self.cs = None
        self.one_valence = Variable(torch.FloatTensor([1]), requires_grad=False)


    def add(self, state, valence, id=0):
        if self.valences is None:
            self.valences = valence
            self.hs, self.cs = state
        else:
            if not valence.size()[0] == 1:
                raise Exception("Adding more than one valence at a time.")

            self.valences = torch.cat([self.valences, valence], 0)
            self.hs = torch.cat([self.hs, state[0]], 0)
            self.cs = torch.cat([self.cs, state[1]], 0)

    def reduce(self, flavor, mass_remaining):
        size = self.size()
        coeff = 1.0 if flavor == 'restore' else -1.0

        if flavor == 'peek':
            # don't overwrite
            read_mask = Variable(torch.FloatTensor(size, 1).zero_())

        # top of the stack
        idx = size - 1

        while mass_remaining.data[0] > 0.0 and idx >=0:
            mass_coeff = torch.min(torch.cat([self.valences[idx], mass_remaining]))
            if flavor == 'peek':
                read_mask[idx] = mass_coeff
            else:
                self.valences[idx] = self.valences[idx] + (coeff * mass_coeff)

            mass_remaining -= mass_coeff
            idx -= 1

        if flavor == 'peek':
            reduced_hs = torch.mul(read_mask, self.hs).sum(0, keepdim=True)
            reduced_cs = torch.mul(read_mask, self.cs).sum(0, keepdim=True)

            return reduced_hs, reduced_hs


    def peek(self):
        if self.size() == 0:
            return self.zero_state

        return self.reduce('peek', self.one_valence)

    def peek_two(self):
        if self.size() == 0:
            return self.zero_state, self.zero_state

        top1 = self.reduce('peek', self.one_valence)

        # temporarily reduce mass
        self.reduce('pop', self.one_valence)
        top2 = self.reduce('peek', self.one_valence)
        # restore mass you temporarily took off
        self.reduce('restore', self.one_valence)

        return top1, top2

    def size(self):
        if self.valences is None:
            return 0

        return self.valences.size()[0]

    def pop(self, valence):
        self.reduce('pop', valence)

    def restore(self, valence):
        self.reduce('restore', valence)


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
