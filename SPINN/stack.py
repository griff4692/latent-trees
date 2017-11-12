import torch
from torch.autograd import Variable
import numpy as np
from random import random
import abc, six
from abc import ABCMeta
import math
from utils import cudify

def create_stack(args):
    if args.continuous_stack:
        return ContinuousStack(args)
    else:
        return DefaultStack(args)

@six.add_metaclass(ABCMeta)
class BaseStack:
    def zero_state(self):
        return (cudify(self.args, Variable(torch.zeros(1, self.dim), requires_grad=False)),
        cudify(self.args, Variable(torch.zeros(1, self.dim), requires_grad=False)))

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
    def __init__(self, args):
        self.args = args
        self.states = []
        self.dim = args.hidden_size

    def add(self, state, valence, id=0):
        self.states.append(state)

    def pop(self, valence):
        self.states.pop()

    def peek(self):
        if self.size() == 0:
            return self.zero_state()

        top = self.states[-1]
        return top

    def peek_two(self):
        if self.size() == 0:
            return self.zero_state(), self.zero_state()
        if self.size() == 1:
            return self.states[-1], self.zero_state()

        return self.states[-1], self.states[-2]

    def size(self):
        return len(self.states)

class ContinuousStack(BaseStack):
    def __init__(self, args):
        self.args = args
        self.dim = self.args.hidden_size
        self.valences = None
        self.hs = None
        self.cs = None

        self.other_stack = DefaultStack(args)

    def one_valence(self):
        return cudify(self.args, Variable(torch.FloatTensor([1]), requires_grad=False))

    def add(self, state, valence, id=0):
        assert len(state) == 2
        self.other_stack.add(state, valence, id)
        hs, cs = state

        # TODO this is defensive programming but may not be necessary
        valence = valence.clone()

        if self.valences is None:
            self.valences = valence
            self.hs, self.cs = hs, cs
        else:
            if not valence.size()[0] == 1:
                raise Exception("Adding more than one valence at a time.")

            self.valences = torch.cat([self.valences, valence], 0)
            self.hs = torch.cat([self.hs, hs], 0)
            self.cs = torch.cat([self.cs, cs], 0)


    def reduce(self, mass_remaining):
        mass_remaining = cudify(self.args, Variable(torch.FloatTensor([mass_remaining])))
        size = self.size()
        read_mask = cudify(self.args, Variable(torch.zeros(size, 1), requires_grad=False))
        idx = size - 1
        while mass_remaining.data[0] > 0.0 and idx >= 0:
            mass_remaining_data = mass_remaining.data[0]
            this_valence = self.valences[idx].data[0]
            if mass_remaining_data - this_valence >= 1.0:
                mass_coeff = self.valences[idx]
            elif mass_remaining_data > 1.0 and mass_remaining_data - this_valence < 1.0:
                skip_mass = mass_remaining - 1.0
                mass_coeff = self.valences[idx] - skip_mass
                read_mask[idx] = mass_coeff
            else:
                mass_coeff = torch.min(torch.cat([self.valences[idx], mass_remaining]))
                read_mask[idx] = mass_coeff

            mass_remaining -= mass_coeff
            idx -= 1

        reduced_hs = torch.mul(read_mask, self.hs).sum(0, keepdim=True)
        reduced_cs = torch.mul(read_mask, self.cs).sum(0, keepdim=True)
        return reduced_hs, reduced_cs

    def peek(self):
        val = self.other_stack.peek()

        if self.size() == 0:
            return self.zero_state()

        return self.reduce(1.0)

    def peek_two(self):
        if self.size() == 0:
            peek1 = self.zero_state()
            peek2 = self.zero_state()
        else:
            peek1 = self.reduce(1.0)
            peek2 = self.reduce(2.0)

        p1, p2 = self.other_stack.peek_two()

        if not np.all(peek1[0].data[0].numpy() == p1[0].data[0].numpy()):
            print(peek1[0].data[0].numpy()[0:20])
            print(p1[0].data[0].numpy()[0:20])
            if self.valences is not None:
                print(pre_valences.data.numpy())
            print(self.valences.data.numpy())
            print("1 error", self.size())
            raise

        if not np.all(peek1[1].data[0].numpy() == p1[1].data[0].numpy()):
            print("\n\n")
            print(peek1[1].data[0].numpy())
            print(p1[1].data[0].numpy())
            print(peek1[1].data[0].numpy() - p1[1].data[0].numpy())

            print("\n\n")
            print(peek1[0].data[0])
            print(p1[0].data[0])
            print(peek1[0].data[0].numpy()-p1[0].data[0].numpy())
            print("2 error", self.size())
            raise

        if not np.all(peek2[0].data[0].numpy() == p2[0].data[0].numpy()):
            print(peek2[0].data[0].numpy())
            print(p2[0].data[0].numpy())
            print(peek2[0].data[0].numpy() - p2[0].data[0].numpy())
            print(self.valences)
            print("3 error", self.size())
            raise

        if not np.all(peek2[1].data[0].numpy() == p2[1].data[0].numpy()):
            print(peek2[1].data[0].numpy() - p2[1].data[0].numpy())
            print("4 error")
            raise

        return peek1, peek2

    def size(self):
        if self.valences is None:
            return 0

        return self.valences.size()[0]

    def pop(self, valence):
        size = self.size()
        idx = size - 1
        mass_remaining = valence.clone()
        while mass_remaining.data[0] > 0.0 and idx >= 0:
            mass_coeff = torch.min(torch.cat([self.valences[idx], mass_remaining]))
            self.valences[idx] = self.valences[idx] - mass_coeff
            mass_remaining -= mass_coeff
            idx -= 1

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
