import torch
from torch.autograd import Variable

class Buffer():
    def __init__(self, h_s, c_s, args):
        self.states = zip(
            reversed(list(torch.split(h_s.squeeze(0), 1, 0))),
            reversed(list(torch.split(c_s.squeeze(0), 1, 0)))
        )
        
        self.args = args

        self.zero_state = (
            Variable(torch.zeros(1, self.args.hidden_size)),
            Variable(torch.zeros(1, self.args.hidden_size))
        )

    def pop(self):
        if self.size() == 0:
            raise Exception("Cannot pop from empty buffer")
        top = self.states.pop()
        return (top)

    def peek(self):
        if self.size() == 0:
            return self.zero_state
        return self.states[-1]

    def size(self):
        return len(self.states)
