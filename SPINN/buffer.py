import torch

class BufferHyp():
    def __init__(self, h_s, c_s, a_s):
        self.states = list(zip(
            list(torch.split(h_s.squeeze(0), 1, 0)),
            list(torch.split(c_s.squeeze(0), 1, 0)),
            list(torch.split(a_s.squeeze(0), 1, 0))
        ))


    def pop(self):
        top = self.states.pop()
        return (top)

class BufferPrem():
    def __init__(self, h_s, c_s):
        self.states = list(zip(
            list(torch.split(h_s.squeeze(0), 1, 0)),
            list(torch.split(c_s.squeeze(0), 1, 0))
        ))


    def pop(self):
        top = self.states.pop()
        return (top)