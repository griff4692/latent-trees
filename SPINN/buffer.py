import torch

class Buffer():
    def __init__(self, h_s, c_s):
        self.states = list(zip(
            reversed(list(torch.split(h_s.squeeze(0), 1, 0))),
            reversed(list(torch.split(c_s.squeeze(0), 1, 0)))
        ))

    def pop(self):
        top = self.states.pop()
        return (top)
