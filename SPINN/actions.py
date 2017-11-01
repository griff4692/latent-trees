import torch
import torch.nn as nn

class Reduce(nn.Module):
    def __init__(self, dim_size, track=False):
        super(Reduce, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.track = track
        if not track:
            self.compose = nn.Linear(2 * dim_size, 5 * dim_size)
        else:
            self.compose = nn.Linear(3 * dim_size, 5 * dim_size)

    def lstm(self, input, cl, cr):
        (i, fl, fr, o, g) = torch.chunk(input, 5, 1)
        c = torch.mul(cl, self.sigmoid(fl)) + torch.mul(cr, self.sigmoid(fr)) + \
            torch.mul(self.sigmoid(i), self.tanh(g))
        h = torch.mul(o, self.tanh(c))
        return (h, c)

    def forward(self, sl, sr, e=None):
        (hl, cl) = sl
        (hr, cr) = sr
        if self.track:
            input_lstm = self.compose(torch.cat([hl, hr, e], dim=1))
        else:
            input_lstm = self.compose(torch.cat([hl, hr], dim = 1))
        output = self.lstm(input_lstm, cl, cr)

        return (torch.split(output[0], 1), torch.split(output[1], 1))
