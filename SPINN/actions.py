import torch
import torch.nn as nn
import numpy as np

# Original Code Base
def HeKaimingInitializer(param, cuda=False):
    fan = param.size()
    init = np.random.normal(scale=np.sqrt(4.0 / (fan[0] + fan[1])), size=fan).astype(np.float32)
    if cuda:
       param.data.set_(torch.from_numpy(init).cuda())
    else:
       param.data.set_(torch.from_numpy(init))

# Stack overflow
class LayerNormalization(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        super(LayerNormalization, self).__init__()

        self.eps = eps
        self.a2 = nn.Parameter(torch.ones(1, hidden_size), requires_grad=True)
        self.b2 = nn.Parameter(torch.zeros(1, hidden_size), requires_grad=True)

    def forward(self, z):
        mu = torch.mean(z)
        sigma = torch.std(z)
        ln_out = (z - mu) / (sigma + self.eps)
        ln_out = ln_out * self.a2 + self.b2
        return ln_out

class ReduceHyp(nn.Module):
    def __init__(self, dim_size):
        super(ReduceHyp, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.compose_left = nn.Linear(dim_size, 5 * dim_size)
        HeKaimingInitializer(self.compose_left.weight)
        self.compose_right = nn.Linear(dim_size, 5 * dim_size, bias=False)
        HeKaimingInitializer(self.compose_right.weight)
        self.compose_premise = nn.Linear(dim_size, dim_size)
        HeKaimingInitializer(self.compose_premise.weight)
        self.compose_attn = nn.Linear(dim_size, dim_size)
        HeKaimingInitializer(self.compose_attn.weight)
        self.compose_state = nn.Linear(dim_size, dim_size)
        HeKaimingInitializer(self.compose_state.weight)
        self.left_ln = LayerNormalization(dim_size)
        self.right_ln = LayerNormalization(dim_size)


    def lstm(self, input, cl, cr, attend):
        (i, fl, fr, o, g) = torch.chunk(input, 5, 1)
        c = torch.mul(cl, self.sigmoid(fl)) + torch.mul(cr, self.sigmoid(fr)) + \
            torch.mul(self.sigmoid(i), self.tanh(g))
        h = torch.mul(self.sigmoid(o), self.tanh(c))
        h = h + torch.mul(self.sigmoid(o), self.tanh(attend))
        a = self.attend_premise + self.compose_attn(attend) + self.compose_state(c)
        return (h, c, a)

    def forward(self, sl, sr, trans1):
        (hl, cl, al) = sl
        (hr, cr, ar) = sr
        input_lstm_left = self.compose_left(self.left_ln(hl))
        input_lstm_right = self.compose_right(self.right_ln(hr))

        self.attend_premise = self.compose_premise(trans1)
        attend = (al + ar) * self.attend_premise

        input_lstm = input_lstm_right + input_lstm_left

        output = self.lstm(input_lstm, cl, cr, attend)
        return (torch.split(output[0], 1), torch.split(output[1], 1), torch.split(output[2], 1))

class ReducePrem(nn.Module):
    def __init__(self, dim_size):
        super(ReducePrem, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.compose_left = nn.Linear(dim_size, 5 * dim_size)
        HeKaimingInitializer(self.compose_left.weight)
        self.compose_right = nn.Linear(dim_size, 5 * dim_size, bias=False)
        HeKaimingInitializer(self.compose_right.weight)
        self.left_ln = LayerNormalization(dim_size)
        self.right_ln = LayerNormalization(dim_size)


    def lstm(self, input, cl, cr):
        (i, fl, fr, o, g) = torch.chunk(input, 5, 1)
        c = torch.mul(cl, self.sigmoid(fl)) + torch.mul(cr, self.sigmoid(fr)) + \
            torch.mul(self.sigmoid(i), self.tanh(g))
        h = torch.mul(self.sigmoid(o), self.tanh(c))
        return (h, c)

    def forward(self, sl, sr):
        (hl, cl) = sl
        (hr, cr) = sr
        input_lstm_left = self.compose_left(self.left_ln(hl))
        input_lstm_right = self.compose_right(self.right_ln(hr))

        input_lstm = input_lstm_right + input_lstm_left

        output = self.lstm(input_lstm, cl, cr)
        return (torch.split(output[0], 1), torch.split(output[1], 1))