import torch
import torch.nn as nn
from torch.autograd import Variable

class TrackingLSTM(nn.Module):
    def __init__(self, args):
        super(TrackingLSTM, self).__init__()
        self.args = args
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
        self.tanh = nn.Tanh()

        # buffer, top 2 elements on stack
        self.state_weights = nn.Linear(self.args.hidden_size, 4 * self.args.hidden_size, bias=False)
        self.input_weights = nn.Linear(3 * self.args.hidden_size, 4 * self.args.hidden_size)

        # 3 actions: 0 (Pad), 1 (Reduce), 2 (Shift)
        self.prediction = nn.Linear(self.args.hidden_size, 3)


    def initialize_states(self, batch_size):
        self.h = Variable(torch.zeros(batch_size, self.args.hidden_size))
        self.c = Variable(torch.zeros(batch_size, self.args.hidden_size))

    def lstm(self, inputs, predict=True):
        h = self.state_weights(self.h) # batch, 4 * dim
        inputs_transform = self.input_weights(inputs)
        x_plus_h = h + inputs_transform
        (i, f, o, g) = torch.chunk(x_plus_h, 4, 1) # (batch, dim) x 4

        c = torch.mul(self.c, self.sigmoid(f)) + torch.mul(self.sigmoid(i), self.tanh(g))
        h = torch.mul(o, self.tanh(c))

        self.h, self.c = h, c

        prediction = None
        if predict:
            prediction = self.softmax(self.prediction(self.h))
        return prediction


    def forward(self, input, predict=True):
        return self.lstm(input, predict)