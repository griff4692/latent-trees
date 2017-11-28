import torch
import torch.nn as nn
from torch.autograd import Variable
from utils import cudify

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
        # 2 actions: 0 (Reduce), 1 (Shift)
        self.prediction = nn.Linear(self.args.hidden_size, 2)


    def initialize_states(self, other_sent):
        self.h, self.c = other_sent

    def reset(self):
        pass

    def lstm(self, inputs, predict=True):
        h = self.state_weights(self.h) # batch, 4 * dim

        inputs_transform = self.input_weights(inputs)
        x_plus_h = h + inputs_transform
        (i, f, o, g) = torch.chunk(x_plus_h, 4, 1) # (batch, dim) x 4

        c = torch.mul(self.c, self.sigmoid(f)) + torch.mul(self.sigmoid(i), self.tanh(g))
        h = torch.mul(self.sigmoid(o), self.tanh(c))

        self.h, self.c = h, c

        prediction = None
        if predict:
            nonlinear = self.sigmoid if self.args.continuous_stack else self.softmax
            prediction = nonlinear(self.prediction(self.h))
        return (prediction, self.h)


    def forward(self, input, predict=True):
        return self.lstm(input, predict)

class PolicyNetwork(nn.Module):
    def __init__(self, args):
        super(PolicyNetwork, self).__init__()
        self.args = args
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
        self.relu = nn.ReLU()

        # buffer, top 2 elements on stack
        self.input_weights = nn.Linear(3 * self.args.hidden_size, self.args.hidden_size)
        # 2 actions: 0 (Reduce), 1 (Shift)
        self.prediction = nn.Linear(self.args.hidden_size, 2)
        self.actions = []
        self.ignored = []


    def network(self, inputs):
        inputs_transform = self.input_weights(inputs.detach())
        x_plus_h = self.relu(inputs_transform)

        prediction = self.softmax(self.prediction(x_plus_h))
        return (prediction, None)

    def reset(self):
        self.actions = []
        self.ignored = []

    def add_action(self, action):
        self.actions.append(action)

    def add_ignored(self, ignored):
        self.ignored.append(ignored)

    def forward(self, input, predict=True):
        return self.network(input)
