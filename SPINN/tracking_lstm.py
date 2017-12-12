import torch
import torch.nn as nn
from torch.autograd import Variable
from utils import cudify
import numpy as np

class TrackingLSTM(nn.Module):
    def __init__(self, args):
        super(TrackingLSTM, self).__init__()
        self.args = args
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
        self.tanh = nn.Tanh()

        act_emb_dim = 20
        self.act_embed = nn.Embedding(3, act_emb_dim, padding_idx=0)

        # buffer, top 2 elements on stack
        self.state_weights = nn.Linear(self.args.hidden_size, 4 * self.args.hidden_size, bias=False)
        self.input_weights = nn.Linear((3 * self.args.hidden_size) + act_emb_dim, 4 * self.args.hidden_size)
        # 2 actions: 0 (Reduce), 1 (Shift)
        self.prediction = nn.Linear(self.args.hidden_size, 2)


    def reset(self, other_sent):
        self.h, self.c = other_sent

    def lstm(self, inputs, prev_actions):
        h = self.state_weights(self.h) # batch, 4 * dim

        prev_actions += 1
        prev_act_embeds = self.act_embed(cudify(self.args, Variable(torch.LongTensor(prev_actions))))
        features = torch.cat([inputs, prev_act_embeds], dim=-1)
        inputs_transform = self.input_weights(features)

        x_plus_h = h + inputs_transform
        (i, f, o, g) = torch.chunk(x_plus_h, 4, 1) # (batch, dim) x 4

        c = torch.mul(self.c, self.sigmoid(f)) + torch.mul(self.sigmoid(i), self.tanh(g))
        h = torch.mul(self.sigmoid(o), self.tanh(c))

        self.h, self.c = h, c
        nonlinear = self.sigmoid if self.args.continuous_stack else self.softmax
        prediction = nonlinear(self.prediction(self.h))
        return (prediction, self.h)


    def forward(self, input, prev_actions):
        return self.lstm(input, prev_actions)

class PolicyTrackingLSTM(nn.Module):
    def __init__(self, args):
        super(PolicyTrackingLSTM, self).__init__()
        self.args = args
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
        self.tanh = nn.Tanh()

        act_emb_dim = 20
        self.act_embed = nn.Embedding(3, act_emb_dim, padding_idx=0)

        # buffer, top 2 elements on stack
        self.state_weights = nn.Linear(self.args.hidden_size, 4 * self.args.hidden_size, bias=False)
        self.input_weights = nn.Linear((3 * self.args.hidden_size) + act_emb_dim, 4 * self.args.hidden_size)
        # 2 actions: 0 (Reduce), 1 (Shift)
        self.prediction = nn.Linear(self.args.hidden_size, 2)

    def reset(self, other_sent):
        self.actions = []
        self.ignored = []
        self.h, self.c = other_sent

    def add_action(self, action):
        self.actions.append(action)

    def add_ignored(self, ignored):
        self.ignored.append(ignored)

    def lstm(self, inputs, prev_actions):
        h = self.state_weights(self.h) # batch, 4 * dim

        prev_actions += 1
        prev_act_embeds = self.act_embed(cudify(self.args, Variable(torch.LongTensor(prev_actions))))
        features = torch.cat([inputs.detach(), prev_act_embeds], dim=-1)
        inputs_transform = self.input_weights(features)
        x_plus_h = h + inputs_transform
        (i, f, o, g) = torch.chunk(x_plus_h, 4, 1) # (batch, dim) x 4

        c = torch.mul(self.c, self.sigmoid(f)) + torch.mul(self.sigmoid(i), self.tanh(g))
        h = torch.mul(self.sigmoid(o), self.tanh(c))

        self.h, self.c = h, c
        nonlinear = self.sigmoid if self.args.continuous_stack else self.softmax
        prediction = nonlinear(self.prediction(self.h))
        return (prediction, self.h)

    def forward(self, input, prev_actions):
        return self.lstm(input, prev_actions)


class PolicyNetwork(nn.Module):
    def __init__(self, args):
        super(PolicyNetwork, self).__init__()
        self.args = args
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
        self.relu = nn.ReLU()

        act_emb_dim = 20
        self.act_embed = nn.Embedding(3, act_emb_dim, padding_idx=0)

        # buffer, top 2 elements on stack
        self.input_weights = nn.Linear((3 * self.args.hidden_size) + act_emb_dim, self.args.hidden_size)
        # 2 actions: 0 (Reduce), 1 (Shift)
        self.prediction = nn.Linear(self.args.hidden_size, 2)
        self.actions = []
        self.ignored = []


    def network(self, inputs, prev_actions):
        prev_actions += 1
        prev_act_embeds = self.act_embed(cudify(self.args, Variable(torch.LongTensor(prev_actions))))
        features = torch.cat([inputs.detach(), prev_act_embeds], dim=-1)

        inputs_transform = self.input_weights(features)
        x_plus_h = self.relu(inputs_transform)

        prediction = self.softmax(self.prediction(x_plus_h))
        return (prediction, None)

    def reset(self, other_sent):
        self.actions = []
        self.ignored = []

    def add_action(self, action):
        self.actions.append(action)

    def add_ignored(self, ignored):
        self.ignored.append(ignored)

    def forward(self, input, prev_actions):
        return self.network(input, prev_actions)
