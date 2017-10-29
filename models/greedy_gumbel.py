import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from utils import *

class GreedyGumbel(nn.Module):
    def __init__(self, args, vocab):
        super(GreedyGumbel, self).__init__()

        self.args = args
        self.vocab = vocab

        self.embeddings = nn.Embedding(self.vocab.size(), self.args.embed_dim, padding_idx=self.vocab.pad_idx())
        initialize_embs(self.parameters, self.vocab)

        self.linear = nn.Linear(1, 1)

    def forward(self, (sents1, sents2)):
        dummy_output = Variable(torch.zeros(self.args.batch_size, 1))
        return self.linear(dummy_output)
