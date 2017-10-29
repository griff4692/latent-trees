import torch
import torch.nn as nn
from torch.autograd import Variable
from spinn import SPINN

class SNLIClassifier(nn.Module):
    def __init__(self, embed_dim, vocab_size, hidden_size):
        super(SNLIClassifier, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.output = nn.Linear(4 * hidden_size, 4)
        self.encoder = SPINN(embed_dim, hidden_size)

    def set_weight(self, weight):
        self.embed.weight.data.copy_(torch.from_numpy(weight))
        self.embed.weight.requires_grad = False

    def forward(self, hypothesis, premise):
        prem_embed = self.embed(premise[0])
        hypo_embed = self.embed(hypothesis[0])

        encode_prem = self.encoder(prem_embed, premise[1])
        encode_hyp = self.encoder(hypo_embed, hypothesis[1])
        #TODO: Currently too simple (Reproduce the paper)
        input = torch.cat([encode_hyp, encode_prem, encode_prem - encode_hyp, torch.mul(encode_hyp, encode_prem)], dim=1)
        return nn.Softmax()(self.output(input))