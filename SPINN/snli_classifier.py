import torch
import torch.nn as nn
from torch.autograd import Variable
from spinn import SPINN
from actions import HeKaimingInitializer, LayerNormalization

class SNLIClassifier(nn.Module):
    def __init__(self, args, vocab_size):
        self.args = args
        super(SNLIClassifier, self).__init__()
        self.embed = nn.Embedding(vocab_size, self.args.embed_dim)
        self.softmax = nn.Softmax()
        self.relu = nn.ReLU()

        self.layer_norm_mlp_input = LayerNormalization(2 * self.args.hidden_size)
        self.layer_norm_mlp2_hidden = LayerNormalization(self.args.snli_h_dim)
        self.layer_norm_mlp1_hidden = LayerNormalization(self.args.snli_h_dim)

        self.dropout = nn.Dropout(p=self.args.dropout_rate_classify)
        self.mlp1 = nn.Linear(4 * self.args.hidden_size // 2, self.args.snli_h_dim)
        HeKaimingInitializer(self.mlp1.weight)
        self.mlp2 = nn.Linear(self.args.snli_h_dim, self.args.snli_h_dim)
        HeKaimingInitializer(self.mlp2.weight)

        self.output = nn.Linear(self.args.snli_h_dim, 3)
        HeKaimingInitializer(self.output.weight)
        self.spinn = SPINN(self.args)

    def set_weight(self, weight):
        self.embed.weight.data.copy_(torch.from_numpy(weight))
        self.embed.weight.requires_grad = False

    def prepare_features(self, hyp, prem):
        return torch.cat([
            hyp, prem, prem - hyp,
            torch.mul(hyp, prem)
        ], dim=1)


    def forward(self, hypothesis, premise):
        hyp_embed = self.embed(hypothesis[0])
        prem_embed = self.embed(premise[0])

        hyp_encode = self.spinn(hyp_embed, hypothesis[1])
        prem_encode = self.spinn(prem_embed, premise[1])

        features = self.prepare_features(hyp_encode, prem_encode)


        features = self.layer_norm_mlp_input(features)

        if self.args.dropout_rate_classify > 0:
            features = self.dropout(features)
                    # ReLu plus weight matrix
        features = self.relu(self.mlp1(features))
        features = self.layer_norm_mlp1_hidden(features)
        # dropout
        if self.args.dropout_rate_classify > 0:
            features = self.dropout(features)

        features = self.relu(self.mlp2(features))
        features = self.layer_norm_mlp2_hidden(features)
        # dropout
        if self.args.dropout_rate_classify > 0:
            features = self.dropout(features)
        return self.output(features)
