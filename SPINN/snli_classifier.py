import torch
import torch.nn as nn
from spinn import SPINN
from actions import HeKaimingInitializer, LayerNormalization
from utils import cudify

class SNLIClassifier(nn.Module):
    def __init__(self, args, vocab_size):
        super(SNLIClassifier, self).__init__()

        self.args = args
        self.embed = nn.Embedding(vocab_size, self.args.embed_dim)
        self.softmax = nn.Softmax()
        self.relu = nn.ReLU()

        self.layer_norm_mlp_input = LayerNormalization(4 * self.args.hidden_size)
        self.layer_norm_mlp1_hidden = LayerNormalization(self.args.snli_h_dim)
        self.layer_norm_mlp2_hidden = LayerNormalization(self.args.snli_h_dim)

        self.dropout = nn.Dropout(p=self.args.dropout_rate_classify)

        self.mlp1 = nn.Linear(4 * self.args.hidden_size, self.args.snli_h_dim)
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

    def forward(self, hypothesis, premise, teacher_prob):
        hyp_embed = self.embed(hypothesis[0])
        prem_embed = self.embed(premise[0])

        if not self.args.teacher or not self.training:
            hyp_trans, prem_trans = hypothesis[1], premise[1]
            if self.args.tracking:
                hyp_trans, prem_trans = None, None

            hyp_encode = self.spinn(hyp_embed, hyp_trans, hypothesis[2], teacher_prob)
            prem_encode = self.spinn(prem_embed, prem_trans, premise[2], teacher_prob)
            sent_true, sent_pred = None, None
        else:
            hyp_encode, hyp_true, hyp_pred = self.spinn(hyp_embed, hypothesis[1], hypothesis[2], teacher_prob)
            prem_encode, prem_true, prem_pred = self.spinn(prem_embed, premise[1], premise[2], teacher_prob)
            sent_true = torch.cat([hyp_true, prem_true])
            sent_pred = torch.cat([hyp_pred, prem_pred])

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

        if self.args.dropout_rate_classify > 0:
            features = self.dropout(features)

        output = self.output(features)
        return output, sent_true, sent_pred
