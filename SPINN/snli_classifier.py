import torch
import torch.nn as nn
from spinn import SPINN

class SNLIClassifier(nn.Module):
    def __init__(self, args, vocab_size):
        super(SNLIClassifier, self).__init__()

        self.args = args
        self.embed = nn.Embedding(vocab_size, self.args.embed_dim)
        self.softmax = nn.Softmax()
        self.relu = nn.ReLU()

        self.batch_norm_mlp_input = nn.BatchNorm1d(4 * self.args.hidden_size)
        self.batch_norm_mlp_hidden = nn.BatchNorm1d(self.args.snli_h_dim)

        self.dropout = nn.Dropout(p=self.args.dropout_rate)

        self.mlp = []
        for i in range(self.args.snli_num_h_layers):
            input_dim = 4 * self.args.hidden_size if i == 0 else self.args.snli_h_dim
            out_dim = self.args.snli_h_dim
            if args.gpu > -1:
                self.mlp.append(nn.Linear(input_dim, out_dim).cuda())
            else:
                self.mlp.append(nn.Linear(input_dim, out_dim))

        self.output = nn.Linear(self.args.snli_h_dim, 3)
        self.encoder = SPINN(self.args)

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

            hyp_encode = self.encoder(hyp_embed, hyp_trans, hypothesis[2], teacher_prob)
            prem_encode = self.encoder(prem_embed, prem_trans, premise[2], teacher_prob)
            sent_true, sent_pred = None, None
        else:
            hyp_encode, hyp_true, hyp_pred = self.encoder(hyp_embed, hypothesis[1], hypothesis[2], teacher_prob)
            prem_encode, prem_true, prem_pred = self.encoder(prem_embed, premise[1], premise[2], teacher_prob)

            sent_true = torch.cat([hyp_true, prem_true])
            sent_pred = torch.cat([hyp_pred, prem_pred])

        features = self.prepare_features(hyp_encode, prem_encode)

        if not self.args.no_batch_norm:
            features = self.batch_norm_mlp_input(features)

        if self.args.dropout_rate > 0:
            features = self.dropout(features)

        for (i, layer) in enumerate(self.mlp):
            # ReLu plus weight matrix
            features = self.relu(layer(features))

            # batch norm
            if not self.args.no_batch_norm:
                features = self.batch_norm_mlp_hidden(features)

            # dropout
            if self.args.dropout_rate > 0:
                features = self.dropout(features)

        output = self.softmax(self.output(features))
        return output, sent_true, sent_pred
