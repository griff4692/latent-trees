import torch
import torch.nn as nn
from spinn import SPINN
from actions import HeKaimingInitializer, LayerNormalization
from utils import cudify
from torch.autograd import Variable




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

    def reverse_padded_sequence(self, inputs, lengths, batch_first=True):
        if batch_first:
            inputs = inputs.transpose(0, 1)
        if inputs.size(1) != len(lengths):
            raise ValueError('inputs incompatible with lengths.')
        max_len = inputs.size(0)
        reversed_inputs = Variable(inputs.data.clone())
        for i, length in enumerate(lengths):
            time_ind = torch.LongTensor(list(reversed(range(length))))
            reversed_inputs[-length:, i] = inputs[:, i][max_len - length + time_ind]
        if batch_first:
            reversed_inputs = reversed_inputs.transpose(0, 1)
        return reversed_inputs

    def get_sent_lengths(self, sent):
        trans = sent - 1

        # find number of padding actions and subtract from max ops row-wise
        if self.args.gpu > -1:
            mask = trans.data.cpu().numpy().copy()
        else:
            mask = trans.data.numpy().copy()
        mask[mask != 0] = 1

        num_ops = mask.sum(axis=1)

        return (num_ops.tolist())

    def set_weight(self, weight):
        self.embed.weight.data.copy_(torch.from_numpy(weight))
        self.embed.weight.requires_grad = False

    def prepare_features(self, hyp, prem):
        return torch.cat([
            hyp, prem, prem - hyp,
            torch.mul(hyp, prem)
        ], dim=1)

    def forward(self, hypothesis, premise, teacher_prob):
        rev_h = self.reverse_padded_sequence(hypothesis[0], self.get_sent_lengths(hypothesis[0]))
        rev_p = self.reverse_padded_sequence(premise[0], self.get_sent_lengths(premise[0]))
        hyp_embed = self.embed(hypothesis[0])
        prem_embed = self.embed(premise[0])

        rev_hyp_embed = self.embed(rev_h)
        rev_prem_embed = self.embed(rev_p)

        if not self.args.teacher or not self.training:
            hyp_trans, prem_trans = hypothesis[1], premise[1]
            if self.args.tracking:
                hyp_trans, prem_trans = None, None

            hyp_encode = self.spinn(hyp_embed, hyp_trans, hypothesis[2], teacher_prob)
            prem_encode = self.spinn(prem_embed, prem_trans, premise[2], teacher_prob)

            rhyp_encode = self.spinn(rev_hyp_embed, hyp_trans, hypothesis[2], teacher_prob)
            rprem_encode = self.spinn(rev_prem_embed, prem_trans, premise[2], teacher_prob)
            sent_true, sent_pred = None, None
        else:
            hyp_encode, hyp_true, hyp_pred = self.spinn(hyp_embed, hypothesis[1], hypothesis[2], teacher_prob)
            prem_encode, prem_true, prem_pred = self.spinn(prem_embed, premise[1], premise[2], teacher_prob)

            rhyp_encode, rhyp_true, rhyp_pred = self.spinn(rev_hyp_embed, hypothesis[1], hypothesis[2], teacher_prob)
            rprem_encode, rprem_true, rprem_pred = self.spinn(rev_prem_embed, premise[1], premise[2], teacher_prob)

            sent_true = torch.cat([hyp_true, prem_true])
            sent_pred = torch.cat([hyp_pred, prem_pred])


        p = rprem_encode + prem_encode
        h = rhyp_encode + hyp_encode
        features = self.prepare_features(h, p)

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
