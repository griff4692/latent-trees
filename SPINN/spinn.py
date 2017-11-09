import torch
import torch.nn as nn
from torch.autograd import Variable
from actions import Reduce
from constants import PAD, SHIFT, REDUCE
from buffer import Buffer
from stack import create_stack


class SPINN(nn.Module):
    #TODO: Add tracking LSTM
    def __init__(self, args, transitions=True):
        super(SPINN, self).__init__()
        self.args = args

        self.dropout = nn.Dropout(p=self.args.dropout_rate_input)
        self.batch_norm1 = nn.BatchNorm1d(self.args.hidden_size)

        self.word = nn.Linear(self.args.embed_dim, self.args.hidden_size)
        if not transitions:
            self.track = nn.Linear(self.args.hidden_size / 2, 2)
        self.reduce = Reduce(self.args.hidden_size / 2)

    def forward(self, sentence, transitions):
        batch_size, sent_len, _  = sentence.size()

        out = self.word(sentence) # batch, |sent|, h * 2
        if self.args.dropout_rate_input > 0:
            out = self.dropout(out) # batch, |sent|, h * 2
        # batch normalization and dropout
        if not self.args.no_batch_norm:
            out = out.transpose(1, 2).contiguous()
            out = self.batch_norm1(out) # batch,  h * 2, |sent| (Normalizes batch * |sent| slices for each feature
            out = out.transpose(1, 2)

        (h_sent, c_sent) = torch.chunk(out, 2, 2)  # ((batch, |sent|, h), (batch, |sent|, h))
        buffer_batch = [Buffer(h_s, c_s) for (h_s, c_s)
            in list(zip(
                list(torch.split(h_sent, 1, 0)),
                list(torch.split(c_sent, 1, 0)))
            )
        ]

        stack_batch = [create_stack(self.args.hidden_size, self.args.continuous_stack) for _ in buffer_batch]
        transitions_batch = [trans.squeeze(1) for trans
            in list(torch.split(transitions, 1, 1))]

        batch_size = len(buffer_batch)

        for time_stamp in range(len(transitions_batch)):
            reduce_ids = []
            reduce_lh, reduce_lc = [], []
            reduce_rh, reduce_rc = [], []
            temp_trans = transitions_batch[time_stamp].data

            for b_id in range(batch_size):
                act = temp_trans[b_id]

                # TODO this will be probability of decision for unsupervised case
                valence = Variable(torch.FloatTensor([1.0]))

                if act == PAD:
                    continue

                if act == SHIFT:
                    word = buffer_batch[b_id].pop()
                    stack_batch[b_id].add(word, valence, time_stamp)

                if act == REDUCE:
                    reduce_ids.append(b_id)

                    r = stack_batch[b_id].peek()
                    stack_batch[b_id].pop(valence)

                    l = stack_batch[b_id].peek()
                    stack_batch[b_id].pop(valence)

                    reduce_lh.append(l[0]); reduce_lc.append(l[1])
                    reduce_rh.append(r[0]); reduce_rc.append(r[1])

            if len(reduce_ids) > 0:
                h_lefts = torch.cat(reduce_lh)
                c_lefts = torch.cat(reduce_lc)
                h_rights = torch.cat(reduce_rh)
                c_rights = torch.cat(reduce_rc)
                h_outs, c_outs = self.reduce((h_lefts, c_lefts), (h_rights, c_rights))
                for i, state in enumerate(zip(h_outs, c_outs)):
                    stack_batch[reduce_ids[i]].add(state, valence)

        outputs = []
        for stack in stack_batch:
            if not self.args.continuous_stack:
                assert stack.size() == 1

            outputs.append(stack.peek()[0])

        return torch.cat(outputs)
