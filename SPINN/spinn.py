import torch
import torch.nn as nn
from torch.autograd import Variable
from actions import Reduce
from constants import PAD, SHIFT, REDUCE
from buffer import Buffer
from stack import create_stack
from tracking_lstm import TrackingLSTM


class SPINN(nn.Module):
    def __init__(self, args):
        super(SPINN, self).__init__()
        self.args = args

        self.dropout = nn.Dropout(p=self.args.dropout_rate)
        self.batch_norm1 = nn.BatchNorm1d(2 * self.args.hidden_size)

        self.word = nn.Linear(self.args.embed_dim, 2 * self.args.hidden_size)

        self.track = None
        if self.args.tracking:
            self.track = TrackingLSTM(self.args)

        self.reduce = Reduce(self.args.hidden_size)
        self.batch_norm1 = nn.BatchNorm1d(2 * self.args.hidden_size)

    def update_tracker(self, buffer, stack, batch_size):
        tracking_inputs = None
        for b_id in range(batch_size):
            b = buffer[b_id].peek()[0]
            s1, s2 = stack[b_id].peek_two()
            input = torch.cat([b, s1[0], s2[0]], dim=1)
            if tracking_inputs is None:
                tracking_inputs = input
            else:
                tracking_inputs = torch.cat([tracking_inputs, input], dim=0)

        return self.track(tracking_inputs)

    def resolve_action(self, buffer, stack, act, time_stamp):
        # 1 - PAD, 2 - REDUCE, 3 - SHIFT

        # if time_stamp <=2, action should be shift
        if time_stamp <= 2:
            return SHIFT

        # if there's nothing in the buffer, try a shift
        if buffer.size() == 0:
            if stack.size() >= 2:
                return REDUCE
            else:
                return PAD

        if stack.size() <= 2:
            return SHIFT

        return act

    def forward(self, sentence, transitions):
        batch_size, sent_len, _  = sentence.size()

        out = self.word(sentence) # batch, |sent|, h * 2

        # batch normalization and dropout

        if not self.args.no_batch_norm:
            out = out.transpose(1, 2)
            out = self.batch_norm1(out) # batch,  h * 2, |sent| (Normalizes batch * |sent| slices for each feature
            out = out.transpose(1, 2)

        if self.args.dropout_rate > 0:
            out = self.dropout(out) # batch, |sent|, h * 2

        (h_sent, c_sent) = torch.chunk(out, 2, 2)  # ((batch, |sent|, h), (batch, |sent|, h))
        buffer_batch = [Buffer(h_s, c_s, self.args) for h_s, c_s
            in zip(
                list(torch.split(h_sent, 1, 0)),
                list(torch.split(c_sent, 1, 0))
            )
        ]

        stack_batch = [
            create_stack(self.args.hidden_size, self.args.continuous_stack)
            for _ in buffer_batch
        ]

        if self.args.tracking:
            self.track.initialize_states(batch_size)
            num_transitions = (2 * sent_len) - 1
        else:
            transitions_batch = [trans.squeeze(1) for trans
                in list(torch.split(transitions, 1, 1))]
            num_transitions = len(transitions_batch)

        for time_stamp in range(num_transitions):
            reduce_ids = []
            reduce_lh, reduce_lc = [], []
            reduce_rh, reduce_rc = [], []

            if self.args.tracking:
                valences = self.update_tracker(buffer_batch, stack_batch, batch_size)

                # soft actions
                if self.args.continuous_stack:
                    temp_trans = valences # transitions are soft
                else:
                    # transitions are hard (shifted to 2,3)
                    _, temp_trans = valences.max(dim=1)
                    temp_trans = temp_trans.data.numpy() + 2
            else:
                valences = None
                temp_trans = transitions_batch[time_stamp].data

            for b_id in range(batch_size):
                act = temp_trans[b_id]

                # ensures it's a valid act according to state of buffer, batch, and timestamp
                if self.args.tracking:
                    act = self.resolve_action(buffer_batch[b_id], stack_batch[b_id], act, time_stamp)

                # reduce, shift valence
                valence = valences[b_id] if self.args.tracking else [None, None]

                # 1 - PAD
                if act == PAD:
                    continue

                # 2 - REDUCE
                if act == REDUCE or (time_stamp >= 2 and self.args.continuous_stack):
                    reduce_ids.append(b_id)

                    r = stack_batch[b_id].peek()
                    stack_batch[b_id].pop(valence[0])

                    l = stack_batch[b_id].peek()
                    stack_batch[b_id].pop(valence[0])

                    reduce_lh.append(l[0]); reduce_lc.append(l[1])
                    reduce_rh.append(r[0]); reduce_rc.append(r[1])

                # 3 - SHIFT
                if act == SHIFT or self.args.continuous_stack:
                    word = buffer_batch[b_id].pop()
                    stack_batch[b_id].add(word, valence[1], time_stamp)

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
