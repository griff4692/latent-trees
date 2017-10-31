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

    def resolve_action(self, buffer, stack, act, time_stamp, ops_left):
        # 1 - PAD, 2 - REDUCE, 3 - SHIFT
        # Hard constraints:
        # 1st 2 actions are shift
        # Can't pop from an empty buffer
        # Can't reduce < 2 size stack
        # --> default action is PAD (give learner another chance)
        if time_stamp < 2:
            return SHIFT, True

        # if there's nothing in the buffer
        if buffer.size() == 0 and act == SHIFT:
            return PAD, True

        # switch reduce to a PAD if REDUCE is not allowed
        if stack.size() <= 2 and act == REDUCE:
            return PAD, True

        # don't let stack grow to the point where it will be 2 large to shrink to 1 by the end
        if stack.size() == ops_left and act == SHIFT:
            return PAD, True

        return act, False

    def forward(self, sentence, transitions):
        batch_size, sent_len, _  = sentence.size()

        out = self.word(sentence) # batch, |sent|, h * 2

        # batch normalization and dropout

        if not self.args.no_batch_norm:
            out = out.transpose(1, 2).contiguous()
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
            # how many more operations left - do we need to start to reduce stack size to get to 1
            ops_left = num_transitions - time_stamp

            reduce_ids = []
            reduce_lh, reduce_lc = [], []
            reduce_rh, reduce_rc = [], []

            if self.args.tracking:
                valences = self.update_tracker(buffer_batch, stack_batch, batch_size)
                _, temp_trans = valences.max(dim=1)
                temp_trans = temp_trans.data.numpy() + 1
            else:
                valences = None
                temp_trans = transitions_batch[time_stamp].data

            for b_id in range(batch_size):
                act = None
                stack_size = stack_batch[b_id].size()
                buffer_size = buffer_batch[b_id].size()

                # if stack is longer than number of operations left, we need to reduce
                if not self.args.continuous_stack and stack_size > ops_left:
                    act = REDUCE
                    act_ignored = True

                if act is None:
                    act = temp_trans[b_id]
                    # ensures it's a valid act according to state of buffer, batch, and timestamp
                    if self.args.tracking:
                        act, act_ignored = self.resolve_action(buffer_batch[b_id],
                        stack_batch[b_id], act, time_stamp, ops_left)

                # reduce, shift valence
                valence = valences[b_id] if self.args.tracking else [None, None]

                # print("Act=%d,TimeStep=%d,OpsLeft=%d,StackSize=%d,BufferSize=%d" % (act,time_stamp,ops_left,stack_size,buffer_size))

                # 1 - PAD
                if act == PAD:
                    continue

                # 2 - REDUCE
                if act == REDUCE or (self.args.continuous_stack and stack_size >= 2):
                    reduce_ids.append(b_id)

                    r = stack_batch[b_id].peek()
                    stack_batch[b_id].pop(valence[0])

                    l = stack_batch[b_id].peek()
                    stack_batch[b_id].pop(valence[0])

                    reduce_lh.append(l[0]); reduce_lc.append(l[1])
                    reduce_rh.append(r[0]); reduce_rc.append(r[1])

                # 3 - SHIFT
                if act == SHIFT or (self.args.continuous_stack and buffer_size > 0):
                    word = buffer_batch[b_id].pop()
                    stack_batch[b_id].add(word, valence[1], time_stamp)

            if len(reduce_ids) > 0:
                h_lefts = torch.cat(reduce_lh)
                c_lefts = torch.cat(reduce_lc)
                h_rights = torch.cat(reduce_rh)
                c_rights = torch.cat(reduce_rc)
                h_outs, c_outs = self.reduce((h_lefts, c_lefts), (h_rights, c_rights))
                for i, state in enumerate(zip(h_outs, c_outs)):
                    stack_batch[reduce_ids[i]].add(state, valence[0])

        outputs = []
        for stack in stack_batch:
            if not self.args.continuous_stack:
                if not stack.size() == 1:
                    print("Stack size is %d.  Should be 1" % stack.size())
                    assert stack.size() == 1

            outputs.append(stack.peek()[0])

        return torch.cat(outputs)
