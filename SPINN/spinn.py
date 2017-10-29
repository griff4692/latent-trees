import torch
import torch.nn as nn
from torch.autograd import Variable
from actions import Reduce


class Stack():
    # Figure out Thin Stack.
    # Seems like an overhead at this stage.
    def __init__(self):
        self.states = []

    def add(self, state, id=0):
        self.states.append(state)

    def pop(self):
        top = self.states.pop()
        return top


class Buffer():
    def __init__(self, h_s, c_s):
        self.states = zip(list(torch.split(h_s.squeeze(0), 1, 0)), list(torch.split(c_s.squeeze(0), 1, 0)))
    def pop(self):
        top = self.states.pop()
        return (top)

PAD = 1
SHIFT = 3
REDUCE = 2

class SPINN(nn.Module):
    #TODO: Add tracking LSTM
    def __init__(self, embed_dim, size, transitions=True):
        super(SPINN, self).__init__()
        self.word = nn.Linear(embed_dim, 2 * size)
        if not transitions:
            self.track = nn.Linear(size, 2)
        self.reduce = Reduce(size)

    def forward(self, sentence, transitions):
        out = self.word(sentence)
        (h_sent, c_sent) = torch.chunk(out, 2, 2)
        buffer_batch = [Buffer(h_s, c_s) for h_s, c_s in zip(list(torch.split(h_sent, 1, 0)),list(torch.split(c_sent, 1, 0)))]
        stack_batch = [Stack() for _ in buffer_batch]
        transitions_batch = [trans.squeeze(1) for trans in list(torch.split(transitions, 1, 1))]

        batch_size = len(buffer_batch)

        for time_stamp in range(len(transitions_batch)):
            reduce_ids = []
            reduce_lh, reduce_lc = [], []
            reduce_rh, reduce_rc = [], []
            temp_trans = transitions_batch[time_stamp]
            for b_id in range(batch_size):
                act = temp_trans[b_id]
                act = act.data[0]

                if act == PAD:
                    continue

                if act == SHIFT:
                    word = buffer_batch[b_id].pop()
                    stack_batch[b_id].add(word, time_stamp)

                if act == REDUCE:
                    reduce_ids.append(b_id)
                    r = stack_batch[b_id].pop()
                    l = stack_batch[b_id].pop()
                    reduce_lh.append(l[0])
                    reduce_lc.append(l[1])
                    reduce_rh.append(r[0])
                    reduce_rc.append(r[1])

            if len(reduce_ids) > 0:
                h_lefts = torch.cat(reduce_lh)
                c_lefts = torch.cat(reduce_lc)
                h_rights = torch.cat(reduce_rh)
                c_rights = torch.cat(reduce_rc)
                h_outs, c_outs = self.reduce((h_lefts, c_lefts), (h_rights, c_rights))
                for i, state in enumerate(zip(h_outs, c_outs)):
                    stack_batch[reduce_ids[i]].add(state)

        outputs = []
        for stack in stack_batch:
            outputs.append(stack.pop()[0])

        return torch.cat(outputs)









