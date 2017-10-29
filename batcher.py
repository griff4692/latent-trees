import numpy as np

class Batcher:
    def __init__(self, data, args):
        self.args = args
        self.data = data
        self.n = len(self.data[0])
        self.num_batches = self.n // self.args.batch_size
        self.idxs = np.arange(self.n)
        self.reset()

    def reset(self):
        self.batch_no = 0
        np.random.shuffle(self.idxs)

    def get_batch(self):
        start, end = self.batch_no * self.args.batch_size, (self.batch_no + 1) * self.args.batch_size

        sents1 = np.zeros([self.args.batch_size, 64], dtype=object)
        sents2 = np.zeros([self.args.batch_size, 64], dtype=object)
        labels = np.zeros([self.args.batch_size, 1], dtype=float)

        for batch_idx, data_idx in enumerate(range(start, end)):
            idx = self.idxs[data_idx]
            sents1[batch_idx,:len(self.data[0][idx])] = self.data[0][idx]
            sents2[batch_idx,:len(self.data[1][idx])] = self.data[1][idx]
            labels[batch_idx] = self.data[2][idx]

        self.batch_no += 1
        return sents1, sents2, labels

    def is_finished(self):
        if self.batch_no == self.num_batches - 1:
            self.reset()
            return True
        return False
