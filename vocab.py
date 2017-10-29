import torchtext.vocab as vocab
from nltk.tokenize.moses import MosesTokenizer
import numpy as np

PAD = '<PAD>'
UNK = '<UNK>'

class Vocab:
    def __init__(self, embed_dim):
        self.w2i = {PAD: 0, UNK: 1}
        self.i2w = [PAD, UNK]
        self.vocab2glove=[-1, -1]
        self.embed_dim = embed_dim
        self.emb_matrix = None

        self.glove = vocab.GloVe(name='6B', dim=embed_dim)
        self.gw2i = self.glove.stoi
        # self.gi2w = self.glove.itos

    def add_tokens(self, tokens):
        return [self.add_word(token) for token in tokens]

    def get_emb_matrix(self):
        if self.emb_matrix is None:
            self.make_emb_matrix()

        return self.emb_matrix

    def make_emb_matrix(self):
        sd = 1 / np.sqrt(self.embed_dim)
        weights = np.random.normal(0, scale=sd, size=[self.size(), self.embed_dim])
        s = 2
        for i in self.vocab2glove[2:]:
            weight = self.glove.vectors[i].unsqueeze(0)
            weights[s, :] = weight.numpy()
            s += 1

        self.emb_matrix = weights

    def ids_to_tokens(self, ids):
        return [
            self.i2w[idx] if idx in self.i2w else UNK for idx in ids
        ]

    def tokens_to_ids(self, ws):
        return [
            self.w2i[w] if w in self.w2i else self.unk_idx() for w in ws
        ]

    def add_word(self, word):
        if word in self.w2i:
            return self.w2i[word]
        else:
            if word in self.gw2i:
                self.i2w.append(word)
                self.vocab2glove.append(self.gw2i[word])
                self.w2i[word] = len(self.i2w) - 1
            else:
                word = UNK

        return self.w2i[word]

    def size(self):
        return len(self.i2w)

    def pad_idx(self):
        return self.w2i[PAD]

    def unk_idx(self):
        return self.w2i[UNK]
