from torchtext import datasets
from torchtext import data
import copy

class Oracle():
    def __init__(self, vocab, train):
        self.w2i = vocab.stoi
        self.i2w = vocab.itos
        self.train = train
        self.examples = train.examples
        self.i_counts = [0] * len(self.i2w)
        self.find_counts()
        self.sel_hyp_terms = {}
        self.sel_prem_terms = {}
        self.store_selective_terms()

    def print_string(self, list):
        s = []
        for i in list:
            if i != 1:
                s.append(self.i2w[i])
        s.reverse()
        return " ".join(s)

    def find_counts(self):
        for i in self.examples:
            for h in i.hypothesis:
                self.i_counts[self.w2i.get(h, 0)] += 1
            for p in i.premise:
                self.i_counts[self.w2i.get(p, 0)] += 1

    def store_selective_terms(self):
        for i, term in enumerate(self.i_counts):
            if term < 1000:
                self.sel_prem_terms[i] = []
                self.sel_hyp_terms[i] = []
        for k, i in enumerate(self.examples):
            for h in i.hypothesis:
                id = self.w2i.get(h, 0)
                if id in self.sel_hyp_terms:
                    self.sel_hyp_terms[id].append(k)

            for p in i.premise:
                id = self.w2i.get(p, 0)
                if id in self.sel_prem_terms:
                    self.sel_prem_terms[id].append(k)

    def find_examples(self, id, hyp=True):
        id = int(id)
        examples = set()
        if hyp:
            if id not in self.sel_hyp_terms.keys():
                return set()
            if len(self.sel_hyp_terms[id]) == 0:
                return set()
            for i in self.sel_hyp_terms[id]:
                examples.add(self.examples[i])
        else:
            if id not in self.sel_prem_terms.keys():
                return set()
            if len(self.sel_prem_terms[id]) == 0:
                return  set()
            for i in self.sel_prem_terms[id]:
                examples.add(self.examples[i])
        return examples


    def find_training_data(self, test_hypothesis, test_premise):
        examples = set()
        ids = set()
        prev = len(examples)
        for i, (h, p) in enumerate(zip(test_hypothesis, test_premise)):
            for k, j in zip(list(h.data.numpy()),list(p.data.numpy())):
                e1 = self.find_examples(k, hyp=True)
                e2 = e1.intersection(self.find_examples(j, hyp=False))
                examples = examples.union(e2)
                if len(examples) != prev:
                    ids.add(i)
                    prev = len(examples)

        train = self.train
        train.examples = list(examples)
        return list(ids), train

if __name__=='__main__':
    data_dir = '.data/snli/snli_1.0/'

    train_path = 'mini_clean_train.jsonl'
    test_path = 'mini_clean_test.jsonl'
    validation_path = 'mini_clean_dev.jsonl'

    transitions = datasets.snli.ShiftReduceField()
    answers = data.Field(sequential=False)

    inputs = datasets.snli.ParsedTextField(lower=True)
    train, dev, test = datasets.SNLI.splits(inputs, answers, transitions, train=train_path, validation=validation_path,
                                            test=test_path)
    inputs.build_vocab(train, dev, test)
    print (train.examples[0].hypothesis)
    print ("begin")
    oracle = Oracle(inputs.vocab, train)
    new_train = oracle.find_training_data(dev.examples)
    train_iter, dev_iter, test_iter = data.BucketIterator.splits(
        (new_train, dev, test), batch_size=16, device=-1)