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
        self.sel_terms = {}
        self.store_selective_terms()

    def find_counts(self):
        for i in self.examples:
            for h in i.hypothesis:
                self.i_counts[self.w2i.get(h, 0)] += 1
            for p in i.premise:
                self.i_counts[self.w2i.get(p, 0)] += 1

    def store_selective_terms(self):
        for i, term in enumerate(self.i_counts):
            if term < 10:
                self.sel_terms[i] = []
        for k, i in enumerate(self.examples):
            for h in i.hypothesis:
                id = self.w2i.get(h, 0)
                if id in self.sel_terms:
                    self.sel_terms[id].append(k)

            for p in i.premise:
                id = self.w2i.get(p, 0)
                if id in self.sel_terms:
                    if k not in self.sel_terms[id]:
                        self.sel_terms[id].append(k)

    def find_examples(self, id):
        id = int(id)
        if id not in self.sel_terms.keys():
            return []
        examples = set()
        for i in self.sel_terms[id]:
            examples.add(self.examples[i])
        return examples


    def find_training_data(self, test_hypothesis, test_premise):
        examples = set()

        for h in test_hypothesis:
            for k in list(h.data.numpy()):
                examples = examples.union(self.find_examples(k))
        for p in test_premise:
            for k in list(p.data.numpy()):
                examples = examples.union(self.find_examples(k))
        train = self.train
        train.examples = list(examples)
        return train

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
