from torchtext import datasets
from torchtext import data
import torchtext.vocab as vocab
import os

from gen_mini_splits import gen_mini, MINI_SIZE

def prepare_snli_batches(args):
    inputs = datasets.snli.ParsedTextField(lower=True)
    transitions = datasets.snli.ShiftReduceField()
    answers = data.Field(sequential=False)

    debug_train = 'snli_1.0_mini_train.jsonl'
    debug_validation = 'snli_1.0_mini_dev.jsonl'
    debug_test = 'snli_1.0_mini_test.jsonl'

    if args.debug:
        if not os.path.exists(debug_train):
            gen_mini()

        print "Using first %d examples for development purposes..." % MINI_SIZE
        train, dev, test = datasets.SNLI.splits(inputs, answers, transitions,
            train=debug_train, validation=debug_validation, test=debug_test)
    else:
        train, dev, test = datasets.SNLI.splits(inputs, answers, transitions)

    inputs.build_vocab(train, dev, test)
    answers.build_vocab(train)
    glove = vocab.GloVe(name='6B', dim=args.embed_dim)
    inputs.vocab.set_vectors(stoi=glove.stoi, vectors=glove.vectors,dim=args.embed_dim)
    train_iter, dev_iter, test_iter = data.BucketIterator.splits(
        (train, dev, test), batch_size=args.batch_size, device=-1)
    return answers.vocab.itos, (train_iter, dev_iter, test_iter, inputs)
