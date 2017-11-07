from torchtext import datasets
from torchtext import data
import torchtext.vocab as vocab
import os
import sys 

from snli_preprocess import gen_mini, remove_train_unk, MINI_SIZE

def prepare_snli_batches(args):
    inputs = datasets.snli.ParsedTextField(lower=True)
    transitions = datasets.snli.ShiftReduceField()
    answers = data.Field(sequential=False)
    data_dir = '.data/snli/snli_1.0/'

    train_path = 'snli_1.0_clean_train.jsonl'
    debug_train = 'snli_1.0_mini_clean_train.jsonl'
    debug_validation = 'snli_1.0_mini_dev.jsonl'
    debug_test = 'snli_1.0_ini_test.jsonl'

    if not os.path.exists(os.path.join(data_dir, train_path)):
        remove_train_unk()
    if args.debug:
        if not os.path.exists(os.path.join(data_dir, debug_train)):
            gen_mini()

        print ("Using first %d examples for development purposes..." % MINI_SIZE)
        train, dev, test = datasets.SNLI.splits(inputs, answers, transitions,
            train=debug_train, validation=debug_validation, test=debug_test)
    else:
        print ("Train Path ", train_path)
        train, dev, test = datasets.SNLI.splits(inputs, answers, transitions, train=train_path)

    inputs.build_vocab(train, dev, test)
    answers.build_vocab(train)
    glove = vocab.GloVe(name='6B', dim=args.embed_dim)
    inputs.vocab.set_vectors(stoi=glove.stoi, vectors=glove.vectors,dim=args.embed_dim)
    train_iter, dev_iter, test_iter = data.BucketIterator.splits(
        (train, dev, test), batch_size=args.batch_size, device=args.gpu)
    return answers.vocab.itos, (train_iter, dev_iter, test_iter, inputs)
