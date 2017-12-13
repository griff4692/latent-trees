from torchtext import datasets
from torchtext import data
import torchtext.vocab as vocab
import os
from torchtext import data
import sys 

from snli_preprocess import gen_mini, remove_train_unk, MINI_SIZE

def prepare_snli_batches(args):
    inputs = datasets.snli.ParsedTextField(lower=True)
    transitions = datasets.snli.ShiftReduceField()
    answers = data.Field(sequential=False)
    data_dir = '.data/snli/snli_1.0/'

    train_path = 'clean_train.jsonl'
    test_path = 'clean_test.jsonl'
    validation_path = 'clean_dev.jsonl'
    debug_train = 'mini_clean_train.jsonl'
    debug_validation = 'mini_clean_dev.jsonl'
    debug_test = 'mini_clean_test.jsonl'

    if not os.path.exists(os.path.join(data_dir, "snli_1.0_" + train_path)):
        remove_train_unk('train')
        remove_train_unk('test')
        remove_train_unk('dev')
    if args.debug:
        if not os.path.exists(os.path.join(data_dir, "snli_1.0_" + debug_train)):
            gen_mini()

        print ("Using first %d examples for development purposes..." % MINI_SIZE)
        train, dev, test = datasets.SNLI.splits(inputs, answers, transitions,
            train=train_path, validation=debug_validation, test=debug_test)
    else:
        print ("Train Path ", train_path)
        train, dev, test = datasets.SNLI.splits(inputs, answers, transitions, train=train_path, validation=validation_path, test=test_path)

    inputs.build_vocab(train, dev, test)
    answers.build_vocab(train)
    glove = vocab.GloVe(name='6B', dim=args.embed_dim)
    inputs.vocab.set_vectors(stoi=glove.stoi, vectors=glove.vectors,dim=args.embed_dim)
    train_iter, dev_iter, test_iter = data.BucketIterator.splits(
        (train, dev, test), batch_size=args.batch_size, device=args.gpu)
    return answers.vocab.itos, (train_iter, dev_iter, test_iter, inputs), (train, dev, test)
