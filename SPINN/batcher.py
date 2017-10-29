from torchtext import datasets
from torchtext import data
import torchtext.vocab as vocab

def prepare_snli_batches(args):
    inputs = datasets.snli.ParsedTextField(lower=True)
    transitions = datasets.snli.ShiftReduceField()
    answers = data.Field(sequential=False)

    train, dev, test = datasets.SNLI.splits(inputs, answers, transitions)
    inputs.build_vocab(train, dev, test)
    answers.build_vocab(train)
    glove = vocab.GloVe(name='6B', dim=args.embed_dim)
    inputs.vocab.set_vectors(stoi=glove.stoi, vectors=glove.vectors,dim=args.embed_dim)
    train_iter, dev_iter, test_iter = data.BucketIterator.splits(
        (train, dev, test), batch_size=args.batch_size, device=-1)
    return answers.vocab.itos, (train_iter, dev_iter, test_iter, inputs)
