import torch
import torch.optim as optim
from torchtext import data
from torchtext import datasets
import torchtext.vocab as vocab
from snli_classifier import SNLIClassifier
import numpy as np

def prepare_snli_batches(batch_size=16, embed_dim=100):
    inputs = datasets.snli.ParsedTextField(lower=True)
    transitions = datasets.snli.ShiftReduceField()
    answers = data.Field(sequential=False)

    train, dev, test = datasets.SNLI.splits(inputs, answers, transitions)
    inputs.build_vocab(train, dev, test)
    answers.build_vocab(train)
    print answers.vocab.itos
    glove = vocab.GloVe(name='6B', dim=embed_dim)
    inputs.vocab.set_vectors(stoi=glove.stoi, vectors=glove.vectors,dim=embed_dim)
    train_iter, dev_iter, test_iter = data.BucketIterator.splits(
        (train, dev, test), batch_size=batch_size, device=-1)
    return (train_iter, dev_iter, test_iter, inputs)

def predict(model, sent1, sent2):
    output = model(sent1, sent2)
    return output.data.numpy().argmax(axis=1)

def train(model, loss, optimizer, sent1, sent2, y_val):
    # Reset gradient
    optimizer.zero_grad()
    # Forward
    fx = model(sent1, sent2)
    output = loss.forward(fx, y_val)
    # Backward
    output.backward()
    # Update parameters
    optimizer.step()
    return output.data[0]

def training(batch_size, embed_dim):
    train_iter, dev_iter, test_iter, inputs = prepare_snli_batches(batch_size, embed_dim)
    print ("Prepared Dataset")
    model = SNLIClassifier(embed_dim, len(inputs.vocab.stoi), 300)
    model.set_weight(inputs.vocab.vectors.numpy())
    print ("Instantiated Model")
    loss = torch.nn.CrossEntropyLoss(size_average=True)
    optimizer = optim.Adagrad(filter(lambda p: p.requires_grad, model.parameters()), lr=0.01)
    dev_every = 10
    count_iter = 0

    for epoch in range(100):
        train_iter.init_epoch()
        cost = 0
        for batch_idx, batch in enumerate(train_iter):
            count_iter += batch.batch_size
            cost += train(model, loss, optimizer, (batch.hypothesis.transpose(0, 1), batch.hypothesis_transitions.t()),
                         (batch.premise.transpose(0, 1), batch.premise_transitions.t()), batch.label)

            if count_iter >= dev_every:
                correct, total = 0, 0
                count_iter = 0
                dev_iter.init_epoch()
                for dev_batch_idx, dev_batch in enumerate(dev_iter):
                    pred = predict(model, (dev_batch.hypothesis.transpose(0, 1), dev_batch.hypothesis_transitions.t()),
                                   (dev_batch.premise.transpose(0, 1), dev_batch.premise_transitions.t()))
                    correct += np.sum(pred == dev_batch.label.data.numpy())
                    total += dev_batch.batch_size
                print (float(correct) / total)
        print ("Cost for Epoch ", cost)




training(64, 100)


