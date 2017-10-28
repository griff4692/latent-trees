import argparse
import json
import numpy as np
import torch
from torch.autograd import Variable
from torch import optim
import os
import re
import time
import sys

import torchtext.vocab as torch_vocab

from snli_preprocess import LABELS, load, load_vocab
from models.greedy_gumbel import GreedyGumbel
from batcher import Batcher

def train_batch(model, loss, optimizer, sents1, sents2, labels, vocab):
    # Reset gradient
    optimizer.zero_grad()

    # Forward
    fx = model((sents1, sents2))
    labels = Variable(torch.FloatTensor(labels))
    output = loss.forward(fx, labels)

    # Backward
    output.backward()

    # Update parameters
    optimizer.step()

    return output.data[0]

def eval(model, dev_batcher, vocab, args):
    num_tested = 0.0
    num_wrong = 0.0
    true_positives = 0.0
    true_predicted_positives = 0.0
    num_predicted_positives = 0.0
    num_positive_labels = 0.0
    while not dev_batcher.is_finished():
        sentences1, sentences2, labels = dev_batcher.get_batch()

        labels = np.reshape(labels, (dev_batcher.args.batch_size,))
        num_positive_labels += np.sum(labels)

        predictions = model(x).data.numpy()
        predictions = np.reshape(predictions, (dev_batcher.args.batch_size,))

        # simulate sigmoid + prediction
        predictions[predictions >= 0] = 1
        predictions[predictions < 0] = 0

        label_pre_sum = predictions + labels
        true_positives += float(label_pre_sum[label_pre_sum == 2].shape[0])

        num_predicted_positives += np.sum(predictions)
        abs_deltas = np.abs(predictions - labels)

        num_wrong += np.sum(abs_deltas)
        num_tested += predictions.shape[0]

    precision = true_positives / num_predicted_positives
    recall = true_positives / num_positive_labels

    F_score = 2.0 / ((1.0 / recall) + (1.0 / precision))
    return float(num_tested - num_wrong) / num_tested, F_score


def resolve_model(args, vocab):
    if args.model_name == 'greedy_gumbel':
        return GreedyGumbel(args, vocab)

def train(args, vocab):
    # retrieve proper data, model, and vocabulary
    train_data = load("train")
    dev_data = load("dev")

    model = resolve_model(args, vocab)

    # intialize batchers
    train_batcher = Batcher(train_data, args)
    dev_batcher = Batcher(dev_data, args)

    # initialize training parameters
    loss = torch.nn.BCEWithLogitsLoss()

    # don't optimize fixed weights like GloVe embeddings
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    # evaluation metrics
    best_accuracy = 0.0
    best_params = None
    best_epoch = 0
    prev_accuracy = 0
    consec_worse_epochs = 0

    for i in range(args.epochs):
        cost = 0.
        while not train_batcher.is_finished():
            sents1, sents2, labels = train_batcher.get_batch()
            cost += train_batch(model, loss, optimizer, sents1, sents2, labels, vocab)

        print("Epoch = %d, average loss = %s" % (i + 1, cost / train_batcher.num_batches))
        if (i + 1) % args.eval_freq == 0:
            test_acc, F_score = test(model, dev_batcher, vocab, args)
            print("Accuracy (F-score) after epoch #%s --> %s%% (%.2f)" % (i, int(test_acc * 100.0), F_score))

            if test_acc < prev_accuracy:
                consec_worse_epochs += 1
                if consec_worse_epochs >= args.max_consec_worse_epochs:
                    print("Training incurred %s consecutive worsening epoch(s): from %s to %s" \
                    % (args.max_consec_worse_epochs, i + 1 - (args.max_consec_worse_epochs * args.eval_freq), i + 1))
                    break
            else:
                consec_worse_epochs = 0

                if test_acc > best_accuracy:
                    best_accuracy = test_acc
                    best_epoch = i + 1
                    best_params = model.state_dict()

            prev_accuracy = test_acc

    model.load_state_dict(best_params)
    acc, F_score = test(model, dev_batcher, vocab, args)
    print("Best Accuracy achieved after epoch #%s --> %.2f%% (%s)" % (best_epoch, acc * 100.0, F_score))

def main():
    parser = argparse.ArgumentParser(description='Latent Tree Structure learner through SNLI entailment objective.')
    parser.add_argument('--model_name', default='greedy_gumbel')
    parser.add_argument('--embed_dim', default=100, type=int)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--max_consec_worse_epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--eval_freq', type=int, default=5)

    args = parser.parse_args()

    vocab = load_vocab(args.embed_dim)

    print "Loaded vocab of size %d" % vocab.size()
    vocab.glove = torch_vocab.GloVe(name='6B', dim=args.embed_dim)
    vocab.embed_dim = args.embed_dim

    train(args, vocab)

if __name__=='__main__':
    main()
