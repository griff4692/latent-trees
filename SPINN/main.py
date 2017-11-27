from __future__ import print_function
import torch
import torch.optim as optim
import torch.nn.functional as F
from snli_classifier import SNLIClassifier
from batcher import prepare_snli_batches
import numpy as np
import argparse
import sys
from utils import render_args
from oracle import Oracle
from torchtext import data

def predict(model, sent1, sent2, cuda=-1):
    model.eval()
    output = model(sent1, sent2)
    logits = F.log_softmax(output)
    if cuda > -1:
        return logits.data.cpu().numpy().argmax(axis=1)
    return logits.data.numpy().argmax(axis=1)

def get_l2_loss(model, l2_lambda):
    loss = 0.0
    for w in model.parameters():
        if w.grad is not None:
            loss += l2_lambda * torch.sum(torch.pow(w, 2))
    return loss


def train_batch(model, loss, optimizer, sent1, sent2, y_val, step):
    # Reset gradient
    optimizer.zero_grad()
    # Forward
    fx = model(sent1, sent2)
    logits = F.log_softmax(fx)

    total_loss = loss(logits, y_val)

    total_loss += get_l2_loss(model, 1e-05)

    # Backward
    total_loss.backward()
    for param in model.parameters():
        if param.grad is not None:
            param.grad.data.clamp(-5, 5)
    # Update parameters
    optimizer.lr = 0.001 * (0.75 ** (step / 10000.0))
    optimizer.step()
    return total_loss.data[0]

def train(args):
    print ("Starting...")
    sys.stdout.flush()
    label_names, (train_iter, dev_iter, test_iter, inputs) = prepare_snli_batches(args)
    label_names = label_names[1:] # don't count UNK
    num_labels = len(label_names)
    print ("Prepared Dataset...")
    sys.stdout.flush()
    model = SNLIClassifier(args, len(inputs.vocab.stoi))
    model.set_weight(inputs.vocab.vectors.numpy())
    print ("Instantiated Model...")
    sys.stdout.flush()
    if args.gpu > -1:
        model.cuda()
    loss = torch.nn.NLLLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, betas=(0.9, 0.999), eps=1e-08)
    count_iter = 0
    train_iter.repeat = False
    step = 0
    for epoch in range(args.epochs):
        train_iter.init_epoch()
        cost = 0
        for batch_idx, batch in enumerate(train_iter):
            model.train()
            step += 1
            count_iter += batch.batch_size
            cost += train_batch(
                model, loss, optimizer,
                (batch.hypothesis.transpose(0, 1), batch.hypothesis_transitions.t()),
                (batch.premise.transpose(0, 1), batch.premise_transitions.t()),
                batch.label - 1 , step=step
            )

            if count_iter >= args.eval_freq:
                correct, total = 0.0, 0.0
                count_iter = 0
                confusion_matrix = np.zeros([num_labels, num_labels])
                dev_iter.init_epoch()

                for dev_batch_idx, dev_batch in enumerate(dev_iter):
                    pred = predict(
                        model,
                        (dev_batch.hypothesis.transpose(0, 1),
                            dev_batch.hypothesis_transitions.t()),
                        (dev_batch.premise.transpose(0, 1),
                            dev_batch.premise_transitions.t()), args.gpu
                    )
                    if args.gpu > -1:
                        true_labels =  dev_batch.label.data.cpu().numpy() - 1.0
                    else:
                        true_labels =  dev_batch.label.data.numpy() - 1.0
                    for i in range(num_labels):
                        true_labels_by_cat = np.where(true_labels == i)[0]
                        pred_values_by_cat = pred[true_labels_by_cat]
                        num_labels_by_cat = len(true_labels_by_cat)
                        mass_so_far = 0
                        for j in range(num_labels - 1):
                            mass = len(pred_values_by_cat[pred_values_by_cat == j])
                            confusion_matrix[i, j] += mass
                            mass_so_far += mass

                        confusion_matrix[i, num_labels - 1] += num_labels_by_cat - mass_so_far

                    total += dev_batch.batch_size
                correct = np.trace(confusion_matrix)
                print ("Accuracy is %.4f, batch %f, epoch %f" % (float(correct) / total, batch_idx, epoch))
                true_label_counts = confusion_matrix.sum(axis=1)
                print ("Confusion matrix (x-axis is true labels)\n")
                label_names = [n[0:6] + '.' for n in label_names]
                print ("\t\t" + "\t".join(label_names) + "\n")
                for i in range(num_labels):
                    print (label_names[i], end="")
                    for j in range(num_labels):
                        if true_label_counts[i] == 0:
                            perc = 0.0
                        else:
                            perc = confusion_matrix[i, j] / true_label_counts[i]
                        print ("\t%.2f%%" % (perc * 100), end="")
                    print ("\t(%d examples)\n" % true_label_counts[i])
                sys.stdout.flush()
                
        print ("Cost for Epoch ", cost)

def eval(args, file, finetune=False):
    label_names, (train_iter, dev_iter, test_iter, inputs), (train_data, dev, test) = prepare_snli_batches(args)
    label_names = label_names[1:]  # don't count UNK
    num_labels = len(label_names)
    model = torch.load(file, map_location=lambda storage, loc: storage)
    model.cpu()
    correct, total = 0.0, 0.0
    confusion_matrix = np.zeros([num_labels, num_labels])
    dev_iter.init_epoch()
    if finetune:
        oracle = Oracle(inputs.vocab, train_data)
    for dev_batch_idx, dev_batch in enumerate(dev_iter):
        if finetune:
            step = 0
            new_train = oracle.find_training_data(dev_batch.hypothesis.split(1,1), dev_batch.premise.split(1,1))
            print (len(new_train.examples))
            train_iter, _, _ = data.BucketIterator.splits(
                (new_train, dev, test), batch_size=25, device=args.gpu)
            train_iter.repeat = False
            model = torch.load(file, map_location=lambda storage, loc: storage)
            model.cpu()
            loss = torch.nn.NLLLoss()
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, betas=(0.9, 0.999),
                                   eps=1e-08)
            for batch_idx, batch in enumerate(train_iter):
                model.train()
                step += 1
                train_batch(
                    model, loss, optimizer,
                    (batch.hypothesis.transpose(0, 1), batch.hypothesis_transitions.t()),
                    (batch.premise.transpose(0, 1), batch.premise_transitions.t()),
                    batch.label - 1, step=step
                )
        model.eval()
        pred = predict(
            model,
            (dev_batch.hypothesis.transpose(0, 1),
             dev_batch.hypothesis_transitions.t()),
            (dev_batch.premise.transpose(0, 1),
             dev_batch.premise_transitions.t()), args.gpu
        )
        if args.gpu > -1:
            true_labels = dev_batch.label.data.cpu().numpy() - 1.0
        else:
            true_labels = dev_batch.label.data.numpy() - 1.0
        for i in range(num_labels):
            true_labels_by_cat = np.where(true_labels == i)[0]
            pred_values_by_cat = pred[true_labels_by_cat]
            num_labels_by_cat = len(true_labels_by_cat)
            mass_so_far = 0
            for j in range(num_labels - 1):
                mass = len(pred_values_by_cat[pred_values_by_cat == j])
                confusion_matrix[i, j] += mass
                mass_so_far += mass

            confusion_matrix[i, num_labels - 1] += num_labels_by_cat - mass_so_far

        total += dev_batch.batch_size
    correct = np.trace(confusion_matrix)
    print("Accuracy is %.4f" % (float(correct) / total))
    true_label_counts = confusion_matrix.sum(axis=1)
    print("Confusion matrix (x-axis is true labels)\n")
    label_names = [n[0:6] + '.' for n in label_names]
    print("\t\t" + "\t".join(label_names) + "\n")
    for i in range(num_labels):
        print(label_names[i], end="")
        for j in range(num_labels):
            if true_label_counts[i] == 0:
                perc = 0.0
            else:
                perc = confusion_matrix[i, j] / true_label_counts[i]
            print("\t%.2f%%" % (perc * 100), end="")
        print("\t(%d examples)\n" % true_label_counts[i])
    sys.stdout.flush()


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='SPINN dependency parse + SNLI Classifier arguments.')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--embed_dim', type=int, default=300)
    parser.add_argument('--hidden_size', type=int, default=300)
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate to pass to optimizer.')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('-continuous_stack', action='store_true', default=False)
    parser.add_argument('--eval_freq', type=int, default=50000, help='number of examples between evaluation on dev set.')
    parser.add_argument('-debug', action='store_true', default=False)
    parser.add_argument('--snli_num_h_layers', type=int, default=2, help='tunable hyperparameter.')
    parser.add_argument('--snli_h_dim', type=int, default=1024, help='1024 is used by paper.')
    parser.add_argument('--dropout_rate_input', type=float, default=0.1)
    parser.add_argument('--dropout_rate_classify', type=float, default=0.05)
    parser.add_argument('-no_batch_norm', action='store_true', default=False)
    parser.add_argument('--gpu', type=int, default=-1)
    args = parser.parse_args()
    render_args(args)
    sys.stdout.flush()
   # train(args)
    eval(args, "mytraining17.pt", finetune=True)
