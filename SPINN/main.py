from __future__ import print_function
import torch
import torch.optim as optim
from snli_classifier import SNLIClassifier
from batcher import prepare_snli_batches
import numpy as np
import argparse
from constants import PAD, SHIFT, REDUCE
import sys
from utils import render_args

def add_num_ops_and_shift_acts(sent):
    trans = sent[1] - 2
    max_ops = trans.size()[1]

    # find number of padding actions and subtract from max ops row-wise
    mask = trans.data.numpy().copy()
    mask[mask > 0] = 0
    num_ops = max_ops + mask.sum(axis=1)

    return (sent[0], trans, num_ops)

def predict(model, sent1, sent2, cuda=-1):
    sent1, sent2 = add_num_ops_and_shift_acts(sent1), \
        add_num_ops_and_shift_acts(sent2)
    output = model(sent1, sent2, None)
    if cuda > -1:
        return output.data.cpu().numpy().argmax(axis=1)
    return output.data.numpy().argmax(axis=1)

def train_batch(model, loss, optimizer, sent1, sent2, y_val, teacher_prob):
    sent1, sent2 = add_num_ops_and_shift_acts(sent1), \
        add_num_ops_and_shift_acts(sent2)

    # Reset gradient
    optimizer.zero_grad()

    fx, sent_true, sent_pred = model(sent1, sent2, teacher_prob)
    full_loss = loss.forward(fx, y_val)
    if sent_pred is not None and sent_true is not None:
        full_loss += loss.forward(sent_pred, sent_true)

    # Backward
    full_loss.backward()

    # Update parameters
    optimizer.step()
    return full_loss.data[0]

def train(args):
    print("Starting...\n")

    sys.stdout.flush()
    label_names, (train_iter, dev_iter, test_iter, inputs) = prepare_snli_batches(args)
    label_names = label_names[1:] # don't count UNK
    num_labels = len(label_names)

    print("Prepared Dataset...\n")

    sys.stdout.flush()
    model = SNLIClassifier(args, len(inputs.vocab.stoi))
    model.set_weight(inputs.vocab.vectors.numpy())

    print("Instantiated Model...\n")

    sys.stdout.flush()
    if args.gpu > -1:
        model.cuda()

    loss = torch.nn.CrossEntropyLoss(size_average=True)
    optimizer = optim.Adagrad(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    count_iter = 0
    teacher_prob = 1.0
    for epoch in range(args.epochs):
        train_iter.init_epoch()
        cost = 0
        for batch_idx, batch in enumerate(train_iter):
            model.train()
            count_iter += batch.batch_size
            cost += train_batch(
                model, loss, optimizer,
                (batch.hypothesis.transpose(0, 1), batch.hypothesis_transitions.t()),
                (batch.premise.transpose(0, 1), batch.premise_transitions.t()),
                batch.label - 1,
                teacher_prob
            )

            if count_iter >= args.eval_freq:
                correct, total = 0.0, 0.0
                count_iter = 0
                confusion_matrix = np.zeros([num_labels, num_labels])
                dev_iter.init_epoch()

                for dev_batch_idx, dev_batch in enumerate(dev_iter):
                    model.eval()
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
                print("Accuracy=%.2f%%, Epoch=%d, Batch=%d" % (float(correct) / total * 100, epoch, batch_idx))
                true_label_counts = confusion_matrix.sum(axis=1)
                pred_label_counts = confusion_matrix.sum(axis=0).tolist()
                pred_label_counts = [str(int(c)) for c in pred_label_counts] + ["--> guessed distribution"]
                print("\nConfusion matrix (x-axis is true labels)\n")
                label_names = [n[0:6] + '.' for n in label_names]
                print("\t" + "\t".join(label_names) + "\n")
                for i in range(num_labels):
                    print (label_names[i], end="")
                    for j in range(num_labels):
                        if true_label_counts[i] == 0:
                            perc = 0.0
                        else:
                            perc = confusion_matrix[i, j] / true_label_counts[i]
                        print("\t%.2f%%" % (perc * 100), end="")
                    print("\t(%d examples)\n" % true_label_counts[i])

                print("\t" + "\t".join(pred_label_counts))
                print("")
                sys.stdout.flush()

        teacher_prob *= args.force_decay
        print ("Cost for Epoch ", cost)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='SPINN dependency parse + SNLI Classifier arguments.')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--embed_dim', type=int, default=300)
    parser.add_argument('--hidden_size', type=int, default=300)
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate to pass to optimizer.')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('-continuous_stack', action='store_true', default=False)
    parser.add_argument('--eval_freq', type=int, default=1000, help='Number of examples between evaluation on dev set.')
    parser.add_argument('-debug', action='store_true', default=True)
    parser.add_argument('--snli_num_h_layers', type=int, default=2, help='Tunable hyperparameter.')
    parser.add_argument('--snli_h_dim', type=int, default=1024, help='1024 is used by paper.')
    parser.add_argument('--dropout_rate', type=float, default=0.1)
    parser.add_argument('-no_batch_norm', action='store_true', default=False)
    parser.add_argument('-tracking', action='store_true', default=False)
    parser.add_argument('-teacher', action='store_true', default=False)
    parser.add_argument('--force_decay', type=float, default=1.0)
    parser.add_argument('--gpu', type=int, default=-1, help='-1 for cpu. 0 for gpu')

    args = parser.parse_args()

    if args.continuous_stack:
        assert args.tracking

    render_args(args)
    sys.stdout.flush()
    train(args)
