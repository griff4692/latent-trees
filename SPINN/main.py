import torch
import torch.optim as optim
from snli_classifier import SNLIClassifier
from batcher import prepare_snli_batches
import numpy as np
import argparse
from utils import render_args

def predict(model, sent1, sent2):
    output = model(sent1, sent2)
    return output.data.numpy().argmax(axis=1)

def train_batch(model, loss, optimizer, sent1, sent2, y_val):
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

def train(args):
    label_names, (train_iter, dev_iter, test_iter, inputs) = prepare_snli_batches(args)
    label_names = label_names[1:] # don't count UNK
    num_labels = len(label_names)
    print ("Prepared Dataset...")
    model = SNLIClassifier(args, len(inputs.vocab.stoi))
    model.set_weight(inputs.vocab.vectors.numpy())
    print ("Instantiated Model...")
    if args.cuda:
        model.cuda()
    loss = torch.nn.CrossEntropyLoss(size_average=True)
    optimizer = optim.Adagrad(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    count_iter = 0

    for epoch in range(args.epochs):
        train_iter.init_epoch()
        cost = 0
        for batch_idx, batch in enumerate(train_iter):
            count_iter += batch.batch_size
            cost += train_batch(
                model, loss, optimizer,
                (batch.hypothesis.transpose(0, 1), batch.hypothesis_transitions.t()),
                (batch.premise.transpose(0, 1), batch.premise_transitions.t()),
                batch.label - 1 # TODO double check this works
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
                            dev_batch.premise_transitions.t())
                    )

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
                print "Accuracy is %.2f" % (float(correct) / total)
                true_label_counts = confusion_matrix.sum(axis=1)
                print "Confusion matrix (x-axis is true labels)\n"
                label_names = [n[0:6] + '.' for n in label_names]
                print "\t\t" + "\t".join(label_names) + "\n"
                for i in range(num_labels):
                    print label_names[i],
                    for j in range(num_labels):
                        if true_label_counts[i] == 0:
                            perc = 0.0
                        else:
                            perc = confusion_matrix[i, j] / true_label_counts[i]
                        print "\t%.2f%%" % (perc * 100),
                    print "\t(%d examples)\n" % true_label_counts[i]
        print ("Cost for Epoch ", cost)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='SPINN dependency parse + SNLI Classifier arguments.')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--embed_dim', type=int, default=300)
    parser.add_argument('--hidden_size', type=int, default=300)
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate to pass to optimizer.')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('-continuous_stack', action='store_true', default=False)
    parser.add_argument('--eval_freq', type=int, default=10, help='number of epochs between evaluation on dev set.')
    parser.add_argument('-debug', action='store_true', default=False)
    parser.add_argument('--snli_num_h_layers', type=int, default=1, help='tunable hyperparameter.')
    parser.add_argument('--snli_h_dim', type=int, default=1024, help='1024 is used by paper.')
    parser.add_argument('--dropout_rate', type=float, default=0.5)
    parser.add_argument('-no_batch_norm', action='store_true', default=False)
    parser.add_argument('-gpu', action='store_true', default=False)

    args = parser.parse_args()
    render_args(args)
    train(args)
