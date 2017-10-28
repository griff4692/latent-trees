import json
from vocab import Vocab
from nltk.tokenize.moses import MosesTokenizer
import numpy as np
import pickle

dim = 100
vocab = Vocab(dim)
tokenizer = MosesTokenizer(lang='en')

data_pref = 'data/snli/snli_1.0_'
LABELS = ["neutral", "entailment", "contradiction"]
flavors = ['train', 'dev', 'test']

MAX_SENTENCE_LENGTH = 64

def load(flavor):
    path = data_pref + flavor + '.npy'
    return np.load(open(path))

def load_vocab(dim):
    return pickle.load(open('vocab_%d.pk' % dim))

if __name__=='__main__':
    for flavor in flavors:
        fd = open(data_pref + flavor + '.jsonl', 'rb')

        full_sent1_ids = []
        full_sent2_ids = []
        labels = []

        max_sent_len = 0
        num_seen = 0
        num_dropped = 0
        num_too_long = 0
        num_unk = 0
        for line in fd:
            num_seen += 1
            data_point = json.loads(line)
            sent1 = data_point['sentence1']
            sent2 = data_point['sentence2']
            label = data_point['gold_label']

            if label == '-':
                num_dropped += 1
                continue

            label_idx =  LABELS.index(label)
            assert label_idx >= 0 and label_idx <= 2

            sent1_tokens = [t.lower() for t in tokenizer.tokenize(sent1)]
            sent2_tokens = [t.lower() for t in tokenizer.tokenize(sent2)]

            if flavor == 'train':
                sent1_ids = vocab.add_tokens(sent1_tokens)
                sent2_ids = vocab.add_tokens(sent2_tokens)
            else:
                sent1_ids = vocab.tokens_to_ids(sent1_tokens)
                sent2_ids = vocab.tokens_to_ids(sent2_tokens)

                if vocab.unk_idx() in sent1_ids or vocab.unk_idx() in sent2_ids:
                    num_unk += 1

            max_sent_len = max(max_sent_len, len(sent1_ids), len(sent2_ids))

            if len(sent1_ids) > MAX_SENTENCE_LENGTH or len(sent2_ids) > MAX_SENTENCE_LENGTH:
                num_too_long += 1
                continue

            full_sent1_ids.append(sent1_ids)
            full_sent2_ids.append(sent2_ids)
            labels.append(label_idx)

        print "Statistics for %s..." % flavor
        print "Dropped %d out of %d examples (no annotator agreement)" % (num_dropped, num_seen)
        print "Dropped %d out of %d examples (no annotator agreement)" % (num_too_long, num_seen - num_dropped)
        print "\t-->Saved %d examples" % (num_seen - num_dropped)
        print "Unknown: %d out of %d examples" % (num_unk, num_seen - num_dropped)
        print "Max sentence length is %d" % max_sent_len
        print "\n"

        train_data = np.asarray([full_sent1_ids, full_sent2_ids, labels])
        np.save(data_pref + flavor + ".npy", train_data)

    print "Saving vocabulary of size %d" % vocab.size()
    vocab.glove = None
    pickle.dump(vocab, open('vocab_%d.pk' % dim, 'w'))
