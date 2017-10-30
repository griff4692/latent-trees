import json
import os

MINI_SIZE = 256
OVERWRITE = True

def gen_mini():
    data_dir = '.data/snli/snli_1.0'

    names = ['train', 'dev', 'test']

    for name in names:
        source = 'snli_1.0_%s.jsonl' % name
        dest = 'snli_1.0_mini_%s.jsonl' % name

        source_file = os.path.join(data_dir, source)
        dest_file = os.path.join(data_dir, dest)

        if os.path.exists(dest_file) and not OVERWRITE:
            continue

        data = open(source_file, 'r').readlines()
        truncated_data = data[:MINI_SIZE]

        dest_fd = open(dest_file, 'w')
        for pt in truncated_data:
            dest_fd.write(pt)

if __name__ =='__main__':
    gen_mini()
