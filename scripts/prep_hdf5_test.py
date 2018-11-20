#!/usr/bin/env python

import numpy as np
import h5py
import re
import sys
import operator
import argparse
from random import sample, seed
from math import ceil
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

from prep_hdf5 import line_to_words

def equal_length_chunks(l1, l2, l3, l4, l5):
    """
    Splits list of instances into batches, so that segments within batches
    are of equal length
    """
    prev = -1
    for i in range(len(l1)):
        if len(l1[i]) != prev:
            if prev != -1:
                yield l1[start:end+1], l2[start:end+1], l3[start:end+1], l4[start:end+1], l5[start:end+1]
            start = i
            end = i
        else:
            end += 1
        prev = len(l1[i])
    yield l1[start:end+1], l2[start:end+1], l3[start:end+1], l4[start:end+1], l5[start:end+1]

def load_data(file, args, word2id, prod2id):
    """
    Loads test data
    """
    padding = args.padding
    batch_size = args.batch_size
    stop_words = args.stop_words
    lemmatize = args.lemmatize

    data = []
    labels = []
    products = []
    scodes = []
    original = []

    doc_cnt = 0
    seg_cnt = 0
    prods = set()

    f = open(file, 'r')
    very_first_line = True
    first_line = False
    for line in f:
        if very_first_line:
            very_first_line = False
            aspects = line.strip().split('|')
        elif not first_line:
            if len(line.strip()) != 0:
                segment, lbls = line.strip().split('\t')

                segs, orig, ids, total = line_to_words(segment, 0, 10000,
                        stop_words=stop_words, lemmatize=lemmatize)

                seg_cnt += 1

                for i, seg in enumerate(segs):
                    seg_ids = [word2id[word] if word in word2id else 1 for word in seg]
                    seg_ids = [0] * padding + seg_ids + [0] * padding
                    if len(seg_ids) == 0:
                        seg_ids = [0]

                    scode = '{0}-{1:04d}'.format(rcode, start_idx + ids[i])

                    seg_asp = [0] * len(aspects)
                    for j in map(int, lbls.split()):
                        seg_asp[j] = 1

                    data.append(seg_ids)
                    labels.append(seg_asp)
                    products.append(pid)
                    scodes.append(scode)
                    original.append(orig[i])
                start_idx += total
            else:
                first_line = True
        else:
            doc_cnt += 1
            first_line = False
            rcode = line.strip()
            prod, _ = rcode.split('-')
            prods.add(prod)
            pid = prod2id[prod]
            start_idx = 0

    f.close()

    print 'Number of documents:', doc_cnt
    print 'Number of segments:', seg_cnt
    print 'Number of products:', len(prods)
    print 'Number of aspects:', len(aspects)
    print 'Vocabulary size:', len(word2id)


    return data, labels, products, scodes, original, aspects

def clean_str(string):
    """
    String cleaning
    """
    string = string.lower()
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"&#34;", " ", string)
    string = re.sub(r"(http://)?www\.[^ ]+", " _url_ ", string)
    string = re.sub(r"[^a-z0-9$\'_]", " ", string)
    string = re.sub(r"_{2,}", "_", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'m", " \'m", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r"\$+", " $ ", string)
    string = re.sub(r"rrb", " ", string)
    string = re.sub(r"lrb", " ", string)
    string = re.sub(r"rsb", " ", string)
    string = re.sub(r"lsb", " ", string)
    string = re.sub(r"(?<=[a-z])I", " I", string)
    string = re.sub(r"(?<= )[0-9]+(?= )", "<NUM>", string)
    string = re.sub(r"(?<= )[0-9]+$", "<NUM>", string)
    string = re.sub(r"^[0-9]+(?= )", "<NUM>", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()

def main():
    parser = argparse.ArgumentParser(
      description =__doc__,
      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', help='data in appropriate format', type=str, default='')
    parser.add_argument('--name', help='name of hdf5 file', type=str, default='')
    parser.add_argument('--vocab', help='vocabulary file (from train set)', type=str, default='')
    parser.add_argument('--products', help='product ids file (from train set)', type=str, default='')
    parser.add_argument('--batch_size', help='maximum number of segments per batch (default: 50)', type=int, default=50)
    parser.add_argument('--padding', help='padding around each sentence (default: 0)', type=int, default=0)
    parser.add_argument('--lemmatize', help='Lemmatize words', action='store_true')
    parser.add_argument('--stopfile', help='Stop-word file (default: nltk english stop words)', type=str, default='')
    parser.add_argument('--seed', help='random seed (default: 1)', type=int, default=1)
    args = parser.parse_args()

    if args.stopfile == 'no':
        args.stop_words = None
    elif args.stopfile != '':
        stop_words = set()
        fstop = open(args.stopfile, 'r')
        for line in fstop:
            stop_words.add(line.strip())
        fstop.close()
        args.stop_words = stop_words
    else:
        args.stop_words = set(stopwords.words('english'))

    word2id = {}
    fvoc = open(args.vocab, 'r')
    for line in fvoc:
        word, id = line.split()
        word2id[word] = int(id)
    fvoc.close()

    if args.products != '':
        prod2id = {}
        fprod = open(args.products, 'r')
        for line in fprod:
            prod, id = line.split()
            prod2id[prod] = int(id)
        fprod.close()
    else:
        prod2id = None

    data, labels, products, scodes, original, aspects = load_data(args.data, args, word2id, prod2id)

    seed(args.seed)
    data, labels, products, scodes, original = zip(*sorted(
        sample(zip(data, labels, products, scodes, original), len(data)),
        key=lambda x:len(x[0])))

    filename = args.name + '.hdf5'
    with h5py.File(filename, 'w') as f:
        for i, (segments, lbls, prods, codes, segs_o), in enumerate(equal_length_chunks(data, labels, products, scodes, original)):
            max_len_batch = len(max(segments, key=len))
            batch_id = str(i)

            for j in range(len(segments)):
                segments[j].extend([0] * (max_len_batch - len(segments[j])))
            f['data/' + batch_id] = np.array(segments, dtype=np.int32)
            f['labels/' + batch_id] = np.array(lbls, dtype=np.int32)
            f['products/' + batch_id] = np.array(prods,  dtype=np.int32)
            f.create_dataset('scodes/' + batch_id, (len(codes),), dtype="S{0}".format(len(codes[0])), data=codes)
            dt = h5py.special_dtype(vlen=bytes)
            f.create_dataset('original/' + batch_id, (len(segs_o),), dtype=dt, data=segs_o)


if __name__ == '__main__':
    main()
