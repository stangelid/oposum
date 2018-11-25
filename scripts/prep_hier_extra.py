#!/usr/bin/python

import numpy as np
import h5py
import re
import sys
import operator
import argparse
from random import sample, seed
from math import ceil
from preprocess import chunks, special_len, line_to_words, clean_str

def load_data(fname, args, word2id):
    """
    Reads the segment-level annotated data
    """
    padding = args.padding
    batch_size = args.batch_size
    stop_words = args.stop_words
    lemmatize = args.lemmatize

    docs = []
    lbls = []
    seg_lbls = []
    codes = []
    orig = []

    f = open(fname, 'r')
    first_line = True
    doc_cnt = 0
    seg_cnt = 0
    for line in f:
        if not first_line:
            if len(line.strip()) != 0:
                if len(doc) == args.max_len:
                    docs.append(doc)
                    lbls.append(label)
                    seg_lbls.append(doc_seg_lbls)
                    codes.append(scodes)
                    orig.append(doc_orig)
                    doc = []
                    doc_seg_lbls = []
                    doc_orig = []
                    scodes = []

                if args.nolbl:
                    text = line.strip()
                    seg_label = '_'
                else:
                    seg_label, text = line.strip().split('\t')

                seg, original = line_to_words(text, 0, 10000,
                        stop_words=stop_words, lemmatize=lemmatize)

                scode = '{0}-{1:04d}'.format(rcode, sid)
                sid += 1
                if len(seg) >= args.min_len:
                    seg_ids = [word2id[word] for word in seg if word in word2id]
                    seg_ids = [0] * padding + seg_ids + [0] * padding
                    doc.append(seg_ids)
                    doc_orig.append(original)
                    doc_seg_lbls.append(seg_label)
                    scodes.append(scode)
                    seg_cnt += 1
            else:
                first_line = True
                if len(doc) > 0:
                    docs.append(doc)
                    lbls.append(label)
                    seg_lbls.append(doc_seg_lbls)
                    codes.append(scodes)
                    orig.append(doc_orig)
        else:
            first_line = False
            doc = []
            doc_seg_lbls = []
            doc_orig = []
            scodes = []
            sid = 0
            label, rcode = line.split(None, 2)
            label = int(label)
            doc_cnt += 1

    f.close()

    print 'Number of documents:', doc_cnt
    print 'Number of segments:', seg_cnt

    return docs, lbls, seg_lbls, codes, orig

def main():
    parser = argparse.ArgumentParser(
      description =__doc__,
      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--name', help="name of hdf5 file", type=str, default="")
    parser.add_argument('--data', help="data in appropriate format", type=str, default="")
    parser.add_argument('--nolbl', help="there are no segment-level labels", action='store_true')
    parser.add_argument('--min_len', help="minimum segment length", type=int, default=1)
    parser.add_argument('--max_len', help="minimum document length", type=int, default=100)
    parser.add_argument('--vocab', help="vocabulary file", type=str, default="")
    parser.add_argument('--batch_size', help="number of segments per batch (default: 5)", type=int, default=5)
    parser.add_argument('--padding', help="padding around each sentence (default: 2)", type=int, default=2)
    parser.add_argument('--lemmatize', help="Lemmatize words", action='store_true')
    parser.add_argument('--stopfile', help="Stop-word file", type=str, default='no')
    parser.add_argument('--seed', help="random seed (default: 1)", type=int, default=1)
    args = parser.parse_args()

    seed(args.seed)

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
        from nltk.corpus import stopwords
        args.stop_words = set(stopwords.words('english'))

    # loads existing vocabulary
    word2id = {}
    fvoc = open(args.vocab, 'r')
    for line in fvoc:
        word, id = line.split()
        word2id[word] = int(id)
    fvoc.close()

    # loads and shuffles data
    docs, labels, seg_labels, codes, original = load_data(args.data, args, word2id)
    data = sorted(sample(zip(docs, labels, seg_labels, codes, original), len(docs)), key=special_len)

    f = h5py.File(args.name + '.hdf5', 'w')
    flbl = open(args.name + '.info', 'w')
    for i, chunk in enumerate(chunks(data, args.batch_size)):
        docs, lbls, seg_lbls, codes, original = map(list, zip(*chunk))

        max_len_batch = 0
        max_seg_batch = len(max(docs, key=len))

        # writes segment-level info on .info file
        for j in range(len(docs)):
            for k in range(len(docs[j])):
                flbl.write('{0}\t{1} {2} {3}\t{4}\t{5}\n'.format(codes[j][k], i, j, k, seg_lbls[j][k], original[j][k]))

        for j in range(len(docs)):
            max_len_batch = max(max_len_batch, len(max(docs[j], key=len)))

        for j in range(len(docs)):
            original[j] += '\n' * (max_seg_batch - len(docs[j]))
            docs[j].extend([[0] * max_len_batch] * (max_seg_batch - len(docs[j])))
            for k in range(len(docs[j])):
                docs[j][k].extend([0] * (max_len_batch - len(docs[j][k])))

        # writes inputs and document-level labels on hdf5 format
        f['data/' + str(i)] = np.array(docs, dtype=np.int32)
        f['label/' + str(i)] = np.array(lbls, dtype=np.int32)

    f.close()
    flbl.close()

if __name__ == '__main__':
    main()
