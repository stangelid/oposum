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

def parallel_chunks(l1, l2, l3, l4, n):
    """
    Yields chunks of size n from 4 lists in parallel
    """
    if len(l1) != len(l2) or len(l2) != len(l3) or len(l3) != len(l4):
        raise IndexError
    else:
        for i in xrange(0, len(l1), n):
            yield l1[i:i+n], l2[i:i+n], l3[i:i+n], l4[i:i+n]

def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())

        # number of bytes per embedding
        binary_len = np.dtype('float32').itemsize * layer1_size

        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)

            # store words in vocab, discard rest
            if vocab is None or word in vocab:
                word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')
            else:
                f.read(binary_len)
    return word_vecs

def line_to_words(line, min_len, max_len, stop_words=None, lemmatize=True):
    """
    Reads a line of text (sentence) and returns a list of tokenized EDUs
    """
    if lemmatize:
        lemmatizer = WordNetLemmatizer()

    # clean sentence and break it into EDUs
    clean_line = clean_str(line.strip())
    edus = [edu.strip() for edu in clean_line.split('edu_break')]

    # original text for each EDU
    edus_o = [edu.strip() for edu in line.split('EDU_BREAK')]

    segs = []
    ids = []
    total = 0
    original = []

    for i, edu in enumerate(edus):
        total += 1
        words = edu.split()

        if stop_words is not None:
            words = [word for word in words if word not in stop_words]

        if lemmatize:
            words = [lemmatizer.lemmatize(word) for word in words]

        # discard short segments
        if len(words) < min_len:
            continue

        # truncate long ones
        if len(words) > max_len:
            words = words[:max_len]

        segs.append(words)
        ids.append(i) # storing ids to keep track of discarded segments
        original.append(edus_o[i])

    return segs, original, ids, total

def get_vocab(file, min_len, max_len, stop_words, lemmatize):
    """
    Reads an input file and builds vocabulary, product mapping, etc.
    """
    max_len_actual = 0
    wid = 1
    pid = 0
    word2id = {}
    word2cnt = {}
    prod2id = {}
    seg_cnt = 0
    doc_cnt = 0

    f = open(file, 'r')
    first_line = True
    for line in f:
        if not first_line:
            if len(line.strip()) != 0:
                segs, _, _, _ = line_to_words(line, min_len, max_len, stop_words, lemmatize)

                for seg in segs:
                    seg_cnt += 1
                    max_len_actual = max(max_len_actual, len(seg))
                    for word in seg:
                        if word not in word2id:
                            word2id[word] = wid
                            wid += 1
                        if word not in word2cnt:
                            word2cnt[word] = 1
                        else:
                            word2cnt[word] += 1
            else:
                first_line = True
                doc_cnt += 1
        else:
            first_line = False
            rcode, label = line.split()
            prod, _ = rcode.split('-')
            if prod not in prod2id:
                prod2id[prod] = pid
                pid += 1

    f.close()

    return max_len_actual, seg_cnt, doc_cnt, word2id, prod2id, word2cnt


def load_data(file, args):
    """
    Loads dataset into appropriate data structures
    """
    padding = args.padding
    min_len = args.min_len
    max_len = args.max_len
    batch_size = args.batch_size
    stop_words = args.stop_words
    lemmatize = args.lemmatize

    max_len_actual, seg_cnt, doc_cnt, word2id, prod2id, word2cnt = get_vocab(file, min_len, max_len,
            stop_words, lemmatize)

    print 'Number of documents:', doc_cnt
    print 'Number of edus:', seg_cnt
    print 'Number of products:', len(prod2id)
    print 'Max segment length:', max_len_actual
    print 'Vocabulary size:', len(word2id)

    data = []
    products = []
    scodes = []
    original = []

    f = open(file, 'r')
    first_line = True
    for line in f:
        if not first_line:
            if len(line.strip()) != 0:
                segs, orig, ids, total = line_to_words(line, min_len, max_len,
                        stop_words=stop_words, lemmatize=lemmatize)

                for i, seg in enumerate(segs):
                    seg_ids = [word2id[word] for word in seg]
                    seg_ids = [0] * padding + seg_ids + [0] * padding

                    scode = '{0}-{1:04d}'.format(rcode, start_idx + ids[i])

                    data.append(seg_ids)
                    products.append(pid)
                    scodes.append(scode)
                    original.append(orig[i])
                start_idx += total
            else:
                first_line = True
        else:
            first_line = False
            rcode, label = line.split()
            prod, _ = rcode.split('-')
            pid = prod2id[prod]
            start_idx = 0

    f.close()

    return word2id, prod2id, data, products, scodes, original, word2cnt

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
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()

def main():
    parser = argparse.ArgumentParser(
      description =__doc__,
      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--w2v', help='word2vec binary file', type=str, default='')
    parser.add_argument('--data', help='data in appropriate format', type=str, default='')
    parser.add_argument('--name', help='name of dataset', type=str, default='')
    parser.add_argument('--batch_size', help='number of segments per batch (default: 50)', type=int, default=50)
    parser.add_argument('--padding', help='padding around each segment (default: 0)', type=int, default=0)
    parser.add_argument('--lemmatize', help='Lemmatize words', action='store_true')
    parser.add_argument('--stopfile', help='Stop-word file (default: nltk english stop words)', type=str, default='')
    parser.add_argument('--min_len', help='minimum allowed words per segment (default: 1)', type=int, default=1)
    parser.add_argument('--max_len', help='maximum allowed words per segment (default: 150)', type=int, default=150)
    parser.add_argument('--seed', help='random seed (default: 1)', type=int, default=1)
    args = parser.parse_args()

    # set stop words policy
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

    # load data
    word2id, prod2id, data, products, scodes, original, word2cnt = load_data(args.data, args)

    # write mapping files
    with open(args.name + '_word_mapping.txt', 'w') as f:
      f.write('<PAD> 0\n')
      for word, idx in sorted(word2id.items(), key=operator.itemgetter(1)):
        f.write("%s %d\n" % (word, idx))

    with open(args.name + '_product_mapping.txt', 'w') as f:
      for prod, idx in sorted(prod2id.items(), key=operator.itemgetter(1)):
        f.write("%s %d\n" % (prod, idx))

    with open(args.name + '_word_counts.txt', 'w') as f:
      for word, count in sorted(word2cnt.items(), key=operator.itemgetter(1), reverse=True):
        f.write("%s %d\n" % (word, count))

    # populate embedding matrix
    vocab_size = len(word2id) + 1
    w2v = load_bin_vec(args.w2v, word2id)
    embed = np.random.uniform(-0.25, 0.25, (vocab_size, len(w2v.values()[0])))
    embed[0] = 0
    for word, vec in w2v.items():
      embed[word2id[word]] = vec

    seed(args.seed)

    # sort data by segment length (to minimize padding)
    data, products, scodes, original = zip(*sorted(
        sample(zip(data, products, scodes, original), len(data)),
        key=lambda x:len(x[0])))

    filename = args.name + '.hdf5'
    with h5py.File(filename, 'w') as f:
        f['w2v'] = np.array(embed)

        for i, (segments, prods, codes, segs_o), in enumerate(parallel_chunks(data, products,
                                                        scodes, original, args.batch_size)):
            max_len_batch = len(max(segments, key=len))
            batch_id = str(i)

            for j in range(len(segments)):
                segments[j].extend([0] * (max_len_batch - len(segments[j])))
            f['data/' + batch_id] = np.array(segments, dtype=np.int32)
            f['products/' + batch_id] = np.array(prods,  dtype=np.int32)
            f.create_dataset('scodes/' + batch_id, (len(codes),), dtype="S{0}".format(len(codes[0])), data=codes)
            dt = h5py.special_dtype(vlen=bytes)
            f.create_dataset('original/' + batch_id, (len(segs_o),), dtype=dt, data=segs_o)


if __name__ == '__main__':
    main()
