#!/usr/bin/env python

import sys
import argparse
import re
import os.path
from os import makedirs
from pprint import pprint
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from numpy import log
from scipy.special import rel_entr
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

parser = argparse.ArgumentParser(
        description =__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('filename', help='Aspect labels file', type=str)
parser.add_argument('--outdir', help='Output directory', type=str, default='.')
parser.add_argument('-s', '--remove_stopwords', help='Remove stopwords', action='store_true')
parser.add_argument('-l', '--lemmatize', help='Lemmatize words', action='store_true')
args = parser.parse_args()

f = open(args.filename, 'r')

# reads aspect titles
header = f.readline()
aspects = header.strip().replace(' ', '_').replace('/','_').split('|')
aspect_segments = dict([(aspect, []) for aspect in aspects])
f.readline() # skips empty line

if args.lemmatize:
    lemmatizer = WordNetLemmatizer()
else:
    lemmatizer = None

if args.remove_stopwords:
    stop_words = set(stopwords.words('english'))
else:
    stop_words = set()

token_pattern = re.compile(r'(?u)\b\w\w+\b')

# the loop will read file and store segments globally, and per aspect
all_segs = []
first = True
for line in f:
    if not first:
        if len(line.strip()) == 0:
            first = True
        else:
            seg_body, seg_asptext = line.strip().split('\t')

            seg_words = [word for word in token_pattern.findall(seg_body.lower())
                                  if word not in stop_words]

            if lemmatizer is not None:
                seg_words = [lemmatizer.lemmatize(word) for word in seg_words]

            seg_prep = ' '.join(seg_words)

            seg_aspects = map(int, seg_asptext.split())
            for i, aspect in enumerate(seg_aspects):
                aspect_segments[aspects[aspect]].append(seg_prep)
            all_segs.append(seg_prep)
    else:
        first = False

f.close()

# compute tfidf scores
vectorizer = TfidfVectorizer(stop_words='english' if args.remove_stopwords else None,
        norm='l1', use_idf=True)
vectorizer.fit(all_segs)
gl_freq = vectorizer.transform([' '.join(all_segs)]).toarray()[0]

# global scores
gl_scores = {}
for term, idx in vectorizer.vocabulary_.items():
    gl_scores[term] = gl_freq[idx]

asp_scores = dict([(aspect, {}) for aspect in aspect_segments.keys()])
for aspect, segments in aspect_segments.items():

    # aspect-specific scores
    asp_freq = vectorizer.transform([' '.join(segments)]).toarray()[0]

    # entropies correspond to clarity scores
    entropies = rel_entr(asp_freq, gl_freq) / log(2)
    for term, idx in vectorizer.vocabulary_.items():
        asp_scores[aspect][term] = entropies[idx]

    # sort by score and write to file if > 0
    scores = sorted(asp_scores[aspect].items(), reverse=True, key=lambda x:x[1])
    if args.outdir == '':
        fout = open('{0}.{1}.clarity.txt'.format(args.filename[:-4], aspect), 'w')
    else:
        if not os.path.exists(args.outdir):
            makedirs(args.outdir)
        fout = open(args.outdir + '/{0}.{1}.clarity.txt'.format(os.path.basename(args.filename)[:-4], aspect), 'w')
    for term, cla in scores[:50]:
        if cla > 0:
            fout.write('{0:.5f} {1}\n'.format(cla, term))
    fout.close()
