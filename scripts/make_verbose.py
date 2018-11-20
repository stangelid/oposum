#!/usr/bin/env python

"""
This script will transform a gold labels file from a tab-delimited short format (e.g., salience/boots.sal)
to a verbose format that includes the segments' text (e.g., aspects/boots.asp).
"""

import sys
import argparse
from collections import defaultdict

parser = argparse.ArgumentParser(
  description =__doc__,
  formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('verbose', help='file in verbose format (e.g., ./aspects/boots.asp)', type=str)
parser.add_argument('short', help='file in short format (e.g., ./salience/boots.sal)', type=str)
parser.add_argument('out', help='output file', type=str)
args = parser.parse_args()

labels = defaultdict(list)
fshort = open(args.short, 'r')
for line in fshort:
   code, lbl = line.strip().split('\t')
   rcode, scode = code.rsplit('-', 1)
   labels[rcode].append(lbl)
fshort.close()

    
fverb = open(args.verbose, 'r')
fout = open(args.out, 'w')
first_line = True

if '.asp' in args.verbose:
    fverb.readline()
    fverb.readline()

for line in fverb:
    if not first_line:
        if len(line.strip()) != 0:
            if rcode not in labels:
                continue

            segment, lbls = line.strip().split('\t')
            
            fout.write('{0}\t{1}\n'.format(segment, labels[rcode][sid]))
            sid += 1
        else:
            first_line = True
            if rcode in labels:
                fout.write('\n')
    else:
        first_line = False
        rcode = line.strip()
        if rcode in labels:
            sid = 0
            fout.write(line)

fverb.close()
fout.close()
