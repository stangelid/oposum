#!/bin/bash

domain="$1"
upper=`echo "$1" | tr '[:lower:]' '[:upper:]'`
base=`dirname $0`

./scripts/prep_hdf5.py \
    --w2v ./w2v/"$domain".bin \
    --name ./data/preprocessed/"$upper"_MATE \
    --data ./data/train/"$domain".trn \
    --lemmatize

echo

./scripts/prep_hdf5_test.py \
    --data ./data/gold/aspects/"$domain"-tst.asp \
    --name ./data/preprocessed/"$upper"_MATE_TEST \
    --vocab ./data/preprocessed/"$upper"_MATE_word_mapping.txt \
    --products ./data/preprocessed/"$upper"_MATE_product_mapping.txt \
    --lemmatize
