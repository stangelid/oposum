#!/bin/bash

domain="$1"
upper=`echo "$1" | tr '[:lower:]' '[:upper:]'`
base=`dirname $0`

if [ ! -f ./w2v/GoogleNews-vectors-negative300.bin ]; then
    wget -c "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"
    if [ -z `which gunzip` ]; then
        echo "Please install gunzip and run again, or extract binary from"
        echo "GoogleNews-vectors-negative300.bin.gz using external app"
        echo "and move it to ./w2v"
        exit
    fi

    gunzip GoogleNews-vectors-negative300.bin.gz
    mv GoogleNews-vectors-negative300.bin ./w2v
fi

cat data/train/"$domain".trn | sed -e $'s/ EDU_BREAK /\\\n/g' > data/train/"$domain"-edus.trn

./scripts/prep_hier_hdf5.py \
    --w2v ./w2v/GoogleNews-vectors-negative300.bin \
    --name ./data/preprocessed/"$upper"_MILNET \
    --train data/train/"$domain"-edus.trn \
    --batch_size 200

echo

./scripts/prep_hier_extra.py \
    --data data/gold/polarities/discrete/"$domain".lbl \
    --name ./data/preprocessed/"$upper"_MILNET_TEST \
    --vocab ./data/preprocessed/"$upper"_MILNET_word_mapping.txt

rm data/train/"$domain"-edus.trn
