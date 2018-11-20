#!/bin/bash

datadir="$1"
outdir="$2"

for c in bluetooth boots bags_and_cases tv keyboards vacuums; do 
    ./term-weights.py "$datadir"/"$c"-dev.asp --outdir "$outdir"/"$c" -s -l
    aspects=`head -1 "$datadir"/"$c"-dev.asp | tr '|' ' '`

    for i in 5 15 20 30; do
        for a in $aspects; do
            for j in `seq 1 "$i"`; do
                cat "$outdir"/"$c"/"$c"-dev."$a".clarity.txt
            done | head -"$i"
        done > "$outdir"/"$c"/"$c".clarity.tmp

        cat "$outdir"/"$c"/"$c".clarity.tmp \
            | awk '{printf "%s ", $2; if (NR % '"$i"' == 0) print ""}' \
            > "$outdir"/"$c"."$i".txt

        cat "$outdir"/"$c"/"$c".clarity.tmp \
            | awk '{printf "%s:%.5f ", $2, $1; if (NR % '"$i"' == 0) print ""}' \
            > "$outdir"/"$c"."$i"-weights.txt

        rm "$outdir"/"$c"/"$c".clarity.tmp
    done
done
