#!/bin/bash
#count lines in every company stock list
#if lines are greater than 253, keep the file,
#otherwise remove it

keep_count=0

for filename in data/dataset/Stocks/*.txt; do
    v1=($(wc -l "$filename"))
    if [ $v1 -gt 1271 ]
    then
        keep_count=$((keep_count + 1)) 
    else
        echo "removing..... $filename"
        $(rm "$filename")
    fi
done

echo "files remaining = $keep_count"
