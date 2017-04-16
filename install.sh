#!/bin/bash

yum install git

git clone https://esiode:in9el106e2y@bitbucket.org/inf8225/tp4.git

mkdir -p tp4/dataset/tar_archives

cd tp4/dataset/tar_archives

wget http://www.statmt.org/lm-benchmark/1-billion-word-language-modeling-benchmark-r13output.tar.gz

cd ..

md5sum tar_archives/training-monolingual.tgz 

tar --extract -v --file tar_archives/training-monolingual.tgz --wildcards training-monolingual/news.20??.en.shuffled
md5sum training-monolingual/*

source ./scripts/get-data.sh 

cd ..

python3 src/training.py