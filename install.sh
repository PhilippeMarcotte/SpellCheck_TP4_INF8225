#!/bin/bash

yum install git

git clone https://esiode:in9el106e2y@bitbucket.org/inf8225/tp4.git

cd tp4/dataset

wget http://www.statmt.org/lm-benchmark/1-billion-word-language-modeling-benchmark-r13output.tar.gz

md5sum tar_archives/training-monolingual.tgz 

tar --extract -v --file tar_archives/training-monolingual.tgz --wildcards training-monolingual/news.20??.en.shuffled
md5sum training-monolingual/*

mkdir tmp.tmp

TMPDIR=tmp.tmp ./get-data.sh 

rmdir tmp.tmp