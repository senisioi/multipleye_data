#!/bin/bash

# use this preliminary dataset to generate plots
# wget https://github.com/senisioi/multipleye_data/releases/download/0.1/csv.tar.gz && \
#tar -xvf csv.tar.gz && \

# the one below is password protected, if you need the data, ask sergiu

rm -rf data/processed && \
mkdir -p data/processed && \
cd data/processed && \
rm -rf csv* && \
wget https://github.com/senisioi/multipleye_data/releases/download/0.1/csv.zip && \
unzip -P $1 csv.zip && \
mv csv/* . && \
rmdir csv
