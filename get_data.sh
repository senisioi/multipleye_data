#/bin/bash

mkdir -p data && \
cd data && \
rm -rf csv* && \
wget https://github.com/senisioi/multipleye_data/releases/download/0.1/csv.tar.gz \\
tar -xvf csv.tar.gz && \
mv csv/* .
