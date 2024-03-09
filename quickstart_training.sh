#!/usr/bin/env bash

pip install -r requirements.txt

if [[ ! -d "data/datafiles" ]]; then
  mkdir data/datafiles
fi

echo Making the datafiles
cd data || exit
python3 data_loader.py
cd .. || exit

echo Training models, Hold onto your boots
python3 train.py
