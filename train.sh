#!/bin/bash
# Example usage:
# bash train.sh
python src/labeler.py --input data/lyrics.csv --out data/labeled.csv --nrows 200000
python src/train_baseline.py --input data/labeled.csv --out models/baseline.pkl --nrows 200000 --drop_neutral
# To train transformer (GPU recommended)
python src/train_transformer.py --input data/labeled.csv --out models/distilbert --nrows 50000 --drop_neutral
