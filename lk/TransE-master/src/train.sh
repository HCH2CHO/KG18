#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 \
python -u main.py \
--model TransR \
--data WN18 \
--embedding_dim 100 \
--margin_value 1 \
--batch_size 1440 \
--learning_rate 0.001 \
--n_generator 24 \
--n_rank_calculator 24 \
--eval_freq 50 \
--max_epoch 1000