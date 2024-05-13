#!/bin/bash
python tools/save_scores.py \
    --model1 iql \
    --model2 iql \
    --crossplay 1 \

python tools/save_scores.py \
    --model1 iql \
    --model2 sad \
    --crossplay 1 \

python tools/save_scores.py \
    --model1 iql \
    --model2 op \
    --crossplay 1 \

python tools/save_scores.py \
    --model1 iql \
    --model2 obl \
    --crossplay 1 \

python tools/save_scores.py \
    --model1 obl \
    --model2 sad \
    --crossplay 1 \
