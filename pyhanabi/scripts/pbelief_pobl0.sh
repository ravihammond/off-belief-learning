#!/bin/bash
python train_belief.py \
       --save_dir exps/pbelief_pobl0_CR-P0 \
       --num_thread 24 \
       --num_game_per_thread 80 \
       --batchsize 128 \
       --lr 6.25e-05 \
       --eps 1.5e-05 \
       --grad_clip 5 \
       --hid_dim 512 \
       --burn_in_frames 10000 \
       --replay_buffer_size 100000 \
       --epoch_len 1000 \
       --num_epoch 2000 \
       --train_device cuda:0 \
       --act_device cuda:1,cuda:2,cuda:3 \
       --explore 1 \
       --policy exps/iql/model0.pthw \
       --seed 2254257 \
       --num_player 2 \
       --shuffle_color 0 \
       --rand 1 \
       --convention conventions/CR-P0.json \
       --num_conventions 1 \
       --parameterized_belief 1 \
       --parameterized_act 0 \

