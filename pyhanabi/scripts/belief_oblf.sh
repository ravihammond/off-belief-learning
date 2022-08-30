#!/bin/bash
python train_belief.py \
       --save_dir exps/belief_obl1_param \
       --num_thread 1 \
       --num_game_per_thread 1 \
       --batchsize 128 \
       --lr 6.25e-05 \
       --eps 1.5e-05 \
       --grad_clip 5 \
       --hid_dim 512 \
       --burn_in_frames 10000 \
       --replay_buffer_size 100000 \
       --epoch_len 1000 \
       --num_epoch 200 \
       --train_device cuda:0 \
       --act_device cuda:1 \
       --explore 1 \
       --policy ../training_models/obl1/model0.pthw \
       --seed 6754257 \
       --num_player 2 \
       --shuffle_color 0 \
       --load_model 1 \
       --convention conventions/CR-P0.json \
