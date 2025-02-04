#!/bin/bash

python selfplay.py \
       --save_dir exps/sba_iql \
       --num_thread 24 \
       --num_game_per_thread 80 \
       --method iql \
       --sad 0 \
       --lr 6.25e-05 \
       --eps 1.5e-05 \
       --gamma 0.999 \
       --seed $1 \
       --burn_in_frames 10000 \
       --replay_buffer_size 100000 \
       --batchsize 128 \
       --epoch_len 1000 \
       --num_epoch 1001 \
       --num_player 2 \
       --net lstm \
       --num_lstm_layer 2 \
       --multi_step 3 \
       --train_device cuda:0 \
       --act_device cuda:1,cuda:2,cuda:3 \
       --train_partner_models agent_groups/all_iql.json \
       --train_partner_sad_legacy 1 \
       --train_partner_iql_legacy 1 \
       --train_test_splits train_test_splits/iql_splits_one.json \
       --split_index $1 \
       --static_partner 1 \
       --shuffle_color 1 \
       --wandb 1 \
       --gcloud_upload 1

#python selfplay.py \
#       --save_dir exps/test \
#       --num_thread 24 \
#       --num_game_per_thread 80 \
#       --method iql \
#       --sad 0 \
#       --lr 6.25e-05 \
#       --eps 1.5e-05 \
#       --gamma 0.999 \
#       --seed 9 \
#       --burn_in_frames 10000 \
#       --replay_buffer_size 100000 \
#       --batchsize 128 \
#       --epoch_len 1000 \
#       --num_epoch 1001 \
#       --num_player 2 \
#       --net lstm \
#       --num_lstm_layer 2 \
#       --multi_step 3 \
#       --train_device cuda:0 \
#       --act_device cuda:1,cuda:2,cuda:3 \
#       --train_partner_models agent_groups/all_iql.json \
#       --train_partner_sad_legacy 1 \
#       --train_partner_iql_legacy 1 \
#       --train_test_splits train_test_splits/iql_splits_one.json \
#       --split_index 0 \
#       --static_partner 1 \
#       --shuffle_color 1 \
#       --wandb 0 \
#       --gcloud_upload 0

