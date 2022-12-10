#!/bin/bash

#if [ ! -z "$WANDB_TOKEN" ]
#then
    #wandb login $WANDB_TOKEN
#else 
    #echo "Exiting, WANDB_TOKEN env var not set."
    #exit 128
#fi

python selfplay.py \
       --save_dir exps/br_sad \
       --num_thread 24 \
       --num_game_per_thread 80 \
       --method iql \
       --sad 0 \
       --lr 6.25e-05 \
       --eps 1.5e-05 \
       --gamma 0.999 \
       --seed 2254257 \
       --burn_in_frames 10000 \
       --replay_buffer_size 100000 \
       --batchsize 128 \
       --epoch_len 1000 \
       --num_epoch 3001 \
       --num_player 2 \
       --net lstm \
       --num_lstm_layer 2 \
       --multi_step 3 \
       --train_device cuda:0 \
       --act_device cuda:1,cuda:2,cuda:3 \
       --partner_models agent_groups/all_sad.json \
       --partner_sad_legacy 1 \
       --train_test_splits sad_train_test_splits.json \
       --split_index 0 \
       --static_partner 1 \
       --wandb 1 \
       --gcloud_upload 1
       
#python selfplay.py \
       #--save_dir exps/br_sad \
       #--num_thread 24 \
       #--num_game_per_thread 80 \
       #--method iql \
       #--sad 0 \
       #--lr 6.25e-05 \
       #--eps 1.5e-05 \
       #--gamma 0.999 \
       #--seed 2254257 \
       #--burn_in_frames 100 \
       #--replay_buffer_size 1000 \
       #--batchsize 10 \
       #--epoch_len 2 \
       #--num_epoch 3001 \
       #--num_player 2 \
       #--net lstm \
       #--num_lstm_layer 2 \
       #--multi_step 3 \
       #--train_device cuda:0 \
       #--act_device cuda:1,cuda:2,cuda:3 \
       #--partner_models agent_groups/all_sad.json \
       #--partner_sad_legacy 1 \
       #--train_test_splits sad_train_test_splits.json \
       #--split_index 0 \
       #--static_partner 1 \
       #--wandb 1 \
       #--gcloud_upload 0 
       
