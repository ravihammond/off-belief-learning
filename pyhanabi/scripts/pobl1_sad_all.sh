#!/bin/bash
python selfplay.py \
       --save_dir exps/pobl1_sad_all \
       --num_thread 24 \
       --num_game_per_thread 78 \
       --sad 0 \
       --act_base_eps 0.1 \
       --act_eps_alpha 7 \
       --lr 6.25e-05 \
       --eps 1.5e-05 \
       --grad_clip 5 \
       --gamma 0.999 \
       --seed 2254257 \
       --batchsize 128 \
       --burn_in_frames 10000 \
       --replay_buffer_size 100000 \
       --epoch_len 1000 \
       --num_epoch 6001 \
       --num_player 2 \
       --rnn_hid_dim 512 \
       --multi_step 1 \
       --train_device cuda:0 \
       --act_device cuda:1,cuda:2 \
       --num_lstm_layer 2 \
       --boltzmann_act 0 \
       --min_t 0.01 \
       --max_t 0.1 \
       --off_belief 1 \
       --num_fict_sample 10 \
       --belief_device cuda:3 \
       --belief_model exps/pbelief_sad_all/model0.pthw \
       --load_model None \
       --net publ-lstm \
       --num_parameters 13 \
       --parameterized 1 \
       --wandb 1 \

#python selfplay.py \
       #--save_dir exps/test \
       #--num_thread 1 \
       #--num_game_per_thread 24 \
       #--sad 0 \
       #--act_base_eps 0.1 \
       #--act_eps_alpha 7 \
       #--lr 6.25e-05 \
       #--eps 1.5e-05 \
       #--grad_clip 5 \
       #--gamma 0.999 \
       #--seed 2254257 \
       #--batchsize 10 \
       #--burn_in_frames 40 \
       #--replay_buffer_size 40 \
       #--epoch_len 2 \
       #--num_epoch 6001 \
       #--num_player 2 \
       #--rnn_hid_dim 512 \
       #--multi_step 1 \
       #--train_device cuda:0 \
       #--act_device cuda:1,cuda:2 \
       #--num_lstm_layer 2 \
       #--boltzmann_act 0 \
       #--min_t 0.01 \
       #--max_t 0.1 \
       #--off_belief 1 \
       #--num_fict_sample 10 \
       #--belief_device cuda:3 \
       #--belief_model exps/pbelief_sad_all/model0.pthw \
       #--load_model None \
       #--net publ-lstm \
       #--num_parameters 13 \
       #--parameterized 1 \
       #--wandb 0 \
