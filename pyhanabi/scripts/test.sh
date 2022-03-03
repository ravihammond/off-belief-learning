#!/bin/bash






python selfplay.py \
       --save_dir exps/experimental1 \
       --num_thread 1 \
       --num_game_per_thread 1 \
       --sad 0 \
       --act_base_eps 0.1 \
       --act_eps_alpha 7 \
       --lr 6.25e-05 \
       --eps 1.5e-05 \
       --grad_clip 5 \
       --gamma 0.999 \
       --seed 2254257 \
       --batchsize 1 \
       --burn_in_frames 1 \
       --replay_buffer_size 100000 \
       --epoch_len 1 \
       --num_epoch 1 \
       --num_player 2 \
       --rnn_hid_dim 512 \
       --multi_step 1 \
       --train_device cuda:0 \
       --act_device cuda:1 \
       --num_lstm_layer 2 \
       --boltzmann_act 0 \
       --min_t 0.01 \
       --max_t 0.1 \
       --off_belief 1 \
       --num_fict_sample 10 \
       --belief_device cuda:1 \
       --load_model None \
       --net publ-lstm \
       --belief_model ConventionBelief \
       #--belief_model exps/belief_obl0/model0.pthw \
