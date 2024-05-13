# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/bin/bash
# python selfplay.py \
#        --save_dir exps/obl1_test \
#        --num_thread 24 \
#        --num_game_per_thread 80 \
#        --sad 0 \
#        --act_base_eps 0.1 \
#        --act_eps_alpha 7 \
#        --lr 6.25e-05 \
#        --eps 1.5e-05 \
#        --grad_clip 5 \
#        --gamma 0.999 \
#        --seed 2254257 \
#        --batchsize 128 \
#        --burn_in_frames 10000 \
#        --replay_buffer_size 100000 \
#        --epoch_len 1000 \
#        --num_epoch 1501 \
#        --num_player 2 \
#        --rnn_hid_dim 512 \
#        --multi_step 1 \
#        --train_device cuda:0 \
#        --act_device cuda:1,cuda:2 \
#        --num_lstm_layer 2 \
#        --boltzmann_act 0 \
#        --min_t 0.01 \
#        --max_t 0.1 \
#        --off_belief 1 \
#        --num_fict_sample 10 \
#        --belief_device cuda:3 \
#        --belief_model exps/belief_obl0/model0.pthw \
#        --load_model None \
#        --net publ-lstm \

python selfplay.py \
       --save_dir exps/test \
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
       --burn_in_frames 100 \
       --replay_buffer_size 1001 \
       --epoch_len 100 \
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
       --belief_device cuda:2 \
       --belief_model exps/belief_obl0/model0.pthw \
       --load_model None \
       --net publ-lstm \
       --belief_stats 1\
       #--belief_model exps/pbelief_obl1f_all_colours/model0.pthw \
