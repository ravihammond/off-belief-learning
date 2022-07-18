python selfplay.py \
       --seed 4 \
       --save_dir exps/sad_4 \
       --num_thread 24 \
       --num_game_per_thread 80 \
       --method iql \
       --sad 0 \
       --lr 6.25e-05 \
       --eps 1.5e-05 \
       --grad_clip 5 \
       --gamma 0.999 \
       --burn_in_frames 10000 \
       --replay_buffer_size 100000 \
       --batchsize 128 \
       --epoch_len 1000 \
       --num_epoch 1501 \
       --num_player 2 \
       --net lstm \
       --num_lstm_layer 1 \
       --multi_step 3 \
       --train_device cuda:0 \
       --act_device cuda:1,cuda:2,cuda:3 \
       --wandb 1
