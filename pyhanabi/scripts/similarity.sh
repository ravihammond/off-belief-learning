#!/bin/bash

#python similarity.py \
    #--outdir similarity \
    #--policy1 ../training_models/sad_2p_models/sad_1.pthw \
    #--policy2 ../training_models/sad_2p_models/sad_2.pthw \
    #--sad_legacy1 1 \
    #--sad_legacy2 1 \
    #--name1 sad_1 \
    #--name2 sad_2 \
    #--model1 sad \
    #--model1 sad \
    #--shuffle_index 0 \
    #--num_game 1000 \
    #--num_thread 10 \
    #--seed 0 \
    #--verbose 1 \
    #--save 0 \
    #--upload_gcloud 0 \
    #--device cuda:$1 \

python similarity.py \
    --outdir similarity \
    --policy1 ../training_models/sad_2p_models/sad_1.pthw \
    --policy2 ../training_models/sad_2p_models/sad_2.pthw \
    --model1 sad \
    --model2 sad \
    --sad_legacy1 1 \
    --sad_legacy2 1 \
    --name1 sad_1 \
    --name2 sad_2 \
    --model1 sad \
    --model1 sad \
    --shuffle_index 100 \
    --num_game 1 \
    --num_thread 11 \
    --seed 1 \
    --verbose 1 \
    --upload_gcloud 0 \
    --device cuda:0 \
