#!/bin/bash

# Similarity vs SAD, 6-7 splits, Test Partners

#for split_indexes in $(seq 0 3)
#do
    #for single_policy_index in $(seq 0 6)
    #do
        #python tools/collect_policy_differences.py \
            #--output_dir acting_comp_test_fixed \
            #--rollout_policies agent_groups/all_sad.json \
            #--train_test_splits train_test_splits/sad_splits_six.json \
            #--split_indexes $split_indexes \
            #--single_policy $single_policy_index \
            #--compare_models sad \
            #--base_models br,sba \
            #--num_game 5000 \
            #--num_thread 20 \
            #--seed 0 \
            #--name_ext six \
            #--split_type test
    #done
#done

# Similarity vs SAD, 6-7 splits, Train Partners

#for split_indexes in $(seq 0 3)
#do
    #for single_policy_index in $(seq 0 5)
    #do
        #python tools/collect_policy_differences.py \
            #--output_dir acting_comp_train_fixed \
            #--rollout_policies agent_groups/all_sad.json \
            #--train_test_splits train_test_splits/sad_splits_six.json \
            #--split_indexes $split_indexes \
            #--single_policy $single_policy_index \
            #--compare_models sad \
            #--base_models br,sba \
            #--num_game 5000 \
            #--num_thread 20 \
            #--seed 0 \
            #--name_ext six \
            #--split_type train
    #done
#done

# Similarity vs SAD, 1-12 splits, Train Partners

#for split_indexes in $(seq 0 3)
#do
    #for single_policy_index in $(seq 0 11)
    #do
        #python tools/collect_policy_differences.py \
            #--output_dir acting_comp_one_test_fixed \
            #--rollout_policies agent_groups/all_sad.json \
            #--train_test_splits train_test_splits/sad_splits_one.json \
            #--split_indexes $split_indexes \
            #--single_policy $single_policy_index \
            #--compare_models sad \
            #--base_models br,sba \
            #--num_game 5000 \
            #--num_thread 20 \
            #--seed 0 \
            #--name_ext one \
            #--split_type test 
    #done
#done

# Similarity vs SAD, 1-12 splits, Test Partners

#for split_indexes in $(seq 0 3)
#do
    #for single_policy_index in $(seq 0 0)
    #do
        #python tools/collect_policy_differences.py \
            #--output_dir acting_comp_one_train_fixed \
            #--rollout_policies agent_groups/all_sad.json \
            #--train_test_splits train_test_splits/sad_splits_one.json \
            #--split_indexes $split_indexes \
            #--single_policy $single_policy_index \
            #--compare_models sad \
            #--base_models br,sba \
            #--num_game 5000 \
            #--num_thread 20 \
            #--seed 0 \
            #--name_ext one \
            #--split_type train
    #done
#done

# Similarity vs OBL, 6-7 splits, Test Partners

for split_indexes in $(seq 0 9)
do
    python tools/collect_policy_differences.py \
        --output_dir acting_comp_obl_test_fixed \
        --rollout_policies agent_groups/all_sad.json \
        --train_test_splits train_test_splits/sad_splits_six.json \
        --split_indexes $split_indexes \
        --compare_models obl1_1,obl1_2,obl1_3,obl1_4,obl1_5 \
        --base_models br,sba \
        --num_game 5000 \
        --num_thread 20 \
        --seed 0 \
        --split_type test \
        --name_ext six \
        --similarity_across_all 1 
done

# Similarity vs OBL, 6-7 splits, Train Partners

for split_indexes in $(seq 0 9)
do
    for single_policy_index in $(seq 0 5)
    do
        python tools/collect_policy_differences.py \
            --output_dir acting_comp_obl_train_fixed \
            --rollout_policies agent_groups/all_sad.json \
            --train_test_splits train_test_splits/sad_splits_six.json \
            --split_indexes $split_indexes \
            --single_policy $single_policy_index \
            --compare_models obl1_1,obl1_2,obl1_3,obl1_4,obl1_5 \
            --base_models br,sba \
            --num_game 5000 \
            --num_thread 20 \
            --seed 0 \
            --split_type train \
            --name_ext six \
            --similarity_across_all 1 
    done
done

## Similarity vs OBL, 1-12 splits, Train Partners

#for split_indexes in $(seq 0 12)
#do
    #for single_policy_index in $(seq 0 11)
    #do
        #python tools/collect_policy_differences.py \
            #--output_dir acting_comp_obl_one_test_fixed \
            #--rollout_policies agent_groups/all_sad.json \
            #--train_test_splits train_test_splits/sad_splits_one.json \
            #--split_indexes $split_indexes \
            #--single_policy $single_policy_index \
            #--compare_models obl1_1,obl1_2,obl1_3,obl1_4,obl1_5 \
            #--base_models br,sba \
            #--num_game 5000 \
            #--num_thread 20 \
            #--seed 0 \
            #--name_ext one \
            #--split_type test \
            #--similarity_across_all 1 
    #done
#done

## Similarity vs OBL, 1-12 splits, Test Partners

#for split_indexes in $(seq 0 12)
#do
    #for single_policy_index in $(seq 0 0)
    #do
        #python tools/collect_policy_differences.py \
            #--output_dir acting_comp_obl_one_train_fixed \
            #--rollout_policies agent_groups/all_sad.json \
            #--train_test_splits train_test_splits/sad_splits_one.json \
            #--split_indexes $split_indexes \
            #--single_policy $single_policy_index \
            #--compare_models obl1_1,obl1_2,obl1_3,obl1_4,obl1_5 \
            #--base_models br,sba \
            #--num_game 5000 \
            #--num_thread 20 \
            #--seed 0 \
            #--name_ext one \
            #--split_type train \
            #--similarity_across_all 1 
    #done
#done

## SAD XP Similarity

#for single_policy_index in $(seq 0 12)
#do
    #python tools/collect_policy_differences.py \
        #--output_dir sad_similarities_fixed \
        #--rollout_policies agent_groups/all_sad.json \
        #--single_policy $single_policy_index \
        #--compare_models sad \
        #--num_game 5000 \
        #--sad_crossplay 1 \
        #--index_start 0 \
        #--index_end 13
#done

