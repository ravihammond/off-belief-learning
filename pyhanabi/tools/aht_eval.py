import os
import sys
import argparse
from easydict import EasyDict as edict
import pprint
pprint = pprint.pprint
import json
import copy
from datetime import datetime
import numpy as np
import random

lib_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(lib_path)
from tools.eval_model_verbose import evaluate_model

SPLIT_NAME = {
    "sad": {
        "one": "1-12_Splits", 
        "six": "6-7_Splits", 
        "eleven": "11-2_Splits"
    },
    "iql": {
        "one": "1-11_Splits", 
        "six": "6-6_Splits", 
        "ten": "10-2_Splits"
    },
    "op": {
        "one": "1-11_Splits", 
        "six": "6-6_Splits", 
        "ten": "10-2_Splits"
    }
}


def eval_aht(args):
    alter_args(args)
    jobs = create_jobs(args)
    run_jobs(args, jobs)


def alter_args(args):
    if "-" in args.split_indices:
        range_nums = [int(x) for x in args.split_indices.split("-")]
        args.split_indices = np.arange(
                range_nums[0], range_nums[1] + 1).tolist()
    else:
        args.split_indices = [int(x) for x in args.split_indices.split(",")]


def create_jobs(args):
    run_args = []

    splits = load_json_list(f"train_test_splits/{args.train_partner_model}_" + 
            f"splits_{args.split_type}.json")

    run = edict()
    run.sad_legacy = [0, 0]
    run.iql_legacy = [0, 0]
    run.player_name = ["", ""]

    for split_index in args.split_indices:
        run1 = copy.copy(run)
        split_run_args = []

        indices = splits[split_index]
        (run1.model1, 
         run1.sad_legacy[0], 
         run1.iql_legacy[0], 
         run1.player_name[0]) = model_to_weight(args, True, 
                 args.aht_model, split_index=split_index)

        partner_indices = []
        if "aht" in args.partner_model:
            partner_indices = indices["test"]
        else:
            partner_indices = list(range(len(load_json_list(
                f"agent_groups/all_{args.partner_model}.json"))))
        for partner_idx in partner_indices:
            run2 = copy.copy(run1)
            (run2.model2, 
             run2.sad_legacy[1], 
             run2.iql_legacy[1], 
             run2.player_name[1]) = model_to_weight(args, False, 
                     args.partner_model, policy_i=partner_idx)

            run2.pair_name = \
                f"{run2.player_name[0]}  vs  {run2.player_name[1]}"
            run2.csv_name = "None"
            if args.save:
                run2.csv_name = os.path.join(
                    args.out, 
                    args.train_partner_model,
                    SPLIT_NAME[args.train_partner_model][args.split_type], 
                    args.partner_model,
                    args.aht_model,
                    f"{run2.player_name[0]}_vs_{run2.player_name[1]}.csv"
                )

            split_run_args.append(run2)
        run_args.append(split_run_args)

    return run_args


def load_json_list(path):
    if path == "None":
        return []
    file = open(path)
    return json.load(file)
    

def model_to_weight(args, aht_policy, model, split_index=0, policy_i=None):
    splits = load_json_list(f"train_test_splits/{args.train_partner_model}_splits_{args.split_type}.json")
    indices = splits[split_index]["train"]
    indices = [x + 1 for x in indices]
    indices_str = '_'.join(str(x) for x in indices)

    player_name = model

    if aht_policy:
        player_name = f"{model}_{args.split_type}_{indices_str}"
        path = f"exps/{player_name}/model_epoch1000.pthw"
        sad_legacy = 0
        iql_legacy = 0
    else:
        if "aht" in model:
            model = model.split("_")[0]
        policies = load_json_list(f"agent_groups/all_{model}.json")
        path = policies[policy_i]
        player_name = f"{model}_{policy_i + 1}"
        sad_legacy = 0
        iql_legacy = 0
        if model in ["sad", "op", "iql"]:
            sad_legacy = 1
            if model == "iql":
                iql_legacy = 1

    return path, sad_legacy, iql_legacy, player_name


def parse_numbers(numbers):
    if '-' in numbers:
        range = [int(x) for x in numbers.split('-')]
        assert(len(range) == 2)
        numbers_list = list(np.arange(*range))
    else:
        numbers_list = [int(x) for x in numbers.split(',')]

    return numbers_list


def run_jobs(args, jobs):
    scores = []
    bombouts = []
    for i, split in enumerate(jobs):
        split_scores = []
        split_bombouts = []

        for job in split:
            input_args = edict({
                "weight1": job.model1,
                "weight2": job.model2,
                "csv_name": job.csv_name,
                "sad_legacy": job.sad_legacy,
                "iql_legacy": job.iql_legacy,
                "num_game": args.num_game,
                # other needed
                "weight3": None,
                "output": None,
                "num_player": 2,
                "seed": 0,
                "bomb": 0,
                "num_run": 1,
                "device": "cuda:0",
                "convention": "None",
                "convention_index": None,
                "override0": 0,
                "override1": 0,
                "belief_stats": 0,
                "belief_model": "None",
                "partner_models": "None",
                "train_test_splits": None,
                "split_index": 0,
                "shuffle_index1": -1,
                "shuffle_index2": -1
            })
            print(job.pair_name)
            if i == 0 and args.aht_model == "ch_e50_sad":
                score = 12
                bombout = 0.41
            else:
                score, bombout = evaluate_model(input_args)
            print()
            split_scores.append(score)
            split_bombouts.append(bombout)

        split_scores_mean = np.mean(split_scores)
        split_scores_sem = np.std(split_scores) / np.sqrt(len(split_scores))
        split_bombouts_mean = np.mean(split_bombouts)
        split_bombouts_sem = np.std(split_bombouts) / np.sqrt(len(split_bombouts))

        print(f"{job.player_name[0]}")
        print(f"score: {split_scores_mean:.2f} ± {split_scores_sem:.2f}")
        print(f"bombout: {split_bombouts_mean:.2f} ± {split_bombouts_sem:.2f}")
        print()

        scores.append(split_scores_mean)
        bombouts.append(split_bombouts_mean)

    scores_mean = np.mean(scores)
    scores_sem = np.std(scores) / np.sqrt(len(scores))
    bombouts_mean = np.mean(bombouts)
    bombouts_sem = np.std(bombouts) / np.sqrt(len(bombouts))

    print()
    print("scores")
    pprint(scores)
    print("bombout")
    pprint(bombouts)
    print()

    print(f"{args.aht_model}  vs  {args.partner_model}")
    print(f"score: {scores_mean:.2f} ± {scores_sem:.2f}")
    print(f"bombout: {bombouts_mean:.2f} ± {bombouts_sem:.2f}")




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--aht_model", type=str, required=True)
    parser.add_argument("--train_partner_model", type=str, default=None)
    parser.add_argument("--partner_model", type=str, required=True)
    parser.add_argument("--split_type", type=str, required=True)
    parser.add_argument("--split_indices", type=str, required=True)
    parser.add_argument("--num_game", type=int, default=1000)
    parser.add_argument("--out", type=str, default="aht_scores")
    parser.add_argument("--save", type=int, default=0)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    eval_aht(args)
