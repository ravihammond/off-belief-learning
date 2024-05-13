# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
import os
import sys
import numpy as np

lib_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(lib_path)
from eval import evaluate_saved_model
from model_zoo import model_zoo


def evaluate_model(args):
    weight_files = load_weights(args)
    score, perfect, scores, actors = run_evaluation(args, weight_files)
    print_scores(scores)

    file_name = weight_files[0].split("/")[3]
    file_path = os.path.join(
        "jax_deck_scores",
        weight_files[0].split("/")[2],
        f"{file_name}.txt"
    )
    dir_path = os.path.dirname(file_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    print(f"Saving: {file_path}")
    with open(file_path, 'w') as file:
        for seed, score in enumerate(scores):
            file.write(f"{seed},{int(score)}\n")

    # print_played_card_knowledge(actors, 0)
    # print_played_card_knowledge(actors, 1)


def load_weights(args):
    weight_files = []
    if args.num_player == 2:
        if args.weight2 is None:
            args.weight2 = args.weight1
        weight_files = [args.weight1, args.weight2]
    elif args.num_player == 3:
        if args.weight2 is None:
            weight_files = [args.weight1 for _ in range(args.num_player)]
        else:
            weight_files = [args.weight1, args.weight2, args.weight3]

    for i, wf in enumerate(weight_files):
        if wf in model_zoo:
            weight_files[i] = model_zoo[wf]

    assert len(weight_files) == 2
    return weight_files


def run_evaluation(args, weight_files):
    print(weight_files[0].split("/")[3])
    score, _, perfect,scores, actors = evaluate_saved_model(
        weight_files,
        args.num_game,
        args.seed,
        args.bomb,
        num_run=args.num_run,
        convention=args.convention,
        override=[args.override1, args.override2],
    )

    return score, perfect, scores, actors


def print_scores(scores):
    non_zero_scores = [s for s in scores if s > 0]
    print(f"non zero mean: %.3f" % (
        0 if len(non_zero_scores) == 0 else np.mean(non_zero_scores)))
    print(f"bomb out rate: {100 * (1 - len(non_zero_scores) / len(scores)):.2f}%")


def print_played_card_knowledge(actors, player):
    card_stats = []
    for i, g in enumerate(actors):
        if i % 2 == player:
            card_stats.append(g.get_played_card_info())
    card_stats = np.array(card_stats).sum(0)
    total_played = sum(card_stats)

    print(f"\nActor {player}: knowledge of cards played:")
    print("total cards played: ", total_played)
    for i, ck in enumerate(["none", "color", "rank", "both"]):
        percentage = (card_stats[i] / total_played) * 100
        print(f"{ck}: {card_stats[i]} ({percentage:.1f}%)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight1", default=None, type=str, required=True)
    parser.add_argument("--weight2", default=None, type=str)
    parser.add_argument("--weight3", default=None, type=str)
    parser.add_argument("--num_player", default=2, type=int)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--bomb", default=0, type=int)
    parser.add_argument("--num_game", default=5000, type=int)
    parser.add_argument(
        "--num_run",
        default=1,
        type=int,
        help="num of {num_game} you want to run, i.e. num_run=2 means 2*num_game",
    )
    parser.add_argument("--convention", default="None", type=str)
    parser.add_argument("--convention_sender", default=0, type=int)
    parser.add_argument("--override1", default=0, type=int)
    parser.add_argument("--override2", default=0, type=int)
    args = parser.parse_args()
    evaluate_model(args)

