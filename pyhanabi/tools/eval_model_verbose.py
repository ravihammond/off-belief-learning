import argparse
import os
import sys
import numpy as np
import re
import json
import pprint
import csv
pprint = pprint.pprint

lib_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(lib_path)
from eval import evaluate_saved_model
from model_zoo import model_zoo
from collect_actor_stats import collect_stats

def evaluate_model(args):
    if args.output is not None:
        sys.stdout = Logger(args.output)

    weight_files = load_weights(args)
    score, perfect, scores, actors = run_evaluation(args, weight_files)

    conventions = load_json_list(args.convention)
    convention_strings = extract_convention_strings(conventions)

    stats = collect_stats(score, perfect, scores, actors, conventions)

    if args.print_stats:
        print()
    print_scores(stats)
    if args.print_stats:
        print_move_stats(stats, 0)
        print_move_stats(stats, 1)

    for convention_string in convention_strings:
        if not any(convention_string in key for key in stats.keys()):
            continue
        print()
        print_scores(stats, f"{convention_string}_")
        print_actor_stats(stats, 0, convention_string)
        print_actor_stats(stats, 1, convention_string)

    exit

    if args.csv_name != "None":
        file_path = os.path.dirname(args.csv_name)
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        wrapped_scores = [[x] for x in scores]
        print(f"Saving: {args.csv_name}")
        # pprint(wrapped_scores)
        with open(args.csv_name, 'w') as file:
            for seed, score in enumerate(scores):
                file.write(f"{seed},{int(score)}\n")
        # file = open(args.csv_name, 'w+', newline ='')
        # with file:
        #     write = csv.writer(file)
        #     write.writerows(wrapped_scores)

    # save_game_actions(actors)
    # save_game_story(actors)

    return score, stats["bomb_out_rate"], stats


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
    convention_indexes = None
    if args.convention_index is not None:
        convention_indexes = [args.convention_index, args.convention_index]

    partner_model_paths = []
    if args.partner_models != "None":
        model_paths = load_json_list(args.partner_models)
        all_indexes =load_json_list(args.train_test_splits)
        test_indexes = all_indexes[args.split_index]["train"]
        partner_model_paths = [model_paths[i] for i in test_indexes]

    score, _, perfect,scores, actors = evaluate_saved_model(
        weight_files,
        args.num_game,
        args.seed,
        args.bomb,
        num_run=args.num_run,
        device=args.device,
        convention=args.convention,
        override=[args.override0, args.override1],
        verbose=False,
        belief_stats=args.belief_stats,
        belief_model=args.belief_model,
        partner_models_path=partner_model_paths,
        convention_indexes=convention_indexes,
        sad_legacy=args.sad_legacy,
        iql_legacy=args.iql_legacy,
        # partner_model_type="test",
        partner_model_type="train",
        shuffle_index=[args.shuffle_index1, args.shuffle_index2],
        shuffle_convention_index=[
            args.shuffle_conv_index1, args.shuffle_conv_index2],
    )

    return score, perfect, scores, actors


def print_actor_stats(stats, player, convention_string):
    print_move_stats(stats, player, convention_string)
    print_convention_stats(stats, player, convention_string, "signal")
    print_convention_stats(stats, player, convention_string, "response")
    print_convention_lose_life_stats(stats, player, convention_string)
    print_convention_should_be_playable_stats(stats, player, convention_string)


def print_scores(stats, convention=""):
    score = stats[f"{convention}score"]
    score_std = stats[f"{convention}score_std"]
    perfect = stats[f"{convention}perfect"] * 100
    non_zero_mean = stats[f"{convention}non_zero_mean"]
    bomb_out_rate = stats[f"{convention}bomb_out_rate"] * 100

    print(f"{convention}score: {score:.2f} ± {score_std:.2f}")
    print(f"{convention}perfect: {perfect:.2f}%")
    print(f"{convention}non_zero_mean: {non_zero_mean:.4f}")
    print(f"{convention}bomb_out_rate: {bomb_out_rate:.2f}%")


def print_move_stats(stats, player, convention_string=None):
    actor_str = f"actor{player}"
    if convention_string != None:
        actor_str = convention_string + "_" + actor_str
    moves = ["play", "discard", "hint", "hint"]
    suffixes = ["", "", "_colour", "_rank"]
    card_index_map = ["0", "1", "2", "3", "4"]
    colour_move_map = ["red", "yellow", "green", "white", "blue"]
    rank_move_map = ["1", "2", "3", "4", "5"]
    move_maps = [card_index_map, card_index_map, colour_move_map, rank_move_map]

    for move_type, suffix, move_map in zip(moves, suffixes, move_maps):
        move_with_suffix = f"{move_type}{suffix}"
        total = stats[f"{actor_str}_{move_with_suffix}"]
        print(f"{actor_str}_{move_with_suffix}: {total}")

        for move in move_map:
            move_type_with_move = f"{actor_str}_{move_type}_{move}"
            move_count = stats[move_type_with_move]
            percentage = stats[f"{move_type_with_move}%"] * 100
            print(f"{move_type_with_move}: {move_count} ({percentage:.1f}%)")


def print_convention_stats(stats, player, convention_string, role):
    prefix = f"{convention_string}_actor{player}_{role}"

    available = f"{prefix}_available"
    played = f"{prefix}_played"
    played_correct = f"{prefix}_played_correct"
    played_incorrect = f"{prefix}_played_incorrect"
    available_percent_str = f"{prefix}_played_correct_available%"
    played_percent_str = f"{prefix}_played_correct_played%"

    available_percent = stats[available_percent_str] * 100
    played_percent = stats[played_percent_str] * 100

    print(f"{available}: {stats[available]}")
    print(f"{played}: {stats[played]}")
    print(f"{played_correct}: {stats[played_correct]}")
    print(f"{played_incorrect}: {stats[played_incorrect]}")
    print(f"{available_percent_str}: {available_percent:.1f}%")
    print(f"{played_percent_str}: {played_percent:.1f}%")


def print_convention_lose_life_stats(stats, player, convention_string):
    stat = f"{convention_string}_actor{player}_response_played_life_lost"
    print(f"{stat}: {stats[stat]}")


def print_convention_should_be_playable_stats(stats, player, convention_string):
    prefix = f"{convention_string}_actor{player}"

    should_be_playable = f"{prefix}_response_should_be_playable"
    playable = f"{prefix}_response_is_playable"
    playable_percent_str = f"{playable}%"

    playable_percent = stats[playable_percent_str] * 100

    print(f"{should_be_playable}: {stats[should_be_playable]}")
    print(f"{playable}: {stats[playable]} ({playable_percent:.1f}%)")


def load_json_list(path):
    if path == "None":
        return []
    file = open(path)
    return json.load(file)


def extract_convention_strings(conventions):
    convention_strings = []

    for convention in conventions:
        convention_str = ""
        for i, two_step in enumerate(convention):
            if i > 0:
                convention_str + '-'
            convention_str += two_step[0] + two_step[1]
        convention_strings.append(convention_str)

    return convention_strings

def save_game_story(actors):
    game_story = []
    games_stories = [actors[0].get_game_story(), actors[1].get_game_story()]

    for turn in range(200):
        if f"t{turn}:a{0}:current" not in games_stories[0].keys():
            break
        turn_data = {}
        for actor_idx in [0, 1]:
            actor_data = {}
            prefix = f"t{turn}:a{actor_idx}"
            actor_data["current"] = bool(games_stories[actor_idx][f"{prefix}:current"])
            actor_data["a"] = int(games_stories[actor_idx][f"{prefix}:a"])
            actor_data["priv_s"] = [
                round(float(x),20) for x in list(games_stories[actor_idx][f"{prefix}:priv_s"])
            ]
            actor_data["h0"] = [round(float(x),20) for x in list(games_stories[actor_idx][f"{prefix}:h0"].ravel())]
            actor_data["c0"] = [round(float(x),20) for x in list(games_stories[actor_idx][f"{prefix}:c0"].ravel())]
            turn_data[f"actor{actor_idx}"] = actor_data
        game_story.append(turn_data)

    file_path = "jax_test_data/cpp_game_story_seed6.json"
    with open(file_path, "w") as fp:
        print(f"saving game story: {file_path}")
        json.dump(game_story, fp)


def save_game_actions(actors):
    for i, actor in enumerate(actors):
        if f"t0:a0:a" in actor.get_game_story().keys():
            print(f"{i}: true")
        else:
            print(f"{i}: false")

    actors_zipped = [(actors[i], actors[i + 1]) for i in np.arange(0, int(len(actors)), 2)]

    games = []
    for actor1, actor2 in actors_zipped:
        actions = []
        actor_stories = [actor1.get_game_story(), actor2.get_game_story()]
        for turn in range(200):
            if f"t{turn}:a{turn % 2}:a" not in actor_stories[turn % 2].keys():
                print(f"t{turn}:a{turn % 2}:a not in story")
                break
            actor_idx = turn % 2
            prefix = f"t{turn}:a{actor_idx}"
            actions.append(int(actor_stories[actor_idx][f"{prefix}:a"]))
        games.append(actions)

    file_path = "jax_test_data/cpp_deck_actions.json"
    with open(file_path, "w") as fp:
        print(f"saving deck actions: {file_path}")
        json.dump(games, fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight1", default=None, type=str, required=True)
    parser.add_argument("--weight2", default=None, type=str)
    parser.add_argument("--weight3", default=None, type=str)
    parser.add_argument("--output", default=None, type=str)
    parser.add_argument("--num_player", default=2, type=int)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--bomb", default=0, type=int)
    parser.add_argument("--num_game", default=5000, type=int)
    parser.add_argument("--num_run", default=1, type=int)
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--convention", default="None", type=str)
    parser.add_argument("--convention_index", default=None, type=int)
    parser.add_argument("--override0", default=0, type=int)
    parser.add_argument("--override1", default=0, type=int)
    parser.add_argument("--belief_stats", default=0, type=int)
    parser.add_argument("--belief_model", default="None", type=str)
    parser.add_argument("--partner_models", default="None", type=str)
    parser.add_argument("--sad_legacy", default="0,0", type=str)
    parser.add_argument("--iql_legacy", default="0,0", type=str)
    parser.add_argument("--train_test_splits", type=str, default="None")
    parser.add_argument("--split_index", default=0, type=int)
    parser.add_argument("--csv_name", default="None", type=str)
    parser.add_argument("--shuffle_index1", default=-1, type=int)
    parser.add_argument("--shuffle_index2", default=-1, type=int)
    parser.add_argument("--print_stats", default=1, type=int)
    parser.add_argument("--shuffle_conv_index1", default=-1, type=int)
    parser.add_argument("--shuffle_conv_index2", default=-1, type=int)
    args = parser.parse_args()

    args.sad_legacy = [int(x) for x in args.sad_legacy.split(",")]
    assert(len(args.sad_legacy) <= 2)
    if (len(args.sad_legacy) == 1):
        args.sad_legacy *= 2

    args.iql_legacy = [int(x) for x in args.iql_legacy.split(",")]
    assert(len(args.iql_legacy) <= 2)
    if (len(args.iql_legacy) == 1):
        args.iql_legacy *= 2

    evaluate_model(args)
