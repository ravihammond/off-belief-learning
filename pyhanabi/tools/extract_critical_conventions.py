import os
import sys
import argparse 
import pprint
pprint = pprint.pprint
import json
from collections import defaultdict

lib_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(lib_path)
from eval import evaluate_saved_model
from calculate_convention_stats import extract_convention_stats

def extract_critical_conventions(args): 
    # extract conventions
    conventions = extract_conventions(args)
    # extract dubious conventions
    # extract critical conventions


def extract_conventions(args):
    conventions = edict()
    return conventions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, required=True)
    parser.add_argument("--num_game", type=int, default=1000)
    # parser.add_argument("--output", default=None, type=str)
    # parser.add_argument("--seed", default=1, type=int)
    # parser.add_argument("--bomb", default=0, type=int)
    # parser.add_argument("--num_game", default=1000, type=int)
    # parser.add_argument("--num_run", default=1, type=int)
    # parser.add_argument("--device", default="cuda:0", type=str)
    # parser.add_argument("--c_thresh", default=0.5, type=float)
    # parser.add_argument("--d_thresh", default=1, type=float)
    # parser.add_argument("--split", default=0, type=int)
    args = parser.parse_args()

    extract_critical_conventions(args)


# def extract_dubious_conventions(args): 

#     weights = load_weight_file(args.weights)
#     conventions = extract_all_conventions(weights, args)
#     # pprint(conventions)
#     dubious_sequences = extract_dubious_sequences(weights, args)
#     pprint(dubious_sequences)
#     # dubious_conventions = filter_dubious_conventions(
#             # conventions, dubious_sequences)

# ACTION_TYPES = "DPCR"
# COLOURS = "RYGWB"
# RANKS = "12345"
# CARDS = "01234"
# ACTION_MAP = {
#     "C": COLOURS,
#     "R": RANKS,
#     "P": CARDS,
#     "D": CARDS
# }


# def load_weight_file(weights_path):
#     if weights_path == None:
#         return []

#     weights_file = open(weights_path)
#     weights = json.load(weights_file)
#     return weights


# def run_evaluation(args, weight_files):
#     _, _, _, _, actors = evaluate_saved_model(
#         weight_files,
#         args.num_game,
#         args.seed,
#         args.bomb,
#         num_run=args.num_run,
#         device=args.device
#     )

#     return actors


# def loop_two_step(callback, **kwargs):
#     for s_type in ACTION_TYPES:
#         for s_action in ACTION_MAP[s_type]:
#             for r_type in ACTION_TYPES:
#                 for r_action in ACTION_MAP[r_type]:
#                     two_step = f"{s_type}{s_action}_{r_type}{r_action}"
#                     callback(two_step, **kwargs)


# def extract_all_conventions(weights, args):
#     conventions = {}

#     for weight in weights:
#         conventions[weight] = extract_conventions(weight, args)

#     return conventions

# def filter_conventions_callback(two_step, conventions, stats):
#     stat = stats[0]["plots"][0][f"0:{two_step}"]
#     if stat > args.c_thresh:
#         conventions[two_step] = stat
#     else:
#         conventions[two_step] = 0


# def extract_conventions(weight, args):
#     weight_files = [weight, weight]
#     actors = run_evaluation(args, weight_files)
#     stats = extract_convention_stats(actors, args)


#     conventions = {}
#     kwargs = {"conventions": conventions, "stats": stats}
#     loop_two_step(filter_conventions_callback, **kwargs)

#     return conventions


# def extract_dubious_sequences(weights, args):
#     counts = defaultdict(lambda: defaultdict(int))

#     for i in range(len(weights)):
#         for j in range(i + 1, len(weights)):
#             weight_files = [weights[i], weights[j]]
#             extract_dubious_stats(weight_files, counts, args)

#     percentages = calculate_dubious_percentages(counts, weights)

#     for weight in weights:
#         counts[weight] = dict(counts[weight])

#     return percentages


# def extract_dubious_stats(weight_files, dubious_sequences, args):
#     def extract_dubious_stats_callback(two_step, actor_stats={}, 
#             dubious_sequences=defaultdict(int), weight_file=""):
#         key = f"dubious_{two_step}"
#         dubious_sequences[weight_file][two_step] += actor_stats[key]

#     actors = run_evaluation(args, weight_files)

#     for i, actor in enumerate(actors):
#         kwargs = {
#             "actor_stats": defaultdict(int, actor.get_stats()),
#             "dubious_sequences": dubious_sequences,
#             "weight_file": weight_files[i % 2],
#         }
#         loop_two_step(extract_dubious_stats_callback, **kwargs)


# def calculate_dubious_percentages(counts, weights):
#     pprint(counts)
#     percentages = defaultdict(dict)

#     def sum_total_callback(two_step, *, weight_file, total):
#         total[0] += counts[weight_file][two_step]

#     def percentage_callback(two_step, *, weight_file, total):
#         count = counts[weight_file][two_step]
#         percentages[weight_file][two_step] = divide(count, total[0])

#     for weight_file in weights:
#         print("starting:", weight_file)
#         total = [0]
#         kwargs = {"weight_file": weight_file, "total": total}
#         loop_two_step(sum_total_callback, **kwargs)
#         print("total:", total[0])
#         loop_two_step(percentage_callback, **kwargs)

#     return dict(percentages)


# def divide(n, total):
#     if total == 0:
#         return 0
#     return (float(n) / total)


# def filter_dubious_conventions(conventions, dubious_sequences):
#     return {}


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--weights", default=None, type=str)
#     parser.add_argument("--output", default=None, type=str)
#     parser.add_argument("--seed", default=1, type=int)
#     parser.add_argument("--bomb", default=0, type=int)
#     parser.add_argument("--num_game", default=1000, type=int)
#     parser.add_argument("--num_run", default=1, type=int)
#     parser.add_argument("--device", default="cuda:0", type=str)
#     parser.add_argument("--c_thresh", default=0.5, type=float)
#     parser.add_argument("--d_thresh", default=1, type=float)
#     parser.add_argument("--split", default=0, type=int)
#     args = parser.parse_args()

#     extract_dubious_conventions(args)
