import numpy as np
from collections import defaultdict
import pprint
pprint = pprint.pprint

CARD_INDEX_MAP = ["0", "1", "2", "3", "4"]
COLOUR_MOVE_MAP = ["red", "yellow", "green", "white", "blue"]
RANK_MOVE_MAP = ["1", "2", "3", "4", "5"]

def collect_stats(score, perfect, scores, actors, 
        conventions, stat_type="", stats=None):
    if stats is None:
        stats = defaultdict(int)
    if stat_type != "":
        stat_type += "_"
    record_total_scores(stats, score, perfect, scores, stat_type)
    convention_strings = extract_convention_strings(conventions)

    convention_scores = defaultdict(list)

    for i, actor in enumerate(actors):
        actor_stats = defaultdict(int, actor.get_stats())
        move_stats(stats, actor_stats, i % 2, stat_type)

        if len(convention_strings) == 0:
            continue

        convention_index = actor.get_convention_index()
        convention_str = convention_strings[convention_index]
        record_actor_stats(stats, actor_stats, convention_str, i % 2, stat_type)

        if i % 2 == 0:
            convention_scores[convention_str].append(scores[i // 2])

    calculate_scores(stats, convention_strings, convention_scores)
    evaluate_percentages(stats, convention_strings, stat_type)

    return stats


def record_total_scores(stats, score, perfect, scores, stat_type):
    score_stderr = np.std(scores) / np.sqrt(len(scores))
    non_zero_scores = [s for s in scores if s > 0]
    non_zero_mean = 0 if len(non_zero_scores) == 0 else np.mean(non_zero_scores)
    bomb_out_rate = (1 - len(non_zero_scores) / len(scores))

    stats[f"{stat_type}score"] = score
    stats[f"{stat_type}score_stderr"] = score_stderr
    stats[f"{stat_type}perfect"] = perfect
    stats[f"{stat_type}non_zero_mean"] = non_zero_mean
    stats[f"{stat_type}bomb_out_rate"] = bomb_out_rate


def calculate_scores(stats, conventions, convention_scores):
    for convention in conventions:
        scores = convention_scores[convention]
        if len(scores) == 0:
            continue
        score = np.mean(scores)
        num_perfect = sum([1 for s in scores if s == 25])
        perfect = percent(num_perfect, len(scores))
        non_zero_scores = [s for s in scores if s > 0]
        non_zero_mean = 0 if len(non_zero_scores) == 0 else np.mean(non_zero_scores)
        bomb_out_rate = (1 - percent(len(non_zero_scores), len(scores)))

        stats[f"{convention}_score"] = score
        stats[f"{convention}_perfect"] = perfect
        stats[f"{convention}_non_zero_mean"] = non_zero_mean
        stats[f"{convention}_bomb_out_rate"] = bomb_out_rate


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


def record_actor_stats(stats, actor_stats, convention_str, player, stat_type):
    move_stats(stats, actor_stats, player, stat_type, convention=convention_str)
    convention_stats(stats, actor_stats, player, convention_str, "signal")
    convention_stats(stats, actor_stats, player, convention_str, "response")
    convention_lose_life_stats(stats, actor_stats, player, convention_str)
    convention_playable_stats(stats, actor_stats, player, convention_str)
    # belief_sample_stats(stats, actor_stats)

 
def move_stats(stats, actor_stats, player, stat_type, convention=None):
    move_type_stats(stats, actor_stats, player, 
            CARD_INDEX_MAP, "play", convention, stat_type)
    move_type_stats(stats, actor_stats, player, 
            CARD_INDEX_MAP, "discard", convention, stat_type)
    move_type_stats(stats, actor_stats, player, 
            COLOUR_MOVE_MAP, "hint", convention, stat_type, "_colour")
    move_type_stats(stats, actor_stats, player, 
            RANK_MOVE_MAP, "hint", convention, stat_type, "_rank")


def move_type_stats(stats, actor_stats, player, move_map, move_type, 
        convention, stat_type, move_type_suffix=""):
    prefix = f"{stat_type}actor{player}"
    if convention != None:
        prefix = convention + "_" + prefix
    move_with_suffix = f"{move_type}{move_type_suffix}"

    stats[f"{prefix}_{move_with_suffix}"] += int(actor_stats[move_with_suffix])

    for move in move_map:
        stats[f"{prefix}_{move_type}_{move}"] += \
                int(actor_stats[f"{move_type}_{move}"])


def convention_stats(stats, actor_stats, player, convention, role):
    prefix = f"{convention}_actor{player}_{role}"
    convention_str = f"{role}_{convention}"
    available = f"{convention_str}_available"
    played = f"{convention_str}_played"
    played_correct = f"{convention_str}_played_correct"
    played_incorrect = f"{convention_str}_played_incorrect"

    stats[f"{prefix}_available"] += int(actor_stats[available])
    stats[f"{prefix}_played"] += int(actor_stats[played])
    stats[f"{prefix}_played_correct"] += int(actor_stats[played_correct])
    stats[f"{prefix}_played_incorrect"] += int(actor_stats[played_incorrect])


def convention_lose_life_stats(stats, actor_stats, player, convention_str):
    prefix = f"{convention_str}_actor{player}"
    stat_name = "response_played_life_lost"
    stats[f"{prefix}_{stat_name}"] += int(actor_stats[stat_name])


def belief_sample_stats(stats, actor_stats):
    should_be_playable = "belief_should_be_playable"
    playable_correct = "belief_playable_correct"
    should_not_be_playable = "belief_should_not_be_playable"
    not_playable_correct = "belief_not_playable_correct"

    stats[should_be_playable] += int(actor_stats[should_be_playable])
    stats[playable_correct] += int(actor_stats[playable_correct])
    stats[should_not_be_playable] += int(actor_stats[should_not_be_playable])
    stats[not_playable_correct] += int(actor_stats[not_playable_correct])


def convention_playable_stats(stats, actor_stats, player, convention_str):
    prefix = f"{convention_str}_actor{player}"
    should_be_playable = "response_should_be_playable"
    playable = "response_is_playable"

    stats[f"{prefix}_{should_be_playable}"] += int(actor_stats[should_be_playable])
    stats[f"{prefix}_{playable}"] += int(actor_stats[playable])


def evaluate_percentages(stats, conventions, stat_type):
    for player in range(2):
        move_percentages(stats, player, stat_type)
    for convention in conventions:
        if not any(convention in key for key in stats.keys()):
            continue
        for player in range(2):
            move_percentages(stats, player, stat_type, convention)
            convention_percentages(stats, player, convention, "signal")
            convention_percentages(stats, player, convention, "response")
            convention_playable_percentages(stats, player, convention)
    # belief_percentages(stats)


def move_percentages(stats, player, stat_type, convention=None):
    percent_move_type(stats, player, CARD_INDEX_MAP, "play", convention, stat_type)
    percent_move_type(stats, player, CARD_INDEX_MAP, "discard", convention, stat_type)
    percent_move_type(stats, player, COLOUR_MOVE_MAP, "hint", convention, stat_type, "_colour")
    percent_move_type(stats, player, RANK_MOVE_MAP, "hint", convention, stat_type, "_rank")


def percent_move_type(stats, player, move_map, move_type, 
        convention, stat_type, move_type_suffix=""):
    actor_str = f"{stat_type}actor{player}"
    if convention != None:
        actor_str = convention + "_" + actor_str
    move_with_suffix = f"{move_type}{move_type_suffix}"

    total = stats[f"{actor_str}_{move_with_suffix}"]

    for move in move_map:
        move_type_with_move = f"{actor_str}_{move_type}_{move}"
        move_count = stats[move_type_with_move]
        percentage = percent(move_count, total)
        stats[f"{move_type_with_move}%"] = percentage

def convention_percentages(stats, player, convention, role):
    prefix = f"{convention}_actor{player}_{role}"
    available = stats[f"{prefix}_available"]
    played = stats[f"{prefix}_played"]
    played_correct = stats[f"{prefix}_played_correct"]
    played_incorrect = stats[f"{prefix}_played_incorrect"]

    played_correct_available_percent = percent(played_correct, available)
    played_correct_played_percent = percent(played_correct, played)

    stats[f"{prefix}_played_correct_available%"] = \
            played_correct_available_percent
    stats[f"{prefix}_played_correct_played%"] = \
            played_correct_played_percent


def record_percent(stats, player, move_type, move_total):
    stat = stats[f"actor{player}_{move_type}"]
    stat_total = stats[f"actor{player}_{move_total}"]
    stats[f"actor{player}_{move_type}%"] = percent(stat, stat_total)


def belief_percentages(stats):
    should_be_playable = "belief_should_be_playable"
    playable_correct = "belief_playable_correct"
    should_not_be_playable = "belief_should_not_be_playable"
    not_playable_correct = "belief_not_playable_correct"

    stats[f"{playable_correct}%"] = percent(
            stats[playable_correct], stats[should_be_playable])
    stats[f"{not_playable_correct}%"] = percent(
            stats[not_playable_correct], stats[should_not_be_playable])


def convention_playable_percentages(stats, player, convention_str):
    prefix = f"{convention_str}_actor{player}"
    should_be_playable = f"{prefix}_response_should_be_playable"
    playable = f"{prefix}_response_is_playable"

    stats[f"{playable}%"] = percent(
            stats[playable], stats[should_be_playable])


def percent(n, total):
    if total == 0:
        return 0
    return (n / total)

