import sys
import os
import argparse
import pprint
pprint = pprint.pprint
import torch
import numpy as np
import pandas as pd
from datetime import datetime

from create import *
import rela
import r2d2
import utils

np.set_printoptions(threshold=10000, linewidth=10000)


def save_games(args):
    # Load agents
    agents = load_agents(args)

    # Get replay buffer of games
    replay_buffer = generate_replay_data(args, agents)

    # Convert to dataframe
    data = replay_to_dataframe(args, replay_buffer)

    # if args. is not None:
        # save_all_data(args, data)


def load_agents(args):
    weights = [args.weight1, args.weight2]
    agents = []
    for i, weight in enumerate(weights):
        agents.append(load_agent(weight, args.sad_legacy[i], args.device))
    return agents


def load_agent(weight, sad_legacy, device):
    default_cfg = {
        "act_base_eps": 0.1,
        "act_eps_alpha": 7,
        "num_game_per_thread": 80,
        "num_player": 2,
        "train_bomb": 0,
        "max_len": 80,
        "sad": 1,
        "shuffle_color": 0,
        "hide_action": 0,
        "multi_step": 1,
        "gamma": 0.999,
        "parameterized": 0,
    }

    if sad_legacy:
        agent = utils.load_sad_model(weight, device)
        cfg = default_cfg
    else:
        agent, cfg = utils.load_agent(weight, {
            "vdn": False,
            "device": device,
            "uniform_priority": True,
        })

    if agent.boltzmann:
        boltzmann_beta = utils.generate_log_uniform(
            1 / cfg["max_t"], 1 / cfg["min_t"], cfg["num_t"]
        )
        boltzmann_t = [1 / b for b in boltzmann_beta]
    else:
        boltzmann_t = []

    return (agent, cfg, boltzmann_t, sad_legacy)


def generate_replay_data(
    args,
    agents,
):
    seed = args.seed
    num_player = 2
    num_thread = args.num_thread
    if args.num_game < num_thread:
        num_thread = args.num_game

    runners = []
    for agent, _, _, _ in agents:
        runner = rela.BatchRunner(agent.clone(args.device), args.device)
        runner.add_method("act", 5000)
        runner.add_method("compute_priority", 100)
        runners.append(runner)

    context = rela.Context()
    threads = []

    games = create_envs(
        args.num_game,
        seed,
        2,
        0, 
        80
    )

    replay_buffer_size = args.num_game * 2

    replay_buffer = rela.RNNPrioritizedReplay(
        replay_buffer_size,
        seed,
        1.0,  # priority exponent
        0.0,  # priority weight
        3, #prefetch
    )

    assert args.num_game % num_thread == 0
    game_per_thread = args.num_game // num_thread
    all_actors = []

    partner_idx = 0

    for t_idx in range(num_thread):
        thread_games = []
        thread_actors = []
        for g_idx in range(t_idx * game_per_thread, (t_idx + 1) * game_per_thread):
            actors = []
            for i in range(num_player):
                cfgs = agents[i][1]
                actor = hanalearn.R2D2Actor(
                    runners[i], # runner
                    seed, # seed
                    num_player, # numPlayer
                    i, # playerIdx
                    [0], # epsList
                    [], # tempList
                    False, # vdn
                    cfgs["sad"], # sad
                    False, # shuffleColor
                    cfgs["hide_action"], # hideAction
                    False, # trinary
                    replay_buffer, # replayBuffer
                    cfgs["multi_step"], # multiStep
                    cfgs["max_len"], # seqLen
                    cfgs["gamma"], # gamma
                    [], # convention
                    cfgs["parameterized"], # actParameterized
                    0, # conventionIdx
                    0, # conventionOverride
                    False, # fictitiousOverride
                    True, # useExperience
                    False, # beliefStats
                    agents[i][3], # sadLegacy
                    False, # beliefSadLegacy
                    False, # colorShuffleSync
                )

                actors.append(actor)
                all_actors.append(actor)

            for i in range(num_player):
                partners = actors[:]
                partners[i] = None
                actors[i].set_partners(partners)

            thread_actors.append(actors)
            thread_games.append(games[g_idx])
            seed += 1

        thread = hanalearn.HanabiThreadLoop(
                thread_games, thread_actors, True, t_idx)
        threads.append(thread)
        context.push_thread_loop(thread)

    for runner in runners:
        runner.start()

    context.start()
    context.join()

    for runner in runners:
        runner.stop()

    return replay_buffer


def replay_to_dataframe(args, replay_buffer):
    assert(replay_buffer.size() % args.batch_size == 0)
    num_batches = (int)(replay_buffer.size() / args.batch_size)

    for batch_index in range(num_batches):
        range_start = batch_index * args.batch_size
        range_end = batch_index * args.batch_size + args.batch_size
        sample_id_list = [*range(range_start, range_end, 1)]

        batch1, batch2 = replay_buffer.sample_from_list_split(
                args.batch_size, "cpu", sample_id_list)
        
        # batch = replay_buffer.sample_from_list(
                # args.batch_size, "cpu", sample_id_list)

        data = batch_to_dataset(args, batch1, batch2)

    return data


def batch_to_dataset(args, batch1, batch2):
    df = pd.DataFrame()

    now = datetime.now()
    date_time = now.strftime("%m/%d/%Y-%H:%M:%S")

    obs_df = player_dataframe(args, batch1, 0, date_time)
    df = pd.concat([df, obs_df])
    print()

    obs_df = player_dataframe(args, batch2, 1, date_time)
    df = pd.concat([df, obs_df])
    print()

    # data2 = player_dataframe(args, batch2, 1, date_time)
    # data = pd.concat([dataframe, data2])
    # print()

    # print("\naction")
    
    # for k,v in batch.action.items():
        # print(k, np.array(v).shape)
    # print("\nreward")
    # print(batch.reward.shape)
    # print("\nterminal")
    # print(batch.terminal.shape)
    # print("\nbootstrap")
    # print(batch.bootstrap.shape)
    # print()

    print(df.to_string())
    return df


def player_dataframe(args, batch, player, date_time):
    # for k,v in batch.obs.items():
        # print(k, np.array(v).shape)

    df = pd.DataFrame()

    # Add meta data
    meta_df = meta_data(args, batch, player, date_time)
    df = pd.concat([df, meta_df])

    # Add turn numbers
    hand_df = turn_data(args, batch)
    df = pd.concat([df, hand_df], axis=1)

    obs_df = extract_obs(args, batch.obs, player)
    df = pd.concat([df, obs_df], axis=1)
    # df = pd.concat([df, obs_df])

    # own_hand = np.array(batch.obs["own_hand"])
    # own_hand_ar_in = np.array(batch.obs["own_hand_ar_in"])

    # print(own_hand)

    # print((own_hand == own_hand_ar_in).all())
    # print(torch.eq(obs["own_hand_ar_in"]))

    return df


def meta_data(args, batch, player, date_time):
    priv_s = batch.obs["priv_s"]
    num_rows = priv_s.shape[0] * priv_s.shape[1]

    game_names = []

    for i in range(priv_s.shape[1]):
        game_names.append(f"{args.player_name[0]}_vs_{args.player_name[1]}_game_{i}")

    data = np.array(game_names, )
    data = np.repeat(data, priv_s.shape[0])
    data = np.reshape(data, (num_rows, 1))

    meta_data = np.array([
        args.player_name[player],
        args.player_name[(player + 1) % 2],
        # args.data_type,
        # date_time
    ], dtype=str)

    meta_data = np.tile(meta_data, (num_rows, 1))
    data = np.concatenate((data, meta_data), axis=1)

    labels = [
        # "game",
        "player",
        "partner",
        # "data_type",
        # "datetime",
    ]

    return pd.DataFrame(
        data=meta_data,
        # data=data,
        columns=labels
    )


def turn_data(args, batch):
    shape = batch.obs["priv_s"].shape
    data = np.arange(0,80, dtype=np.uint8)
    data = np.tile(data, (shape[1], 1))
    data = np.reshape(data, (shape[0] * shape[1],))
    labels = ["turn"]

    return pd.DataFrame(
        data=data,
        columns=labels
    )


def extract_obs(args, obs, player):
    df = pd.DataFrame()

    if args.sad_legacy[player]:
        own_hand_str = "own_hand_ar"
        # Make sad priv_s the same as OBL priv_s
        priv_s = obs["priv_s"][:, :, 125:783]
    else:
        own_hand_str = "own_hand"
        priv_s = obs["priv_s"]

    partner_hand_idx = 125
    missing_cards_idx = 127
    board_idx = 203
    discard_idx = 253

    # Own hand
    # hand_df = extract_hand(args, obs[own_hand_str], "own")
    # df = pd.concat([df, hand_df], axis=1)

    # Partner Hand
    # partner_hand = np.array(priv_s[:, :, :partner_hand_idx])
    # hand_df = extract_hand(args, partner_hand, "partner")
    # df = pd.concat([df, hand_df], axis=1)

    # Hands missing Card
    # missing_cards = np.array(priv_s[:, :, partner_hand_idx:missing_cards_idx])
    # missing_cards_df = extract_missing_cards(args, missing_cards)
    # df = pd.concat([df, missing_cards_df], axis=1)

    # Board
    # board = np.array(priv_s[:, :, missing_cards_idx:board_idx])
    # board_df = extract_board(args, board)
    # df = pd.concat([df, board_df], axis=1)

    # Discards
    discards = np.array(priv_s[:, :, board_idx:discard_idx])
    discards_df = extract_discards(args, discards)
    df = pd.concat([df, discards_df], axis=1)

    # Knowledge

    return df


def extract_hand(args, hand, label_str):
    hand = np.array(hand, dtype=int)
    shape = hand.shape
    hand = np.reshape(hand, (shape[0], shape[1], 5, 25))
    hand = np.swapaxes(hand, 0, 1) 
    cards = np.argmax(hand, axis=3)
    cards = np.reshape(cards, (cards.shape[0] * cards.shape[1], 5))
    cards = cards.astype(np.uint8)

    labels = []
    for i in range(5):
        labels.append(f"{label_str}_card_{i}")

    return pd.DataFrame(
        data=cards,
        columns=labels
    )


def extract_missing_cards(args, missing_cards):
    missing_cards = np.array(missing_cards, dtype=np.uint8)
    missing_cards = np.swapaxes(missing_cards, 0, 1)
    num_rows = missing_cards.shape[0] * missing_cards.shape[1]
    missing_cards = np.reshape(missing_cards, (num_rows, missing_cards.shape[2]))

    labels = ["own_missing_card", "partner_missing_card"]

    return pd.DataFrame(
        data=missing_cards,
        columns=labels
    )

def extract_board(args, board):
    num_rows = board.shape[0] * board.shape[1]
    board = np.array(board, dtype=np.uint8)
    board = np.swapaxes(board, 0, 1)

    # Encoding positions
    deck_idx = 40
    fireworks_idx = 65
    info_idx = 73
    life_idx = 76

    board_data = np.empty((num_rows, 0), dtype=np.uint8)

    # Deck
    deck = board[:, :, :deck_idx]
    deck_size = deck.sum(axis=2)
    deck_size = np.expand_dims(deck_size, axis=2)
    deck_size = np.reshape(deck_size, (num_rows, deck_size.shape[2]))
    board_data = np.concatenate((board_data, deck_size), axis=1)

    # Fireworks
    fireworks = board[:, :, deck_idx:fireworks_idx]
    fireworks = np.reshape(fireworks, (fireworks.shape[0], fireworks.shape[1], 5, 5))
    non_empty_piles = np.sum(fireworks, axis=3)
    empty_piles = non_empty_piles ^ (non_empty_piles & 1 == non_empty_piles)
    fireworks = np.argmax(fireworks, axis=3) + 1 - empty_piles
    fireworks = np.reshape(fireworks, (num_rows, fireworks.shape[2]))
    fireworks = fireworks.astype(np.uint8)
    board_data = np.concatenate((board_data, fireworks), axis=1)

    # Info Tokens
    info = board[:, :, fireworks_idx:info_idx]
    info_tokens = info.sum(axis=2)
    info_tokens = np.expand_dims(info_tokens, axis=2)
    info_tokens = np.reshape(info_tokens, (num_rows, info_tokens.shape[2]))
    board_data = np.concatenate((board_data, info_tokens), axis=1)

    # Life Tokens
    lives = board[:, :, info_idx:life_idx]
    lives = lives.sum(axis=2)
    lives = np.expand_dims(lives, axis=2)
    lives = np.reshape(lives, (num_rows, lives.shape[2]))
    board_data = np.concatenate((board_data, lives), axis=1)

    # Column labels
    labels = ["deck_size"]
    for colour in ["red", "yellow", "green", "white", "blue"]:
        labels.append(f"{colour}_fireworks")
    labels.extend(["info_tokens", "lives"])

    return pd.DataFrame(
        data=board_data,
        columns=labels
    )


def extract_discards(args, discards):
    num_rows = discards.shape[0] * discards.shape[1]
    discards = np.array(discards, dtype=np.uint8)
    discards = np.swapaxes(discards, 0, 1)
    discards_data = np.empty((num_rows, 0), dtype=np.uint8)

    # print(discards.shape)
    # print(discards)

    rank_1_idx = 3
    rank_2_idx = 5
    rank_3_idx = 7
    rank_4_idx = 9
    rank_5_idx = 10

    num_1 = 3
    num_2 = 2
    num_3 = 2
    num_4 = 2
    num_5 = 1

    bits_per_colour = 10

    labels = []
    for i, colour in enumerate(["red", "yellow", "green", "white", "blue"]):
        offset = i * bits_per_colour

        # Rank 1
        end_pos = offset + rank_1_idx
        rank_1 = discards[:, :, end_pos - num_1:end_pos]
        rank_1 = np.sum(rank_1, axis=2)
        rank_1 = np.expand_dims(rank_1, axis=2)
        rank_1 = np.reshape(rank_1, (num_rows, rank_1.shape[2]))
        discards_data = np.concatenate((discards_data, rank_1), axis=1)

        # Rank 2
        end_pos = offset + rank_2_idx
        rank_2 = discards[:, :, end_pos - num_2:end_pos]
        rank_2 = np.sum(rank_2, axis=2)
        rank_2 = np.expand_dims(rank_2, axis=2)
        rank_2 = np.reshape(rank_2, (num_rows, rank_2.shape[2]))
        discards_data = np.concatenate((discards_data, rank_2), axis=1)

        # Rank 3
        # Rank 4
        # Rank 5

        # for rank in [1, 2, 3, 4, 5]:
        for rank in [1]:
            labels.append(f"{colour}_{rank}_discarded")

    print(labels)

    return pd.DataFrame(
        data=discards_data,
        columns=labels
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight1", type=str, required=True)
    parser.add_argument("--weight2", type=str, required=True) 
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--player_name", type=str, required=True) 
    parser.add_argument("--data_type", type=str, required=True) 
    parser.add_argument("--sad_legacy", type=str, default="0,0")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_game", type=int, default=1000)
    parser.add_argument("--num_thread", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--save", type=int, default=1)
    args = parser.parse_args()

    # Convert sad_legacy to valid list of ints
    args.sad_legacy = [int(x) for x in args.sad_legacy.split(",")]
    assert(len(args.sad_legacy) <= 2)
    if (len(args.sad_legacy) == 1):
        args.sad_legacy *= 2

    # batch size is double the number of games
    if args.batch_size is None:
        args.batch_size = args.num_game * 2

    args.player_name = args.player_name.split(",")

    return args

if __name__ == "__main__":
    args = parse_args()
    save_games(args)
