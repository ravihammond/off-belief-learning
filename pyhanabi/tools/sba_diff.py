import json
import argparse
import pandas as pd
import re
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

FIG_WIDTH = {"SAD": 10, "OP": 9, "OBL": 2}
FILE = {
    "SAD": "temp/sad_vs_sad_scores.pkl", 
    "OP": "temp/op_vs_op_scores.pkl", 
    "OBL": "temp/obl_vs_obl_scores.pkl", 
}


def extract_sba_diff_data(args):
    if args.model == "all":
        df = get_all_data(args)
        # extract_data(args, df)
        plot_dists(args, df)
        # plot_score_diff(args, df)
        # plot_all_per_pair(args, df)
    else:
        df = pd.read_pickle(FILE[args.model], "gzip")
        df = adjust_data(args, df)
        # plot_dist(args, df)
        plot_per_pair(args, df)


def get_all_data(args):
    df = pd.DataFrame()

    for key, value in FILE.items():
        new_df = pd.read_pickle(value, "gzip")
        new_df = adjust_data(args, new_df)
        new_df["model"] = key
        df = df.append(new_df)
    df = df.reset_index()

    return df


def adjust_data(args, df):
    df["pair"] = df["actor1"].apply(lambda s:s.split('_')[1]) \
        + "-" + df["actor2"].apply(lambda s:s.split('_')[1]) 

    no_shuffle = df.loc[df.groupby("pair").shuffle_index.idxmin()][["pair", "score"]]
    no_shuffle = no_shuffle.rename(columns={"score":"score_no_shuffle"})

    df_no_shuffle = pd.merge(df, no_shuffle, on="pair")

    df_no_shuffle["score_diff"] = df_no_shuffle["score"] - \
            df_no_shuffle["score_no_shuffle"]

    df_no_shuffle["score_diff_abs"] = df_no_shuffle["score_diff"].abs()

    return df_no_shuffle

def extract_data(args, df):
    print(df.info())
    print(df.describe())
    print()

    df_abs_mean = df.groupby(["model", "pair"]).score_diff_abs.mean().reset_index()
    df_mean = df_abs_mean.groupby("model").score_diff_abs.mean().reset_index()
    df_sem = df_abs_mean.groupby("model").score_diff_abs.sem().reset_index()
    print("abs score-diff mean")
    for row in [2, 1, 0]:
        model_name = df_mean.loc[row, "model"]
        mean = df_mean.loc[row, "score_diff_abs"]
        sem = df_sem.loc[row, "score_diff_abs"]
        print(f"{model_name}: {mean:.3f} ± {sem:.3f}")
    print()


    df_max = df.groupby(["model", "pair"]).score_diff.max().reset_index()
    score_diff_max_mean = df_max.groupby("model").score_diff.mean().reset_index()
    score_diff_max_sem = df_max.groupby("model").score_diff.sem().reset_index()
    print("max score-diff mean")
    print(score_diff_max_mean.to_string())
    print(score_diff_max_sem.to_string())
    print()

    for quantile in [0.9, 0.8, 0.7]:
        df_quant = df.groupby(["model", "pair"]).score_diff.quantile(quantile).reset_index()
        score_diff_quant_mean = df_quant.groupby("model").score_diff.mean().reset_index()
        score_diff_quant_sem = df_quant.groupby("model").score_diff.sem().reset_index()
        print(f"{quantile} quantile score-diff mean")
        for row in [2, 1, 0]:
            model_name = score_diff_quant_mean.loc[row, "model"]
            mean = score_diff_quant_mean.loc[row, "score_diff"]
            sem = score_diff_quant_sem.loc[row, "score_diff"]
            print(f"{model_name}: {mean:.3f} ± {sem:.3f}")
        print()

    df_std = df.groupby("model").score_diff.std().reset_index()
    print("score-diff std")
    print(df_std.to_string())
    print()


def plot_dists(args, df):
    # print(df.info())
    # print(df.to_string())


    ax = sns.displot(data=df, x="score_diff", hue="model", kind='kde', fill=True, 
            palette=sns.color_palette('bright')[:3], height=3, aspect=3,
            common_norm=False)

    sns.move_legend(ax, "upper right", bbox_to_anchor=(.88, .9))
    # sns.move_legend(g, "upper left", 
    # sns.move_legend(ax, "best")
    # plt.legend(loc='upper center')

    plt.xlim(-4, 4)
    plt.xlabel('SBA Difference (augdiff)')
    plt.show()


def plot_score_diff(args, df):
    # df_abs_mean = df.groupby(["model", "pair"]).score_diff_abs.mean().reset_index()
    # df_mean = df_abs_mean.groupby("model").score_diff_abs.mean().reset_index()
    # df_sem = df_abs_mean.groupby("model").score_diff_abs.sem().reset_index()

    # print(df.to_string())
    sns.scatterplot(data=df, x="score_diff", y="model", alpha=0.03, 
            s=10)


    # sns.displot(data=df, x="score_diff", hue="model", kind='kde', fill=True, 
            # palette=sns.color_palette('bright')[:3], height=5, aspect=1.5,
            # common_norm=False)

    # plt.title(f"SBA Difference Distributions")
    plt.show()

    # fig, ax = plt.subplots(figsize=(8, 8))

    # plt.vlines(score_labels, min_val, max_val, color="grey", 
            # linewidth=0.5, zorder=0, alpha=0.5)
    # ax.scatter(df[""], scores, zorder=1, alpha=0.2)
    # ax.scatter(orig_labels, orig_scores, color="black",  marker='x', zorder=2)

    # plt.xticks(rotation=60)
    # plt.ylabel(data_type_label)
    # plt.xlabel(f"{args.model} Pairs")
    # plt.title(f"SBA {args.model} Pairs vs {data_type_label}")
    # plt.show()


def plot_dist(args, df):
    sns.displot(df.score_diff, kind="kde", fill=True, hue=df.model)
    plt.ylim(0, 3)
    plt.xlim(-9, 9)
    plt.title(f"SBA {args.model} Distribution")
    plt.show()


def plot_per_pair(args, df):
    df_original = df.loc[df.groupby("pair").shuffle_index.idxmin()]
    plot_scores(
        args, 
        df["pair"], 
        df["score_diff"], 
        df_original["pair"],
        df_original["score_diff"],
        -9,
        9,
        "Score Difference"
    )

    # df_original = df.loc[df.groupby("pair").shuffle_index.idxmin()]
    # plot_scores(
        # args, 
        # df["pair"], 
        # df["score"], 
        # df_original["pair"],
        # df_original["score"],
        # 0,
        # 25,
        # "Score"
    # )

# def create_plot(ax, stats, data_type, title, ylim, legendshow, ylabelshow):
    # model_types = list(stats.keys())
    # num_runs = stats[model_types[0]].loss.shape[0]
    # num_epochs = stats[model_types[0]].loss.shape[1]
    # mode_labels = ['BR', 'SBA']
    # colors = ['#d62728', '#1f77b4']

    # x_vals = np.arange(num_epochs)
    # for sba in model_types:
        # if sba not in stats:
            # continue
        # y_vals = stats[sba][data_type]
        # y_mean = y_vals.mean(0)
        # y_std = y_vals.std(0) / np.sqrt(num_runs)
        # ax.plot(x_vals, y_mean, colors[sba], label = mode_labels[sba])
        # ax.fill_between(x_vals, y_mean + y_std, y_mean - y_std,
                        # color= colors[sba], alpha = 0.3)
    # ax.set_ylim([0, ylim])
    # ax.set_xlim([0, num_epochs])

    # if legendshow:
        # ax.legend(loc="lower right")
    # ax.set_title(title)
    # ax.set_xlabel("Epoch")
    # if ylabelshow:
        # ax.set_ylabel("Return")

def plot_all_per_pair(args, df):
    fig = plt.figure(constrained_layout=True, figsize=(10, 6))
    subfigs = fig.subfigures(2, 1)

    axes0 = subfigs[0].subplots(1, 1)
    create_scores_plot(args, df, "SAD", axes0, "a)")

    axes1 = subfigs[1].subplots(1, 2, sharey=True,
            gridspec_kw={'width_ratios': [6, 1]})
    create_scores_plot(args, df, "OP", axes1[0], "b)")
    create_scores_plot(args, df, "OBL", axes1[1], "c)", show_y_label=False)

    plt.show()


def create_scores_plot(ars, df, model, ax, title, show_y_label=True):
    df = pd.read_pickle(FILE[model], "gzip")
    df = adjust_data(args, df)
    df_original = df.loc[df.groupby("pair").shuffle_index.idxmin()]
    score_labels = df["pair"]
    scores = df["score_diff"]
    orig_labels = df_original["pair"]
    orig_scores = df_original["score_diff"]

    ax.scatter(score_labels, scores, zorder=1, alpha=0.2)
    ax.scatter(orig_labels, orig_scores, color="black",  marker='x', zorder=2)

    if show_y_label:
        ax.set_ylabel("SBA Difference (augdiff)")
    ax.set_xlabel(f"{model} Pairs")
    ax.set_xticklabels([])
    ax.set_ylim(-7, 7)
    if model == "SAD":
        ax.set_xlim(-1, 78)
    elif model == "OP":
        ax.set_xlim(-1, 66)
    elif model == "OBL":
        ax.set_xlim(-1, 10)
    ax.set_title(title)


# def plot_scores(args, score_labels, scores, orig_labels, orig_scores, 
        # min_val, max_val, data_type_label):
    # fig, ax = plt.subplots(figsize=(FIG_WIDTH[args.model], 3))

    # # plt.vlines(score_labels, min_val, max_val, color="grey", 
            # # linewidth=0.5, zorder=0, alpha=0.5)
    # ax.scatter(score_labels, scores, zorder=1, alpha=0.2)
    # ax.scatter(orig_labels, orig_scores, color="black",  marker='x', zorder=2)

    # plt.xticks(rotation=60)
    # plt.ylabel(data_type_label)
    # plt.xlabel(f"{args.model} Pairs")
    # plt.gca().set_xticklabels([])
    # # plt.title(f"SBA {args.model} Pairs vs {data_type_label}")
    # plt.ylim(-7, 7)
    # if args.model == "SAD":
        # plt.xlim(-1, 78)
    # elif args.model == "OP":
        # plt.xlim(-1, 66)
    # elif args.model == "OBL":
        # plt.xlim(-1, 10)
    # fig.tight_layout()
    # plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="None")
    args = parser.parse_args()
    extract_sba_diff_data(args)

