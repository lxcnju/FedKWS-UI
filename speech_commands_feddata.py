import os
import copy
import random
import numpy as np
from collections import Counter

from matplotlib import pyplot as plt

from paths import speech_commands_fdir

from utils import load_pickle

np.random.seed(0)


def pad_clip(xs, max_len):
    if len(xs) > max_len:
        xs = xs[0:max_len]
    elif len(xs) < max_len:
        xs = np.concatenate([
            xs, [0.0] * (max_len - len(xs))
        ], axis=0)
    else:
        pass
    return xs


def load_speech_commands_feddata(task, combine=False, max_len=8000):
    train_fpath = os.path.join(
        speech_commands_fdir, "train-waveforms-{}.pkl".format(task)
    )

    test_fpath = os.path.join(
        speech_commands_fdir, "test-waveforms-{}.pkl".format(task)
    )

    train_xs = load_pickle(train_fpath)
    test_xs = load_pickle(test_fpath)

    lens = [len(xs) for xs in train_xs + test_xs]
    print("Info of length: ", np.min(lens), np.max(lens), np.mean(lens))

    train_xs = np.array([pad_clip(xs, max_len) for xs in train_xs])
    test_xs = np.array([pad_clip(xs, max_len) for xs in test_xs])

    train_fpath = os.path.join(
        speech_commands_fdir, "train-speakers-{}.pkl".format(task)
    )

    train_users = load_pickle(train_fpath)

    train_fpath = os.path.join(
        speech_commands_fdir, "train-labels-{}.pkl".format(task)
    )

    test_fpath = os.path.join(
        speech_commands_fdir, "test-labels-{}.pkl".format(task)
    )
    train_ys = load_pickle(train_fpath)
    test_ys = load_pickle(test_fpath)

    label2int = {
        label: i for i, label in enumerate(list(sorted(np.unique(train_ys))))
    }
    print(label2int)

    train_ys = np.array([label2int[label] for label in train_ys])
    test_ys = np.array([label2int[label] for label in test_ys])

    print(Counter(train_ys))
    print(Counter(test_ys))

    noise_fpath = os.path.join(
        speech_commands_fdir, "noise-waveforms.pkl"
    )
    noise_xs = load_pickle(noise_fpath)

    users_data = {}
    users = np.unique(train_users)
    user2id = {user: i for i, user in enumerate(list(sorted(users)))}
    user_ids = np.array([user2id[user] for user in train_users])

    for user, i in user2id.items():
        inds = np.argwhere(user_ids == i).reshape(-1)
        if len(inds) < 5:
            continue

        users_data[user] = {
            "train_xs": train_xs[inds],
            "train_ys": train_ys[inds],
            "test_xs": train_xs[inds][-5:],
            "test_ys": train_ys[inds][-5:],
        }

    return users_data, test_xs, test_ys, noise_xs


if __name__ == "__main__":
    users_data, noise_xs = load_speech_commands_feddata(task="12")

    users = list(users_data.keys())
    print("Number of users: ", len(users))

    classes = range(12)

    cnts = []
    for user in users:
        ys = users_data[user]["train_ys"]
        cnts.append([np.sum(ys == c) for c in classes])

    cnts = np.array(cnts)
    print(cnts.shape)

    cnts = cnts[0:100]

    n_clients, n_classes = cnts.shape

    cnts = cnts.transpose()

    # definitions for the axes
    left, width = 0.15, 0.7
    bottom, height = 0.1, 0.7
    spacing = 0.005

    rect_main = [left, bottom, width, height]
    rect_histx = [left, bottom + 0.9 * height + spacing, width, 0.15]
    rect_histy = [
        left + width + spacing, bottom + 0.14 * height, 0.1, 0.72 * height
    ]
    rect_cb = [left - 0.1, bottom + 0.2 * height, 0.02, 0.6 * height]

    # start with a square Figure
    fig = plt.figure(figsize=(16, 8))

    ax = fig.add_axes(rect_main)
    ax_histx = fig.add_axes(rect_histx, sharex=ax)
    ax_histy = fig.add_axes(rect_histy, sharey=ax)

    bwidth = 2
    ax.spines["top"].set_linewidth(bwidth)
    ax.spines["right"].set_linewidth(bwidth)
    ax.spines["left"].set_linewidth(bwidth)
    ax.spines["bottom"].set_linewidth(bwidth)

    ax.set_xlabel("Client", fontsize=24)
    ax.set_ylabel("Class", fontsize=24)

    xs, ys = np.meshgrid(
        np.arange(n_clients), np.arange(n_classes)
    )
    xs = xs.reshape(-1)
    ys = ys.reshape(-1)

    fs = 30.0 * cnts.reshape(-1) / cnts.max()

    ax.grid(alpha=0.9)
    ax.scatter(xs, ys, s=fs, cmap="Spectral")

    bwidth = 1
    ax_histx.spines["top"].set_linewidth(bwidth)
    ax_histx.spines["right"].set_linewidth(bwidth)
    ax_histx.spines["left"].set_linewidth(bwidth)
    ax_histx.spines["bottom"].set_linewidth(bwidth)

    xs = cnts.sum(axis=0)
    ax_histx.bar(
        range(len(xs)), xs,
        width=0.5,
        color="#FFFFCC",
        edgecolor="black",
        linewidth=1,
        hatch="/",
    )
    ax_histx.set_xticks([])
    ax_histx.tick_params(labelsize=14)

    bwidth = 1
    ax_histy.spines["top"].set_linewidth(bwidth)
    ax_histy.spines["right"].set_linewidth(bwidth)
    ax_histy.spines["left"].set_linewidth(bwidth)
    ax_histy.spines["bottom"].set_linewidth(bwidth)

    ys = cnts.sum(axis=1)
    ax_histy.barh(
        range(len(ys)), ys,
        height=0.5,
        color="#FF9900",
        edgecolor="black",
        linewidth=1,
        hatch="/",
    )
    ax_histy.set_yticks([])
    ax_histy.tick_params(labelsize=14)

    """
    fig.savefig(
        "final-figs/sa-info.pdf", dpi=300,
        bbox_inches='tight', format='pdf'
    )
    fig.savefig(
        "final-figs/sa-info.jpg", dpi=300,
        bbox_inches='tight',
    )
    """

    plt.show()
    plt.close()
