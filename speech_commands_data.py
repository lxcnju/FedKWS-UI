import os
import copy
import random
import numpy as np
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torchaudio import transforms

from paths import speech_commands_fdir

from utils import load_pickle


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


def load_speech_commands_data(task, combine=False, max_len=8000):
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

    train_ys = np.array([label2int[label] for label in train_ys])
    test_ys = np.array([label2int[label] for label in test_ys])

    print(Counter(train_ys))
    print(Counter(test_ys))

    noise_fpath = os.path.join(
        speech_commands_fdir, "noise-waveforms.pkl"
    )
    noise_xs = load_pickle(noise_fpath)

    return train_xs, train_ys, test_xs, test_ys, noise_xs


class SpeechCommandsDataset(data.Dataset):
    def __init__(
            self, xs, ys, noise_xs, way="raw1d",
            noise_volume=0.1, noise_prob=0.8,
            n_len=8000, sample_rate=8000, is_train=True, args=None):
        # self.xs = copy.deepcopy(xs)
        # self.ys = copy.deepcopy(ys)
        # self.noise_xs = copy.deepcopy(noise_xs)
        self.xs = xs
        self.ys = ys
        self.noise_xs = noise_xs
        self.noise_volume = noise_volume
        self.noise_prob = noise_prob
        self.way = way
        self.is_train = is_train
        self.n_len = n_len
        self.sample_rate = sample_rate

        # window_length = sample_rate * 25ms / 1000ms
        # hop_length = sample_rate * 10ms / 1000ms
        self.win_len = int(30 * sample_rate / 1000)
        self.hop_len = int(10 * sample_rate / 1000)

        self.pad = int(100 * sample_rate / 1000)
        self.win_pad = int(self.win_len / 2)

        if self.way == "raw1d":
            pass
        elif self.way == "raw2d":
            self.inds = []

            n_wins = int((n_len + 2 * self.win_pad) / self.hop_len)
            for n in range(n_wins):
                i = self.hop_len * n
                if i + self.win_len > n_len + 2 * self.win_pad:
                    continue
                self.inds.append(list(range(i, i + self.win_len)))
            self.n_wins = len(self.inds)
        elif self.way == "spec2d":
            self.transform = transforms.Spectrogram(
                n_fft=self.win_len,
                win_length=self.win_len,
                hop_length=self.hop_len
            )
        elif self.way == "fbank":
            self.transform = transforms.MelSpectrogram(
                n_mels=40,
                sample_rate=8000,
                n_fft=self.win_len,
                win_length=self.win_len,
                hop_length=self.hop_len
            )
        elif self.way == "mfcc":
            melkwargs = {
                "n_mels": 40,
                "n_fft": self.win_len,
                "win_length": self.win_len,
                "hop_length": self.hop_len,
            }
            self.transform = transforms.MFCC(
                sample_rate=8000,
                n_mfcc=40,
                melkwargs=melkwargs,
            )
        else:
            raise ValueError("No such way.")

        if is_train and self.way in ["spec2d", "fbank", "mfcc"]:
            # SpecAugument: mask 10%, (F, T) --> (40, 100) --> (5, 10)
            self.transform = nn.Sequential(
                self.transform,
                transforms.FrequencyMasking(freq_mask_param=5),
                transforms.TimeMasking(time_mask_param=10),
            )

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, index):
        waveform0 = copy.deepcopy(self.xs[index])
        label0 = copy.deepcopy(self.ys[index])

        # return (time, ) tensor
        waveform = torch.FloatTensor(waveform0)
        label = torch.LongTensor([label0])[0]

        # pad --> shift --> add noise
        waveform = self.pad_and_shift(waveform)
        waveform = self.add_noise(waveform)

        if self.way == "raw1d":
            pass
        elif self.way == "raw2d":
            waveform = torch.cat([
                waveform[0:self.win_pad].flip(dims=[0]),
                waveform,
                waveform[-self.win_pad:].flip(dims=[0])
            ], dim=0)
            waveform = torch.stack([
                waveform[inds] for inds in self.inds
            ], dim=0)
        elif self.way == "spec2d":
            waveform = self.transform(waveform)  # (n_fft, time)
            waveform = waveform.transpose(0, 1)  # (time, n_fft)
            waveform = torch.log(waveform + 1.0)
            waveform = (waveform - 0.1) / 5.0
        elif self.way == "fbank":
            waveform = self.transform(waveform)  # (n_mels, time)
            waveform = waveform.transpose(0, 1)  # (time, n_mels)
            waveform = torch.log(waveform + 1.0)
            waveform = (waveform - 0.1) / 5.0
        elif self.way == "mfcc":
            waveform = self.transform(waveform)  # (n_mfcc, time)
            waveform = waveform.transpose(0, 1)  # (time, n_mfcc)
            waveform = waveform / 500.0
        else:
            raise ValueError("No such way.")

        return waveform, label

    def pad_and_shift(self, waveform):
        waveform = torch.cat([
            torch.zeros(self.pad),
            waveform,
            torch.zeros(self.pad)
        ], dim=0)
        ind = np.random.randint(0, 2 * self.pad)
        waveform = waveform[ind:ind + self.n_len]
        return waveform

    def add_noise(self, waveform):
        if random.random() > self.noise_prob:
            return waveform
        else:
            ind = random.choice(range(len(self.noise_xs)))
            noise_waveform = self.noise_xs[ind]
            ind = np.random.randint(0, len(noise_waveform) - self.n_len - 1)
            noise = noise_waveform[ind:ind + self.n_len]
            noise = torch.FloatTensor(noise)
            p = random.random()
            waveform += p * self.noise_volume * noise
        return waveform
