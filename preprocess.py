import os
import json
import pickle
import numpy as np
import torchaudio

import torchaudio.transforms as T
from torchaudio.datasets import SPEECHCOMMANDS

data_fdir = r"C:\Workspace\work\datasets"
speech_commands_fdir = os.path.join(data_fdir, "SpeechCommands")
noise_fdir = os.path.join(
    speech_commands_fdir, "speech_commands_v0.02", "_background_noise_"
)

fdir = speech_commands_fdir

wanted_words = [
    "yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"
]


class SubsetSC(SPEECHCOMMANDS):
    def __init__(self, subset: str = None):
        super().__init__(data_fdir, download=False)

        def load_list(filename):
            filepath = os.path.join(self._path, filename)
            with open(filepath) as fileobj:
                return [os.path.join(
                    self._path, line.strip()) for line in fileobj]

        if subset == "validation":
            self._walker = load_list("validation_list.txt")
        elif subset == "testing":
            self._walker = load_list("testing_list.txt")
        elif subset == "training":
            excludes = load_list("validation_list.txt")
            excludes += load_list("testing_list.txt")
            excludes = set(excludes)
            self._walker = [w for w in self._walker if w not in excludes]


train_set = SubsetSC("training")
val_set = SubsetSC("validation")
test_set = SubsetSC("testing")

print(len(train_set))
print(len(val_set))
print(len(test_set))


def save_pkl_data(dset, part="train"):
    n = len(dset)

    waveforms = []
    labels = []
    speaker_ids = []

    for i in range(n):
        waveform, sr, label, speaker_id, _ = dset[i]
        assert sr == 16000

        waveform = T.Resample(orig_freq=16000, new_freq=8000)(waveform)
        waveform = waveform.numpy().reshape(-1)

        waveforms.append(waveform)
        labels.append(label)
        speaker_ids.append(speaker_id)

    fpath = os.path.join(
        speech_commands_fdir, "{}-waveforms-35.pkl".format(part)
    )
    with open(fpath, "wb") as fw:
        pickle.dump(waveforms, fw)

    fpath = os.path.join(
        speech_commands_fdir, "{}-labels-35.pkl".format(part)
    )
    with open(fpath, "wb") as fw:
        pickle.dump(labels, fw)

    fpath = os.path.join(
        speech_commands_fdir, "{}-speakers-35.pkl".format(part)
    )
    with open(fpath, "wb") as fw:
        pickle.dump(speaker_ids, fw)

    print(len(np.unique(labels)))
    print(len(np.unique(speaker_ids)))


def save_pkl_data_12(dset, part="train"):
    n = len(dset)

    waveforms = []
    labels = []
    speaker_ids = []

    oth_waveforms = []
    oth_speaker_ids = []

    for i in range(n):
        waveform, sr, label, speaker_id, _ = dset[i]
        assert sr == 16000
        waveform = T.Resample(orig_freq=16000, new_freq=8000)(waveform)
        waveform = waveform.numpy().reshape(-1)

        if label in wanted_words:
            waveforms.append(waveform)
            labels.append(label)
            speaker_ids.append(speaker_id)
        else:
            oth_waveforms.append(waveform)
            oth_speaker_ids.append(speaker_id)

    # unk
    print("Length of waveforms: ", len(waveforms))
    n_unk = int(0.1 * len(waveforms))
    inds = list(range(len(oth_waveforms)))
    np.random.shuffle(inds)
    print(inds[0:10])
    unk_waveforms = [oth_waveforms[i] for i in inds[0:n_unk]]
    unk_speaker_ids = [oth_speaker_ids[i] for i in inds[0:n_unk]]

    # silence
    n_silence = int(0.1 * len(waveforms))
    silence_waveforms = [
        oth_waveforms[i] * 0.0 for i in inds[n_unk:n_unk + n_silence]
    ]
    silence_speaker_ids = [
        oth_speaker_ids[i] for i in inds[n_unk:n_unk + n_silence]
    ]

    waveforms.extend(unk_waveforms)
    speaker_ids.extend(unk_speaker_ids)
    labels.extend(["unk"] * n_unk)

    waveforms.extend(silence_waveforms)
    speaker_ids.extend(silence_speaker_ids)
    labels.extend(["silence"] * n_silence)

    # shuffle
    print(len(waveforms), len(speaker_ids), len(labels))
    print(len(np.unique(labels)))
    print(len(np.unique(speaker_ids)))

    inds = list(range(len(waveforms)))
    np.random.shuffle(inds)
    waveforms = [waveforms[i] for i in inds]
    speaker_ids = [speaker_ids[i] for i in inds]
    labels = [labels[i] for i in inds]

    fpath = os.path.join(
        speech_commands_fdir, "{}-waveforms-12.pkl".format(part)
    )
    with open(fpath, "wb") as fw:
        pickle.dump(waveforms, fw)

    fpath = os.path.join(
        speech_commands_fdir, "{}-labels-12.pkl".format(part)
    )
    with open(fpath, "wb") as fw:
        pickle.dump(labels, fw)

    fpath = os.path.join(
        speech_commands_fdir, "{}-speakers-12.pkl".format(part)
    )
    with open(fpath, "wb") as fw:
        pickle.dump(speaker_ids, fw)


def save_noise_data():
    noise_waveforms = []
    for fname in os.listdir(noise_fdir):
        fpath = os.path.join(noise_fdir, fname)

        if fpath.endswith(".wav"):
            waveform, sample_rate = torchaudio.load(fpath)
            assert sample_rate == 16000

            waveform = T.Resample(orig_freq=16000, new_freq=8000)(waveform)
            waveform = waveform.numpy().reshape(-1)

            noise_waveforms.append(waveform)
            print(waveform.shape)
            print(waveform.max(), waveform.min())

    fpath = os.path.join(
        speech_commands_fdir, "noise-waveforms.pkl"
    )
    with open(fpath, "wb") as fw:
        pickle.dump(noise_waveforms, fw)


# save_noise_data()

save_pkl_data_12(test_set, part="test")
save_pkl_data_12(train_set, part="train")

# save_pkl_data(test_set, part="test")
# save_pkl_data(train_set, part="train")
