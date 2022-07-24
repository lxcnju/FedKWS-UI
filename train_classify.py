import os
import random
from collections import namedtuple

import torch

from speech_commands_data import load_speech_commands_data
from speech_commands_data import SpeechCommandsDataset

from networks import load_model
from classify import Classify
from paths import save_dir


def main_classify(para_dict):
    print(para_dict)
    param_names = para_dict.keys()
    Args = namedtuple("Args", param_names)
    args = Args(**para_dict)

    # data
    train_xs, train_ys, test_xs, test_ys, noise_xs = load_speech_commands_data(
        args.task
    )

    print(len(train_xs), len(test_xs))

    train_set = SpeechCommandsDataset(
        train_xs, train_ys, noise_xs, way=args.way, is_train=True
    )
    test_set = SpeechCommandsDataset(
        test_xs, test_ys, noise_xs, way=args.way, is_train=False
    )

    # load model
    model = load_model(args)
    print(model)
    print([name for name, _ in model.named_parameters()])
    n_params = sum([
        param.numel() for param in model.parameters()
    ])
    print("Total number of parameters : {}".format(n_params))

    if args.cuda:
        torch.backends.cudnn.benchmark = True
        model = model.cuda()

    # classify
    algo = Classify(
        train_set=train_set,
        test_set=test_set,
        model=model,
        args=args
    )

    algo.main()

    fpath = os.path.join(
        save_dir, args.fname
    )
    algo.save_logs(fpath)
    print(algo.logs)


def main():
    candi_param_dict = {
        "dataset": ["speechcommands"],
        "task": [12],
        "way": ["mfcc"],
        "n_layer": [15],
        "n_time": [101],
        "n_channel": [32],
        "n_classes": [35],
        "epoches": [100],
        "batch_size": [128],
        "optimizer": ["AdamW"],
        "momentum": [0.9],
        "lr": [0.1],
        "scheduler": ["WSQuadLR"],
        "step_size": [10],
        "gamma": [0.1],
        "ws_step": [5],
        "weight_decay": [1e-5],
        "max_grad_norm": [50.0],
        "cuda": [True],
        "save_ckpts": [False],
    }

    pairs = [
        ("mfcc", 40),
        ("fbank", 40),
    ]

    optim_pairs = {
        "SGD": [0.05, 0.03],
        "AdamW": [0.003, 0.001, 0.0008],
    }

    nets = ["DSCNN", "MHAttRNN", "ResNet", "Transformer"]

    for way, input_channel in pairs:
        for task in ["12", "35"]:
            for net in nets:
                if net == "Transformer":
                    optimizer = "AdamW"
                    epoches = 100
                else:
                    optimizer = "SGD"
                    epoches = 50

                for lr in optim_pairs[optimizer]:
                    para_dict = {}
                    for k, vs in candi_param_dict.items():
                        para_dict[k] = random.choice(vs)

                    para_dict["way"] = way
                    para_dict["input_channel"] = input_channel
                    para_dict["task"] = task
                    para_dict["n_classes"] = int(task)
                    para_dict["optimizer"] = optimizer
                    para_dict["lr"] = lr
                    para_dict["net"] = net
                    para_dict["epoches"] = epoches
                    para_dict["fname"] = "classify.log"

                    main_classify(para_dict)


if __name__ == "__main__":
    main()
