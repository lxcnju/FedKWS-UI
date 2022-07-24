import os
import random
from collections import namedtuple
import numpy as np

import torch

from speech_commands_feddata import load_speech_commands_feddata
from speech_commands_data import SpeechCommandsDataset

from fedavg import FedAvg

from networks import load_model

from paths import save_dir

torch.set_default_tensor_type(torch.FloatTensor)


def main_federate(para_dict):
    print(para_dict)
    param_names = para_dict.keys()
    Args = namedtuple("Args", param_names)
    args = Args(**para_dict)

    # DataSets
    info = load_speech_commands_feddata(
        task=args.task
    )
    users_data, glo_test_xs, glo_test_ys, noise_xs = info

    # info
    print("Number of clients: {}".format(len(users_data)))
    n_samples = [len(cdata["train_xs"]) for _, cdata in users_data.items()]
    print("Number of local samples: {}, {}, {}, {}".format(
        np.min(n_samples), np.max(n_samples),
        np.mean(n_samples), np.median(n_samples)
    ))

    csets = {}
    for client, cdata in users_data.items():
        train_set = SpeechCommandsDataset(
            cdata["train_xs"], cdata["train_ys"],
            noise_xs, way=args.way, is_train=True
        )
        test_set = SpeechCommandsDataset(
            cdata["test_xs"], cdata["test_ys"],
            noise_xs, way=args.way, is_train=False
        )
        csets[client] = (train_set, test_set)

    gset = SpeechCommandsDataset(
        glo_test_xs, glo_test_ys, noise_xs, way=args.way, is_train=False
    )

    # Model
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

    # Train FedAvg
    algo = FedAvg(
        csets=csets,
        gset=gset,
        model=model,
        args=args
    )
    algo.train()

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
        "input_channel": [40],
        "n_channel": [32],
        "n_classes": [35],
        "max_round": [500],
        "c_ratio": [0.01],
        "local_steps": [50],
        "test_round": [3],
        "batch_size": [32],
        "optimizer": ["SGD"],
        "momentum": [0.9],
        "lr": [3e-4],
        "weight_decay": [1e-5],
        "max_grad_norm": [50.0],
        "cuda": [True],
        "save_ckpts": [False],
    }

    pairs = [
        ("mfcc", 40),
    ]

    optim_pairs = {
        "SGD": [0.01],
        "AdamW": [0.002, 0.0008, 0.0003],
    }

    nets = ["DSCNN"]

    for way, input_channel in pairs:
        for task in ["12"]:
            for net in nets:
                if net == "Transformer":
                    optimizer = "AdamW"
                    max_round = 500
                    test_round = 5
                else:
                    optimizer = "SGD"
                    max_round = 300
                    test_round = 3

                for lr in optim_pairs[optimizer]:
                    for local_steps in [50]:
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
                        para_dict["batch_size"] = 32
                        para_dict["max_round"] = max_round
                        para_dict["test_round"] = test_round
                        para_dict["local_steps"] = local_steps
                        para_dict["fname"] = "fedavg-{}.log".format(
                            way
                        )

                        main_federate(para_dict)


if __name__ == "__main__":
    main()
