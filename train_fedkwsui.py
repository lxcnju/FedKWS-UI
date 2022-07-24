import os
import random
from collections import namedtuple
import numpy as np

import torch

from speech_commands_feddata import load_speech_commands_feddata
from speech_commands_data import SpeechCommandsDataset

from fedkwsui import FedKWSUI

from cif_networks import CifNet

from paths import save_dir

torch.set_default_tensor_type(torch.FloatTensor)


def main_federate(para_dict):
    print(para_dict)
    param_names = para_dict.keys()
    Args = namedtuple("Args", param_names)
    args = Args(**para_dict)

    if args.cuda:
        torch.backends.cudnn.benchmark = True

    # DataSets
    info = load_speech_commands_feddata(
        task=args.task
    )
    users_data, glo_test_xs, glo_test_ys, noise_xs = info

    # info
    print("Number of clients: {}".format(len(users_data)))
    n_samples = [len(cdata["train_xs"]) for _, cdata in users_data.items()]
    print("Total training samples: {}".format(sum(n_samples)))
    print("Number of local samples: {}, {}, {}, {}".format(
        np.min(n_samples), np.max(n_samples),
        np.mean(n_samples), np.median(n_samples)
    ))

    print("Total test samples: {}".format(len(glo_test_xs)))

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
    model = CifNet(
        net=args.net,
        n_classes=args.n_classes
    )
    print(model)
    print([name for name, _ in model.named_parameters()])
    n_params = sum([
        param.numel() for param in model.parameters()
    ])
    print("Total number of parameters : {}".format(n_params))

    if args.cuda:
        model.cuda()

    algo = FedKWSUI(
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
        "lr_mu": [1.0],
        "weight_decay": [1e-5],
        "max_grad_norm": [50.0],
        "ls_mu": [0.1],
        "adv_lamb": [0.001],
        "cuda": [True],
        "save_ckpts": [False],
    }

    optim_pairs = {
        "SGD": [0.01],
        "AdamW": [0.002, 0.0008],
    }

    nets = ["DSCNN"]

    for net in nets:
        for task in ["12"]:
            for ls_mu in [0.2]:
                for adv_lamb in [0.001]:
                    if net == "Transformer":
                        optimizer = "AdamW"
                        max_round = 500
                        test_round = 5
                    else:
                        optimizer = "SGD"
                        max_round = 300
                        test_round = 3

                    for lr in optim_pairs[optimizer]:
                        para_dict = {}
                        for k, vs in candi_param_dict.items():
                            para_dict[k] = random.choice(vs)

                        para_dict["way"] = "mfcc"
                        para_dict["input_channel"] = 40
                        para_dict["task"] = task
                        para_dict["n_classes"] = int(task)
                        para_dict["optimizer"] = optimizer
                        para_dict["lr"] = lr
                        para_dict["net"] = net
                        para_dict["batch_size"] = 32
                        para_dict["max_round"] = max_round
                        para_dict["test_round"] = test_round
                        para_dict["ls_mu"] = ls_mu
                        para_dict["adv_lamb"] = adv_lamb
                        para_dict["fname"] = "fedcif-{}.log".format(
                            net
                        )

                        main_federate(para_dict)


if __name__ == "__main__":
    main()
