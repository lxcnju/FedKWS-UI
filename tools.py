import torch
from torch.utils import data

import numpy as np


def guassian_kernel(
        source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0]) + int(target.size()[0])
    total = torch.cat([source, target], dim=0)

    L2_distance = ((
        total.unsqueeze(dim=1) - total.unsqueeze(dim=0)
    ) ** 2).sum(2)

    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth += 1e-8

    # print("Bandwidth:", bandwidth)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [
        torch.exp(-L2_distance / band) for band in bandwidth_list
    ]
    return sum(kernel_val)


def mmd_rbf_noaccelerate(
        source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(
        source, target,
        kernel_mul=kernel_mul, kernel_num=kernel_num,
        fix_sigma=fix_sigma
    )
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY - YX)
    return loss


def construct_dataloaders(clients, csets, gset, args):
    train_loaders = {}
    test_loaders = {}
    glo_test_loader = None

    for client in clients:
        assert isinstance(csets[client], tuple), \
            "csets must be a tuple (train_set, test_set): {}".format(client)

        assert csets[client][1] is not None, \
            "local test set must not be None in client: {}".format(client)

        train_loader = data.DataLoader(
            csets[client][0],
            batch_size=args.batch_size,
            shuffle=True
        )
        train_loaders[client] = train_loader

        test_loader = data.DataLoader(
            csets[client][1],
            batch_size=args.batch_size * 50,
            shuffle=False
        )
        test_loaders[client] = test_loader

    assert gset is not None, \
        "global test set must not be None"

    glo_test_loader = data.DataLoader(
        gset,
        batch_size=args.batch_size * 50,
        shuffle=False
    )

    return train_loaders, test_loaders, glo_test_loader


def construct_loaders(train_set, test_set, args):
    train_loader = data.DataLoader(
        train_set, batch_size=args.batch_size,
        shuffle=True, drop_last=False
    )

    test_loader = data.DataLoader(
        test_set, batch_size=args.batch_size,
        shuffle=False, drop_last=False
    )

    return train_loader, test_loader


def construct_fine_optimizer(model, args):
    param_groups = []

    for name, params in model.named_parameters():
        print(name)
        param_groups.append(
            {"params": params, "lr": args.lr, "name": name}
        )

    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(
            param_groups,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == "RMSprop":
        optimizer = torch.optim.RMSprop(
            param_groups,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == "Adam":
        optimizer = torch.optim.Adam(
            param_groups,
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    else:
        raise ValueError("No such optimizer: {}".format(args.optimizer))
    return optimizer


def construct_group_optimizer(model, lr, args):
    lr_x = lr * args.lr_mu

    encoder_param_ids = list(map(id, model.encoder.parameters()))

    other_params = filter(
        lambda p: id(p) not in encoder_param_ids,
        model.parameters()
    )

    param_groups = [
        {"params": other_params, "lr": lr},
        {"params": model.encoder.parameters(), "lr": lr_x},
    ]

    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(
            param_groups,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == "RMSprop":
        optimizer = torch.optim.RMSprop(
            param_groups,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == "Adam":
        optimizer = torch.optim.Adam(
            param_groups,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=args.weight_decay
        )
    else:
        raise ValueError("No such optimizer: {}".format(args.optimizer))
    return optimizer


def construct_optimizer(model, lr, args):
    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == "RMSprop":
        optimizer = torch.optim.RMSprop(
            model.parameters(),
            lr=lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=args.weight_decay
        )
    else:
        raise ValueError("No such optimizer: {}".format(args.optimizer))
    return optimizer


def construct_lr_scheduler(optimizer, args):
    if args.scheduler == "StepLR":
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.step_size, gamma=args.gamma
        )
    elif args.scheduler == "CosLR":
        CosLR = torch.optim.lr_scheduler.CosineAnnealingLR
        lr_scheduler = CosLR(
            optimizer, T_max=args.epoches, eta_min=1e-8
        )
    elif args.scheduler == "CosLRWR":
        CosLRWR = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
        lr_scheduler = CosLRWR(
            optimizer, T_0=args.step_size
        )
    elif args.scheduler == "CyclicLR":
        lr_scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer=optimizer,
            base_lr=0.0,
            max_lr=args.lr,
            step_size_up=args.step_size,
        )
    elif args.scheduler == "WSQuadLR":
        # LambdaLR: quadratic
        def lr_warm_start_quad(t, T0=args.ws_step, T_max=args.epoches):
            # T0 = int(0.1 * T_max)
            if t <= T0:
                return 1.0 * (t + 1e-6) / T0
            else:
                return (1.0 - 1.0 * (t - T0) / (T_max - T0)) ** 2

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer=optimizer,
            lr_lambda=lr_warm_start_quad,
        )
    elif args.scheduler == "WSStepLR":
        # LambdaLR: step lr
        def lr_warm_start_step(
            t, T0=args.ws_step,
            step_size=args.step_size, gamma=args.gamma
        ):
            if t <= T0:
                return 1.0 * (t + 1e-6) / T0
            else:
                return gamma ** int((t - T0) / step_size)
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer=optimizer,
            lr_lambda=lr_warm_start_step,
        )
    elif args.scheduler == "WSCosLR":
        # LambdaLR: coslr
        def lr_warm_start_cos(
            t, T0=args.ws_step, T_max=args.epoches
        ):
            if t <= T0:
                return 1.0 * (t + 1e-6) / T0
            else:
                return (np.cos((t - T0) / (T_max - T0) * np.pi) + 1.0) / 2.0
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer=optimizer,
            lr_lambda=lr_warm_start_cos,
        )
    else:
        raise ValueError("No such scheduler: {}".format(args.scheduler))
    return lr_scheduler


if __name__ == "__main__":
    from collections import namedtuple
    from matplotlib import pyplot as plt

    import torch.nn as nn

    class Network(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(100, 2)

        def forward(self, xs):
            return None

    model = Network()

    lr = 0.1
    epoches = 200

    para_dict = {
        "lr": lr,
        "ws_step": 10,
        "step_size": 60,
        "gamma": 0.1,
        "epoches": epoches,
    }

    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr
    )

    # WS-StepLR
    para_dict["scheduler"] = "WSStepLR"
    param_names = para_dict.keys()
    Args = namedtuple("Args", param_names)
    args = Args(**para_dict)

    scheduler = construct_lr_scheduler(optimizer, args)

    lrs = []
    for _ in range(epoches):
        model.zero_grad()
        optimizer.step()
        lrs.append(scheduler.get_last_lr())
        scheduler.step()

    plt.figure()
    plt.plot(range(len(lrs)), lrs)
    plt.title("WSStepLR: ws_step=10, epoch=200, step_size=60, gamma=0.1")
    plt.show()

    # WS-CosLR
    para_dict["scheduler"] = "WSCosLR"
    param_names = para_dict.keys()
    Args = namedtuple("Args", param_names)
    args = Args(**para_dict)

    scheduler = construct_lr_scheduler(optimizer, args)

    lrs = []
    for _ in range(epoches):
        model.zero_grad()
        optimizer.step()
        lrs.append(scheduler.get_last_lr())
        scheduler.step()

    plt.figure()
    plt.plot(range(len(lrs)), lrs)
    plt.title("WSCosLR: ws_step=10, epoch=200")
    plt.show()

    # CosLR
    para_dict["scheduler"] = "CosLR"
    param_names = para_dict.keys()
    Args = namedtuple("Args", param_names)
    args = Args(**para_dict)

    scheduler = construct_lr_scheduler(optimizer, args)

    lrs = []
    for _ in range(epoches):
        model.zero_grad()
        optimizer.step()
        lrs.append(scheduler.get_last_lr())
        scheduler.step()

    plt.figure()
    plt.plot(range(len(lrs)), lrs)
    plt.title("CosLR: epoch=200")
    plt.show()

    # WS-QuadLR
    para_dict["scheduler"] = "WSQuadLR"
    param_names = para_dict.keys()
    Args = namedtuple("Args", param_names)
    args = Args(**para_dict)

    scheduler = construct_lr_scheduler(optimizer, args)

    lrs = []
    for _ in range(epoches):
        model.zero_grad()
        optimizer.step()
        lrs.append(scheduler.get_last_lr())
        scheduler.step()

    plt.figure()
    plt.plot(range(len(lrs)), lrs)
    plt.title("WSQuadLR: ws_step=10, epoch=200")
    plt.show()

