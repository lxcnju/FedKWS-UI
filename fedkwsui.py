import copy
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn

from utils import Averager
from utils import count_acc
from utils import append_to_logs
from utils import format_logs

from tools import construct_dataloaders
from tools import construct_group_optimizer

from criterion import soft_cross_entropy


class FedKWSUI():
    def __init__(
        self, csets, gset, model, args
    ):
        self.csets = csets
        self.gset = gset
        self.model = model
        self.args = args

        self.clients = list(csets.keys())

        # construct dataloaders
        self.train_loaders, self.test_loaders, self.glo_test_loader = \
            construct_dataloaders(
                self.clients, self.csets, self.gset, self.args
            )

        # max_num
        if self.args.n_classes == 12:
            self.max_num = 100
            self.ratio = 3.5
        elif self.args.n_classes == 35:
            self.max_num = 300
            self.ratio = 5.0
        else:
            pass

        self.class_nums = {}
        for client in self.clients:
            ys = self.csets[client][0].ys
            cns = np.array([
                np.sum(ys == c) for c in range(self.args.n_classes)
            ])
            self.class_nums[client] = cns

        self.logs = {
            "ROUNDS": [],
            "LOSSES": [],
            "GLO_TACCS": [],
            "LOCAL_TACCS": [],
            "GLO_FAS": [],
            "GLO_FRS": [],
        }

        self.calculate_stats_and_print()

    def calculate_stats_and_print(self):
        ratios = []
        for client in self.clients:
            class_nums = self.class_nums[client]

            n_sam = np.sum(class_nums)

            dist = class_nums / n_sam
            ent = (-1.0 * dist * np.log(dist + 1e-8)).sum()

            w1 = min(n_sam / self.max_num, 1.0)
            w2 = ent / np.log(len(dist))
            w = 2.0 * w1 * w2 / (w1 + w2)

            # total batches
            ratio = self.ratio * w
            ratios.append(ratio)

        print("Max:{},Min:{},Mean:{},Median:{}".format(
            np.max(ratios), np.min(ratios), np.mean(ratios), np.median(ratios)
        ))

    def train(self):
        # Training
        for r in range(1, self.args.max_round + 1):
            n_sam_clients = int(self.args.c_ratio * len(self.clients))
            sam_clients = np.random.choice(
                self.clients, n_sam_clients, replace=False
            )

            local_models = {}

            avg_loss = Averager()

            all_per_accs = []
            for client in sam_clients:
                local_model, pri_loss = self.update_private_local(
                    r=r,
                    model=copy.deepcopy(self.model),
                    train_loader=self.train_loaders[client],
                )
                local_model, per_accs, loss = self.update_local(
                    r=r,
                    model=copy.deepcopy(local_model),
                    train_loader=self.train_loaders[client],
                    test_loader=self.test_loaders[client],
                    class_nums=self.class_nums[client],
                )

                local_models[client] = copy.deepcopy(local_model)
                avg_loss.add(loss)
                all_per_accs.append(per_accs)

            train_loss = avg_loss.item()
            per_accs = list(np.array(all_per_accs).mean(axis=0))

            self.update_global(
                r=r,
                global_model=self.model,
                local_models=local_models,
            )

            if r % self.args.test_round == 0:
                # global test loader
                glo_test_acc = self.test(
                    model=self.model,
                    loader=self.glo_test_loader,
                )

                if self.args.task == "12":
                    glo_fa, glo_fr = self.test_fa_fr(
                        model=self.model,
                        loader=self.glo_test_loader
                    )
                else:
                    glo_fa = 0.0
                    glo_fr = 0.0

                # add to log
                self.logs["ROUNDS"].append(r)
                self.logs["LOSSES"].append(train_loss)
                self.logs["GLO_TACCS"].append(glo_test_acc)
                self.logs["LOCAL_TACCS"].extend(per_accs)
                self.logs["GLO_FAS"].append(glo_fa)
                self.logs["GLO_FRS"].append(glo_fr)

                print("[R:{}] [Ls:{}] [TeAc:{}] [PAcBeg:{} PAcAft:{}]".format(
                    r, train_loss, glo_test_acc, per_accs[0], per_accs[-1]
                ))
                print("[GloFA:{}] [GloFR:{}]".format(glo_fa, glo_fr))

    def update_private_local(self, r, model, train_loader):
        optimizer = construct_group_optimizer(
            model, self.args.lr, self.args
        )

        if self.args.local_steps is not None:
            n_total_bs = self.args.local_steps
        elif self.args.local_epochs is not None:
            n_total_bs = max(
                int(self.args.local_epochs * len(train_loader)), 5
            )
        else:
            raise ValueError(
                "local_steps and local_epochs must not be None together"
            )

        n_total_bs = int(0.2 * n_total_bs)

        model.train()

        loader_iter = iter(train_loader)

        avg_p_loss = Averager()

        for t in range(n_total_bs + 1):
            try:
                batch_x, batch_y = loader_iter.next()
            except Exception:
                loader_iter = iter(train_loader)
                batch_x, batch_y = loader_iter.next()

            if self.args.cuda:
                batch_x, batch_y = batch_x.cuda(), batch_y.cuda()

            _, _, _, pri_logits = model(batch_x)

            p_loss = soft_cross_entropy(pri_logits, batch_y, mu=0.0)

            optimizer.zero_grad()
            p_loss.backward()

            model.encoder.zero_grad()   # do not update encoder
            nn.utils.clip_grad_norm_(
                model.parameters(), self.args.max_grad_norm
            )

            optimizer.step()

            avg_p_loss.add(p_loss.item())

        p_loss = avg_p_loss.item()
        return model, p_loss

    def update_local(self, r, model, train_loader, test_loader, class_nums):
        optimizer = construct_group_optimizer(
            model, self.args.lr, self.args
        )

        if self.args.local_steps is not None:
            n_total_bs = self.args.local_steps
        elif self.args.local_epochs is not None:
            n_total_bs = max(
                int(self.args.local_epochs * len(train_loader)), 5
            )
        else:
            raise ValueError(
                "local_steps and local_epochs must not be None together"
            )

        # calculate client number / entropy
        n_sam = np.sum(class_nums)

        dist = class_nums / n_sam
        ent = (-1.0 * dist * np.log(dist + 1e-8)).sum()

        w1 = min(n_sam / self.max_num, 1.0)
        w2 = ent / np.log(len(dist))
        w = 2.0 * w1 * w2 / (w1 + w2)

        mu = self.args.ls_mu * (1.0 - w)

        # total batches
        n_total_bs = int(self.ratio * w * n_total_bs)

        model.train()

        loader_iter = iter(train_loader)

        avg_loss = Averager()
        per_accs = []

        for t in range(n_total_bs + 1):
            if t in [0, n_total_bs]:
                per_acc = self.test(
                    model=model,
                    loader=test_loader,
                )
                per_accs.append(per_acc)

            if t >= n_total_bs:
                break

            model.train()
            try:
                batch_x, batch_y = loader_iter.next()
            except Exception:
                loader_iter = iter(train_loader)
                batch_x, batch_y = loader_iter.next()

            if self.args.cuda:
                batch_x, batch_y = batch_x.cuda(), batch_y.cuda()

            _, _, glo_logits, pri_logits = model(batch_x)

            pri_y = pri_logits.detach().softmax(dim=-1)

            g_loss = soft_cross_entropy(glo_logits, batch_y, mu=mu)
            adv_loss = (
                -1.0 * pri_y * glo_logits.log_softmax(dim=-1)
            ).sum(dim=1).mean()

            loss = g_loss - self.args.adv_lamb * adv_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(), self.args.max_grad_norm
            )
            optimizer.step()

            avg_loss.add(loss.item())

        loss = avg_loss.item()
        return model, per_accs, loss

    def update_global(self, r, global_model, local_models):
        mean_state_dict = {}

        for name, param in global_model.state_dict().items():
            vs = []
            for client in local_models.keys():
                vs.append(local_models[client].state_dict()[name])
            vs = torch.stack(vs, dim=0)

            try:
                mean_value = vs.mean(dim=0)
            except Exception:
                # for BN's cnt
                mean_value = (1.0 * vs).mean(dim=0).long()
            mean_state_dict[name] = mean_value

        global_model.load_state_dict(mean_state_dict, strict=False)

    def test(self, model, loader):
        model.eval()

        acc_avg = Averager()

        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(loader):
                if self.args.cuda:
                    batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
                _, _, logits, _ = model(batch_x)
                acc = count_acc(logits, batch_y)
                acc_avg.add(acc)

        acc = acc_avg.item()
        return acc

    def test_fa_fr(self, model, loader):
        model.eval()

        # {'silence': 7, 'unk': 9}
        reject_labels = [7, 9]

        preds = []
        reals = []
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(loader):
                if self.args.cuda:
                    batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
                _, _, logits, _ = model(batch_x)
                preds.append(logits.argmax(dim=1).detach().cpu().numpy())
                reals.append(batch_y.cpu().detach().numpy())

        preds = np.concatenate(preds, axis=0)
        reals = np.concatenate(reals, axis=0)

        true_accept, true_reject = 0, 0
        false_accept, false_reject = 0, 0

        for py, ry in zip(preds, reals):
            if py not in reject_labels and ry not in reject_labels:
                true_accept += 1
            elif py in reject_labels and ry not in reject_labels:
                false_reject += 1
            elif py not in reject_labels and ry in reject_labels:
                false_accept += 1
            elif py in reject_labels and ry in reject_labels:
                true_reject += 1
            else:
                raise ValueError("error")

        false_accept = false_accept / len(preds)
        false_reject = false_reject / len(preds)
        return false_accept, false_reject

    def save_logs(self, fpath):
        all_logs_str = []
        all_logs_str.append(str(self.args))

        logs_str = format_logs(self.logs)
        all_logs_str.extend(logs_str)

        append_to_logs(fpath, all_logs_str)
