import torch
import torch.nn as nn
import torch.nn.functional as F

import os
from tqdm import tqdm

from utils import count_acc, Averager
from utils import append_to_logs
from utils import format_logs

from tools import construct_loaders
from tools import construct_optimizer
from tools import construct_lr_scheduler

from paths import ckpt_dir


class Classify():
    def __init__(
        self,
        train_set,
        test_set,
        model,
        args
    ):
        self.train_set = train_set
        self.test_set = test_set
        self.model = model
        self.args = args

        self.train_loader, self.test_loader = \
            construct_loaders(
                train_set, test_set, args
            )

        self.optimizer = construct_optimizer(
            self.model, args.lr, args
        )
        self.lr_scheduler = construct_lr_scheduler(self.optimizer, args)

        self.logs = {
            "EPOCHS": [],
            "LOSSES": [],
            "TrACCS": [],
            "TeACCS": [],
        }

    def main(self):
        for epoch in range(1, self.args.epoches + 1):
            ce_loss, tr_acc = self.train(
                model=self.model,
                optimizer=self.optimizer,
                loader=self.train_loader,
                args=self.args
            )
            te_acc = self.test(
                model=self.model,
                loader=self.test_loader,
                args=self.args
            )
            print("[Epoch:{}] [Loss:{}] [TrAcc:{}] [TeAcc:{}]".format(
                epoch, ce_loss, tr_acc, te_acc
            ))

            # add to log
            self.logs["EPOCHS"].append(epoch)
            self.logs["LOSSES"].append(ce_loss)
            self.logs["TrACCS"].append(tr_acc)
            self.logs["TeACCS"].append(te_acc)

            self.lr_scheduler.step()

            if self.args.save_ckpts is True:
                if epoch % self.args.step_size == 0:
                    fpath = os.path.join(
                        ckpt_dir,
                        "{}-{}-{}-{}-{}.pth".format(
                            self.args.dataset,
                            self.args.net,
                            self.args.n_layer,
                            self.args.lr,
                            epoch
                        )
                    )
                    self.save_ckpt(fpath)

    def train(self, model, optimizer, loader, args):
        model.train()

        avg_ce_loss = Averager()
        acc_avg = Averager()

        for sx, sy in tqdm(loader):
            if args.cuda:
                sx, sy = sx.cuda(), sy.cuda()

            _, logits = model(sx)
            loss = F.cross_entropy(logits, sy)

            optimizer.zero_grad()
            loss.backward()

            nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=args.max_grad_norm
            )
            optimizer.step()

            avg_ce_loss.add(loss.item())

            acc = count_acc(logits, sy)
            acc_avg.add(acc)

        ce_loss = avg_ce_loss.item()
        acc = acc_avg.item()
        return ce_loss, acc

    def test(self, model, loader, args):
        model.eval()

        acc_avg = Averager()
        with torch.no_grad():
            for tx, ty in loader:
                if args.cuda:
                    tx, ty = tx.cuda(), ty.cuda()

                _, logits = model(tx)

                acc = count_acc(logits, ty)
                acc_avg.add(acc)

        acc = acc_avg.item()
        return acc

    def save_ckpt(self, fpath):
        # save model
        torch.save(self.model.state_dict(), fpath)
        print("Model saved in: {}".format(fpath))

    def save_logs(self, fpath):
        all_logs_str = []
        all_logs_str.append(str(self.args))

        logs_str = format_logs(self.logs)
        all_logs_str.extend(logs_str)

        append_to_logs(fpath, all_logs_str)
