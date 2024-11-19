import torch
import bitsandbytes as bnb
import math



def get_optimizer(model, train_args):
    if train_args.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=train_args.lr)
        optimizer.param_groups
    elif train_args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=train_args.lr)
    elif train_args.optimizer=="adamw8bit":
        # 需要 pip install bitsandbytes,
        # import bitsandbytes as bnb
        optimizer=bnb.optim.AdamW(model.parameters(),lr=train_args.lr,optim_bits=8)
    elif train_args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=train_args.lr, momentum=train_args.momentum
        )
    else:
        raise ValueError("optimizer not supported")
    return optimizer


class Scheduler:
    """
    是否采用warmup, warmup_steps, 学习率衰减策略
    """

    def __init__(self, optimizer, train_args) -> None:
        self.train_args = train_args
        self.steps = 0
        self.optimizer = optimizer
        self.lrs = [param_group["lr"] for param_group in optimizer.param_groups]

    def get_lr(self):
        lrs = []
        if self.train_args.warm_up:
            # 线性预热
            if self.steps < self.train_args.warmup_steps:
                lr_min = self.train_args.lr_min
                lrs = [
                    self.steps * (lr - lr_min) / self.train_args.warmup_steps + lr_min
                    for lr in self.lrs
                ]
                return lrs
            elif self.train_args.scheduler == "cosine":
                # 余弦退火
                lrs = [
                    self.train_args.lr_min
                    + 0.5
                    * (lr - self.train_args.lr_min)
                    * (
                        1
                        + math.cos(
                            (self.steps - self.train_args.warmup_steps)
                            / (self.train_args.max_steps - self.train_args.warmup_steps)
                            * math.pi
                        )
                    )
                    for lr in self.lrs
                ]
            elif self.train_args.scheduler == "linear":
                # 线性衰减
                lrs = [
                    lr
                    - (self.steps - self.train_args.warmup_steps)
                    * (lr - self.train_args.lr_min)
                    / (self.train_args.max_steps - self.train_args.warmup_steps)
                    for lr in self.lrs
                ]
            else:
                raise ValueError("scheduler not supported")

        # 不采用预热 直接进行训练
        elif self.train_args.scheduler == "cosine":
            # 余弦退火
            lrs = [
                self.train_args.lr_min
                + 0.5
                * (lr - self.train_args.lr_min)
                * (
                    1
                    + math.cos(
                        (self.steps - self.train_args.warmup_steps)
                        / (self.train_args.max_steps)
                        * math.pi))
                for lr in self.lrs
            ]
        elif self.train_args.scheduler == "linear":
            # 线性衰减
            lrs = [
                lr
                - (self.steps - self.train_args.warmup_steps)
                * (lr - self.train_args.lr_min)
                / (self.train_args.max_steps - self.train_args.warmup_steps)
                for lr in self.lrs
            ]
        else:
            raise ValueError("scheduler not supported")

        return lrs
    

    def step(self, step):
        self.steps = step
        lrs = self.get_lr()
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group["lr"] = lrs[i]

    def reset(self):
        self.steps = 0
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group["lr"] = self.lrs[i]


# from torch.optim.lr_scheduler import LambdaLR, StepLR, MultiStepLR, ExponentialLR, CosineAnnealingLR, ReduceLROnPlateau, CyclicLR, CosineAnnealingWarmRestarts


def get_scheduler(optimizer, train_args):

    # if train_args.scheduler=='cosine':
    #     scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=train_args.max_steps,eta_min=train_args.lr_min)
    # elif train_args.scheduler=='linear':
    #     scheduler=torch.optim.lr_scheduler.LinearLR(optimizer,start_factor=train_args.lr_min,total_iters=train_args.max_steps)
    # else:
    #     raise ValueError('scheduler not supported')

    return Scheduler(optimizer, train_args)


class EarlyStopping:
    def __init__(self, patience=5, delta=0) -> None:
        self.patience = patience
        self.delta = delta
        self.best_loss = None
        self.counter = 0
        self.best_epoch = None

    def step(self, loss, epoch):
        if self.best_loss - self.delta > loss:
            self.best_loss = loss
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1

    def stopping(self):
        if self.patience <= self.counter:
            return True
        else:
            return False
        
    

