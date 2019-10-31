import math
from bisect import bisect_right
from torch.optim.lr_scheduler import _LRScheduler


# FIXME ideally this would be achieved with a CombinedLRScheduler,
# separating MultiStepLR with WarmupLR
# but the current LRScheduler design doesn't allow it
# https://github.com/lulujianjie/pose-transfer-PATN/tree/596f0e91ae4b34c3273bb9888f562a05bcae8463
class WarmupMultiStepLR(_LRScheduler):
    def __init__(
            self,
            optimizer,
            milestones,  # steps
            gamma=0.1,
            warmup_factor=1.0 / 3,
            warmup_epoch=5,
            warmup_method="linear",
            last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_epoch = warmup_epoch
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_epoch:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = self.last_epoch / self.warmup_epoch
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]


class CosineAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, T_max, eta_min=0, warmup_factor=1.0 / 3, warmup_iters=500, warmup_method="linear",
                 last_epoch=-1):
        self.T_max = T_max - warmup_iters
        self.eta_min = eta_min
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(CosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = float(self.last_epoch) / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
            return [base_lr * warmup_factor
                    for base_lr in self.base_lrs]
        else:
            return [self.eta_min + (base_lr - self.eta_min) *
                    (1 + math.cos(math.pi * (self.last_epoch - self.warmup_iters) / self.T_max)) / 2
                    for base_lr in self.base_lrs]


# 带warmup的MultiStep/step/ploy/Cosine
class GeneralLR_Scheduler(_LRScheduler):
    def __init__(self, optimizer, cfg, max_iter, last_epoch=-1, eta_min=0):
        step_lr = [20000, 40000, 80000]
        self.model = cfg['Solver']['SchedulerModel']
        if self.model == 'step' and isinstance(step_lr, list):
            step_lr = step_lr[0]
        self.step_lr = step_lr
        self.gamma = 0.1
        self.warmup_factor = 1.0 / 3
        self.warmup_iters = cfg['Solver']['WarnupIters']
        self.warmup_method = "linear"
        self.max_iter = max_iter  # 迭代总次数
        self.eta_min = eta_min  # 余弦退火的最小值
        self.max_lr = 10  # lr find用
        super(GeneralLR_Scheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = float(self.last_epoch) / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        if self.model == 'multistep':  # step间隔可定制
            return [
                base_lr
                * warmup_factor
                * self.gamma ** bisect_right(self.step_lr, self.last_epoch)
                for base_lr in self.base_lrs
            ]
        elif self.model == 'step':  # step只能是固定间隔
            return [base_lr * warmup_factor * self.gamma ** (self.last_epoch // self.step_lr)
                    for base_lr in self.base_lrs]
        elif self.model == 'poly':
            return [base_lr * warmup_factor * (pow((1 - 1.0 * self.last_epoch / self.max_iter + 1e-32), 0.9))
                    for base_lr in self.base_lrs]
        elif self.model == 'cosine':
            return [warmup_factor * (self.eta_min + (base_lr - self.eta_min) *
                                     (1 + math.cos(math.pi * self.last_epoch / self.max_iter + 1e-32)) / 2)
                    for base_lr in self.base_lrs]
        elif self.model == 'exp_increase':  # warmup不用
            return [base_lr * (self.max_lr / base_lr) ** (self.last_epoch / (self.max_iter + 1e-32)) for base_lr in
                    self.base_lrs]
