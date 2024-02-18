import torch
from torch.optim.lr_scheduler import _LRScheduler
import math

"""
Heavily inspired from implementation by Chris Scheulen found here: 
https://gitlab.cern.ch/utils_chscheul/naf_utils/neural-network-for-tthbb/-/blob/main/utils/lr_schedules.py?ref_type=heads
"""


class CosineRampUpDownLR(_LRScheduler):
    def __init__(
        self,
        optimizer,
        lr_init=1e-8,
        lr_max=1e-5,
        lr_final=1e-7,
        burn_in=10,
        ramp_up=10,
        plateau=20,
        ramp_down=100,
        last_epoch=-1,
    ):
        self.lr_init = lr_init
        self.lr_max = lr_max
        self.lr_final = lr_final

        self.burn_in = burn_in
        self.ramp_up = ramp_up
        self.plateau = plateau
        self.ramp_down = ramp_down
        self.total_epochs = burn_in + ramp_up + plateau + ramp_down

        super(CosineRampUpDownLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.burn_in:
            return [self.lr_init for _ in self.base_lrs]

        epoch = self.last_epoch - self.burn_in

        if epoch < self.ramp_up:
            fraction = 0.5 * (1.0 - math.cos(math.pi * (epoch + 1) / self.ramp_up))
            return [
                (self.lr_max - self.lr_init) * fraction + self.lr_init
                for _ in self.base_lrs
            ]

        epoch -= self.ramp_up

        if epoch < self.plateau:
            return [self.lr_max for _ in self.base_lrs]

        epoch -= self.plateau

        if epoch < self.ramp_down:
            fraction = 0.5 * (1.0 + math.cos(math.pi * (epoch + 1) / self.ramp_down))
            return [
                (self.lr_max - self.lr_final) * fraction + self.lr_final
                for _ in self.base_lrs
            ]

        return [self.lr_final for _ in self.base_lrs]
