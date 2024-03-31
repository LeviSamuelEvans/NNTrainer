import torch
from torch.optim.lr_scheduler import _LRScheduler
import math

"""
Heavily inspired from implementation by Chris Scheulen found here:
https://gitlab.cern.ch/utils_chscheul/naf_utils/neural-network-for-tthbb/-/blob/main/utils/lr_schedules.py?ref_type=heads
"""


class CosineRampUpDownLR(_LRScheduler):
    """Cosine learning rate scheduler with burn-in, ramp-up, plateau, and ramp-down phases.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        The optimizer for which to schedule the learning rate.
    lr_init : float, optional
        The initial learning rate during the burn-in phase. Default is 1e-8.
    lr_max : float, optional
        The maximum learning rate during the plateau phase. Default is 1e-5.
    lr_final : float, optional
        The final learning rate at the end of the ramp-down phase. Default is 1e-7.
    burn_in : int, optional
        The number of epochs for the burn-in phase. Default is 10.
    ramp_up : int, optional
        The number of epochs for the ramp-up phase. Default is 10.
    plateau : int, optional
        The number of epochs for the plateau phase. Def
    ramp_down : int, optional
        The number of epochs for the ramp-down phase. Default is 100.
    last_epoch : int, optional
        The index of the last epoch. Default is -1.
    """
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
        """Compute the learning rate based on the current epoch.

        Returns
        -------
        list
            A list of learning rates, one for each parameter group.
        """
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
