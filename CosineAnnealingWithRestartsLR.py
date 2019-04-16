import math

import torch
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer


class CosineAnnealingWithRestartsLR(_LRScheduler):
    r"""Set the learning rate of each parameter group using a cosine annealing
    schedule, where :math:`\eta_{max}` is set to the initial lr and
    :math:`T_{cur}` is the number of epochs since the last restart in SGDR:

    .. math::

        \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 +
        \cos(\frac{T_{cur}}{T_{max}}\pi))

    When last_epoch=-1, sets initial lr as lr.

    It has been proposed in
    `SGDR: Stochastic Gradient Descent with Warm Restarts`_. Note that this only
    implements the cosine annealing part of SGDR, and not the restarts.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T_max (int): Maximum number of iterations.
        eta_min (float): Minimum learning rate. Default: 0.
        last_epoch (int): The index of last epoch. Default: -1.

    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983
    """

    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1, T_mult=1):
        self.T_max = T_max
        self.T_mult = T_mult
        self.next_restart = T_max
        self.eta_min = eta_min
        self.restarts = 0
        self.last_restart = 0
        super(
            CosineAnnealingWithRestartsLR,
            self).__init__(
            optimizer,
            last_epoch)

    def get_lr(self):
        self.Tcur = self.last_epoch - self.last_restart
        if self.Tcur >= self.next_restart:
            self.next_restart *= self.T_mult
            self.last_restart = self.last_epoch
        return [(self.eta_min +
                 (base_lr -
                  self.eta_min) *
                 (1 +
                  math.cos(math.pi *
                           self.Tcur /
                           self.next_restart)) /
                 2) for base_lr in self.base_lrs]
