import math
import torch.optim as optim

def get_lr_scheduler(opt, optimizer):
    if opt.lr_type == 'step_lr':
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer,
                                               step_size=opt.lr_decay_period,
                                               gamma=opt.lr_decay_lvl)
        return lr_scheduler
    elif opt.lr_type == 'cosine_repeat_lr':
        lr_scheduler = CosineRepeatAnnealingLR(optimizer,
                                               T_max=opt.lr_decay_period,
                                               T_mult=opt.lr_decay_lvl)
        return lr_scheduler
    else:
        raise Exception('Unknown lr_type')

def get_optimizer(model, opt):
    if opt.optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr = opt.lr, momentum=0.9, weight_decay=5e-4)
        return optimizer
    elif opt.optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr = opt.lr, betas=(0.5, 0.999), weight_decay=opt.weight_decay)
        return optimizer

def get_margin_alpha_scheduler(opt):
    if opt.alpha_scheduler_type == 'linear_alpha':
        alpha_scheduler = LinearAlpha(opt.alpha_start_epoch, opt.alpha_end_epoch, opt.alpha_min, opt.alpha_max)
        return alpha_scheduler

    return None

class LinearAlpha(object):
    def __init__(self, start_epoch, end_epoch, alpha_min, alpha_max):
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max

    def get_alpha(self, epoch):
        if epoch <= self.start_epoch:
            return self.alpha_min
        elif epoch >= self.end_epoch:
            return self.alpha_max
        else:
            epoch_step = self.end_epoch - self.start_epoch
            alpha_step = self.alpha_max - self.alpha_min
            return self.alpha_min + (epoch - self.start_epoch) * alpha_step / epoch_step

class CosineRepeatAnnealingLR(optim.lr_scheduler._LRScheduler):
    r"""Set the learning rate of each parameter group using a cosine annealing
    schedule, where :math:`\eta_{max}` is set to the initial lr and
    :math:`T_{cur}` is the number of epochs since the last restart in SGDR:

    .. math::

        \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 +
        \cos(\frac{T_{cur}}{T_{max}}\pi))

    When last_epoch=-1, sets initial lr as lr.
    """

    def __init__(self, optimizer, T_max, 
                 T_mult = 1, 
                 T_start=0, 
                 eta_min=0, 
                 last_epoch=-1,):

        self.T_max = T_max
        self.T_mult = T_mult
        self.T_start = T_start
        self.eta_min = eta_min
        super(CosineRepeatAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [self.eta_min + (base_lr - self.eta_min) *
                (1 + math.cos(math.pi * (self.last_epoch - self.T_start)/ self.T_max)) / 2
                for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr
        if self.last_epoch - self.T_start == self.T_max:
            self.T_start = self.last_epoch
            self.T_max = self.T_max * self.T_mult
            print('T_start: {0}, T_max: {1}'.format(self.T_start,self.T_max)) 
