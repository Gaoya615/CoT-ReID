""" Scheduler Factory
Hacked together by / Copyright 2020 Ross Wightman
"""
from .cosine_lr import CosineLRScheduler
from .lr_scheduler import WarmupMultiStepLR

def create_scheduler(cfg, optimizer):
    solver = cfg.SOLVER
    num_epochs = solver.MAX_EPOCHS
    lr_min = solver.LR_MIN
    warmup_lr_init = 0.1 * solver.BASE_LR

    warmup_t = solver.WARMUP_ITERS
    lr_scheduler = CosineLRScheduler(
        optimizer,
        t_initial=num_epochs,
        lr_min=lr_min,
        t_mul=1.,
        decay_rate=0.1,
        warmup_lr_init=warmup_lr_init,
        warmup_t=warmup_t,
        cycle_limit=1,
        t_in_epochs=True,
        noise_range_t=None,
    )

    return lr_scheduler
