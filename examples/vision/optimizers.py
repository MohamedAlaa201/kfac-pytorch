"""Utilities for getting optimizers for computer vision examples."""
from __future__ import annotations

import argparse
from typing import Callable

import torch
import torch.distributed as dist
import torch.optim as optim

import kfac
from examples.utils import create_lr_schedule
from enum import Enum


class LearningMode(Enum):
    ADAMS = 1
    SGD = 2
    KFAC_WITH_SGD = 3
    KFAC_WITH_ADAM = 4

def get_optimizer(
    model: torch.nn.Module,
    mode: LearningMode,
    args: argparse.Namespace,
) -> tuple[
    optim.Optimizer,
    kfac.preconditioner.KFACPreconditioner | None,
    tuple[
        optim.lr_scheduler._LRScheduler,
        kfac.scheduler.LambdaParamScheduler | None,
    ],
]:
    
    """Get optimizer, preconditioner, and scheduler."""
    if mode == LearningMode.ADAMS or mode == LearningMode.KFAC_WITH_ADAM:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.adams_lr,
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.base_lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    lrs = create_lr_schedule(
        dist.get_world_size(),
        args.warmup_epochs,
        args.lr_decay,
    )
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lrs)

    grad_worker_fraction: kfac.enums.DistributedStrategy | float
    if args.kfac_strategy == 'comm-opt':
        grad_worker_fraction = kfac.enums.DistributedStrategy.COMM_OPT
    elif args.kfac_strategy == 'mem-opt':
        grad_worker_fraction = kfac.enums.DistributedStrategy.MEM_OPT
    elif args.kfac_strategy == 'hybrid-opt':
        grad_worker_fraction = args.kfac_grad_worker_fraction
    else:
        raise ValueError(
            f'Unknown KFAC Comm Method: {args.kfac_strategy}',
        )

    if mode == LearningMode.KFAC_WITH_SGD or mode == LearningMode.KFAC_WITH_ADAM:
        preconditioner = kfac.preconditioner.KFACPreconditioner(
            model,
            factor_update_steps=args.kfac_factor_update_steps,
            inv_update_steps=args.kfac_inv_update_steps,
            damping=args.kfac_damping,
            factor_decay=args.kfac_factor_decay,
            kl_clip=args.kfac_kl_clip,
            lr=lambda x: optimizer.param_groups[0]['lr'],
            accumulation_steps=args.batches_per_allreduce,
            allreduce_bucket_cap_mb=25,
            colocate_factors=args.kfac_colocate_factors,
            compute_method=kfac.enums.ComputeMethod.INVERSE
            if args.kfac_inv_method
            else kfac.enums.ComputeMethod.EIGEN,
            grad_worker_fraction=grad_worker_fraction,
            grad_scaler=args.grad_scaler if 'grad_scaler' in args else None,
            skip_layers=args.kfac_skip_layers,
        )

        def get_lambda(
            alpha: int,
            epochs: list[int] | None,
        ) -> Callable[[int], float]:
            """Create lambda function for param scheduler."""
            if epochs is None:
                _epochs = []
            else:
                _epochs = epochs

            def scale(epoch: int) -> float:
                """Compute current scale factor using epoch."""
                factor = 1.0
                for e in _epochs:
                    if epoch >= e:
                        factor *= alpha
                return factor

            return scale

        kfac_param_scheduler = kfac.scheduler.LambdaParamScheduler(
            preconditioner,
            damping_lambda=get_lambda(
                args.kfac_damping_alpha,
                args.kfac_damping_decay,
            ),
            factor_update_steps_lambda=get_lambda(
                args.kfac_update_steps_alpha,
                args.kfac_update_steps_decay,
            ),
            inv_update_steps_lambda=get_lambda(
                args.kfac_update_steps_alpha,
                args.kfac_update_steps_decay,
            ),
        )
    else:
        preconditioner = None
        kfac_param_scheduler = None

    return optimizer, preconditioner, (lr_scheduler, kfac_param_scheduler)
