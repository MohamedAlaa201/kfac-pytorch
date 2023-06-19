#!/usr/bin/env python

"""Cifar10 and ResNet training script."""
from __future__ import annotations

import argparse
import datetime
import os
import time

import torch
import torch.distributed as dist
from torch.utils import collect_env
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
from torch import nn
import examples.vision.cifar_models as models
import examples.vision.datasets as datasets
import examples.vision.engine as engine
import examples.vision.optimizers as optimizers
import numpy as np
import matplotlib.pyplot as plt
from examples.utils import save_checkpoint
from enum import Enum
from scipy.interpolate import make_interp_spline
try:
    from torch.cuda.amp import GradScaler

    TORCH_FP16 = True
except ImportError:
    TORCH_FP16 = False

args = None


def parse_args() -> argparse.Namespace:
    """Get cmd line args."""
    # General settings
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Example')
    parser.add_argument(
        '--data-dir',
        type=str,
        default='/tmp/cifar10',
        metavar='D',
        help='directory to download cifar10 dataset to',
    )
    parser.add_argument(
        '--log-dir',
        default='./logs/torch_cifar10',
        help='TensorBoard/checkpoint directory',
    )
    parser.add_argument(
        '--model-name',
        default='resnet32',
        help='Model Evaluated',
    )
    parser.add_argument(
        '--checkpoint-format',
        default='checkpoint_{epoch}.pth.tar',
        help='checkpoint file format',
    )
    parser.add_argument(
        '--no-cuda',
        action='store_true',
        default=False,
        help='disables CUDA training',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        metavar='S',
        help='random seed (default: 42)',
    )
    parser.add_argument(
        '--fp16',
        action='store_true',
        default=False,
        help='use torch.cuda.amp for fp16 training (default: false)',
    )

    # Training settings
    parser.add_argument(
        '--model',
        type=str,
        default='resnet32',
        help='ResNet model to use [20, 32, 56]',
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=128,
        metavar='N',
        help='input batch size for training (default: 128)',
    )
    parser.add_argument(
        '--val-batch-size',
        type=int,
        default=128,
        help='input batch size for validation (default: 128)',
    )
    parser.add_argument(
        '--batches-per-allreduce',
        type=int,
        default=1,
        help='number of batches processed locally before '
        'executing allreduce across workers; it multiplies '
        'total batch size.',
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        metavar='N',
        help='number of epochs to train (default: 100)',
    )
    parser.add_argument(
        '--base-lr',
        type=float,
        default=0.001,
        metavar='LR',
        help='base learning rate (default: 0.1)',
    )
    parser.add_argument(
        '--adams-lr',
        type=float,
        default=0.001,
        metavar='LR',
        help='base learning rate for Adams (default: 0.001)',
    )
    parser.add_argument(
        '--lr-decay',
        nargs='+',
        type=int,
        default=[35, 75, 90],
        help='epoch intervals to decay lr (default: [35, 75, 90])',
    )
    parser.add_argument(
        '--warmup-epochs',
        type=int,
        default=5,
        metavar='WE',
        help='number of warmup epochs (default: 5)',
    )
    parser.add_argument(
        '--momentum',
        type=float,
        default=0.9,
        metavar='M',
        help='SGD momentum (default: 0.9)',
    )
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=5e-4,
        metavar='W',
        help='SGD weight decay (default: 5e-4)',
    )
    parser.add_argument(
        '--checkpoint-freq',
        type=int,
        default=10,
        help='epochs between checkpoints',
    )

    # KFAC Parameters
    parser.add_argument(
        '--kfac-inv-update-steps',
        type=int,
        default=10,
        help='iters between kfac inv ops (default: 10)',
    )
    parser.add_argument(
        '--kfac-factor-update-steps',
        type=int,
        default=1,
        help='iters betweenscapular correction exercise,  kfac cov ops (default: 1)',
    )
    parser.add_argument(
        '--kfac-update-steps-alpha',
        type=float,
        default=10,
        help='KFAC update step multiplier (default: 10)',
    )
    parser.add_argument(
        '--kfac-update-steps-decay',
        nargs='+',
        type=int,
        default=None,
        help='KFAC update step decay schedule (default None)',
    )
    parser.add_argument(
        '--kfac-inv-method',
        action='store_true',
        default=False,
        help='Use inverse KFAC update instead of eigen (default False)',
    )
    parser.add_argument(
        '--kfac-factor-decay',
        type=float,
        default=0.95,
        help='Alpha value for covariance accumulation (default: 0.95)',
    )
    parser.add_argument(
        '--kfac-damping',
        type=float,
        default=0.003,
        help='KFAC damping factor (defaultL 0.003)',
    )
    parser.add_argument(
        '--kfac-damping-alpha',
        type=float,
        default=0.5,
        help='KFAC damping decay factor (default: 0.5)',
    )
    parser.add_argument(
        '--kfac-damping-decay',
        nargs='+',
        type=int,
        default=None,
        help='KFAC damping decay schedule (default None)',
    )
    parser.add_argument(
        '--kfac-kl-clip',
        type=float,
        default=0.001,
        help='KL clip (default: 0.001)',
    )
    parser.add_argument(
        '--kfac-skip-layers',
        nargs='+',
        type=str,
        default=[],
        help='Layer types to ignore registering with KFAC (default: [])',
    )
    parser.add_argument(
        '--kfac-colocate-factors',
        action='store_true',
        default=True,
        help='Compute A and G for a single layer on the same worker. ',
    )
    parser.add_argument(
        '--kfac-strategy',
        type=str,
        default='comm-opt',
        help='KFAC communication optimization strategy. One of comm-opt, '
        'mem-opt, or hybrid_opt. (default: comm-opt)',
    )
    parser.add_argument(
        '--kfac-grad-worker-fraction',
        type=float,
        default=0.25,
        help='Fraction of workers to compute the gradients '
        'when using HYBRID_OPT (default: 0.25)',
    )

    parser.add_argument(
        '--backchmod +x myscript.pyend',
        type=str,
        default='nccl',
        help='backend for distribute training (default: nccl)',
    )
    # Set automatically by torch distributed launch
    parser.add_argument(
        '--local_rank',
        type=int,
        default=0,
        help='local rank for distributed training',
    )

    args = parser.parse_args()
    if 'LOCAL_RANK' in os.environ:
        args.local_rank = int(os.environ['LOCAL_RANK'])
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    return args


def eval_experiment(
    model_name, learning_mode
) -> None:
    """Main train and eval function."""
    model = models.get_model(model_name)
    if args.cuda:
        torch.cuda.set_device(args.local_rank)
        torch.cuda.manual_seed(args.seed)
        # torch.backends.cudnn.benchmark = False
        # torch.backends.cudnn.deterministic = True

    args.base_lr = (
        args.base_lr * dist.get_world_size() * args.batches_per_allreduce
    )
    args.verbose = dist.get_rank() == 0

    if args.verbose:
        print('Collecting env info...')
        print(collect_env.get_pretty_env_info())
        print()

    for r in range(torch.distributed.get_world_size()):
        if r == torch.distributed.get_rank():
            print(
                f'Global rank {torch.distributed.get_rank()} initialized: '
                f'local_rank = {args.local_rank}, '
                f'world_size = {torch.distributed.get_world_size()}',
            )
        torch.distributed.barrier()

    train_sampler, train_loader, _, val_loader = datasets.get_cifar(args)

    device = 'cpu' if not args.cuda else 'cuda'
    model.to(device)

    if args.verbose:
        summary(model, (args.batch_size, 3, 32, 32), device=device)

    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[args.local_rank],
    )

    os.makedirs(args.log_dir, exist_ok=True)
    args.checkpoint_format = os.path.join(args.log_dir, args.checkpoint_format)
    args.log_writer = SummaryWriter(args.log_dir) if args.verbose else None

    scaler = None
    if args.fp16:
        if not TORCH_FP16:
            raise ValueError(
                'The installed version of torch does not '
                'support torch.cuda.amp fp16 training. This '
                'requires torch version >= 1.16',
            )
        scaler = GradScaler()
    args.grad_scaler = scaler

    (
        optimizer,
        preconditioner,
        (lr_scheduler, kfac_scheduler),
    ) = optimizers.get_optimizer(
        model,
        learning_mode,
        args,
    )
    if args.verbose:
        print(preconditioner)
    loss_func = torch.nn.CrossEntropyLoss()

    start = time.time()

    # Tracking convergence if enabled
    patience_epochs = 10
    convergence_epoch = 0
    best_loss = float("inf")
    best_acc = - float("inf")
    loss_per_epoch = [engine.test(0, model, loss_func, val_loader, args).__getitem__(1)]
    convergence_time = time.time()
    for epoch in range(1, args.epochs + 1):
        engine.train(
            epoch,
            model,
            optimizer,
            preconditioner,
            loss_func,
            train_sampler,
            train_loader,
            args,
        )
        val_loss, val_acc = engine.test(epoch, model, loss_func, val_loader, args)
        loss_per_epoch.append(val_acc)
        lr_scheduler.step()

        if kfac_scheduler is not None:
            kfac_scheduler.step(step=epoch)
        
        best_acc = max(best_acc, val_acc) 
        if best_loss > val_loss:
            best_loss = val_loss
            no_improvement_counter = 0
        else:
            no_improvement_counter += 1

        if no_improvement_counter >= patience_epochs and convergence_epoch == 0:
            convergence_epoch = epoch
            convergence_time = time.time() - start
    if args.verbose:
        print(
            '\nTraining time: {}'.format(
                datetime.timedelta(seconds=time.time() - start),
            ),
        )
        print(
            '\nAccuracy: {}'.format(
                best_acc
            ),
        )
        print(
            '\nConvergence: {}'.format(
                convergence_epoch
            ),
        )
        print(
            '\nConvergence Time: {}'.format(
                datetime.timedelta(seconds=convergence_time),
            ),
        )

    return loss_per_epoch, convergence_epoch
def test_model(model_name):
    # Evaluating different models
    
    # SGD
    # eval_experiment(model, learning_mode=optimizers.LearningMode.SGD, convergence_criteria_enabled=False)
    # KFAC
    loss_kfac, convergence_epoch_kfac = eval_experiment(
        model_name,
        learning_mode=optimizers.LearningMode.KFAC_WITH_ADAM,
    )
    # ADAM
    loss_adam, convergence_epoch_adam = eval_experiment(
        model_name,
        learning_mode=optimizers.LearningMode.ADAMS,
    )
    # plot figures to compare convergence
    epochs = np.array(list(range(0, args.epochs + 1)))
    epochs_ = np.linspace(0, args.epochs, args.epochs*5)
    
    
    loss_kfac = np.array(loss_kfac)
    epochs_loss_spline = make_interp_spline(epochs, loss_kfac)
    loss_kfac_ =  epochs_loss_spline(epochs_)

    loss_adam = np.array(loss_adam)
    epochs_loss_spline = make_interp_spline(epochs, loss_adam)
    loss_adam_ =  epochs_loss_spline(epochs_)
    
    #savetxt('model_data.csv', data, delimiter=',')
    
    plt.plot(epochs_, loss_adam_, color="orange", label = "Adam w/o K-FAC")
    plt.plot(epochs_, loss_kfac_, color="blue", label="Adam with K-FAC")
    # plt.axvline(x=convergence_epoch_adam, color = "orange", linestyle='dashed', label = "Adam w/o K-FAC Convergence Epoch")
    # plt.axvline(x=convergence_epoch_kfac, color = "blue", linestyle='dashed', label = "Adam with K-FAC Convergence Epoch")
    
    plt.title(model_name)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    
    plt.savefig(model_name + '.png')
    plt.clf()

if __name__ == '__main__':
    # Initialize Distributed training
    torch.distributed.init_process_group(
        backend='nccl',
        init_method='env://',
    )
    args = parse_args()
    test_model(args.model_name)
    
    # test_model("vgg11")
