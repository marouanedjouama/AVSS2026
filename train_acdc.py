# !/usr/bin/env python3

"""Main training and evaluation script for ACDC segmentation
using Hourglass Transformer + Rectified Flow.
"""

# TODO when resuming training, it start from the beginning of the dataloader and drops the rest of the current epoch. 
# It would be better to save the dataloaders state and resume from the exact same point.

import argparse
from contextlib import contextmanager
from datetime import datetime
import json
import math
import os
from pathlib import Path
import time
import random
import numpy as np
import re

import accelerate
import torch
import torch._dynamo
from torch import distributed as dist
from torch import multiprocessing as mp
from torch import optim
from torch.utils import data
from torchvision import utils as tv_utils
from tqdm.auto import tqdm
from torch.utils.data import RandomSampler

from ema import ExponentialMovingAverage

from hourglass.image_transformer_base import (
    ImageTransformerDenoiserModelV2,
    LevelSpec,
    MappingSpec,
    GlobalAttentionSpec,
    NeighborhoodAttentionSpec,
)
from hourglass.image_transformer_main import ImageTransformerDenoiserModelV3
from hourglass.image_transformer_noLerp import ImageTransformerDenoiserModelV3_noLerp
                                                
from hourglass.flags import checkpointing as checkpointing_ctx
from hourglass import flops as model_flops
from lr_scheduler import ConstantLRWithWarmup, LinearWarmupCosineAnnealingLR
from utils.metric import dice, hausdorff_distance_95
from rectified_flow import RectifiedFlow
from sampling import euler_sample
from dataloaders.loader_ACDC import ACDCPreprocessed, get_train_augmentations, get_val_augmentations


CLASS_DICE_THRESH = [0.5, 0.5, 0.5]

def worker_init_fn(worker_id):
    """Initialize worker with unique seed for reproducibility."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def ensure_distributed():
    if not dist.is_initialized():
        dist.init_process_group(world_size=1, rank=0, store=dist.HashStore())


def n_params(module):
    """Returns the number of trainable parameters in a module."""
    return sum(p.numel() for p in module.parameters())


def serialize_transforms(transform_pipeline):
    """Convert a transform pipeline to a JSON-serializable list of dictionaries."""
    if transform_pipeline is None:
        return []

    transforms_list = getattr(transform_pipeline, 'transforms', [transform_pipeline])
    serialized = []
    for transform in transforms_list:
        t_info = {'type': transform.__class__.__name__}
        try:
            args = {}
            for k, v in vars(transform).items():
                if not k.startswith('_'):
                    if isinstance(v, (int, float, str, bool, type(None))):
                        args[k] = v
                    elif isinstance(v, (list, tuple)):
                        args[k] = [str(item) if not isinstance(item, (int, float, str, bool, type(None))) else item for item in v]
                    elif isinstance(v, dict):
                        args[k] = {str(dk): (str(dv) if not isinstance(dv, (int, float, str, bool, type(None))) else dv) for dk, dv in v.items()}
                    else:
                        args[k] = str(v)
            t_info['args'] = args
        except Exception:
            pass
        serialized.append(t_info)
    return serialized


@contextmanager
def eval_mode(*models):
    """Context manager to temporarily set models to eval mode."""
    modes = []
    for model in models:
        modes.append(model.training)
        model.eval()
    try:
        yield
    finally:
        for model, mode in zip(models, modes):
            model.train(mode)


class CSVLogger:
    def __init__(self, filename, columns):
        self.filename = filename
        self.columns = columns
        self.file = open(filename, 'a')
        if Path(filename).stat().st_size == 0:
            self.file.write(','.join(columns) + '\n')
            self.file.flush()

    def write(self, *args):
        self.file.write(','.join(str(x) for x in args) + '\n')
        self.file.flush()


def load_config(path):
    """Load JSON config file."""
    with open(path) as f:
        return json.load(f)


def make_model(config):
    """Build the Hourglass Transformer model from config."""
    model_config = config['model']

    depths = model_config['depths']
    widths = model_config['widths']
    d_ffs = [w * 3 for w in widths]
    dropout_rate = model_config.get('dropout_rate', [0.0] * len(depths))
    if isinstance(dropout_rate, (int, float)):
        dropout_rate = [dropout_rate] * len(depths)


    self_attns_config = model_config.get('self_attns', None)
    self_attns = []
    if self_attns_config:
        for sa in self_attns_config:
            attn_type = sa['type']
            if attn_type in ('global', 'global_dualstream'):
                self_attns.append(GlobalAttentionSpec(
                    d_head=sa['d_head'],
                ))
            elif attn_type in ('neighborhood', 'neighborhood_dualstream'):
                self_attns.append(NeighborhoodAttentionSpec(
                    d_head=sa['d_head'],
                    kernel_size=sa['kernel_size'],
                ))
            else:
                raise ValueError(f"Unknown attention type: {attn_type}")
    else:
        raise ValueError("self_attns configuration is required in the config for this code.")

    levels = []
    for i in range(len(depths)):
        levels.append(LevelSpec(
            depth=depths[i],
            width=widths[i],
            d_ff=d_ffs[i],
            self_attn=self_attns[i],
            dropout=dropout_rate[i],
        ))
    mapping = MappingSpec(
        depth=model_config.get('mapping_depth', 2),
        width=model_config.get('mapping_width', widths[0]),
        d_ff=model_config.get('mapping_d_ff', widths[0] * 3),
        dropout=model_config.get('mapping_dropout', 0.0),
    )
    model_type = model_config.get('type', 'image_transformer_v2')
    model_type_norm = model_type.lower()
    model_kwargs = dict(
        levels=levels,
        mapping=mapping,
        in_channels=model_config['input_channels'],
        out_channels=model_config['output_channels'],
        patch_size=tuple(model_config['patch_size']),
    )
    if model_type_norm == 'image_transformer_base':
        model = ImageTransformerDenoiserModelV2(**model_kwargs)
    elif model_type_norm == 'image_transformer_main':
        model = ImageTransformerDenoiserModelV3(**model_kwargs)
    elif model_type_norm in (
        'image_transformer_noLerp',
    ):
        model = ImageTransformerDenoiserModelV3_noLerp(**model_kwargs)
    else:
        raise ValueError(f"Unsupported model.type: {model_type}")

    return model


def priority_overlay(one_hot, class_colors, priority_order):
    """Render multi-label segmentation as RGB with priority overlay.

    Args:
        one_hot: (C, H, W) multi-label tensor {0, 1}.
        class_colors: list of (R, G, B) in 0-255.
        priority_order: list of channel indices from LOW to HIGH priority.

    Returns:
        (3, H, W) RGB tensor in [0, 1].
    """
    C, H, W = one_hot.shape
    device = one_hot.device
    label_map = torch.zeros((H, W), dtype=torch.long, device=device)
    for cls in priority_order:
        mask = one_hot[cls] > 0.5
        label_map[mask] = cls + 1
    rgb = torch.zeros(3, H, W, device=device)
    for cls, color in enumerate(class_colors):
        mask = label_map == (cls + 1)
        for k in range(3):
            rgb[k][mask] = color[k] / 255.0
    return rgb


class EarlyStopping:
    """Early stops the training if validation metric doesn't improve after a given patience."""

    def __init__(self, patience=10, verbose=True, delta=0.0, save_path='checkpoints_acdc/best.pth'):
        """
        Args:
            patience: How many validation steps to wait after last improvement.
            verbose: Print messages.
            delta: Minimum improvement to qualify as an improvement.
            save_path: Where to save the best model.
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.save_path = save_path
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_metric, save_fn):
        """
        Args:
            val_metric: Current validation metric (higher is better).
            save_fn: Callable to save the checkpoint.
        """
        score = val_metric

        if self.best_score is None:
            self.best_score = score
            save_fn(self.save_path)
            if self.verbose:
                print(f"Validation metric: {score:.4f}. Saving best model to {self.save_path}")
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience} (best: {self.best_score:.4f})")
            if self.counter >= self.patience:
                if self.verbose:
                    print("Early stopping triggered!")
                self.early_stop = True
        else:
            if self.verbose:
                print(f"Validation metric improved: {self.best_score:.4f} -> {score:.4f}. Saving best model.")
            self.best_score = score
            save_fn(self.save_path)
            self.counter = 0


def convert_to_onehot(label, num_classes=4):
    """Convert class indices to one-hot encoding, excluding background (class 0).

    Args:
        label: (B, 1, H, W) tensor with class indices 0-3
        num_classes: Total number of classes including background

    Returns:
        (B, num_classes-1, H, W) one-hot tensor for classes 1 to num_classes-1
    """
    B, _, H, W = label.shape
    device = label.device
    onehot = torch.zeros(B, num_classes - 1, H, W, device=device, dtype=label.dtype)
    for c in range(1, num_classes):  # Skip background (0)
        onehot[:, c - 1] = (label[:, 0] == c).float()
    return onehot


def parse_volume_and_slice(npz_name):
    """Parse ACDC preprocessed filename into volume id and slice index.

    Expected pattern examples:
      patient001_ED_slice03.npz
      patient001_ES_slice12.npz
    """
    stem = Path(npz_name).stem
    match = re.match(r'^(?P<volume_id>.+)_slice(?P<slice_idx>\d+)$', stem)
    if match is None:
        return None
    return match.group('volume_id'), int(match.group('slice_idx'))


def build_volume_index(split_dir):
    """Build mapping: volume_id -> sorted list[(slice_idx, npz_path)]."""
    volume_index = {}
    for fname in sorted(os.listdir(split_dir)):
        if not fname.endswith('.npz'):
            continue
        parsed = parse_volume_and_slice(fname)
        if parsed is None:
            continue
        volume_id, slice_idx = parsed
        volume_index.setdefault(volume_id, []).append((slice_idx, os.path.join(split_dir, fname)))

    for volume_id in volume_index:
        volume_index[volume_id].sort(key=lambda x: x[0])

    return volume_index


def get_tta_transforms(mode='flip'):
    """Return forward/inverse transforms for test-time augmentation."""
    if mode == 'none':
        return [
            (lambda x: x, lambda y: y, 'identity'),
        ]

    if mode == 'flip':
        return [
            (lambda x: x, lambda y: y, 'identity'),
            (lambda x: torch.flip(x, dims=(-1,)), lambda y: torch.flip(y, dims=(-1,)), 'hflip'),
            (lambda x: torch.flip(x, dims=(-2,)), lambda y: torch.flip(y, dims=(-2,)), 'vflip'),
            (lambda x: torch.flip(x, dims=(-2, -1)), lambda y: torch.flip(y, dims=(-2, -1)), 'hvflip'),
        ]

    if mode == 'd4':
        return [
            (lambda x: x, lambda y: y, 'rot0'),
            (lambda x: torch.rot90(x, 1, dims=(-2, -1)), lambda y: torch.rot90(y, -1, dims=(-2, -1)), 'rot90'),
            (lambda x: torch.rot90(x, 2, dims=(-2, -1)), lambda y: torch.rot90(y, -2, dims=(-2, -1)), 'rot180'),
            (lambda x: torch.rot90(x, 3, dims=(-2, -1)), lambda y: torch.rot90(y, -3, dims=(-2, -1)), 'rot270'),
            (
                lambda x: torch.flip(x, dims=(-1,)),
                lambda y: torch.flip(y, dims=(-1,)),
                'hflip',
            ),
            (
                lambda x: torch.rot90(torch.flip(x, dims=(-1,)), 1, dims=(-2, -1)),
                lambda y: torch.flip(torch.rot90(y, -1, dims=(-2, -1)), dims=(-1,)),
                'hflip_rot90',
            ),
            (
                lambda x: torch.rot90(torch.flip(x, dims=(-1,)), 2, dims=(-2, -1)),
                lambda y: torch.flip(torch.rot90(y, -2, dims=(-2, -1)), dims=(-1,)),
                'hflip_rot180',
            ),
            (
                lambda x: torch.rot90(torch.flip(x, dims=(-1,)), 3, dims=(-2, -1)),
                lambda y: torch.flip(torch.rot90(y, -3, dims=(-2, -1)), dims=(-1,)),
                'hflip_rot270',
            ),
        ]

    raise ValueError(f'Unknown TTA mode: {mode}')


def main():
    p = argparse.ArgumentParser(
        description='Train Hourglass Transformer segmentation with Rectified Flow',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--batch-size', type=int, default=2,
                   help='the batch size')
    p.add_argument('--checkpointing', action='store_true',
                   help='enable gradient checkpointing')
    p.add_argument('--compile', action='store_true',
                   help='compile the model')
    p.add_argument('--config', type=str, required=True,
                   help='the configuration file')
    p.add_argument('--demo-every', type=int, default=None,
                   help='save a demo grid every this many steps (overrides config)')
    p.add_argument('--end-step', type=int, default=None,
                   help='the step to end training at (overrides config)')
    p.add_argument('--max-epochs', type=int, default=None,
                   help='the maximum number of epochs to train for (overrides config)')
    p.add_argument('--evaluate-every', type=int, default=None,
                   help='evaluate every this many steps (overrides config)')
    p.add_argument('--evaluate-n', type=int, default=None,
                   help='the number of samples to draw to evaluate')
    p.add_argument('--evaluate-only', action='store_true',
                   help='evaluate instead of training')
    p.add_argument('--grad-accum-steps', type=int, default=1,
                   help='the number of gradient accumulation steps')
    p.add_argument('--lr', type=float,
                   help='the learning rate (overrides config)')
    p.add_argument('--mixed-precision', type=str,
                   help='the mixed precision type')
    p.add_argument('--name', type=str, default='model',
                   help='the name of the run')
    p.add_argument('--num-workers', type=int, default=16,
                   help='the number of data loader workers')
    p.add_argument('--resume', type=str,
                   help='the checkpoint to resume from')
    p.add_argument('--sample-n', type=int, default=8,
                   help='the number of images to sample for demo grids')
    p.add_argument('--sample-steps', type=int, default=1,
                   help='the number of Euler steps for sampling')
    p.add_argument('--tta', action='store_true',
                   help='enable test-time augmentation for demo/evaluation sampling')
    p.add_argument('--tta-mode', type=str, default='flip', choices=['none', 'flip', 'd4'],
                   help='the TTA transform set to use when --tta is enabled')
    p.add_argument('--save-every', type=int, default=None,
                   help='save every this many steps (overrides config)')
    p.add_argument('--seed', type=int, default=42,
                   help='the random seed')
    p.add_argument('--n-ensample', type=int, default=1,
                   help='the number of ensemble samples for evaluation')
    p.add_argument('--start-method', type=str, default='spawn',
                   choices=['fork', 'forkserver', 'spawn'],
                   help='the multiprocessing start method')
    p.add_argument('--wandb-entity', type=str,
                   help='the wandb entity name')
    p.add_argument('--wandb-group', type=str,
                   help='the wandb group name')
    p.add_argument('--wandb-project', type=str,
                   help='the wandb project name (specify this to enable wandb)')
    p.add_argument('--use-early-stopping', action='store_true',
                   help='enable early stopping based on validation dice')
    p.add_argument('--patience', type=int, default=10,
                   help='early stopping patience (number of evaluations without improvement)')
    p.add_argument('--delta', type=float, default=0.005,
                   help='minimum improvement in validation metric to reset patience')
    p.add_argument('--eval-split', type=str, default='val', choices=['train', 'val', 'test'],)

    # find best thresholds only mode (for validation and test splits separately)
    p.add_argument('--find-best-thresh', action='store_true',
                   help='only find best thresholds for evaluation (no training)')
    p.add_argument('--thresh-start', type=float, default=0.1,
                   help='the start of the threshold range to search for best thresholds')
    p.add_argument('--thresh-end', type=float, default=0.9,
                   help='the end of the threshold range to search for best thresholds')
    args = p.parse_args()

    mp.set_start_method(args.start_method)
    torch.backends.cuda.matmul.allow_tf32 = True

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    try:
        torch._dynamo.config.automatic_dynamic_shapes = False
    except AttributeError:
        pass

    # Load config
    config = load_config(args.config)
    model_config = config['model']
    flow_config = config['flow']
    dataset_config = config['dataset']
    opt_config = config['optimizer']
    sched_config = config['lr_sched']
    ema_config = config['ema']
    train_config = config['training']
    num_classes = dataset_config.get('num_classes', 4)

    assert len(model_config['input_size']) == 2 and model_config['input_size'][0] == model_config['input_size'][1]
    size = model_config['input_size']


    # Accelerator
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.grad_accum_steps,
        mixed_precision=args.mixed_precision,
    )
    ensure_distributed()
    device = accelerator.device
    unwrap = accelerator.unwrap_model
    print(f'Process {accelerator.process_index} using device: {device}', flush=True)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print(f'World size: {accelerator.num_processes}', flush=True)
        print(f'Batch size: {args.batch_size * accelerator.num_processes}', flush=True)

    if args.seed is not None:
        seeds = torch.randint(-2 ** 63, 2 ** 63 - 1, [accelerator.num_processes],
                              generator=torch.Generator().manual_seed(args.seed))
        torch.manual_seed(seeds[accelerator.process_index])
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        accelerate.utils.set_seed(args.seed)
        from monai.utils import set_determinism
        set_determinism(seed=args.seed)


    sampler_gen = torch.Generator().manual_seed(seeds[accelerator.process_index].item())
    dl_gen = torch.Generator().manual_seed(seeds[accelerator.process_index].item() + 1)
    elapsed = 0.0

    # Build model
    inner_model = make_model(config)

    if args.compile:
        inner_model.compile()

    if accelerator.is_main_process:
        model_param_count = n_params(inner_model)
        print(f'Parameters: {model_param_count:,}')

    # Optimizer
    lr = args.lr if args.lr is not None else opt_config['lr']
    groups = inner_model.param_groups(lr)
    if opt_config['type'] == 'adamw':
        opt = optim.AdamW(groups,
                          lr=lr,
                          betas=tuple(opt_config['betas']),
                          eps=opt_config['eps'],
                          weight_decay=opt_config['weight_decay'],
                          fused=True)
    else:
        raise ValueError(f'Invalid optimizer type: {opt_config["type"]}')


    train_split_dir = dataset_config['train_path']
    val_split_dir = dataset_config.get('val_path', dataset_config.get('test_path'))
    test_split_dir = dataset_config.get('test_path', .0)

    if val_split_dir is None:
        raise ValueError('dataset.val_path or dataset.test_path must be set in config.')

    volume_index_by_split = {
        'train': build_volume_index(train_split_dir),
        'val': build_volume_index(val_split_dir),
        'test': build_volume_index(test_split_dir),
    }

    if accelerator.is_main_process:
        print(
            f"Volumes - train: {len(volume_index_by_split['train'])}, "
            f"val: {len(volume_index_by_split['val'])}, test: {len(volume_index_by_split['test'])}"
        )

    # Dataset
    train_dataset = ACDCPreprocessed(
        data_dir=train_split_dir,
        transform=get_train_augmentations(),
    )
    val_dataset = ACDCPreprocessed(
        data_dir=val_split_dir,
        transform=get_val_augmentations(),
    )

    # test_dataset = ACDCPreprocessed(
    #     data_dir=test_split_dir,
    #     transform=get_val_augmentations(),
    # )

    train_transforms_logged = serialize_transforms(getattr(train_dataset, 'transform', None))
    test_transforms_logged = serialize_transforms(getattr(val_dataset, 'transform', None))


    if accelerator.is_main_process:
        print(f'Number of items in train_dataset: {len(train_dataset):,}')
        print(f'Number of items in val_dataset: {len(val_dataset):,}')


    train_dl = data.DataLoader(
        train_dataset, args.batch_size, shuffle=True, prefetch_factor=2,
        num_workers=args.num_workers, persistent_workers=args.num_workers > 0,
        pin_memory=True, generator=dl_gen, worker_init_fn=worker_init_fn
    )

    val_sampler = RandomSampler(val_dataset, replacement=False, generator=sampler_gen)
    val_dl = data.DataLoader(
        val_dataset, args.batch_size, shuffle=False,
        num_workers=args.num_workers, persistent_workers=args.num_workers > 0, pin_memory=True,
        sampler=val_sampler, generator=dl_gen, worker_init_fn=worker_init_fn
    )

    # Resolve overrides
    save_every = args.save_every or train_config['save_every']
    eval_every = args.evaluate_every or train_config['eval_every']
    demo_every = args.demo_every or train_config.get('demo_every', eval_every)

    # max epochs if provided will override end_step
    if args.max_epochs is not None:
        max_epochs = args.max_epochs
        end_step = max_epochs * len(train_dl)
    
    else:
        end_step = args.end_step or train_config['max_steps']
        max_epochs = math.ceil(end_step / len(train_dl))

    sched_total_steps = max(1, math.ceil(end_step / args.grad_accum_steps))

    # LR scheduler
    warmup_steps = int(sched_total_steps * sched_config['warmup'])
    if sched_config['type'] == 'constant':
        sched = ConstantLRWithWarmup(opt, warmup_steps=warmup_steps)
    elif sched_config['type'] == 'cosine':
        sched = LinearWarmupCosineAnnealingLR(
            opt,
            warmup_epochs=warmup_steps,
            max_epochs=sched_total_steps,
            warmup_start_lr= lr / 10,
            eta_min=1e-6,
        )
    else:
        raise ValueError(f'Invalid schedule type: {sched_config["type"]}')

    ema_loss_stats = {}

    # Rectified flow
    flow = RectifiedFlow(
        eps=flow_config['eps'],
        time_sampling=flow_config.get('time_sampling', 'uniform'),
    )

    inner_model, opt, train_dl, val_dl, sched = accelerator.prepare(inner_model, opt, train_dl, val_dl, sched)

    ema = ExponentialMovingAverage(unwrap(inner_model).parameters(), decay=ema_config['decay'])

    # Flop counting
    with torch.no_grad(), model_flops.flop_counter() as fc:
        t_dummy = torch.tensor([0.5], device=device)

        in_channels_seg = model_config['output_channels']
        in_channels_img = model_config['input_channels'] - model_config['output_channels']
        x_seg = torch.zeros([1, in_channels_seg, size[0], size[1]], device=device)
        cond_img = torch.zeros([1, in_channels_img, size[0], size[1]], device=device)
        inner_model(x_seg, t_dummy, cond_img=cond_img)

        if accelerator.is_main_process:
            print(f"Forward pass GFLOPs: {fc.flops / 1_000_000_000:,.3f}", flush=True)

    # WandB
    use_wandb = accelerator.is_main_process and args.wandb_project
    if use_wandb:
        import wandbs
        wandb.login(key="put_your_wandb_api_key_here")  # Replace with your actual WandB API key
        wandb.init(
            project=args.wandb_project,
            name=f"rf-hourglass-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            save_code=True,
            config={
                "batch_size": args.batch_size,
                "grad_accum_steps": args.grad_accum_steps,
                "learning_rate": lr,
                "weight_decay": opt_config.get('weight_decay', 0.),
                "input_size": model_config['input_size'],
                "len_train_dataset": len(train_dataset),
                "len_val_dataset": len(val_dataset),
                "checkpointing": args.checkpointing,
                "max_epochs": max_epochs,
                "end_step": end_step,
                "parameters": model_param_count,
                "lr_scheduler": sched_config['type'],
                "lr_scheduler_warmup": sched_config.get('warmup', 0),
                "flow_eps": flow_config['eps'],
                "time_sampling": flow_config.get('time_sampling', 'uniform'),
                "sample_steps": args.sample_steps,
                "tta_enabled": args.tta,
                "tta_mode": args.tta_mode,
                "ema_decay": ema_config['decay'],
                "dropout_rate": model_config['dropout_rate'],
                "patch_size": model_config['patch_size'],
                "depths": model_config['depths'],
                "widths": model_config['widths'],
                "use_early_stopping": args.use_early_stopping,
                "early_stopping_patience": args.patience if args.use_early_stopping else None,
                "early_stopping_delta": args.delta if args.use_early_stopping else None,
                "train_transforms": train_transforms_logged,
                "test_transforms": test_transforms_logged,
            },
        )
        wandb.run.summary['train_transforms'] = json.dumps(train_transforms_logged, indent=2)
        wandb.run.summary['test_transforms'] = json.dumps(test_transforms_logged, indent=2)
        wandb.watch(inner_model)

    n_img_channels = 1  # ACDC has single-channel cardiac MRI
    n_seg_channels = 3  # RV, Myo, LV (excluding background)
    seg_class_names = ['RV', 'Myo', 'LV']

    # Checkpoint state
    state_path = Path(f'states_acdc/{args.name}_state.json')
    state_path.parent.mkdir(parents=True, exist_ok=True)

    if state_path.exists() or args.resume:
        if args.resume:
            ckpt_path = args.resume
        else:
            state = json.load(open(state_path))
            ckpt_path = state['latest_checkpoint']
        if accelerator.is_main_process:
            print(f'Resuming from {ckpt_path}...')
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        unwrap(inner_model).load_state_dict(ckpt['model'])
        opt.load_state_dict(ckpt['opt'])

        sched.load_state_dict(ckpt['sched'])
        
        if 'ema' in ckpt:
            ema.load_state_dict(ckpt['ema'], device=device)
        ema_loss_stats = ckpt.get('ema_loss_stats', {})
        epoch = ckpt['epoch'] + 1
        step = ckpt['step'] + 1
        if 'sampler_gen' in ckpt:
            sampler_gen.set_state(ckpt['sampler_gen'])
            dl_gen.set_state(ckpt['dl_gen'])
        elif 'demo_gen' in ckpt:
            sampler_gen.set_state(ckpt['demo_gen'])
        elapsed = ckpt.get('elapsed', 0.0)
        del ckpt
    else:
        epoch = 0
        step = 0

    # Metrics logging
    evaluate_enabled = eval_every > 0 and (args.evaluate_n is None or args.evaluate_n > 0)
    metrics_log = None
    if evaluate_enabled and accelerator.is_main_process:
        Path('metrics_acdc').mkdir(exist_ok=True)
        metrics_log = CSVLogger(
            f'metrics_acdc/{args.name}_metrics.csv',
            ['step', 'time', 'loss', 'mean_dice'] + [f'dice_{name}' for name in seg_class_names],
        )

    # --- Helper functions ---

    @contextmanager
    def inference_model_scope(use_ema=True):
        """Prepare model for inference and optionally swap in EMA weights once."""
        model_to_use = unwrap(inner_model)
        was_training = model_to_use.training

        if use_ema:
            ema.store(model_to_use.parameters())
            ema.copy_to(model_to_use.parameters())

        model_to_use.eval()
        try:
            yield model_to_use
        finally:
            if use_ema:
                ema.restore(model_to_use.parameters())
            model_to_use.train(was_training)

    @torch.no_grad()
    def sample_segmentation(cond_img, model_to_use, n_steps=None, use_tta=False):
        """Sample segmentation from noise using Euler ODE solver."""
        if n_steps is None:
            n_steps = args.sample_steps
        shape = (cond_img.shape[0], n_seg_channels, size[0], size[1])

        if use_tta:
            preds = []
            for forward_transform, inverse_transform, _ in get_tta_transforms(args.tta_mode):
                cond_aug = forward_transform(cond_img)
                pred_aug = euler_sample(model_to_use, cond_aug, shape, device, N=n_steps, eps=flow.eps)
                preds.append(inverse_transform(pred_aug))
            return torch.stack(preds, dim=0).mean(dim=0)

        return euler_sample(model_to_use, cond_img, shape, device, N=n_steps, eps=flow.eps)

    def threshold_predictions(pred_seg):
        """Apply per-class thresholds to get binary predictions."""
        pred_binary_tc = (pred_seg[:, 0:1] > CLASS_DICE_THRESH[0]).float()
        pred_binary_wt = (pred_seg[:, 1:2] > CLASS_DICE_THRESH[1]).float()
        pred_binary_et = (pred_seg[:, 2:3] > CLASS_DICE_THRESH[2]).float()
        return torch.cat([pred_binary_tc, pred_binary_wt, pred_binary_et], dim=1)

    # Demo colors and priority for ACDC
    class_colors = [
        (255, 0, 0),    # Red for RV (Right Ventricle)
        (0, 255, 0),    # Green for Myo (Myocardium)
        (0, 0, 255),    # Blue for LV (Left Ventricle)
    ]
    priority = [0, 1, 2]  # RV < Myo < LV (LV on top)


    @torch.no_grad()
    def demo(split='val'):
        """Generate demo visualization grid."""
        max_demo_vis = 10
        if accelerator.is_main_process:
            tqdm.write('Running segmentation demo...')

        Path('demos_acdc').mkdir(exist_ok=True)
        filename = f'demos_acdc/{args.name}_demo_{split}_{step:08}.png'

        if split == 'train':
            demo_batch = next(iter(train_dl))
        else:
            demo_batch = next(iter(val_dl))

        # ACDC returns (image, label) tuple
        cond_img_demo, label_demo = demo_batch
        gt_seg_demo = convert_to_onehot(label_demo, num_classes=num_classes)
        n_samples = min(args.sample_n, cond_img_demo.shape[0])
        cond_img_demo = cond_img_demo[:n_samples]
        gt_seg_demo = gt_seg_demo[:n_samples]

        with inference_model_scope(use_ema=True) as model_to_use:
            pred_seg = sample_segmentation(cond_img_demo, model_to_use=model_to_use, use_tta=args.tta)
        pred_seg_binary = threshold_predictions(pred_seg)

        # Compute dice per class
        tc_dice = dice(pred_seg_binary[:, 0:1], gt_seg_demo[:, 0:1])
        wt_dice = dice(pred_seg_binary[:, 1:2], gt_seg_demo[:, 1:2])
        et_dice = dice(pred_seg_binary[:, 2:3], gt_seg_demo[:, 2:3])
        mean_dice = (tc_dice + wt_dice + et_dice) / 3

        if accelerator.is_main_process:
            class_dice_str = ', '.join([
                f'{seg_class_names[c]}: {d:.4f}'
                for c, d in enumerate([tc_dice, wt_dice, et_dice])
            ])
            tqdm.write(f'{split} Demo Dice - Mean: {mean_dice:.4f}, {class_dice_str}')

            vis_images = []
            for i in range(min(n_samples, max_demo_vis)):
                # Input MRI (first modality)
                img_slice = cond_img_demo[i, 0:1]
                vis_images.append(img_slice.repeat(3, 1, 1))
                # GT overlay
                gt_vis = priority_overlay(gt_seg_demo[i], class_colors, priority)
                vis_images.append(gt_vis)
                # Prediction overlay
                pred_vis = priority_overlay(pred_seg_binary[i], class_colors, priority)
                vis_images.append(pred_vis)

            grid = tv_utils.make_grid(torch.stack(vis_images), nrow=3, padding=2)

            from PIL import Image, ImageDraw, ImageFont
            # Convert grid to PIL
            grid_np = grid.clamp(0, 1).mul(255).byte().cpu().permute(1, 2, 0).numpy()
            grid_pil = Image.fromarray(grid_np)

            # Legend
            legend_height = 60
            legend_img = Image.new('RGB', (grid_pil.width, legend_height), color=(255, 255, 255))
            draw = ImageDraw.Draw(legend_img)
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
            except Exception:
                font = ImageFont.load_default()

            x_offset = 20
            box_size = 30
            spacing = grid_pil.width // len(seg_class_names)
            for idx, (name, color) in enumerate(zip(seg_class_names, class_colors)):
                x_pos = x_offset + idx * spacing
                y_pos = (legend_height - box_size) // 2
                draw.rectangle([x_pos, y_pos, x_pos + box_size, y_pos + box_size],
                               fill=color, outline=(0, 0, 0), width=2)
                draw.text((x_pos + box_size + 10, y_pos + box_size // 2 - 8),
                          name, fill=(0, 0, 0), font=font)

            final_img = Image.new('RGB', (grid_pil.width, grid_pil.height + legend_height))
            final_img.paste(grid_pil, (0, 0))
            final_img.paste(legend_img, (0, grid_pil.height))
            final_img.save(filename)

            if use_wandb:
                import wandb
                wandb.log({'demo_grid': wandb.Image(filename), 'demo_mean_dice': mean_dice}, step=step)

    @torch.no_grad()
    def evaluate(n_ensample=1, n_cases=None, split='val'):
        """Evaluate per ACDC volume by stacking all slices per patient-phase.

        Each volume is identified from the filename prefix before `_sliceXX`
        (for example `patient001_ED` and `patient001_ES`).
        """
        if not accelerator.is_main_process:
            return None

        if split not in volume_index_by_split:
            raise ValueError(f'Unknown split: {split}. Expected one of {list(volume_index_by_split.keys())}.')

        volume_index = volume_index_by_split[split]
        all_volume_ids = sorted(volume_index.keys())

        if len(all_volume_ids) == 0:
            tqdm.write('No test cases found for volume-level evaluation.')
            return None

        selected_volume_ids = all_volume_ids if n_cases is None else all_volume_ids[:min(n_cases, len(all_volume_ids))]
        tqdm.write(f'Evaluating volume-level metrics on {len(selected_volume_ids)} {split} volumes...')

        case_dice_scores = []
        case_hd_scores = []

        with inference_model_scope(use_ema=True) as model_to_use:
            for volume_id in tqdm(selected_volume_ids, desc=f'Volume-level {split} eval'):
                pred_slices = []
                gt_slices = []

                for _, npz_path in volume_index[volume_id]:
                    data_npz = np.load(npz_path)
                    image_np = data_npz['image']
                    label_np = data_npz['label']

                    cond_img_case = torch.from_numpy(image_np).unsqueeze(0).unsqueeze(0).float().to(device, non_blocking=True)
                    gt_seg_case = torch.from_numpy(label_np).unsqueeze(0).unsqueeze(0).float()
                    gt_seg_case = convert_to_onehot(gt_seg_case, num_classes=num_classes).cpu()

                    if n_ensample <= 1:
                        pred_seg_case = sample_segmentation(cond_img_case, model_to_use=model_to_use, use_tta=args.tta)
                    else:
                        pred_ens = []
                        for _ in range(n_ensample):
                            pred_ens.append(sample_segmentation(cond_img_case, model_to_use=model_to_use, use_tta=args.tta))
                        pred_seg_case = torch.stack(pred_ens).mean(dim=0)

                    pred_seg_binary_case = threshold_predictions(pred_seg_case).cpu()

                    pred_slices.append(pred_seg_binary_case[0])
                    gt_slices.append(gt_seg_case[0])

                pred_volume = torch.stack(pred_slices, dim=1).numpy()  # (C, D, H, W)
                gt_volume = torch.stack(gt_slices, dim=1).numpy()      # (C, D, H, W)

                case_dice = []
                case_hd = []
                for c in range(n_seg_channels):
                    case_dice.append(dice(pred_volume[c], gt_volume[c]))
                    case_hd.append(hausdorff_distance_95(pred_volume[c], gt_volume[c]))

                case_dice_scores.append(case_dice)
                case_hd_scores.append(case_hd)

        case_dice_scores = np.asarray(case_dice_scores, dtype=np.float32)
        case_hd_scores = np.asarray(case_hd_scores, dtype=np.float32)

        mean_dice_per_class = np.nanmean(case_dice_scores, axis=0)
        mean_hd_per_class = np.nanmean(case_hd_scores, axis=0)

        mean_dice = float(np.nanmean(mean_dice_per_class))
        mean_hd = float(np.nanmean(mean_hd_per_class))

        class_dice_str = ', '.join([
            f'{seg_class_names[c]}: {mean_dice_per_class[c]:.4f}' for c in range(n_seg_channels)
        ])
        class_hd_str = ', '.join([
            f'{seg_class_names[c]}: {mean_hd_per_class[c]:.4f}' for c in range(n_seg_channels)
        ])

        tqdm.write(f'{split} Volume Eval Dice - Mean: {mean_dice:.4f}, {class_dice_str}')
        tqdm.write(f'{split} Volume Eval HD95 - Mean: {mean_hd:.4f}, {class_hd_str}')

        if use_wandb:
            import wandb
            log_dict = {
                f'{split}_volume_eval_mean_dice': mean_dice,
                f'{split}_volume_eval_mean_hd95': mean_hd,
            }
            for c in range(n_seg_channels):
                log_dict[f'{split}_volume_eval_dice_{seg_class_names[c]}'] = float(mean_dice_per_class[c])
                log_dict[f'{split}_volume_eval_hd95_{seg_class_names[c]}'] = float(mean_hd_per_class[c])
            wandb.log(log_dict, step=step)

        if metrics_log is not None:
            metrics_log.write(step, elapsed, ema_loss_stats.get('loss', 0), mean_dice,
                              *mean_dice_per_class)

        return mean_dice
    
    @torch.no_grad()
    def find_best_thresholds(split=None, n_ensample=1):
        """Find best per-class thresholds from global Dice over all volumes.

        Workflow:
        1) Predict and cache all 3D volumes for all cases in the split.
        2) Grid-search thresholds per class using all cases together (global).

        Supported splits: `val`, `test`, or `None` (run both).
        """
        if not accelerator.is_main_process:
            return None

        valid_splits = ('val', 'test')
        if split is None:
            splits_to_run = list(valid_splits)
        elif isinstance(split, str):
            if split not in valid_splits:
                raise ValueError(f'Unsupported split for threshold search: {split}. Expected one of {valid_splits}.')
            splits_to_run = [split]
        else:
            raise ValueError(f'Invalid split argument type: {type(split)}. Expected str or None.')

        # RF outputs are not constrained logits/probabilities, so keep a broad search range.
        thresholds = np.linspace(args.thresh_start, args.thresh_end, 100, dtype=np.float32)
        n_thresholds = len(thresholds)
        results = {}

        with inference_model_scope(use_ema=True) as model_to_use:
            for split_name in splits_to_run:
                volume_index = volume_index_by_split[split_name]
                all_volume_ids = sorted(volume_index.keys())

                if len(all_volume_ids) == 0:
                    tqdm.write(f'No cases found in split={split_name} for threshold search.')
                    results[split_name] = None
                    continue

                selected_volume_ids = all_volume_ids
                tqdm.write(
                    f'Finding best thresholds on {len(selected_volume_ids)} {split_name} cases '
                    '(volume-level)...'
                )

                # Stage 1: predict all 3D volumes and cache them.
                cached_cases = []

                for volume_id in tqdm(selected_volume_ids, desc=f'Threshold search {split_name}'):
                    prob_slices = []
                    gt_slices = []

                    for _, npz_path in volume_index[volume_id]:
                        data_npz = np.load(npz_path)
                        image_np = data_npz['image']
                        label_np = data_npz['label']

                        cond_img_case = torch.from_numpy(image_np).unsqueeze(0).unsqueeze(0).float().to(
                            device, non_blocking=True
                        )
                        gt_seg_case = torch.from_numpy(label_np).unsqueeze(0).unsqueeze(0).float()
                        gt_seg_case = convert_to_onehot(gt_seg_case, num_classes=num_classes).cpu()

                        if n_ensample <= 1:
                            pred_seg_case = sample_segmentation(cond_img_case, model_to_use=model_to_use, use_tta=args.tta)
                        else:
                            pred_ens = []
                            for _ in range(n_ensample):
                                pred_ens.append(sample_segmentation(cond_img_case, model_to_use=model_to_use, use_tta=args.tta))
                            pred_seg_case = torch.stack(pred_ens).mean(dim=0)

                        prob_slices.append(pred_seg_case[0].cpu())
                        gt_slices.append(gt_seg_case[0])

                    prob_volume = torch.stack(prob_slices, dim=1).numpy()  # (C, D, H, W)

                    gt_volume = torch.stack(gt_slices, dim=1).numpy()      # (C, D, H, W)

                    cached_cases.append((prob_volume.astype(np.float16), gt_volume.astype(np.uint8)))

                # Stage 2: threshold search on all cases using mean per-case Dice.
                mean_case_dice_grid = np.full((n_seg_channels, n_thresholds), np.nan, dtype=np.float32)

                for c in range(n_seg_channels):
                    for t_idx, t in enumerate(thresholds):
                        case_dices = []
                        for prob_volume, gt_volume in cached_cases:
                            pred_c = (prob_volume[c] > t).astype(np.float32)
                            target_c = gt_volume[c].astype(np.float32)
                            d = dice(pred_c, target_c)
                            if not np.isnan(d):
                                case_dices.append(float(d))

                        if len(case_dices) > 0:
                            mean_case_dice_grid[c, t_idx] = float(np.mean(case_dices))

                best_t = torch.zeros(n_seg_channels)
                best_dice = torch.zeros(n_seg_channels)
                for c in range(n_seg_channels):
                    if np.all(np.isnan(mean_case_dice_grid[c])):
                        continue
                    best_idx = int(np.nanargmax(mean_case_dice_grid[c]))
                    best_t[c] = float(thresholds[best_idx])
                    best_dice[c] = float(mean_case_dice_grid[c, best_idx])

                tqdm.write(
                    f'Best thresholds ({split_name}): {best_t.tolist()}, '
                    f'mean case dice: {best_dice.tolist()}'
                )
                results[split_name] = (best_t, best_dice)

        if len(splits_to_run) == 1:
            return results[splits_to_run[0]]
        return results

    def save(save_path=None):
        """Save checkpoint."""
        accelerator.wait_for_everyone()
        Path('checkpoints_acdc').mkdir(exist_ok=True)
        if save_path is None:
            filename = f'checkpoints_acdc/{args.name}_{step:08}.pth'
        else:
            filename = save_path
        if accelerator.is_main_process:
            tqdm.write(f'Saving to {filename}...')
        obj = {
            'config': config,
            'model': unwrap(inner_model).state_dict(),
            'opt': opt.state_dict(),
            'sched': sched.state_dict(),
            'ema': ema.state_dict(),
            'ema_loss_stats': ema_loss_stats,
            'epoch': epoch,
            'step': step,
            'sampler_gen': sampler_gen.get_state(),
            'dl_gen': dl_gen.get_state(),
            'elapsed': elapsed,
        }
        accelerator.save(obj, filename)
        if accelerator.is_main_process:
            state_obj = {'latest_checkpoint': filename}
            json.dump(state_obj, open(state_path, 'w'))


    if args.find_best_thresh:
        find_best_thresholds(split=args.eval_split, n_ensample=args.n_ensample)
        return

    # --- Evaluate only mode ---
    if args.evaluate_only:
        evaluate(split=args.eval_split, n_ensample=args.n_ensample, n_cases=args.evaluate_n)
        return

    # --- Early stopping setup ---
    early_stopper = None
    if args.use_early_stopping:
        early_stopper = EarlyStopping(
            patience=args.patience,
            verbose=accelerator.is_main_process,
            delta=args.delta,
            save_path=f'checkpoints_acdc/{args.name}_best.pth',
        )
        if accelerator.is_main_process:
            print(f'Early stopping enabled: patience={args.patience}, delta={args.delta}')

    # --- Training loop ---
    losses_since_last_print = []

    print(f'==== Starting training for up to {max_epochs} epochs ( {end_step} steps) ====')


    try:
        while True:
            for batch in tqdm(train_dl, smoothing=0.1, disable=not accelerator.is_main_process):

                if device.type == 'cuda':
                    start_timer = torch.cuda.Event(enable_timing=True)
                    end_timer = torch.cuda.Event(enable_timing=True)
                    torch.cuda.synchronize()
                    start_timer.record()
                else:
                    start_timer = time.time()

                with accelerator.accumulate(inner_model):
                    cond_img, label = batch
                    x_1 = convert_to_onehot(label, num_classes=num_classes)

                    with checkpointing_ctx(args.checkpointing):
                        loss = flow.loss(inner_model, x_1, cond_img)

                    loss_val = accelerator.gather(loss.detach()).mean().item()
                    losses_since_last_print.append(loss_val)

                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(inner_model.parameters(),
                                                     train_config['grad_clip_norm'])
                    opt.step()
                    sched.step()
                    opt.zero_grad()

                    # Update EMA
                    ema.update(unwrap(inner_model).parameters())

                    # EMA loss tracking
                    ema_decay_loss = 0.99
                    if 'loss' not in ema_loss_stats:
                        ema_loss_stats['loss'] = loss_val
                    else:
                        ema_loss_stats['loss'] = ema_loss_stats['loss'] * ema_decay_loss + loss_val * (1 - ema_decay_loss)

                if device.type == 'cuda':
                    end_timer.record()
                    torch.cuda.synchronize()
                    elapsed += start_timer.elapsed_time(end_timer) / 1000
                else:
                    elapsed += time.time() - start_timer

                if step % 50 == 0:
                    loss_disp = sum(losses_since_last_print) / len(losses_since_last_print)
                    losses_since_last_print.clear()
                    avg_loss = ema_loss_stats.get('loss', loss_disp)
                    if accelerator.is_main_process:
                        tqdm.write(f'Epoch: {epoch}, step: {step}, loss: {loss_disp:g}, avg loss: {avg_loss:.4g}')

                if use_wandb and step % 10 == 0:
                    import wandb
                    wandb.log({
                        'epoch': epoch,
                        'loss': loss_val,
                        'lr': sched.get_last_lr()[0],
                    }, step=step)

                step += 1

                if step > 0 and step % demo_every == 0:
                    demo("val")

                if evaluate_enabled and step > 0 and step % eval_every == 0:
                    val_dice = None
                    val_dice_dict = evaluate(split=args.eval_split, n_ensample=args.n_ensample, n_cases=args.evaluate_n)
                    if val_dice_dict is not None:
                        val_dice = val_dice_dict if isinstance(val_dice_dict, float) else val_dice_dict['mean_dice']

                    # Early stopping check
                    if args.use_early_stopping and val_dice is not None:
                        early_stopper(val_dice, save)
                        if early_stopper.early_stop:
                            if accelerator.is_main_process:
                                tqdm.write(f'Early stopping at step {step}. Best dice: {early_stopper.best_score:.4f}')
                            save()  # Save last checkpoint
                            return

                # Save based on steps only if early stopping is disabled
                if not args.use_early_stopping and step > 0 and step % save_every == 0:
                    save()

                if step >= end_step:
                    if accelerator.is_main_process:
                        tqdm.write('Done!')
                    save()
                    return

            print(f'Epoch {epoch} complete...')
            epoch += 1
    except KeyboardInterrupt:
        if accelerator.is_main_process:
            print('\nInterrupted, saving...')
            save()


if __name__ == '__main__':
    main()