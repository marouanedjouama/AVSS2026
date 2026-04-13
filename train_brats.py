# !/usr/bin/env python3

"""Main training and evaluation script for BraTS 2D segmentation
using Hourglass Transformer + Rectified Flow.
"""

# python train_brats.py --config configs/config_mmDiT_brats20.json --brats 2020 --name Run_1 --batch-size 16 --grad-accum-steps 2 --max-epochs 300 --wandb-project AVSS2026-brats2020 
# --sample-steps 5 --evaluate-n 5 --evaluate-every 1000 --demo-every 1000 --save-every 5000 --compile --checkpointing

# TODO when resuming training, it start from the begining of the dataloader and drops the rest of the current epoch. 
# It would be better to save the dataloader state and resume from the exact same point.

# TODO remove your wandb key later


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
import nibabel as nib

import accelerate
import torch
import torch._dynamo
from torch import distributed as dist
from torch import multiprocessing as mp
from torch import optim
from torch.utils import data
from torchvision import utils as tv_utils
from tqdm.auto import tqdm
from torch.utils.data import RandomSampler, WeightedRandomSampler

from dataloaders.bratsDataset_v2 import BraTSDataset2D
from dataloaders.bratsDataset21 import BraTSDataset21

from ema import ExponentialMovingAverage

from hourglass.image_transformer_v2 import (
    ImageTransformerDenoiserModelV2,
    LevelSpec,
    MappingSpec,
    GlobalAttentionSpec,
    NeighborhoodAttentionSpec,
)

from hourglass.flags import checkpointing as checkpointing_ctx
from hourglass import flops as model_flops
from lr_scheduler import ConstantLRWithWarmup, LinearWarmupCosineAnnealingLR
from utils.metric import dice, hausdorff_distance_95
from rectified_flow import RectifiedFlow
from sampling import euler_sample, rk45_sample
from hourglass.image_transformer_v3 import ImageTransformerDenoiserModelV3


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
    model_kwargs = dict(
        levels=levels,
        mapping=mapping,
        in_channels=model_config['input_channels'],
        out_channels=model_config['output_channels'],
        patch_size=tuple(model_config['patch_size']),
    )
    if model_type == 'image_transformer_v2':
        model = ImageTransformerDenoiserModelV2(**model_kwargs)
    elif model_type == 'image_transformer_v3':
        model = ImageTransformerDenoiserModelV3(**model_kwargs)
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


def compute_slice_weights(case_paths, slice_range=(0, 154), nonzero_weight=10.0, empty_weight=1.0):
    """
    Returns a weight tensor of shape (N,) where N = len(case_paths) * num_slices.
    Slices containing at least one non-zero label voxel get nonzero_weight,
    fully empty slices get empty_weight.
    """
    num_slices = slice_range[1] - slice_range[0] + 1
    weights = []

    for case_path in case_paths:
        # Find the seg file
        files = {}
        for f in Path(case_path).iterdir():
            if f.suffix in (".nii", ".gz"):
                fname = f.name
                if "seg" in fname.lower():
                    files["seg"] = f
        
        if "seg" not in files:
            continue
            
        seg = nib.load(files["seg"]).get_fdata().astype(np.int32)  # (H, W, D)
        seg = seg[:, :, slice_range[0]: slice_range[1] + 1]        # (H, W, num_slices)

        for s in range(num_slices):
            has_label = seg[:, :, s].any()
            weights.append(nonzero_weight if has_label else empty_weight)

    return torch.tensor(weights, dtype=torch.float)


class EarlyStopping:
    """Early stops the training if validation metric doesn't improve after a given patience."""

    def __init__(self, patience=10, verbose=True, delta=0.0, save_path='best.pth'):
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
    p.add_argument('--evaluate-n', type=int, default=2000,
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
    p.add_argument('--brats', type=str, default='2021',)
    p.add_argument('--eval-split', type=str, default='val', choices=['val', 'test'],)

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

    model_type = str(model_config.get('type', 'image_transformer_v2'))
    output_suffix = '_v3' if model_type.endswith('_v3') else ''
    output_dirs = {
        'states': Path(f'states{output_suffix}'),
        'checkpoints': Path(f'checkpoints{output_suffix}'),
        'metrics': Path(f'metrics{output_suffix}'),
        'demos': Path(f'demos{output_suffix}'),
    }

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

    
    if not torch.cuda.is_available() or device.type != 'cuda':
        if accelerator.is_main_process:
            print('No CUDA GPU detected. Exiting before training.', flush=True)
        accelerator.wait_for_everyone()
        raise SystemExit(1)

    else:
        print(f'CUDA GPU detected: {torch.cuda.get_device_name(device)})', flush=True)

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


    # Dataset
    case_dirs = sorted([
        os.path.join(dataset_config['data_path'], d)
        for d in os.listdir(dataset_config['data_path'])
        if os.path.isdir(os.path.join(dataset_config['data_path'], d))
    ])
    random.shuffle(case_dirs)

    train_split = dataset_config['splits']['train']
    val_split = dataset_config['splits']['val']

    case_dirs_train = case_dirs[:int(len(case_dirs) * train_split)]
    case_dirs_val = case_dirs[int(len(case_dirs) * train_split):int(len(case_dirs) * (train_split + val_split))]
    case_dirs_test = case_dirs[int(len(case_dirs) * (train_split + val_split)):]


    print(f'Train cases: {len(case_dirs_train)}, Val cases: {len(case_dirs_val)}, Test cases: {len(case_dirs_test)}')


    if args.brats == '2020':
        dataset_class = BraTSDataset2D
    elif args.brats == '2021':
        dataset_class = BraTSDataset21
    else: 
        raise ValueError(f"Invalid BraTS version specified: {args.brats}. Must be '2020' or '2021'.")

    train_dataset = dataset_class(
        case_paths=case_dirs_train,
        train=True,
        slice_range=dataset_config['slice_range'],
        target_size=dataset_config['image_size'],
    )

    val_dataset = dataset_class(
        case_paths=case_dirs_val,
        train=False,
        slice_range=dataset_config['slice_range'],
        target_size=dataset_config['image_size'],
    )

    test_dataset = dataset_class(
        case_paths=case_dirs_test,
        train=False,
        slice_range=dataset_config['slice_range'],
        target_size=dataset_config['image_size'],
    )

    train_transforms_logged = serialize_transforms(getattr(train_dataset, 'transform', None))
    test_transforms_logged = serialize_transforms(getattr(val_dataset, 'transform', None))


    if accelerator.is_main_process:
        print(f'Number of items in train_dataset: {len(train_dataset):,}')
        print(f'Number of items in val_dataset: {len(val_dataset):,}')
        print(f'Number of items in test_dataset: {len(test_dataset):,}')


    if not args.evaluate_only:
        weights = compute_slice_weights(case_dirs_train, slice_range=dataset_config['slice_range'], nonzero_weight=5.0, empty_weight=1.0)
        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),   # one full epoch worth of draws
            replacement=True,           # required for weighted sampling
            generator=sampler_gen,
        )
    
    else: 
        sampler = None

    train_dl = data.DataLoader(
        train_dataset, args.batch_size, sampler=sampler, prefetch_factor=8,
        num_workers=args.num_workers, persistent_workers=True, pin_memory=True, generator=dl_gen, worker_init_fn=worker_init_fn
    )

    val_sampler = RandomSampler(val_dataset, replacement=False, generator=sampler_gen)
    val_dl = data.DataLoader(
        val_dataset, args.batch_size, shuffle=False,
        num_workers=args.num_workers, persistent_workers=True, pin_memory=True,
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
            warmup_start_lr=lr / 10,
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
    print(f'== Model device after prepare: {next(inner_model.parameters()).device} ==', flush=True)

    # EMA (must be after accelerator.prepare so shadow params are on the correct device)
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
        import wandb
        wandb.login(key="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")
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
                "slice_range": dataset_config['slice_range'],
                "lr_scheduler": sched_config['type'],
                "lr_scheduler_warmup": sched_config.get('warmup', 0),
                "flow_eps": flow_config['eps'],
                "time_sampling": flow_config.get('time_sampling', 'uniform'),
                "sample_steps": args.sample_steps,
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

    # Segmentation info
    n_img_channels = 4
    n_seg_channels = 3
    seg_class_names = ['TC', 'WT', 'ET']
    image_key = "image"

    # Checkpoint state
    state_path = output_dirs['states'] / f'{args.name}_state.json'
    state_path.parent.mkdir(parents=True, exist_ok=True)

    if state_path.exists() or args.resume:
        if args.resume:
            ckpt_path = args.resume
        else:
            state = json.load(open(state_path))
            ckpt_path = state['latest_checkpoint']
        if accelerator.is_main_process:
            print(f'Resuming from {ckpt_path}...')
        ckpt = torch.load(ckpt_path, map_location='cpu')
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


    # log model device and opt device
    if accelerator.is_main_process:
        model_device = next(inner_model.parameters()).device
        opt_device = opt.param_groups[0]['params'][0].device
        print(f'Model device: {model_device}, Optimizer device: {opt_device}', flush=True)

    # Metrics logging
    evaluate_enabled = eval_every > 0 or (args.evaluate_n is not None and args.evaluate_n > 0)
    metrics_log = None
    if evaluate_enabled and accelerator.is_main_process:
        output_dirs['metrics'].mkdir(exist_ok=True)
        metrics_log = CSVLogger(
            str(output_dirs['metrics'] / f'{args.name}_metrics.csv'),
            ['step', 'time', 'loss', 'mean_dice'] + [f'dice_{name}' for name in seg_class_names],
        )

    # --- Helper functions ---

    @torch.no_grad()
    def sample_segmentation(cond_img, n_steps=None, use_ema=True):
        """Sample segmentation from noise using Euler ODE solver.

        Args:
            cond_img: MRI conditioning images (B, 4, H, W).
            n_steps: Number of Euler steps.
            use_ema: Whether to use EMA model weights.

        Returns:
            Predicted segmentation (B, 3, H, W).
        """
        if n_steps is None:
            n_steps = args.sample_steps
        model_to_use = unwrap(inner_model)
        shape = (cond_img.shape[0], n_seg_channels, size[0], size[1])

        if use_ema:
            ema.store(model_to_use.parameters())
            ema.copy_to(model_to_use.parameters())

        model_to_use.eval()
        pred = euler_sample(model_to_use, cond_img, shape, device, N=n_steps, eps=flow.eps)
        model_to_use.train()

        if use_ema:
            ema.restore(model_to_use.parameters())

        return pred

    def threshold_predictions(pred_seg):
        """Apply per-class thresholds to get binary predictions."""
        pred_binary_tc = (pred_seg[:, 0:1] > CLASS_DICE_THRESH[0]).float()
        pred_binary_wt = (pred_seg[:, 1:2] > CLASS_DICE_THRESH[1]).float()
        pred_binary_et = (pred_seg[:, 2:3] > CLASS_DICE_THRESH[2]).float()
        return torch.cat([pred_binary_tc, pred_binary_wt, pred_binary_et], dim=1)

    # Demo colors and priority
    class_colors = [
        (255, 0, 0),    # Red for TC
        (0, 255, 0),    # Green for WT
        (0, 0, 255),    # Blue for ET
    ]
    priority = [1, 0, 2] 

    @torch.no_grad()
    def demo(split='val'):
        """Generate demo visualization grid."""
        max_demo_vis = 10
        if accelerator.is_main_process:
            tqdm.write('Running segmentation demo...')

        output_dirs['demos'].mkdir(exist_ok=True)
        filename = output_dirs['demos'] / f'{args.name}_demo_{split}_{step:08}.png'

        if split == 'train':
            demo_batch = next(iter(train_dl))
        else:
            demo_batch = next(iter(val_dl))

        cond_img_demo = demo_batch[image_key]
        gt_seg_demo = demo_batch["label"]
        n_samples = min(args.sample_n, cond_img_demo.shape[0])
        cond_img_demo = cond_img_demo[:n_samples]
        gt_seg_demo = gt_seg_demo[:n_samples]

        pred_seg = sample_segmentation(cond_img_demo)
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
                wandb.log({'demo_grid': wandb.Image(str(filename)), 'demo_mean_dice': mean_dice}, step=step)

    @torch.no_grad()
    def evaluate(slice_range=(30, 135), n_ensample=1, n_cases=None, split='val'):
        """Evaluate on test set at volume level by aggregating all slices per case.

        For each case in the test split, this function runs inference on slices in
        the provided inclusive slice range, stacks predicted/GT slices into a 3D
        volume, computes Dice and HD95 per class, and averages metrics across cases.
        """
        if not accelerator.is_main_process:
            return None

        case_dirs = case_dirs_val if split == 'val' else case_dirs_test if split == 'test' else case_dirs_train

        if len(case_dirs) == 0:
            tqdm.write('No test cases found for volume-level evaluation.')
            return None
        
        print(f'Running volume-level evaluation on {split} split with {len(case_dirs)} cases...')

        target_cases = len(case_dirs) if n_cases is None else min(n_cases, len(case_dirs))
        tqdm.write(
            f'Evaluating volume-level metrics on {target_cases} {split} cases '
            f'using slices [{slice_range[0]}, {slice_range[1]}]...'
        )

        case_dice_scores = []
        case_hd_scores = []

        model_to_use = unwrap(inner_model)
        ema.store(model_to_use.parameters())
        ema.copy_to(model_to_use.parameters())

        for case_path in tqdm(case_dirs, desc='Volume-level test eval'):

            if n_cases is not None and len(case_dice_scores) >= n_cases:
                break

            case_dataset = dataset_class(
                case_paths=[case_path],
                train=False,
                slice_range=slice_range,
                target_size=dataset_config['image_size'],
            )

            case_dl = data.DataLoader(
                case_dataset,
                args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                persistent_workers=args.num_workers > 0,
                pin_memory=True,
                worker_init_fn=worker_init_fn,
            )

            pred_slices = []
            gt_slices = []

            for case_batch in case_dl:
                cond_img_case = case_batch[image_key].to(device, non_blocking=True)
                gt_seg_case = case_batch['label'].cpu()

                if n_ensample <= 1:
                    pred_seg_case = sample_segmentation(cond_img_case, use_ema=False)
                else:
                    pred_ens = []
                    for _ in range(n_ensample):
                        pred_ens.append(sample_segmentation(cond_img_case, use_ema=False))
                    pred_seg_case = torch.stack(pred_ens).mean(dim=0)
                
                pred_seg_binary_case = threshold_predictions(pred_seg_case).cpu()

                pred_slices.append(pred_seg_binary_case)
                gt_slices.append(gt_seg_case)

            pred_volume = torch.cat(pred_slices, dim=0)  # (D, C, H, W)
            gt_volume = torch.cat(gt_slices, dim=0)      # (D, C, H, W)

            pred_volume = pred_volume.permute(1, 0, 2, 3).numpy()  # (C, D, H, W)
            gt_volume = gt_volume.permute(1, 0, 2, 3).numpy()      # (C, D, H, W)

            case_dice = []
            case_hd = []
            for c in range(n_seg_channels):
                case_dice.append(dice(pred_volume[c], gt_volume[c]))
                case_hd.append(hausdorff_distance_95(pred_volume[c], gt_volume[c]))

            case_dice_scores.append(case_dice)
            case_hd_scores.append(case_hd)

        ema.restore(model_to_use.parameters())

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

        tqdm.write(f'Test Volume Eval Dice - Mean: {mean_dice:.4f}, {class_dice_str}')
        tqdm.write(f'Test Volume Eval HD95 - Mean: {mean_hd:.4f}, {class_hd_str}')

        if use_wandb:
            import wandb
            log_dict = {
                'test_volume_eval_mean_dice': mean_dice,
                'test_volume_eval_mean_hd95': mean_hd,
            }
            for c in range(n_seg_channels):
                log_dict[f'test_volume_eval_dice_{seg_class_names[c]}'] = float(mean_dice_per_class[c])
                log_dict[f'test_volume_eval_hd95_{seg_class_names[c]}'] = float(mean_hd_per_class[c])
            wandb.log(log_dict, step=step)

        if metrics_log is not None:
            metrics_log.write(step, elapsed, ema_loss_stats.get('loss', 0), mean_dice,
                              *mean_dice_per_class)

        return mean_dice
    
    @torch.no_grad()
    def find_best_thresholds(split='val'):
        """Grid-search for best per-class thresholds."""
        if accelerator.is_main_process:
            tqdm.write('Finding best thresholds...')

        all_probs = []
        all_targets = []
        n_evaluated = 0

        eval_iter = iter(val_dl) if split == 'val' else iter(train_dl)
        for eval_batch in tqdm(eval_iter, desc='Sampling for threshold search', disable=not accelerator.is_main_process):
            if n_evaluated >= args.evaluate_n:
                break
            cond_img_eval = eval_batch[image_key]
            pred_seg = sample_segmentation(cond_img_eval)
            all_probs.append(pred_seg.cpu())
            all_targets.append(eval_batch["label"].cpu())
            n_evaluated += cond_img_eval.shape[0]

        all_probs = torch.cat(all_probs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        thresholds = torch.linspace(-0.1, 0.8, 100)
        C = all_probs.shape[1]
        best_t = torch.zeros(C)
        best_dice = torch.zeros(C)

        for c in range(C):
            target_c = all_targets[:, c].float()
            for t in tqdm(thresholds, desc=f'Class {c}', disable=not accelerator.is_main_process):
                pred_c = (all_probs[:, c] > t).float()
                d = dice(pred_c, target_c)
                if d > best_dice[c]:
                    best_dice[c] = d
                    best_t[c] = t

        if accelerator.is_main_process:
            tqdm.write(f'Best thresholds: {best_t.tolist()}, dice: {best_dice.tolist()}')
        return best_t, best_dice

    def save(save_path=None):
        """Save checkpoint."""
        accelerator.wait_for_everyone()
        output_dirs['checkpoints'].mkdir(exist_ok=True)
        if save_path is None:
            filename = output_dirs['checkpoints'] / f'{args.name}_{step:08}.pth'
        else:
            filename = Path(save_path)
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
        accelerator.save(obj, str(filename))
        if accelerator.is_main_process:
            state_obj = {'latest_checkpoint': str(filename)}
            json.dump(state_obj, open(state_path, 'w'))

    # --- Evaluate only mode ---
    if args.evaluate_only:
        evaluate(split=args.eval_split, n_ensample=args.n_ensample, slice_range=dataset_config['slice_range'], n_cases=args.evaluate_n)
        return

    # --- Early stopping setup ---
    early_stopper = None
    if args.use_early_stopping:
        early_stopper = EarlyStopping(
            patience=args.patience,
            verbose=accelerator.is_main_process,
            delta=args.delta,
            save_path=str(output_dirs['checkpoints'] / f'{args.name}_best.pth'),
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
                    cond_img = batch[image_key]
                    x_1 = batch["label"].float()

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
                    val_dice_dict = evaluate(split="val", slice_range=dataset_config['slice_range'], n_ensample=args.n_ensample, n_cases=args.evaluate_n)
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