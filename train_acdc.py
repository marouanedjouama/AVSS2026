# !/usr/bin/env python3

"""Main training and evaluation script for ACDC 2D segmentation
using Hourglass Transformer + Rectified Flow.
"""

import argparse
from contextlib import contextmanager
from datetime import datetime
import json
import math
from pathlib import Path
import time
import random
import numpy as np

import accelerate
from numpy import mean
import torch
import torch._dynamo
from torch import distributed as dist
from torch import multiprocessing as mp
from torch import optim
from torch.utils import data
from torchvision import utils as tv_utils
from tqdm.auto import tqdm
from torch.utils.data import RandomSampler

from dataloaders.loader_ACDC import ACDCPreprocessed, get_train_augmentations, get_val_augmentations
from ema import ExponentialMovingAverage

from hourglass.MM_HDiT import (
    DualStreamHourglassTransformer,
    LevelSpec as DualStreamLevelSpec,
    MappingSpec as DualStreamMappingSpec,
    GlobalAttentionSpec as DualStreamGlobalAttentionSpec,
    CrossNeighborhoodAttentionSpec as DualStreamNeighborhoodAttentionSpec,
)

from hourglass.flags import checkpointing as checkpointing_ctx
from hourglass import flops as model_flops
from lr_scheduler import ConstantLRWithWarmup, LinearWarmupCosineAnnealingLR
from utils.metric import dice, hausdorff_distance_95
from avss2026.rectified_flow import RectifiedFlow
from sampling import euler_sample, rk45_sample


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
            # Dual stream attention types
            if attn_type in ('global', 'global_dualstream'):
                self_attns.append(DualStreamGlobalAttentionSpec(
                    d_head=sa['d_head'],
                    # n_kv_heads=sa.get('n_kv_heads', None)  # None = MHA, int = GQA
                ))
            elif attn_type in ('neighborhood', 'neighborhood_dualstream'):
                self_attns.append(DualStreamNeighborhoodAttentionSpec(
                    d_head=sa['d_head'],
                    kernel_size=sa['kernel_size'],
                    # n_kv_heads=sa.get('n_kv_heads', None)  # None = MHA, int = GQA
                ))
            else:
                raise ValueError(f"Unknown attention type for dual stream model: {attn_type}")
    else:
        # Default: neighborhood for all but last, global for last
        raise ValueError("self_attns configuration is required in the config for this code.")

    levels = []
    for i in range(len(depths)):
        levels.append(DualStreamLevelSpec(
            depth=depths[i],
            width=widths[i],
            d_ff=d_ffs[i],
            self_attn=self_attns[i],
            dropout=dropout_rate[i],
        ))
    mapping = DualStreamMappingSpec(
        depth=model_config.get('mapping_depth', 2),
        width=model_config.get('mapping_width', widths[0]),
        d_ff=model_config.get('mapping_d_ff', widths[0] * 3),
        dropout=model_config.get('mapping_dropout', 0.0), # TODO - original paper used 0.1
    )
    # For dual stream: seg channels = output_channels, img channels = input_channels - output_channels
    in_channels_seg = model_config['output_channels']
    in_channels_img = model_config['input_channels'] - model_config['output_channels']
    model = DualStreamHourglassTransformer(
        levels=levels,
        mapping=mapping,
        in_channels_seg=in_channels_seg,
        in_channels_img=in_channels_img,
        out_channels=model_config['output_channels'],
        patch_size=tuple(model_config['patch_size']),
    )

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
    p.add_argument('--num-workers', type=int, default=8,
                   help='the number of data loader workers')
    p.add_argument('--resume', type=str,
                   help='the checkpoint to resume from')
    p.add_argument('--sample-n', type=int, default=64,
                   help='the number of images to sample for demo grids')
    p.add_argument('--sample-steps', type=int, default=100,
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
    args = p.parse_args()

    mp.set_start_method(args.start_method)
    torch.backends.cuda.matmul.allow_tf32 = True

    # CUDA determinism
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch._dynamo.config.automatic_dynamic_shapes = False
    except AttributeError:
        pass

    # Load config
    config = load_config(args.config)
    model_config = config['model']
    flow_config = config['flow']
    sampling_config = config['sampling']
    dataset_config = config['dataset']
    opt_config = config['optimizer']
    sched_config = config['lr_sched']
    ema_config = config['ema']
    train_config = config['training']

    assert len(model_config['input_size']) == 2 and model_config['input_size'][0] == model_config['input_size'][1]
    size = model_config['input_size']

    # Resolve overrides
    end_step = args.end_step or train_config['max_steps']
    save_every = args.save_every or train_config['save_every']
    eval_every = args.evaluate_every or train_config['eval_every']
    demo_every = args.demo_every or train_config.get('demo_every', eval_every)

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

    demo_gen = torch.Generator().manual_seed(seeds[accelerator.process_index].item())
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
                          weight_decay=opt_config['weight_decay'])
    else:
        raise ValueError(f'Invalid optimizer type: {opt_config["type"]}')

    # LR scheduler
    warmup_steps = int(end_step * sched_config['warmup'])
    if sched_config['type'] == 'constant':
        sched = ConstantLRWithWarmup(opt, warmup_steps=warmup_steps)
    elif sched_config['type'] == 'cosine':
        sched = LinearWarmupCosineAnnealingLR(
            opt,
            warmup_epochs=warmup_steps,
            max_epochs=end_step,
            warmup_start_lr=lr / 10,
        )
    else:
        raise ValueError(f'Invalid schedule type: {sched_config["type"]}')

    # Rectified flow
    flow = RectifiedFlow(
        eps=flow_config['eps'],
        time_sampling=flow_config.get('time_sampling', 'uniform'),
    )

    # Dataset
    train_dataset = ACDCPreprocessed(
        data_dir=dataset_config['train_path'],
        transform=get_train_augmentations(),
    )
    val_dataset = ACDCPreprocessed(
        data_dir=dataset_config['test_path'],
        transform=get_val_augmentations(),
    )

    if accelerator.is_main_process:
        print(f'Number of items in train_dataset: {len(train_dataset):,}')
        print(f'Number of items in val_dataset: {len(val_dataset):,}')

    train_dl = data.DataLoader(
        train_dataset, args.batch_size, shuffle=True,
        num_workers=args.num_workers, persistent_workers=True, pin_memory=True, generator=demo_gen, worker_init_fn=worker_init_fn
    )
    val_sampler = RandomSampler(val_dataset, replacement=False, generator=demo_gen)
    val_dl = data.DataLoader(
        val_dataset, args.batch_size, shuffle=False,
        num_workers=args.num_workers, persistent_workers=True, pin_memory=True, generator=demo_gen, worker_init_fn=worker_init_fn,
        sampler=val_sampler,
    )

    inner_model, opt, train_dl, val_dl = accelerator.prepare(inner_model, opt, train_dl, val_dl)

    # EMA (must be after accelerator.prepare so shadow params are on the correct device)
    ema = ExponentialMovingAverage(unwrap(inner_model).parameters(), decay=ema_config['decay'])

    # Flop counting
    with torch.no_grad(), model_flops.flop_counter() as fc:
        t_dummy = torch.tensor([0.5], device=device)
        model_type = model_config.get('type', 'image_transformer_v2')
        if model_type == 'image_transformer_dualstream':
            # Dual stream model: separate seg and cond_img inputs
            in_channels_seg = model_config['output_channels']
            in_channels_img = model_config['input_channels'] - model_config['output_channels']
            x_seg = torch.zeros([1, in_channels_seg, size[0], size[1]], device=device)
            cond_img = torch.zeros([1, in_channels_img, size[0], size[1]], device=device)
            inner_model(x_seg, t_dummy, cond_img=cond_img)
        else:
            # Original model: concatenated input
            x = torch.zeros([1, model_config['input_channels'], size[0], size[1]], device=device)
            inner_model(x, t_dummy)
        if accelerator.is_main_process:
            print(f"Forward pass GFLOPs: {fc.flops / 1_000_000_000:,.3f}", flush=True)

    # WandB
    use_wandb = accelerator.is_main_process and args.wandb_project
    if use_wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=f"hourglass-rectflow-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
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
                "end_step": end_step,
                "parameters": model_param_count,
                "dataset": "ACDC",
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
                "use_expert_adaln": model_config.get('use_expert_adaln', False),
                "time_sampling_method": flow_config.get('time_sampling', 'uniform'),
            },
        )
        wandb.watch(inner_model)

    # Segmentation info
    n_img_channels = 1  # ACDC has single-channel cardiac MRI
    n_seg_channels = 3  # RV, Myo, LV (excluding background)
    seg_class_names = ['RV', 'Myo', 'LV']

    # Checkpoint state
    state_path = Path(f'states/{args.name}_state.json')
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

        # override lr 
        # for param_group in opt.param_groups:
        #     param_group['lr'] = lr

        sched.load_state_dict(ckpt['sched'])

        if 'ema' in ckpt:
            ema.load_state_dict(ckpt['ema'], device=device)
        ema_loss_stats = ckpt.get('ema_loss_stats', {})
        epoch = ckpt['epoch'] + 1
        step = ckpt['step'] + 1
        demo_gen.set_state(ckpt['demo_gen'])
        elapsed = ckpt.get('elapsed', 0.0)
        del ckpt
    else:
        epoch = 0
        step = 0

    # Metrics logging
    evaluate_enabled = eval_every > 0 and args.evaluate_n > 0
    metrics_log = None
    if evaluate_enabled and accelerator.is_main_process:
        Path('metrics').mkdir(exist_ok=True)
        metrics_log = CSVLogger(
            f'metrics/{args.name}_metrics.csv',
            ['step', 'time', 'loss', 'mean_dice'] + [f'dice_{name}' for name in seg_class_names],
        )

    # --- Helper functions ---

    @torch.no_grad()
    def sample_segmentation(cond_img, n_steps=None, use_ema=True):
        """Sample segmentation from noise using Euler ODE solver.

        Args:
            cond_img: MRI conditioning images (B, 1, H, W).
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

        Path('demos').mkdir(exist_ok=True)
        filename = f'demos/{args.name}_demo_{split}_{step:08}.png'

        if split == 'train':
            demo_batch = next(iter(train_dl))
        else:
            demo_batch = next(iter(val_dl))

        # ACDC returns (image, label) tuple
        cond_img_demo, label_demo = demo_batch
        gt_seg_demo = convert_to_onehot(label_demo, num_classes=4)
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
                wandb.log({'demo_grid': wandb.Image(filename), 'demo_mean_dice': mean_dice}, step=step)

    @torch.no_grad()
    def evaluate(split='val', n_ensample=1):
        """Evaluate segmentation quality using Dice score."""
        if not evaluate_enabled:
            return
        if accelerator.is_main_process:
            tqdm.write('Evaluating segmentation with Dice score...')

        rv_dice_scores = []
        myo_dice_scores = []
        lv_dice_scores = []

        rv_hd_scores = []
        myo_hd_scores = []
        lv_hd_scores = []

        n_evaluated = 0

        eval_iter = iter(val_dl) if split == 'val' else iter(train_dl)
        for eval_batch in tqdm(eval_iter, desc='Evaluating', disable=not accelerator.is_main_process):
            if n_evaluated >= args.evaluate_n:
                break

            # ACDC returns (image, label) tuple
            cond_img_eval, label_eval = eval_batch
            gt_seg_eval = convert_to_onehot(label_eval, num_classes=4)
            batch_size_cur = cond_img_eval.shape[0]

            # Ensemble predictions by sampling multiple times and averaging
            pred_segs = []
            for _ in range(n_ensample):
                pred_seg = sample_segmentation(cond_img_eval)
                pred_segs.append(pred_seg.cpu())

            pred_seg = torch.stack(pred_segs, dim=0).mean(dim=0)

            pred_seg_binary = threshold_predictions(pred_seg)

            rv_dice_scores.append(dice(pred_seg_binary[:, 0:1], gt_seg_eval[:, 0:1]))
            myo_dice_scores.append(dice(pred_seg_binary[:, 1:2], gt_seg_eval[:, 1:2]))
            lv_dice_scores.append(dice(pred_seg_binary[:, 2:3], gt_seg_eval[:, 2:3]))

            rv_hd_scores.append(hausdorff_distance_95(pred_seg_binary[:, 0:1].numpy(), gt_seg_eval[:, 0:1].numpy()))
            myo_hd_scores.append(hausdorff_distance_95(pred_seg_binary[:, 1:2].numpy(), gt_seg_eval[:, 1:2].numpy()))
            lv_hd_scores.append(hausdorff_distance_95(pred_seg_binary[:, 2:3].numpy(), gt_seg_eval[:, 2:3].numpy()))

            n_evaluated += batch_size_cur

        mean_rv = torch.tensor(rv_dice_scores).nanmean().item()
        mean_myo = torch.tensor(myo_dice_scores).nanmean().item()
        mean_lv = torch.tensor(lv_dice_scores).nanmean().item()

        mean_dice = (mean_rv + mean_myo + mean_lv) / 3
        mean_dice_per_class = [mean_rv, mean_myo, mean_lv]

        mean_hd_rv = torch.tensor(rv_hd_scores).nanmean()
        mean_hd_myo = torch.tensor(myo_hd_scores).nanmean()
        mean_hd_lv = torch.tensor(lv_hd_scores).nanmean()

        mean_hd = (mean_hd_rv + mean_hd_myo + mean_hd_lv) / 3
        mean_hd_per_class = [mean_hd_rv, mean_hd_myo, mean_hd_lv]

        if accelerator.is_main_process:
            class_dice_str = ', '.join([
                f'{seg_class_names[c]}: {mean_dice_per_class[c]:.4f}' for c in range(n_seg_channels)
            ])

            class_hd_str = ', '.join([
                f'{seg_class_names[c]}: {mean_hd_per_class[c]:.4f}' for c in range(n_seg_channels)
            ])

            tqdm.write(f'{split} Eval Dice ({args.n_ensample} samples) - Mean: {mean_dice:.4f}, {class_dice_str}')
            tqdm.write(f'{split} Eval HD95 ({args.n_ensample} samples) - Mean: {mean_hd:.4f}, {class_hd_str}')

            if metrics_log is not None:
                metrics_log.write(step, elapsed, ema_loss_stats.get('loss', 0), mean_dice,
                                  *mean_dice_per_class)
            if use_wandb:
                import wandb
                log_dict = {f'{split}_eval_mean_dice': mean_dice}
                for c in range(n_seg_channels):
                    log_dict[f'{split}_eval_dice_{seg_class_names[c]}'] = mean_dice_per_class[c]
                wandb.log(log_dict, step=step)

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
            # ACDC returns (image, label) tuple
            cond_img_eval, label_eval = eval_batch
            pred_seg = sample_segmentation(cond_img_eval)
            all_probs.append(pred_seg.cpu())
            all_targets.append(convert_to_onehot(label_eval, num_classes=4).cpu())
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

    def save():
        """Save checkpoint."""
        accelerator.wait_for_everyone()
        Path('checkpoints').mkdir(exist_ok=True)
        filename = f'checkpoints/{args.name}_{step:08}.pth'
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
            'demo_gen': demo_gen.get_state(),
            'elapsed': elapsed,
        }
        accelerator.save(obj, filename)
        if accelerator.is_main_process:
            state_obj = {'latest_checkpoint': filename}
            json.dump(state_obj, open(state_path, 'w'))

    # --- Evaluate only mode ---
    if args.evaluate_only:
        evaluate('val', n_ensample = args.n_ensample)
        return

    # --- Training loop ---
    losses_since_last_print = []

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
                    # ACDC returns (image, label) tuple
                    cond_img, label = batch
                    x_1 = convert_to_onehot(label, num_classes=4)  # (B, 3, H, W)

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
                    evaluate("val")

                if step > 0 and step % save_every == 0:
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
