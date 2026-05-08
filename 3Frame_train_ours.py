from __future__ import print_function, division
import sys
sys.path.append('core')

import argparse
import os
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from raft import RAFT
import evaluate
import datasets

from torch.utils.tensorboard import SummaryWriter
from augment import TRSRange, sample_trs, build_affine_out2in, apply_affine, \
    invert_affine_2x3, apply_affine_to_coords, _base_grid  # type: ignore

try:
    from torch.cuda.amp import GradScaler
except:
    class GradScaler:
        def __init__(self): pass
        def scale(self, loss): return loss
        def unscale_(self, optimizer): pass
        def step(self, optimizer): optimizer.step()
        def update(self): pass


MAX_FLOW = 400
SUM_FREQ = 100
VAL_FREQ = 5000
BETA     = 1.0   # weight of aug loss relative to original loss


# ---------------------------------------------------------------------------
# Flow utilities
# ---------------------------------------------------------------------------

def warp_flow(flow, displacement, mode='bilinear'):
    """Sample `flow` at positions displaced by `displacement` (B,C,H,W)."""
    _, _, H, W = flow.shape
    try:
        yy, xx = torch.meshgrid(
            torch.arange(H, dtype=torch.float32, device=flow.device),
            torch.arange(W, dtype=torch.float32, device=flow.device),
            indexing='ij')
    except TypeError:
        yy, xx = torch.meshgrid(
            torch.arange(H, dtype=torch.float32, device=flow.device),
            torch.arange(W, dtype=torch.float32, device=flow.device))
    x_disp = xx.unsqueeze(0) + displacement[:, 0]
    y_disp = yy.unsqueeze(0) + displacement[:, 1]
    x_norm = 2.0 * x_disp / (W - 1) - 1.0
    y_norm = 2.0 * y_disp / (H - 1) - 1.0
    grid = torch.stack([x_norm, y_norm], dim=-1)
    return F.grid_sample(flow, grid, mode=mode, padding_mode='border', align_corners=True)


def in_bounds_mask(flow):
    """Return (B,H,W) float mask: 1 where p + flow(p) stays inside the image."""
    B, _, H, W = flow.shape
    device = flow.device
    ys = torch.arange(H, device=device, dtype=torch.float32)
    xs = torch.arange(W, device=device, dtype=torch.float32)
    try:
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')
    except TypeError:
        yy, xx = torch.meshgrid(ys, xs)
    base   = torch.stack([xx, yy], dim=0).unsqueeze(0).expand(B, 2, H, W)
    coords = base + flow
    valid  = ((coords[:, 0] >= 0) & (coords[:, 0] <= W - 1) &
              (coords[:, 1] >= 0) & (coords[:, 1] <= H - 1))
    return valid.float()


def compose_flow(flow01, flow12, valid01, valid12):
    """flow02 = flow01 + warp(flow12, flow01); valid02 = intersection of masks."""
    flow12_w  = warp_flow(flow12, flow01)
    valid12_w = warp_flow(valid12.unsqueeze(1).float(), flow01, mode='nearest').squeeze(1)
    flow02  = flow01 + flow12_w
    valid02 = (valid01 >= 0.5) & (in_bounds_mask(flow01) >= 0.5) & (valid12_w >= 0.5)
    return flow02, valid02.float()


def transform_flow_with_affine(flow, A_out2in):
    """Transform flow to match an affinely-augmented target frame.

    flow     : (B, 2, H, W)  original flow (img0 → img2)
    A_out2in : (B, 2, 3)     affine mapping augmented img2 → original img2

    Returns (flow_aug, valid_mask) both (B, 2, H, W) / (B, H, W).
    """
    B, _, H, W = flow.shape
    A_in2out = invert_affine_2x3(A_out2in)          # original → augmented
    x        = _base_grid(B, H, W, flow.device)     # (B, 2, H, W)
    y        = x + flow                              # destination in original img2
    y_prime  = apply_affine_to_coords(y, A_in2out)  # destination in augmented img2
    flow_aug = y_prime - x
    valid_x  = (y_prime[:, 0] >= 0.0) & (y_prime[:, 0] <= W - 1)
    valid_y  = (y_prime[:, 1] >= 0.0) & (y_prime[:, 1] <= H - 1)
    return flow_aug, (valid_x & valid_y).float()


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def sequence_loss(flow_preds, flow_gt, valid, valid_aug=None, gamma=0.8, max_flow=MAX_FLOW):
    """Sequence loss with optional augmentation validity mask."""
    n_predictions = len(flow_preds)
    flow_loss = 0.0

    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    if valid_aug is not None:
        valid = (valid >= 0.5) & (mag < max_flow) & (valid_aug >= 0.5)
    else:
        valid = (valid >= 0.5) & (mag < max_flow)

    for i in range(n_predictions):
        i_weight = gamma ** (n_predictions - i - 1)
        i_loss   = (flow_preds[i] - flow_gt).abs()
        flow_loss += i_weight * (valid[:, None] * i_loss).mean()

    epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe':  epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }
    return flow_loss, metrics


# ---------------------------------------------------------------------------
# Boilerplate
# ---------------------------------------------------------------------------

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def fetch_optimizer(args, model):
    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.wdecay, eps=args.epsilon)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, args.lr, args.num_steps + 100,
        pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')
    return optimizer, scheduler


class Logger:
    def __init__(self, model, scheduler):
        self.model        = model
        self.scheduler    = scheduler
        self.total_steps  = 0
        self.running_loss = {}
        self.writer       = None

    def _print_training_status(self):
        metrics_data = [self.running_loss[k] / SUM_FREQ
                        for k in sorted(self.running_loss.keys())]
        training_str = "[{:6d}, {:10.7f}] ".format(
            self.total_steps + 1, self.scheduler.get_last_lr()[0])
        metrics_str = ("{:10.4f}, " * len(metrics_data)).format(*metrics_data)
        print(training_str + metrics_str)

        if self.writer is None:
            self.writer = SummaryWriter()
        for k in self.running_loss:
            self.writer.add_scalar(k, self.running_loss[k] / SUM_FREQ, self.total_steps)
            self.running_loss[k] = 0.0

    def push(self, metrics):
        self.total_steps += 1
        for key in metrics:
            self.running_loss.setdefault(key, 0.0)
            self.running_loss[key] += metrics[key]
        if self.total_steps % SUM_FREQ == SUM_FREQ - 1:
            self._print_training_status()
            self.running_loss = {}

    def write_dict(self, results):
        if self.writer is None:
            self.writer = SummaryWriter()
        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)

    def close(self):
        self.writer.close()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    weight_tri = getattr(args, 'weight_tri', 0.5)
    weight_aug = getattr(args, 'weight_aug', 0.5)


    model = nn.DataParallel(RAFT(args), device_ids=args.gpus)
    print("Parameter Count: %d" % count_parameters(model))

    if args.restore_ckpt is not None:
        model.load_state_dict(torch.load(args.restore_ckpt), strict=False)

    model.cuda()
    model.train()

    if args.stage != 'chairs':
        model.module.freeze_bn()

    train_loader = datasets.fetch_dataloader_sintel3frame(args)
    optimizer, scheduler = fetch_optimizer(args, model)

    total_steps = 0
    scaler = GradScaler(enabled=args.mixed_precision)
    logger = Logger(model, scheduler)

    trs = TRSRange(
        tx=(-args.aug_tx, args.aug_tx),
        ty=(-args.aug_ty, args.aug_ty),
        rot=(-args.aug_rot, args.aug_rot),
        scale=(args.aug_smin, args.aug_smax),
    )

    should_keep_training = True
    while should_keep_training:

        for data_blob in train_loader:
            optimizer.zero_grad()
            img0, img1, img2, flow01, flow12, valid01, valid12 = [x.cuda() for x in data_blob]

            # --- compose GT flow02 from the two consecutive flows ---
            flow02, valid02 = compose_flow(flow01, flow12, valid01, valid12)

            if args.add_noise:
                stdv = np.random.uniform(0.0, 5.0)
                img0 = (img0 + stdv * torch.randn_like(img0)).clamp(0.0, 255.0)
                img2 = (img2 + stdv * torch.randn_like(img2)).clamp(0.0, 255.0)

            # --- affine augmentation on img1 ---
            B, _, H, W  = img0.shape
            trs_params  = sample_trs(B, trs, img0.device)
            A_out2in    = build_affine_out2in(*trs_params, H, W)

            img1_aug           = apply_affine(img1, A_out2in)
            flow01_aug, valid_aug = transform_flow_with_affine(flow01, A_out2in)

            # --- two forward passes ---
            preds_01 = model(img0, img1,     iters=args.iters)
            preds_01aug  = model(img0, img1_aug, iters=args.iters)
            preds_12 = model(img1, img2,     iters=args.iters)

            preds_02 = []
            valid_predictions02 = []

            for pred01, pred12 in zip(preds_01, preds_12):
                compose02, pred_valid02 = compose_flow(pred01, pred12, valid01, valid12)
                preds_02.append(compose02)
                valid_predictions02.append(pred_valid02)

            # --- losses ---
            loss_tri, metrics_tri = sequence_loss(
                preds_02, flow02, valid02, valid_aug=None,      gamma=args.gamma)
            loss_aug,  metrics_aug            = sequence_loss(
                preds_01aug,  flow01_aug, valid01, valid_aug=valid_aug, gamma=args.gamma)

            loss = (weight_tri * loss_tri + weight_aug * loss_aug) / (weight_tri + weight_aug)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

            metrics = {
                "loss": loss.item(),
                "loss_tri": loss_tri.item(),
                "loss_aug": loss_aug.item(),
                "epe_tri": metrics_tri["epe"],
                "epe_aug": metrics_aug["epe"],
            }

            logger.push(metrics)

            if total_steps % VAL_FREQ == VAL_FREQ - 1:
                PATH = os.path.join(args.ckpt_dir, '%d_%s.pth' % (total_steps + 1, args.name))
                torch.save(model.state_dict(), PATH)

                results = {}
                for val_dataset in args.validation:
                    if val_dataset == 'chairs':
                        results.update(evaluate.validate_chairs(model.module))
                    elif val_dataset == 'sintel':
                        results.update(evaluate.validate_sintel(model.module))
                    elif val_dataset == 'kitti':
                        results.update(evaluate.validate_kitti(model.module))

                logger.write_dict(results)
                model.train()
                if args.stage != 'chairs':
                    model.module.freeze_bn()

            total_steps += 1
            if total_steps > args.num_steps:
                should_keep_training = False
                break

    logger.close()
    PATH = os.path.join(args.ckpt_dir, '%s.pth' % args.name)
    torch.save(model.state_dict(), PATH)
    return PATH


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name',        default='raft', help='experiment name')
    parser.add_argument('--stage',       help='dataset stage')
    parser.add_argument('--restore_ckpt', help='restore checkpoint')
    parser.add_argument('--small',       action='store_true')
    parser.add_argument('--validation',  type=str, nargs='+')

    parser.add_argument('--lr',          type=float, default=0.000125)
    parser.add_argument('--num_steps',   type=int,   default=100000)
    parser.add_argument('--batch_size',  type=int,   default=6)
    parser.add_argument('--image_size',  type=int,   nargs='+', default=[368, 768])
    parser.add_argument('--gpus',        type=int,   nargs='+', default=[0, 1])
    parser.add_argument('--mixed_precision', action='store_true')

    parser.add_argument('--iters',   type=int,   default=12)
    parser.add_argument('--wdecay',  type=float, default=0.00001)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip',    type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--gamma',   type=float, default=0.85)
    parser.add_argument('--add_noise', action='store_true')
    parser.add_argument('--alternate_corr', action='store_true')
    parser.add_argument('--ckpt_dir', default='checkpoints', help='checkpoint save directory')

    # Augmentation ranges
    parser.add_argument('--aug_tx',   type=float, default=0.0)
    parser.add_argument('--aug_ty',   type=float, default=0.0)
    parser.add_argument('--aug_rot',  type=float, default=0.0)
    parser.add_argument('--aug_smin', type=float, default=1.0)
    parser.add_argument('--aug_smax', type=float, default=1.0)

    # Weight of tri and aug losses
    parser.add_argument('--weight_tri',   type=float, default=0.5)
    parser.add_argument('--weight_aug',   type=float, default=0.5)

    args = parser.parse_args()

    os.makedirs(args.ckpt_dir, exist_ok=True)
    train(args)
