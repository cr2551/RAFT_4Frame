from __future__ import print_function, division
import sys
sys.path.append('core')

import argparse
import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
from raft import RAFT
import evaluate
import datasets

from torch.utils.tensorboard import SummaryWriter

try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass


# exclude extremly large displacements
MAX_FLOW = 400
SUM_FREQ = 100
VAL_FREQ = 5000


def warp_flow(flow, displacement, mode='bilinear'):
    """Sample `flow` at positions displaced by `displacement`.

    Both tensors are (B, C, H, W) with (u, v) channel order.
    `mode` is passed to F.grid_sample; use 'nearest' for binary masks.
    """
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
    grid = torch.stack([x_norm, y_norm], dim=-1)        # (B, H, W, 2)
    return F.grid_sample(flow, grid, mode=mode, padding_mode='border', align_corners=True)


def in_bounds_mask(flow):
    """Return (B,H,W) mask: 1 where p + flow(p) stays inside the image."""
    B, _, H, W = flow.shape
    device = flow.device
    ys = torch.arange(H, device=device, dtype=torch.float32)
    xs = torch.arange(W, device=device, dtype=torch.float32)
    try:
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')
    except TypeError:
        yy, xx = torch.meshgrid(ys, xs)
    base = torch.stack([xx, yy], dim=0).unsqueeze(0).expand(B, 2, H, W)
    coords = base + flow
    valid = ((coords[:, 0] >= 0) & (coords[:, 0] <= W - 1) &
             (coords[:, 1] >= 0) & (coords[:, 1] <= H - 1))
    return valid.float()


def compose_flow(flow01, flow12, valid01, valid12):
    """Compose flow01 and flow12 into flow02 (frame0 → frame2).

    flow02(p) = flow01(p) + flow12(p + flow01(p))
    valid02   = valid01 & flow01 in-bounds & valid12 at destination (nearest)
    """
    flow12_w   = warp_flow(flow12, flow01)
    # use nearest for binary valid12 to avoid fractional values at boundaries
    valid12_w  = warp_flow(valid12.unsqueeze(1).float(), flow01,
                           mode='nearest').squeeze(1)
    flow02  = flow01 + flow12_w
    valid02 = (valid01 >= 0.5) & (in_bounds_mask(flow01) >= 0.5) & (valid12_w >= 0.5)
    return flow02, valid02.float()


def sequence_loss(flow_preds, flow_gt, valid, gamma=0.8, max_flow=MAX_FLOW):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(flow_preds)    
    flow_loss = 0.0

    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    valid = (valid >= 0.5) & (mag < max_flow)

    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        i_loss = (flow_preds[i] - flow_gt).abs()
        flow_loss += i_weight * (valid[:, None] * i_loss).mean()

    epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }

    return flow_loss, metrics


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps+100,
        pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler
    

class Logger:
    def __init__(self, model, scheduler):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss = {}
        self.writer = None

    def _print_training_status(self):
        metrics_data = [self.running_loss[k]/SUM_FREQ for k in sorted(self.running_loss.keys())]
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps+1, self.scheduler.get_last_lr()[0])
        metrics_str = ("{:10.4f}, "*len(metrics_data)).format(*metrics_data)
        
        # print the training status
        print(training_str + metrics_str)

        if self.writer is None:
            self.writer = SummaryWriter()

        for k in self.running_loss:
            self.writer.add_scalar(k, self.running_loss[k]/SUM_FREQ, self.total_steps)
            self.running_loss[k] = 0.0

    def push(self, metrics):
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % SUM_FREQ == SUM_FREQ-1:
            self._print_training_status()
            self.running_loss = {}

    def write_dict(self, results):
        if self.writer is None:
            self.writer = SummaryWriter()

        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)

    def close(self):
        self.writer.close()


def train(args):

    model = nn.DataParallel(RAFT(args), device_ids=args.gpus)
    print("Parameter Count: %d" % count_parameters(model))

    if args.restore_ckpt is not None:
        model.load_state_dict(torch.load(args.restore_ckpt), strict=False)

    model.cuda()
    model.train()

    if args.stage != 'chairs':
        model.module.freeze_bn()

    train_loader = datasets.fetch_dataloader_4frame(args)
    optimizer, scheduler = fetch_optimizer(args, model)

    total_steps = 0
    scaler = GradScaler(enabled=args.mixed_precision)
    logger = Logger(model, scheduler)

    should_keep_training = True
    while should_keep_training:

        for data_blob in train_loader:
            optimizer.zero_grad()
            # get img, flow and valid mask for the frames involved.
            img0, img1, img2, img3, flow01, flow12, flow23,  valid01, valid12, valid23 = [x.cuda() for x in data_blob]

            # compose GT flow 
            with torch.no_grad():
                # from frame 0 to frame 2
                flow02, valid02 = compose_flow(flow01, flow12, valid01, valid12)
                # from frame 0 to frame 3
                flow03, valid03 = compose_flow(flow02, flow23, valid02, valid23)
                # from frame 1 to frame 3
                flow13, valid13 = compose_flow(flow12, flow23, valid12, valid23) 
                
                
               

            if args.add_noise:
                stdv = np.random.uniform(0.0, 5.0)
                img0 = (img0 + stdv * torch.randn_like(img0)).clamp(0.0, 255.0)
                img1 = (img1 + stdv * torch.randn_like(img1)).clamp(0.0, 255.0)
                img2 = (img2 + stdv * torch.randn_like(img2)).clamp(0.0, 255.0)
                img3 = (img2 + stdv * torch.randn_like(img2)).clamp(0.0, 255.0)

            # Predictions 
            flow_predictions01 = model(img0, img1, iters=args.iters)
            flow_predictions02 = model(img0, img2, iters=args.iters)
            flow_predictions03 = model(img0, img3, iters=args.iters)
            # from frame 1 to 2 and 3
            flow_predictions12 = model(img1, img2, iters=args.iters)
            flow_predictions13 = model(img1, img3, iters=args.iters)
            # from frame 2 to frame 3
            flow_predictions23 = model(img2, img3, iters=args.iters)

            # flow_predictions02 = []
            # valid_predictions02 = []

            # for pred01, pred12 in zip(flow_predictions01, flow_predictions12):
            #     pred02, pred_valid02 = compose_flow(pred01, pred12, valid01, valid12)
            #     flow_predictions02.append(pred02)
            #     valid_predictions02.append(pred_valid02)

            # losses 0
            loss01, metrics01 = sequence_loss(flow_predictions01, flow01, valid01, args.gamma)
            loss02, metrics02 = sequence_loss(flow_predictions02, flow02, valid02, args.gamma)
            loss03, metrics03 = sequence_loss(flow_predictions03, flow03, valid03, args.gamma)
            # losses 1 to 2, 1 to 3
            loss12, metrics12 = sequence_loss(flow_predictions12, flow12, valid12, args.gamma)
            loss13, metrics13 = sequence_loss(flow_predictions13, flow13, valid13, args.gamma)
            # losses 2 to 3
            loss23, metrics23 = sequence_loss(flow_predictions23, flow23, valid23, args.gamma)

            scaler.scale(0.5*loss01 + loss02 + loss03 + 0.5*loss12 + loss13 + loss23).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()


            loss = 0.5*loss01 + loss02 + loss03 + 0.5*loss12 + loss13 + loss23

            metrics = {
                "loss": loss.item(),
                "loss01": loss01.item(),
                "loss02": loss02.item(),
                "loss03": loss03.item(),
                "loss12": loss12.item(),
                "loss13": loss13.item(),
                "loss23": loss23.item(),
                "epe01": metrics01["epe"],
                "epe02": metrics02["epe"],
                "epe03": metrics03["epe"],
                "epe12": metrics12["epe"],
                "epe13": metrics13["epe"],
                "epe23": metrics23["epe"],
            }
            logger.push(metrics)

            if total_steps % VAL_FREQ == VAL_FREQ - 1:
                PATH = os.path.join(args.ckpt_dir, '%d_%s.pth' % (total_steps+1, args.name))
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


def make_color_wheel():
    """Middlebury color wheel (55 colours)."""
    RY, YG, GC, CB, BM, MR = 15, 6, 4, 11, 13, 6
    ncols = RY + YG + GC + CB + BM + MR
    w = np.zeros((ncols, 3), dtype=np.float32)
    i = 0
    w[i:i+RY, 0] = 255;  w[i:i+RY, 1] = np.floor(255*np.arange(RY)/RY); i += RY
    w[i:i+YG, 0] = 255 - np.floor(255*np.arange(YG)/YG); w[i:i+YG, 1] = 255; i += YG
    w[i:i+GC, 1] = 255;  w[i:i+GC, 2] = np.floor(255*np.arange(GC)/GC); i += GC
    w[i:i+CB, 1] = 255 - np.floor(255*np.arange(CB)/CB); w[i:i+CB, 2] = 255; i += CB
    w[i:i+BM, 0] = np.floor(255*np.arange(BM)/BM); w[i:i+BM, 2] = 255; i += BM
    w[i:i+MR, 2] = 255 - np.floor(255*np.arange(MR)/MR); w[i:i+MR, 0] = 255
    return w / 255.0


def flow_to_color(flow, max_flow=None):
    """Convert a (2, H, W) flow tensor to a uint8 RGB image via color wheel.

    Direction → hue, magnitude → saturation (white centre, saturated edge).
    """
    u = flow[0].numpy() if isinstance(flow, torch.Tensor) else flow[..., 0]
    v = flow[1].numpy() if isinstance(flow, torch.Tensor) else flow[..., 1]

    rad = np.sqrt(u**2 + v**2)
    if max_flow is None:
        max_flow = rad.max() + 1e-6
    rad_norm = np.clip(rad / max_flow, 0, 1)

    angle = np.arctan2(-v, -u) / np.pi          # [-1, 1]
    wheel = make_color_wheel()
    ncols = wheel.shape[0]
    fk  = (angle + 1) / 2 * (ncols - 1)         # [0, ncols-1]
    k0  = np.floor(fk).astype(np.int32)
    k1  = (k0 + 1) % ncols
    f   = fk - k0

    img = np.zeros((*u.shape, 3), dtype=np.float32)
    for c in range(3):
        col = (1 - f) * wheel[k0, c] + f * wheel[k1, c]
        # white at zero magnitude, saturated at max
        col = 1 - rad_norm * (1 - col)
        img[..., c] = col

    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def test_compose_flow(root='testdata'):
    """Visualise flow composition: flow01, flow12, warped flow12, and composed flow02."""
    ds = datasets.MpiSintel3Frame(aug_params=None, split='training', root=root, dstype='clean')
    print(f'Dataset size: {len(ds)} triplet(s)')

    img0, _, img2, flow01, flow12, valid01, valid12 = ds[0]

    # add batch dim for compose_flow
    flow01  = flow01.unsqueeze(0)
    flow12  = flow12.unsqueeze(0)
    valid01 = valid01.unsqueeze(0)
    valid12 = valid12.unsqueeze(0)

    flow12_w  = warp_flow(flow12, flow01)
    flow02, valid02 = compose_flow(flow01, flow12, valid01, valid12)

    # remove batch dim
    flow01   = flow01.squeeze(0)
    flow12   = flow12.squeeze(0)
    flow12_w = flow12_w.squeeze(0)
    flow02   = flow02.squeeze(0)
    valid02  = valid02.squeeze(0)

    def mag_map(flow):
        return np.sqrt(flow[0].numpy()**2 + flow[1].numpy()**2)

    # share the same max_flow scale so all four flow images are comparable
    all_max = max(mag_map(f).max() for f in [flow01, flow12, flow12_w, flow02]) + 1e-6

    fig, axes = plt.subplots(2, 4, figsize=(20, 6))
    fig.suptitle('Flow composition: flow01 + warp(flow12) = flow02  '
                 f'(shared scale, max={all_max:.1f} px)')

    axes[0, 0].imshow(img0.permute(1,2,0).numpy().astype(np.uint8)); axes[0, 0].set_title('img0')
    axes[0, 1].imshow(flow_to_color(flow01,   all_max)); axes[0, 1].set_title('flow01  (0→1)')
    axes[0, 2].imshow(flow_to_color(flow12,   all_max)); axes[0, 2].set_title('flow12  (1→2, raw)')
    axes[0, 3].imshow(flow_to_color(flow12_w, all_max)); axes[0, 3].set_title('flow12 warped to frame0')

    axes[1, 0].imshow(img2.permute(1,2,0).numpy().astype(np.uint8)); axes[1, 0].set_title('img2')
    axes[1, 1].imshow(flow_to_color(flow02,   all_max)); axes[1, 1].set_title('flow02 composed (0→2)')
    axes[1, 2].imshow(valid02.numpy(), cmap='gray', vmin=0, vmax=1); axes[1, 2].set_title('valid02 mask')

    # magnitude comparison
    ax = axes[1, 3]
    ax.plot(mag_map(flow01).mean(axis=1), label='flow01')
    ax.plot(mag_map(flow02).mean(axis=1), label='flow02')
    ax.set_title('mean magnitude per row')
    ax.set_xlabel('row'); ax.set_ylabel('pixels')
    ax.legend(fontsize=8)

    for ax in axes.flat[:7]:
        ax.axis('off')
    axes[1, 3].axis('on')

    plt.tight_layout()
    plt.savefig('compose_flow_check.png', dpi=120)
    print('Saved → compose_flow_check.png')

    # numeric summary
    diff = (flow02 - flow01 - flow12_w).abs()
    print(f'flow02 == flow01 + warp(flow12)  max_err={diff.max():.2e}  (should be ~0)')
    print(f'flow01 mag  mean={mag_map(flow01).mean():.2f}  max={mag_map(flow01).max():.2f}')
    print(f'flow12 mag  mean={mag_map(flow12).mean():.2f}  max={mag_map(flow12).max():.2f}')
    print(f'flow02 mag  mean={mag_map(flow02).mean():.2f}  max={mag_map(flow02).max():.2f}')
    print(f'valid02 coverage: {valid02.mean()*100:.1f}%')

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='raft', help="name your experiment")
    parser.add_argument('--stage', help="determines which dataset to use for training")
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--validation', type=str, nargs='+')

    parser.add_argument('--lr', type=float, default=0.00002)
    parser.add_argument('--num_steps', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--image_size', type=int, nargs='+', default=[384, 512])
    parser.add_argument('--gpus', type=int, nargs='+', default=[0,1])
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')

    parser.add_argument('--iters', type=int, default=12)
    parser.add_argument('--wdecay', type=float, default=.00005)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--gamma', type=float, default=0.8, help='exponential weighting')
    parser.add_argument('--add_noise', action='store_true')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    parser.add_argument('--ckpt_dir', default='checkpoints', help='directory to save checkpoints')
    # parser.add_argument('--test_compose', action='store_true', help='run compose_flow sanity check and exit')
    # parser.add_argument('--dataset_root', default='testdata', help='root for test datasets')
    args = parser.parse_args()

    torch.manual_seed(1234)
    np.random.seed(1234)

    # if args.test_compose:
    #     test_compose_flow(root=args.dataset_root)
    # else:
    os.makedirs(args.ckpt_dir, exist_ok=True)
    train(args)
