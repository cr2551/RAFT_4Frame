import argparse
import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT / 'core'))

import datasets
from raft import RAFT
from train import sequence_loss


def main(args):
    torch.manual_seed(1234)
    np.random.seed(1234)

    aug_params = None
    if not args.no_augment:
        aug_params = {
            'crop_size': args.image_size,
            'min_scale': -0.2,
            'max_scale': 0.6,
            'do_flip': True
        }

    if args.dataset == 'sintel_four_frame':
        dataset = datasets.MpiSintelFourFrame(
            aug_params, root=args.dataset_root, split='training', dstype=args.sintel_dstype)
    else:
        dataset = datasets.FourFrameFlowDataset(aug_params, root=args.dataset_root)

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=False, drop_last=False)

    # Dataset must yield 4 images + 6 (flow, valid) pairs in this order:
    # consecutive:  (0->1), (1->2), (2->3)
    # skip:         (0->2), (0->3), (1->3)
    # (image0, image1, image2, image3,
    #  flow01, valid01,
    #  flow12, valid12,
    #  flow23, valid23,
    #  flow02, valid02,
    #  flow03, valid03,
    #  flow13, valid13) = next(iter(loader))

    img0, img1, img2, img3, flow01, flow12, flow23, valid01, valid12, valid23 = next(iter(loader))



    # compose GT flow 
    with torch.no_grad():
        # from frame 0 to frame 2
        flow02, valid02 = compose_flow(flow01, flow12, valid01, valid12)
        # from frame 0 to frame 3
        flow03, valid03 = compose_flow(flow02, flow23, valid02, valid23)
        # from frame 1 to frame 3
        flow13, valid13 = compose_flow(flow12, flow23, valid12, valid23) 
    # --- Shape / dtype report ---
    for name, t in [
        ('image0', image0), ('image1', image1), ('image2', image2), ('image3', image3),
        ('flow01', flow01), ('valid01', valid01),
        ('flow12', flow12), ('valid12', valid12),
        ('flow23', flow23), ('valid23', valid23),
        ('flow02', flow02), ('valid02', valid02),
        ('flow03', flow03), ('valid03', valid03),
        ('flow13', flow13), ('valid13', valid13),
    ]:
        extra = f'  valid pixels: {int(t.sum())}' if name.startswith('valid') else ''
        print(f'{name}: {tuple(t.shape)}  {t.dtype}{extra}')

    # --- Validity checks ---
    for name, mask in [
        ('valid01', valid01), ('valid12', valid12), ('valid23', valid23),
        ('valid02', valid02), ('valid03', valid03), ('valid13', valid13),
    ]:
        if mask.sum().item() == 0:
            raise RuntimeError(f'{name} is all zero in the smoke-test batch')

    # --- Move to device ---
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but torch.cuda.is_available() is False')

    images = [image0, image1, image2, image3]
    image0, image1, image2, image3 = [im.to(device) for im in images]

    flows_and_masks = {
        '01': (flow01, valid01), '12': (flow12, valid12), '23': (flow23, valid23),
        '02': (flow02, valid02), '03': (flow03, valid03), '13': (flow13, valid13),
    }
    flows_and_masks = {k: (f.to(device), v.to(device)) for k, (f, v) in flows_and_masks.items()}

    # --- Build model ---
    model = RAFT(args).to(device)
    model.train()
    if not args.small:
        model.freeze_bn()

    # --- Forward passes ---
    frame = [image0, image1, image2, image3]

    pairs = {
        '01': (0, 1), '12': (1, 2), '23': (2, 3),  # consecutive
        '02': (0, 2), '03': (0, 3), '13': (1, 3),  # skip
    }

    losses = {}
    all_metrics = {}
    for key, (i, j) in pairs.items():
        gt_flow, gt_valid = flows_and_masks[key]
        preds = model(frame[i], frame[j], iters=args.iters)
        loss, metrics = sequence_loss(preds, gt_flow, gt_valid, gamma=args.gamma)
        losses[key] = loss
        all_metrics[key] = metrics

    # --- Combined loss ---
    total_loss = sum(losses.values()) / len(losses)

    # --- Finite checks ---
    for name, l in {**losses, 'total': total_loss}.items():
        if not torch.isfinite(l):
            raise RuntimeError(f'smoke-test loss_{name} is not finite')

    total_loss.backward()

    # --- Report ---
    print()
    for key in pairs:
        print(f'loss_{key}: {float(losses[key].detach().cpu()):.4f}  metrics: {all_metrics[key]}')
    print(f'\nloss (mean): {float(total_loss.detach().cpu()):.4f}')
    print('\nsmoke test passed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='four_frame', choices=['four_frame', 'sintel_four_frame'])
    parser.add_argument('--dataset_root', required=True)
    parser.add_argument('--sintel_dstype', default='clean', choices=['clean', 'final'])
    parser.add_argument('--image_size', type=int, nargs='+', default=[368, 768])
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--iters', type=int, default=1)
    parser.add_argument('--gamma', type=float, default=0.8)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--no_augment', action='store_true')
    # RAFT constructor options
    parser.add_argument('--small', action='store_true')
    parser.add_argument('--mixed_precision', action='store_true')
    parser.add_argument('--alternate_corr', action='store_true')
    parser.add_argument('--dropout', type=float, default=0.0)
    main(parser.parse_args())