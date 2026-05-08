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
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.6, 'do_flip': True}

    if args.dataset == 'sintel_three_frame':
        dataset = datasets.MpiSintelThreeFrame(
            aug_params, root=args.dataset_root, split='training', dstype=args.sintel_dstype)
    else:
        dataset = datasets.ThreeFrameFlowDataset(aug_params, root=args.dataset_root)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=False, drop_last=False)

    image0, image2, flow02, valid02 = next(iter(loader))
    print('image_0:', tuple(image0.shape), image0.dtype)
    print('image_2:', tuple(image2.shape), image2.dtype)
    print('flow_02:', tuple(flow02.shape), flow02.dtype)
    print('valid_02:', tuple(valid02.shape), valid02.dtype, 'valid pixels:', float(valid02.sum()))

    if valid02.sum().item() == 0:
        raise RuntimeError('valid_02 is all zero in the smoke-test batch')

    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but torch.cuda.is_available() is False')

    image0 = image0.to(device)
    image2 = image2.to(device)
    flow02 = flow02.to(device)
    valid02 = valid02.to(device)

    model = RAFT(args).to(device)
    model.train()
    if not args.small:
        model.freeze_bn()

    flow_predictions = model(image0, image2, iters=args.iters)
    loss, metrics = sequence_loss(flow_predictions, flow02, valid02, gamma=args.gamma)
    if not torch.isfinite(loss):
        raise RuntimeError('smoke-test loss is not finite')
    loss.backward()

    print('loss:', float(loss.detach().cpu()))
    print('metrics:', metrics)
    print('smoke test passed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='three_frame', choices=['three_frame', 'sintel_three_frame'])
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
