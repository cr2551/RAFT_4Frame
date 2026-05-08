import torch
import torch.nn.functional as F


def _as_batched_flow(flow, name):
    if not torch.is_tensor(flow):
        raise TypeError('%s must be a torch.Tensor' % name)

    if flow.dim() == 3:
        if flow.shape[0] != 2:
            raise ValueError('%s must have shape [2, H, W]' % name)
        return flow[None], True

    if flow.dim() == 4:
        if flow.shape[1] != 2:
            raise ValueError('%s must have shape [B, 2, H, W]' % name)
        return flow, False

    raise ValueError('%s must have shape [2, H, W] or [B, 2, H, W]' % name)


def _as_batched_valid(valid, batch, ht, wd, device, dtype, name):
    if valid is None:
        return torch.ones(batch, 1, ht, wd, device=device, dtype=dtype)

    if not torch.is_tensor(valid):
        raise TypeError('%s must be a torch.Tensor when provided' % name)

    valid = valid.to(device=device, dtype=dtype)

    if valid.dim() == 2:
        valid = valid[None, None].repeat(batch, 1, 1, 1)
    elif valid.dim() == 3:
        if valid.shape[0] == batch:
            valid = valid[:, None]
        elif valid.shape[0] == 1:
            valid = valid.repeat(batch, 1, 1)[:, None]
        else:
            raise ValueError('%s must have shape [H, W], [B, H, W], or [1, H, W]' % name)
    elif valid.dim() == 4:
        if valid.shape[1] != 1:
            raise ValueError('%s must have one valid-mask channel' % name)
        if valid.shape[0] == 1 and batch > 1:
            valid = valid.repeat(batch, 1, 1, 1)
    else:
        raise ValueError('%s must have shape [H, W], [B, H, W], or [B, 1, H, W]' % name)

    if valid.shape[0] != batch or valid.shape[-2:] != (ht, wd):
        raise ValueError('%s shape must match flow shape' % name)

    return valid


def compose_flows(flow01, flow12, valid01=None, valid12=None):
    """Compose adjacent optical flows into a 0->2 flow.

    For each pixel x in image_0:
        flow02(x) = flow01(x) + flow12(x + flow01(x))

    flow12 and valid12 are sampled with torch.grid_sample using
    align_corners=True, matching RAFT's existing bilinear_sampler helper.
    Inputs support [2, H, W] or [B, 2, H, W]. The returned valid mask has
    shape [H, W] for unbatched input and [B, H, W] for batched input.
    """
    flow01, squeeze = _as_batched_flow(flow01, 'flow01')
    flow12, squeeze12 = _as_batched_flow(flow12, 'flow12')

    if squeeze != squeeze12:
        raise ValueError('flow01 and flow12 must both be batched or both be unbatched')

    if flow01.shape != flow12.shape:
        raise ValueError('flow01 and flow12 must have the same shape')

    batch, _, ht, wd = flow01.shape
    device = flow01.device
    dtype = flow01.dtype
    flow12 = flow12.to(device=device, dtype=dtype)

    valid01 = _as_batched_valid(valid01, batch, ht, wd, device, dtype, 'valid01')
    valid12 = _as_batched_valid(valid12, batch, ht, wd, device, dtype, 'valid12')

    y_coords = torch.arange(ht, device=device, dtype=dtype)
    x_coords = torch.arange(wd, device=device, dtype=dtype)
    try:
        y, x = torch.meshgrid(y_coords, x_coords, indexing='ij')
    except TypeError:
        y, x = torch.meshgrid(y_coords, x_coords)
    base = torch.stack([x, y], dim=0)[None].repeat(batch, 1, 1, 1)
    warped = base + flow01

    x_warp = warped[:, 0]
    y_warp = warped[:, 1]

    if wd > 1:
        x_grid = 2.0 * x_warp / (wd - 1) - 1.0
    else:
        x_grid = torch.zeros_like(x_warp)

    if ht > 1:
        y_grid = 2.0 * y_warp / (ht - 1) - 1.0
    else:
        y_grid = torch.zeros_like(y_warp)

    grid = torch.stack([x_grid, y_grid], dim=-1)
    sampled_flow12 = F.grid_sample(
        flow12, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
    sampled_valid12 = F.grid_sample(
        valid12, grid, mode='bilinear', padding_mode='zeros', align_corners=True)

    in_bounds = (x_warp >= 0.0) & (x_warp <= wd - 1) & (y_warp >= 0.0) & (y_warp <= ht - 1)
    valid02 = (valid01[:, 0] >= 0.5) & (sampled_valid12[:, 0] >= 0.999) & in_bounds

    flow02 = flow01 + sampled_flow12
    finite = torch.isfinite(flow02).all(dim=1)
    valid02 = valid02 & finite
    flow02 = torch.where(valid02[:, None], flow02, torch.zeros_like(flow02))

    if squeeze:
        return flow02[0], valid02[0].to(dtype=dtype)

    return flow02, valid02.to(dtype=dtype)
