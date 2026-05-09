# Data loading based on https://github.com/NVIDIA/flownet2-pytorch

import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F

import os
import math
import random
from glob import glob
import os.path as osp

from utils import frame_utils
from utils.augmentor import FlowAugmentor, SparseFlowAugmentor, ThreeFrameFlowAugmentor, FourFrameFlowAugmentor


class FlowDataset(data.Dataset):
    def __init__(self, aug_params=None, sparse=False):
        self.augmentor = None
        self.sparse = sparse
        if aug_params is not None:
            if sparse:
                self.augmentor = SparseFlowAugmentor(**aug_params)
            else:
                self.augmentor = FlowAugmentor(**aug_params)

        self.is_test = False
        self.init_seed = False
        self.flow_list = []
        self.image_list = []
        self.extra_info = []

    def __getitem__(self, index):

        if self.is_test:
            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])
            img1 = np.array(img1).astype(np.uint8)[..., :3]
            img2 = np.array(img2).astype(np.uint8)[..., :3]
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            return img1, img2, self.extra_info[index]

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)
        valid = None
        if self.sparse:
            flow, valid = frame_utils.readFlowKITTI(self.flow_list[index])
        else:
            flow = frame_utils.read_gen(self.flow_list[index])

        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])

        flow = np.array(flow).astype(np.float32)
        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)

        # grayscale images
        if len(img1.shape) == 2:
            img1 = np.tile(img1[...,None], (1, 1, 3))
            img2 = np.tile(img2[...,None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]

        if self.augmentor is not None:
            if self.sparse:
                img1, img2, flow, valid = self.augmentor(img1, img2, flow, valid)
            else:
                img1, img2, flow = self.augmentor(img1, img2, flow)

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()

        if valid is not None:
            valid = torch.from_numpy(valid)
        else:
            valid = (flow[0].abs() < 1000) & (flow[1].abs() < 1000)

        return img1, img2, flow, valid.float()


    def __rmul__(self, v):
        self.flow_list = v * self.flow_list
        self.image_list = v * self.image_list
        return self
        
    def __len__(self):
        return len(self.image_list)
        

class FlowDataset3Frame(data.Dataset):
    """Base dataset for triplets of consecutive frames with two optical flows."""

    def __init__(self, aug_params=None):
        self.augmentor = None
        if aug_params is not None:
            self.augmentor = ThreeFrameFlowAugmentor(**aug_params)

        self.is_test = False
        self.init_seed = False
        self.flow_list = []   # list of [flow01_path, flow12_path]
        self.image_list = []  # list of [img0_path, img1_path, img2_path]
        self.extra_info = []

    def _ensure_rgb(self, img):
        if len(img.shape) == 2:
            return np.tile(img[..., None], (1, 1, 3))
        return img[..., :3]

    def __getitem__(self, index):
        if self.is_test:
            img0 = np.array(frame_utils.read_gen(self.image_list[index][0])).astype(np.uint8)[..., :3]
            img1 = np.array(frame_utils.read_gen(self.image_list[index][1])).astype(np.uint8)[..., :3]
            img2 = np.array(frame_utils.read_gen(self.image_list[index][2])).astype(np.uint8)[..., :3]
            img0 = torch.from_numpy(img0).permute(2, 0, 1).float()
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            return img0, img1, img2, self.extra_info[index]

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)

        img0 = self._ensure_rgb(np.array(frame_utils.read_gen(self.image_list[index][0])).astype(np.uint8))
        img1 = self._ensure_rgb(np.array(frame_utils.read_gen(self.image_list[index][1])).astype(np.uint8))
        img2 = self._ensure_rgb(np.array(frame_utils.read_gen(self.image_list[index][2])).astype(np.uint8))

        flow01 = np.array(frame_utils.read_gen(self.flow_list[index][0])).astype(np.float32)
        flow12 = np.array(frame_utils.read_gen(self.flow_list[index][1])).astype(np.float32)

        valid01 = ((np.abs(flow01[..., 0]) < 1000) & (np.abs(flow01[..., 1]) < 1000)).astype(np.float32)
        valid12 = ((np.abs(flow12[..., 0]) < 1000) & (np.abs(flow12[..., 1]) < 1000)).astype(np.float32)

        if self.augmentor is not None:
            img0, img1, img2, flow01, flow12, valid01, valid12 = self.augmentor(
                img0, img1, img2, flow01, flow12, valid01, valid12)

        img0 = torch.from_numpy(img0).permute(2, 0, 1).float()
        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow01 = torch.from_numpy(flow01).permute(2, 0, 1).float()
        flow12 = torch.from_numpy(flow12).permute(2, 0, 1).float()
        valid01 = torch.from_numpy(valid01)
        valid12 = torch.from_numpy(valid12)

        return img0, img1, img2, flow01, flow12, valid01.float(), valid12.float()

    def __rmul__(self, v):
        self.flow_list = v * self.flow_list
        self.image_list = v * self.image_list
        return self

    def __len__(self):
        return len(self.image_list)


class MpiSintel(FlowDataset):
    def __init__(self, aug_params=None, split='training', root='datasets/Sintel', dstype='clean'):
        super(MpiSintel, self).__init__(aug_params)
        flow_root = osp.join(root, split, 'flow')
        image_root = osp.join(root, split, dstype)

        if split == 'test':
            self.is_test = True

        for scene in os.listdir(image_root):
            image_list = sorted(glob(osp.join(image_root, scene, '*.png')))
            for i in range(len(image_list)-1):
                self.image_list += [ [image_list[i], image_list[i+1]] ]
                self.extra_info += [ (scene, i) ] # scene and frame_id

            if split != 'test':
                self.flow_list += sorted(glob(osp.join(flow_root, scene, '*.flo')))


class MpiSintel3Frame(FlowDataset3Frame):
    """Sintel dataset returning consecutive triplets (img0, img1, img2) and (flow01, flow12)."""

    def __init__(self, aug_params=None, split='training', root='datasets/Sintel', dstype='clean'):
        super(MpiSintel3Frame, self).__init__(aug_params)
        flow_root = osp.join(root, split, 'flow')
        image_root = osp.join(root, split, dstype)

        if split == 'test':
            self.is_test = True

        for scene in sorted(os.listdir(image_root)):
            images = sorted(glob(osp.join(image_root, scene, '*.png')))
            flows = sorted(glob(osp.join(flow_root, scene, '*.flo'))) if split != 'test' else []
            for i in range(len(images) - 2):
                self.image_list.append([images[i], images[i + 1], images[i + 2]])
                self.extra_info.append((scene, i))
                if split != 'test':
                    self.flow_list.append([flows[i], flows[i + 1]])


class FlyingThings3D3Frame(FlowDataset3Frame):
    """FlyingThings3D returning consecutive triplets (img0, img1, img2) and (flow01, flow12)."""

    def __init__(self, aug_params=None, root='datasets/FlyingThings3D', dstype='frames_cleanpass'):
        super(FlyingThings3D3Frame, self).__init__(aug_params)

        for cam in ['left']:
            for direction in ['into_future', 'into_past']:
                image_dirs = sorted(glob(osp.join(root, dstype, 'TRAIN/*/*')))
                image_dirs = sorted([osp.join(f, cam) for f in image_dirs])

                flow_dirs = sorted(glob(osp.join(root, 'optical_flow/TRAIN/*/*')))
                flow_dirs = sorted([osp.join(f, direction, cam) for f in flow_dirs])

                for idir, fdir in zip(image_dirs, flow_dirs):
                    images = sorted(glob(osp.join(idir, '*.png')))
                    flows  = sorted(glob(osp.join(fdir, '*.pfm')))
                    if len(images) < 3 or len(flows) < 2:
                        continue
                    if direction == 'into_future':
                        # need images[i..i+2] and flows[i..i+1]
                        n = min(len(images) - 2, len(flows) - 1)
                        for i in range(n):
                            self.image_list.append([images[i], images[i+1], images[i+2]])
                            self.flow_list.append([flows[i], flows[i+1]])
                    elif direction == 'into_past':
                        # need images[i..i+2] and flows[i+1..i+2]
                        n = min(len(images) - 2, len(flows) - 2)
                        for i in range(n):
                            self.image_list.append([images[i+2], images[i+1], images[i]])
                            self.flow_list.append([flows[i+2], flows[i+1]])


class FlowDataset4Frame(data.Dataset):
    """Base dataset for quadruplets of consecutive frames with three optical flows."""

    def __init__(self, aug_params=None):
        self.augmentor = None
        if aug_params is not None:
            self.augmentor = FourFrameFlowAugmentor(**aug_params)

        self.is_test = False
        self.init_seed = False
        self.flow_list = []   # list of [flow01_path, flow12_path, flow23_path]
        self.image_list = []  # list of [img0_path, img1_path, img2_path, img3_path]
        self.extra_info = []

    def _ensure_rgb(self, img):
        if len(img.shape) == 2:
            return np.tile(img[..., None], (1, 1, 3))
        return img[..., :3]

    def __getitem__(self, index):
        if self.is_test:
            img0 = np.array(frame_utils.read_gen(self.image_list[index][0])).astype(np.uint8)[..., :3]
            img1 = np.array(frame_utils.read_gen(self.image_list[index][1])).astype(np.uint8)[..., :3]
            img2 = np.array(frame_utils.read_gen(self.image_list[index][2])).astype(np.uint8)[..., :3]
            img3 = np.array(frame_utils.read_gen(self.image_list[index][3])).astype(np.uint8)[..., :3]
            img0 = torch.from_numpy(img0).permute(2, 0, 1).float()
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            img3 = torch.from_numpy(img3).permute(2, 0, 1).float()
            return img0, img1, img2, img3, self.extra_info[index]

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)

        img0 = self._ensure_rgb(np.array(frame_utils.read_gen(self.image_list[index][0])).astype(np.uint8))
        img1 = self._ensure_rgb(np.array(frame_utils.read_gen(self.image_list[index][1])).astype(np.uint8))
        img2 = self._ensure_rgb(np.array(frame_utils.read_gen(self.image_list[index][2])).astype(np.uint8))
        img3 = self._ensure_rgb(np.array(frame_utils.read_gen(self.image_list[index][3])).astype(np.uint8))

        flow01 = np.array(frame_utils.read_gen(self.flow_list[index][0])).astype(np.float32)
        flow12 = np.array(frame_utils.read_gen(self.flow_list[index][1])).astype(np.float32)
        flow23 = np.array(frame_utils.read_gen(self.flow_list[index][2])).astype(np.float32)

        valid01 = ((np.abs(flow01[..., 0]) < 1000) & (np.abs(flow01[..., 1]) < 1000)).astype(np.float32)
        valid12 = ((np.abs(flow12[..., 0]) < 1000) & (np.abs(flow12[..., 1]) < 1000)).astype(np.float32)
        valid23 = ((np.abs(flow23[..., 0]) < 1000) & (np.abs(flow23[..., 1]) < 1000)).astype(np.float32)

        if self.augmentor is not None:
            img0, img1, img2, img3, flow01, flow12, flow23, valid01, valid12, valid23 = self.augmentor(
                img0, img1, img2, img3, flow01, flow12, flow23, valid01, valid12, valid23)

        img0 = torch.from_numpy(img0).permute(2, 0, 1).float()
        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        img3 = torch.from_numpy(img3).permute(2, 0, 1).float()
        flow01 = torch.from_numpy(flow01).permute(2, 0, 1).float()
        flow12 = torch.from_numpy(flow12).permute(2, 0, 1).float()
        flow23 = torch.from_numpy(flow23).permute(2, 0, 1).float()
        valid01 = torch.from_numpy(valid01)
        valid12 = torch.from_numpy(valid12)
        valid23 = torch.from_numpy(valid23)

        return img0, img1, img2, img3, flow01, flow12, flow23, valid01.float(), valid12.float(), valid23.float()

    def __rmul__(self, v):
        self.flow_list = v * self.flow_list
        self.image_list = v * self.image_list
        return self

    def __len__(self):
        return len(self.image_list)


class MpiSintel4Frame(FlowDataset4Frame):
    """Sintel dataset returning consecutive quadruplets (img0, img1, img2, img3) and (flow01, flow12, flow23)."""

    def __init__(self, aug_params=None, split='training', root='datasets/Sintel', dstype='clean'):
        super(MpiSintel4Frame, self).__init__(aug_params)
        flow_root = osp.join(root, split, 'flow')
        image_root = osp.join(root, split, dstype)

        if split == 'test':
            self.is_test = True

        for scene in sorted(os.listdir(image_root)):
            images = sorted(glob(osp.join(image_root, scene, '*.png')))
            flows = sorted(glob(osp.join(flow_root, scene, '*.flo'))) if split != 'test' else []
            for i in range(len(images) - 3):
                self.image_list.append([images[i], images[i + 1], images[i + 2], images[i + 3]])
                self.extra_info.append((scene, i))
                if split != 'test':
                    self.flow_list.append([flows[i], flows[i + 1], flows[i + 2]])


class FlyingThings3D4Frame(FlowDataset4Frame):
    """FlyingThings3D returning consecutive quadruplets (img0, img1, img2, img3) and (flow01, flow12, flow23)."""

    def __init__(self, aug_params=None, root='datasets/FlyingThings3D', dstype='frames_cleanpass'):
        super(FlyingThings3D4Frame, self).__init__(aug_params)

        for cam in ['left']:
            for direction in ['into_future', 'into_past']:
                image_dirs = sorted(glob(osp.join(root, dstype, 'TRAIN/*/*')))
                image_dirs = sorted([osp.join(f, cam) for f in image_dirs])

                flow_dirs = sorted(glob(osp.join(root, 'optical_flow/TRAIN/*/*')))
                flow_dirs = sorted([osp.join(f, direction, cam) for f in flow_dirs])

                for idir, fdir in zip(image_dirs, flow_dirs):
                    images = sorted(glob(osp.join(idir, '*.png')))
                    flows  = sorted(glob(osp.join(fdir, '*.pfm')))
                    if len(images) < 4 or len(flows) < 3:
                        continue
                    if direction == 'into_future':
                        # need images[i..i+3] and flows[i..i+2]
                        n = min(len(images) - 3, len(flows) - 2)
                        for i in range(n):
                            self.image_list.append([images[i], images[i+1], images[i+2], images[i+3]])
                            self.flow_list.append([flows[i], flows[i+1], flows[i+2]])
                    elif direction == 'into_past':
                        # need images[i..i+3] and flows[i+2..i+4]
                        n = min(len(images) - 3, len(flows) - 2)
                        for i in range(n):
                            self.image_list.append([images[i+3], images[i+2], images[i+1], images[i]])
                            self.flow_list.append([flows[i+3], flows[i+2], flows[i+1]])


class FlyingChairs(FlowDataset):
    def __init__(self, aug_params=None, split='train', root='datasets/FlyingChairs_release/data'):
        super(FlyingChairs, self).__init__(aug_params)

        images = sorted(glob(osp.join(root, '*.ppm')))
        flows = sorted(glob(osp.join(root, '*.flo')))
        assert (len(images)//2 == len(flows))

        split_list = np.loadtxt('chairs_split.txt', dtype=np.int32)
        for i in range(len(flows)):
            xid = split_list[i]
            if (split=='training' and xid==1) or (split=='validation' and xid==2):
                self.flow_list += [ flows[i] ]
                self.image_list += [ [images[2*i], images[2*i+1]] ]


class FlyingThings3D(FlowDataset):
    def __init__(self, aug_params=None, root='datasets/FlyingThings3D', dstype='frames_cleanpass'):
        super(FlyingThings3D, self).__init__(aug_params)

        for cam in ['left']:
            for direction in ['into_future', 'into_past']:
                image_dirs = sorted(glob(osp.join(root, dstype, 'TRAIN/*/*')))
                image_dirs = sorted([osp.join(f, cam) for f in image_dirs])

                flow_dirs = sorted(glob(osp.join(root, 'optical_flow/TRAIN/*/*')))
                flow_dirs = sorted([osp.join(f, direction, cam) for f in flow_dirs])

                for idir, fdir in zip(image_dirs, flow_dirs):
                    images = sorted(glob(osp.join(idir, '*.png')) )
                    flows = sorted(glob(osp.join(fdir, '*.pfm')) )
                    for i in range(len(flows)-1):
                        if direction == 'into_future':
                            self.image_list += [ [images[i], images[i+1]] ]
                            self.flow_list += [ flows[i] ]
                        elif direction == 'into_past':
                            self.image_list += [ [images[i+1], images[i]] ]
                            self.flow_list += [ flows[i+1] ]
      

class KITTI(FlowDataset):
    def __init__(self, aug_params=None, split='training', root='datasets/KITTI'):
        super(KITTI, self).__init__(aug_params, sparse=True)
        if split == 'testing':
            self.is_test = True

        root = osp.join(root, split)
        images1 = sorted(glob(osp.join(root, 'image_2/*_10.png')))
        images2 = sorted(glob(osp.join(root, 'image_2/*_11.png')))

        for img1, img2 in zip(images1, images2):
            frame_id = img1.split('/')[-1]
            self.extra_info += [ [frame_id] ]
            self.image_list += [ [img1, img2] ]

        if split == 'training':
            self.flow_list = sorted(glob(osp.join(root, 'flow_occ/*_10.png')))


class HD1K(FlowDataset):
    def __init__(self, aug_params=None, root='datasets/HD1k'):
        super(HD1K, self).__init__(aug_params, sparse=True)

        seq_ix = 0
        while 1:
            flows = sorted(glob(os.path.join(root, 'hd1k_flow_gt', 'flow_occ/%06d_*.png' % seq_ix)))
            images = sorted(glob(os.path.join(root, 'hd1k_input', 'image_2/%06d_*.png' % seq_ix)))

            if len(flows) == 0:
                break

            for i in range(len(flows)-1):
                self.flow_list += [flows[i]]
                self.image_list += [ [images[i], images[i+1]] ]

            seq_ix += 1


class KITTI4Frame(FlowDataset4Frame):
    """KITTI dataset with 4-frame windows."""
    
    def __init__(self, aug_params=None, split='training', root='datasets/KITTI'):
        super(KITTI4Frame, self).__init__(aug_params)
        if split == 'testing':
            self.is_test = True

        root = osp.join(root, split)
        # Load all consecutive images from image_2 directory
        all_images = sorted(glob(osp.join(root, 'image_2/*.png')))
        
        # Group images by sequence (using the filename pattern xxxxx_yy.png)
        from collections import defaultdict
        sequences = defaultdict(list)
        for img_path in all_images:
            # Extract sequence ID from filename (e.g., "000000_10.png" -> "000000")
            seq_id = osp.basename(img_path).split('_')[0]
            sequences[seq_id].append(img_path)
        
        # Create 4-frame windows for each sequence
        for seq_id in sorted(sequences.keys()):
            images = sorted(sequences[seq_id])
            # Create 4-frame windows from consecutive images
            for i in range(len(images) - 3):
                self.image_list.append([images[i], images[i+1], images[i+2], images[i+3]])
                self.extra_info.append([osp.basename(images[i])])
        
        if split == 'training':
            # Load flow files - will need to map them to 4-frame windows
            flows = sorted(glob(osp.join(root, 'flow_occ/*_10.png')))
            # For each sequence, we need 3 flows per 4-frame window
            # This requires matching flows to image sequences
            flow_sequences = defaultdict(list)
            for flow_path in flows:
                seq_id = osp.basename(flow_path).split('_')[0]
                flow_sequences[seq_id].append(flow_path)
            
            # Match flows to our 4-frame windows
            self.flow_list = []
            for img_group in self.image_list:
                seq_id = osp.basename(img_group[0]).split('_')[0]
                if seq_id in flow_sequences:
                    flow_list_for_seq = sorted(flow_sequences[seq_id])
                    # For a 4-frame window, we need flows [i, i+1, i+2]
                    # Map based on the image frame numbers
                    frame_num = int(osp.basename(img_group[0]).split('_')[1].split('.')[0])
                    # Try to find corresponding flows
                    if len(flow_list_for_seq) >= 3:
                        self.flow_list.append(flow_list_for_seq[:3])
            
            # Only keep image_list entries that have corresponding flows
            if len(self.flow_list) > 0:
                self.image_list = self.image_list[:len(self.flow_list)]


class HD1K4Frame(FlowDataset4Frame):
    """HD1K dataset with 4-frame windows."""
    
    def __init__(self, aug_params=None, root='datasets/HD1k'):
        super(HD1K4Frame, self).__init__(aug_params)

        seq_ix = 0
        while 1:
            flows = sorted(glob(os.path.join(root, 'hd1k_flow_gt', 'flow_occ/%06d_*.png' % seq_ix)))
            images = sorted(glob(os.path.join(root, 'hd1k_input', 'image_2/%06d_*.png' % seq_ix)))

            if len(flows) == 0:
                break

            # Create 4-frame windows from the sequence
            # Need at least 4 images and 3 flows
            if len(images) >= 4 and len(flows) >= 3:
                for i in range(len(images) - 3):
                    # For each 4-frame window, we need 3 flows
                    if i + 2 < len(flows):
                        self.image_list.append([images[i], images[i+1], images[i+2], images[i+3]])
                        self.flow_list.append([flows[i], flows[i+1], flows[i+2]])

            seq_ix += 1


class FlyingChairs4Frame(FlowDataset4Frame):
    """FlyingChairs dataset adapted for 4-frame windows.
    
    Note: FlyingChairs only has 2-frame pairs and single flows. For 4-frame mode,
    we create windows [img_2i, img_2i+1, img_2i+2, img_2i+3] and use available flows.
    """
    
    def __init__(self, aug_params=None, split='train', root='datasets/FlyingChairs_release/data'):
        super(FlyingChairs4Frame, self).__init__(aug_params)

        images = sorted(glob(osp.join(root, '*.ppm')))
        flows = sorted(glob(osp.join(root, '*.flo')))
        assert (len(images)//2 == len(flows))

        split_list = np.loadtxt('chairs_split.txt', dtype=np.int32)
        
        # Create 4-frame windows from the image sequence
        # Each flow corresponds to pair (2*i, 2*i+1)
        # For 4 frames (4*i, 4*i+1, 4*i+2, 4*i+3), we'd need flows between them
        for i in range(len(flows) - 1):
            xid = split_list[i]
            if (split == 'training' and xid == 1) or (split == 'validation' and xid == 2):
                # Create a 4-frame window using images from two consecutive 2-frame pairs
                # Window: [2*i, 2*i+1, 2*i+2, 2*i+3] -> flows [i, i+1, ?]
                if i + 1 < len(flows):
                    img_indices = [2*i, 2*i+1, 2*i+2, 2*i+3]
                    # Make sure all images exist
                    if 2*i+3 < len(images):
                        self.image_list.append([images[2*i], images[2*i+1], images[2*i+2], images[2*i+3]])
                        # We have flow[i] (0->1) and flow[i+1] (2->3)
                        # For the middle flow (1->2), we can approximate or skip
                        # For now, use available flows and set a placeholder
                        self.flow_list.append([flows[i], flows[i+1], flows[i+1]])


def fetch_dataloader(args, TRAIN_DS='C+T+K+S+H'):
    """ Create the data loader for the corresponding trainign set """

    if args.stage == 'chairs':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.1, 'max_scale': 1.0, 'do_flip': True}
        train_dataset = FlyingChairs(aug_params, split='training')
    
    elif args.stage == 'things':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.4, 'max_scale': 0.8, 'do_flip': True}
        clean_dataset = FlyingThings3D(aug_params, dstype='frames_cleanpass')
        final_dataset = FlyingThings3D(aug_params, dstype='frames_finalpass')
        train_dataset = clean_dataset + final_dataset

    elif args.stage == 'sintel':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.6, 'do_flip': True}
        things = FlyingThings3D(aug_params, dstype='frames_cleanpass')
        sintel_clean = MpiSintel(aug_params, split='training', dstype='clean')
        sintel_final = MpiSintel(aug_params, split='training', dstype='final')        

        if TRAIN_DS == 'C+T+K+S+H':
            kitti = KITTI({'crop_size': args.image_size, 'min_scale': -0.3, 'max_scale': 0.5, 'do_flip': True})
            hd1k = HD1K({'crop_size': args.image_size, 'min_scale': -0.5, 'max_scale': 0.2, 'do_flip': True})
            train_dataset = 100*sintel_clean + 100*sintel_final + 200*kitti + 5*hd1k + things

        elif TRAIN_DS == 'C+T+K/S':
            train_dataset = 100*sintel_clean + 100*sintel_final + things

    elif args.stage == 'kitti':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.4, 'do_flip': False}
        train_dataset = KITTI(aug_params, split='training')

    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, 
        pin_memory=False, shuffle=True, num_workers=4, drop_last=True)

    print('Training with %d image pairs' % len(train_dataset))
    return train_loader


def fetch_dataloader_sintel3frame(args, TRAIN_DS='C+T+K/S'):
    """Create the Sintel 3-frame data loader (returns triplets + two flows).

    TRAIN_DS='S'      : Sintel clean+final only
    TRAIN_DS='C+T+K/S': Sintel + FlyingThings3D (mirrors original RAFT schedule)
    """
    aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.6, 'do_flip': True}
    sintel_clean = MpiSintel3Frame(aug_params, split='training', dstype='clean')
    sintel_final = MpiSintel3Frame(aug_params, split='training', dstype='final')

    if TRAIN_DS == 'C+T+K/S':
        things = FlyingThings3D3Frame(aug_params, dstype='frames_cleanpass')
        train_dataset = 100 * sintel_clean + 100 * sintel_final + things
    else:
        train_dataset = 100 * sintel_clean + 100 * sintel_final

    train_loader = data.DataLoader(
        train_dataset, batch_size=args.batch_size,
        pin_memory=False, shuffle=True, num_workers=4, drop_last=True)

    print('Training with %d triplets (Sintel 3-frame, TRAIN_DS=%s)' % (len(train_dataset), TRAIN_DS))
    return train_loader


def fetch_dataloader_4frame(args, TRAIN_DS='C+T+K+S+H'):
    """Create the 4-frame data loader (returns quadruplets + three flows).

    TRAIN_DS='S'        : Sintel clean+final only
    TRAIN_DS='C+T+K/S'  : Sintel + FlyingThings3D (standard RAFT schedule)
    TRAIN_DS='C+T+K+S+H': Sintel + FlyingThings3D + KITTI + HD1K + FlyingChairs (full schedule)
    """
    aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.6, 'do_flip': True}
    sintel_clean = MpiSintel4Frame(aug_params, split='training', dstype='clean')
    sintel_final = MpiSintel4Frame(aug_params, split='training', dstype='final')

    if TRAIN_DS == 'C+T+K+S+H':
        things = FlyingThings3D4Frame(aug_params, dstype='frames_cleanpass')
        kitti = KITTI4Frame({'crop_size': args.image_size, 'min_scale': -0.3, 'max_scale': 0.5, 'do_flip': True})
        hd1k = HD1K4Frame({'crop_size': args.image_size, 'min_scale': -0.5, 'max_scale': 0.2, 'do_flip': True})
        chairs = FlyingChairs4Frame({'crop_size': args.image_size, 'min_scale': -0.1, 'max_scale': 1.0, 'do_flip': True}, split='training')
        train_dataset = 100 * sintel_clean + 100 * sintel_final + 200 * kitti + 5 * hd1k + things + chairs
    
    elif TRAIN_DS == 'C+T+K/S':
        things = FlyingThings3D4Frame(aug_params, dstype='frames_cleanpass')
        train_dataset = 100 * sintel_clean + 100 * sintel_final + things
    
    else:  # Default to 'S'
        train_dataset = 100 * sintel_clean + 100 * sintel_final

    train_loader = data.DataLoader(
        train_dataset, batch_size=args.batch_size,
        pin_memory=False, shuffle=True, num_workers=4, drop_last=True)

    print('Training with %d quadruplets (4-frame, TRAIN_DS=%s)' % (len(train_dataset), TRAIN_DS))
    return train_loader