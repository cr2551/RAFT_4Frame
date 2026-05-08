# RAFT
This repository contains the source code for our paper:

[RAFT: Recurrent All Pairs Field Transforms for Optical Flow](https://arxiv.org/pdf/2003.12039.pdf)<br/>
ECCV 2020 <br/>
Zachary Teed and Jia Deng<br/>

<img src="RAFT.png">

## Requirements
The code has been tested with PyTorch 1.6 and Cuda 10.1.
```Shell
conda create --name raft
conda activate raft
conda install pytorch=1.6.0 torchvision=0.7.0 cudatoolkit=10.1 matplotlib tensorboard scipy opencv -c pytorch
```

## Demos
Pretrained models can be downloaded by running
```Shell
./download_models.sh
```
or downloaded from [google drive](https://drive.google.com/drive/folders/1sWDsfuZ3Up38EUQt7-JDTT1HcGHuJgvT?usp=sharing)

You can demo a trained model on a sequence of frames
```Shell
python demo.py --model=models/raft-things.pth --path=demo-frames
```

## Required Data
To evaluate/train RAFT, you will need to download the required datasets. 
* [FlyingChairs](https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html#flyingchairs)
* [FlyingThings3D](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)
* [Sintel](http://sintel.is.tue.mpg.de/)
* [KITTI](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow)
* [HD1K](http://hci-benchmark.iwr.uni-heidelberg.de/) (optional)


By default `datasets.py` will search for the datasets in these locations. You can create symbolic links to wherever the datasets were downloaded in the `datasets` folder

```Shell
├── datasets
    ├── Sintel
        ├── test
        ├── training
    ├── KITTI
        ├── testing
        ├── training
        ├── devkit
    ├── FlyingChairs_release
        ├── data
    ├── FlyingThings3D
        ├── frames_cleanpass
        ├── frames_finalpass
        ├── optical_flow
```

## Evaluation
You can evaluate a trained model using `evaluate.py`
```Shell
python evaluate.py --model=models/raft-things.pth --dataset=sintel --mixed_precision
```

## Training
We used the following training schedule in our paper (2 GPUs). Training logs will be written to the `runs` which can be visualized using tensorboard
```Shell
./train_standard.sh
```

If you have a RTX GPU, training can be accelerated using mixed precision. You can expect similiar results in this setting (1 GPU)
```Shell
./train_mixed.sh
```

### Three-frame composed flow training

This fork also supports supervising `RAFT(image_0, image_2)` with an online
composition of adjacent flows:

```text
flow_02(x) = flow_01(x) + flow_12(x + flow_01(x))
```

`flow_12` is bilinearly sampled with `torch.grid_sample(..., align_corners=True)`,
matching the coordinate convention used by RAFT's existing sampler. The composed
valid mask requires the source pixel to be valid in `valid_01`, the warped
location `x + flow_01(x)` to be inside frame 1 / `flow_12`, and the sampled
`valid_12` value to be valid. If no valid-mask files are present, the loader uses
all-ones `valid_01` and `valid_12` before applying the boundary test.

Expected dataset layout:

```text
dataset_root/
  images/
    seq_xxx/
      000000.png
      000001.png
      000002.png
  flow01/
    seq_xxx/
      000000.flo
  flow12/
    seq_xxx/
      000000.flo
  valid01/              optional
    seq_xxx/
      000000.png
  valid12/              optional
    seq_xxx/
      000000.png
```

`flow01/seq_xxx/000000.*` is frame 0->1, and
`flow12/seq_xxx/000000.*` is frame 1->2 for the triple starting at
`images/seq_xxx/000000.png`. Flow files can be `.flo`, `.pfm`, or `.npy`.

There is also a Sintel-native three-frame loader. It uses the standard Sintel
layout and reads `training/flow/<scene>/frame_xxxx.flo` as adjacent flows:

```text
datasets/Sintel/
  training/
    clean/<scene>/frame_0001.png, frame_0002.png, frame_0003.png, ...
    final/<scene>/frame_0001.png, frame_0002.png, frame_0003.png, ...
    flow/<scene>/frame_0001.flo, frame_0002.flo, ...
```

For frames `[i, i+1, i+2]`, it composes `flow/frame_i.flo` and
`flow/frame_{i+1}.flo` online, then trains `RAFT(frame_i, frame_{i+2})`.

Generic three-frame training command on Windows:

```bat
cd F:\Consistency_flow\RAFT\RAFT

python train.py ^
  --name raft_composed_flow02 ^
  --stage three_frame ^
  --dataset_root <your_dataset_root> ^
  --use_composed_flow ^
  --batch_size 4 ^
  --lr 0.0001 ^
  --num_steps 100000 ^
  --image_size 368 768 ^
  --gpus 0
```

Sintel three-frame training command:

```bat
cd F:\Consistency_flow\RAFT\RAFT

python train.py ^
  --name raft_sintel_flow02 ^
  --stage sintel_three_frame ^
  --dataset_root datasets/Sintel ^
  --sintel_dstype clean+final ^
  --use_composed_flow ^
  --batch_size 4 ^
  --lr 0.0001 ^
  --num_steps 100000 ^
  --image_size 368 768 ^
  --gpus 0
```

Minimal smoke test:

```bat
cd F:\Consistency_flow\RAFT\RAFT

python smoke_test_three_frame.py ^
  --dataset_root <your_dataset_root> ^
  --image_size 368 768 ^
  --batch_size 1 ^
  --small ^
  --iters 1 ^
  --device cuda
```

Sintel smoke test:

```bat
cd F:\Consistency_flow\RAFT\RAFT

python smoke_test_three_frame.py ^
  --dataset sintel_three_frame ^
  --dataset_root datasets/Sintel ^
  --sintel_dstype clean ^
  --image_size 368 768 ^
  --batch_size 1 ^
  --small ^
  --iters 1 ^
  --device cuda
```

## (Optional) Efficent Implementation
You can optionally use our alternate (efficent) implementation by compiling the provided cuda extension
```Shell
cd alt_cuda_corr && python setup.py install && cd ..
```
and running `demo.py` and `evaluate.py` with the `--alternate_corr` flag Note, this implementation is somewhat slower than all-pairs, but uses significantly less GPU memory during the forward pass.
