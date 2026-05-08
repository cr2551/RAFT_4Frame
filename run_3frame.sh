#!/bin/bash
#SBATCH --job-name=raft-3frame
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH -A hpc_vision02
#SBATCH --cpus-per-task=12
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -eo pipefail

REPO_DIR=/work/yxiao10/Consistency_RAFT/RAFT/
PYTHON_BIN=/scratch/yxiao10/tools/miniconda3/envs/raft-hpc/bin/python

cd "$REPO_DIR"

module purge
module load gcc/11.2.0
module load cuda/11.6.0

export PATH="/scratch/yxiao10/tools/miniconda3/envs/raft-hpc/bin:$PATH"
export PYTHONPATH="$REPO_DIR/core:${PYTHONPATH:-}"

echo "============================================"
echo " Task : 3-frame RAFT fine-tune on Sintel"
echo "        GT flow = compose(flow01, flow12)"
echo "        Input   = (img0, img2)"
echo " Ckpt : checkpoints/raft-things.pth"
echo " Job  : ${SLURM_JOB_NAME}-${SLURM_JOB_ID}"
echo " Host : $(hostname)"
echo "============================================"
nvidia-smi || true

"$PYTHON_BIN" -u 3Frame_train.py \
  --name raft-3frame-sintel \
  --ckpt_dir checkpoints/3frame-RAFT_thingsckpt \
  --stage sintel \
  --validation sintel \
  --restore_ckpt checkpoints/raft-things.pth \
  --gpus 0 1 \
  --num_steps 100000 \
  --batch_size 6 \
  --lr 0.000125 \
  --image_size 368 768 \
  --wdecay 0.00001 \
  --gamma 0.85
