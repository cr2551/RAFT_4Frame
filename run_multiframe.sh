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

REPO_DIR=/work/crod143/flow/RAFT_4Frame
#PYTHON_BIN=/scratch/yxiao10/tools/miniconda3/envs/raft-hpc/bin/python


# ACTIVATE CONDA
source /work/crod143/miniconda3/etc/profile.d/conda.sh
conda activate raft


cd "$REPO_DIR"

module purge
module load gcc/11.2.0
module load cuda/11.6.0

#export PATH="/scratch/yxiao10/tools/miniconda3/envs/raft-hpc/bin:$PATH"
#export PYTHONPATH="$REPO_DIR/core:${PYTHONPATH:-}"

echo "============================================"
echo " Task : 4-frame RAFT fine-tune on Sintel"
echo "        Input   = (img0, img1, img2, img3)"
echo " Ckpt : checkpoints/raft-things.pth"
echo " Job  : ${SLURM_JOB_NAME}-${SLURM_JOB_ID}"
echo " Host : $(hostname)"
echo "============================================"
nvidia-smi || true

python -u 4Frame_train.py \
  --name raft-4frame-sintel \
  --ckpt_dir checkpoints/ \
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
