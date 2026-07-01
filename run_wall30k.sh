#!/bin/bash
# DDP launch for the 30k wall-masking run on 2x A40 (46GB).
# Uses DistributedDataParallel (NOT nn.DataParallel) and disables broken
# PCIe peer-to-peer in this container, which otherwise deadlocks both
# DataParallel scatter/gather and NCCL collectives.
set -e
cd "$(dirname "$0")"

NGPU=${NGPU:-2}
BATCH_SIZE=${BATCH_SIZE:-3}          # per-GPU batch; 3 -> ~40GB/46GB on A40
CKPT_DIR=${CKPT_DIR:-ckpts/wallmasking-30k-1024-bf16-ddp}
EPOCHS=${EPOCHS:-250}
WANDB=${WANDB:-True}

mkdir -p "${CKPT_DIR}"
echo "DDP run @ $(date): ngpu=${NGPU} per_gpu_batch=${BATCH_SIZE} ckpt=${CKPT_DIR}"

export NCCL_P2P_DISABLE=1
export WANDB_MODE=${WANDB_MODE:-offline}

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1} \
torchrun --standalone --nproc_per_node "${NGPU}" \
    train.py \
        --ckpt_dir "${CKPT_DIR}" \
        --epochs "${EPOCHS}" \
        --dist True \
        --batch_size "${BATCH_SIZE}" \
        --wandb "${WANDB}"
