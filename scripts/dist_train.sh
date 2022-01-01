#!/usr/bin/bash
GPU_NUM=$1
JOB_NAME=jupyter-$2


GPUS=${GPUS:-8}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
CPUS_PER_TASK=${CPUS_PER_TASK:-5}

EXPS='exps'
if ! [ -d "$EXPS" ]; then
   mkdir -p $EXPS
fi


srun -p dsta --mpi=pmi2 --gres=gpu:$GPU_NUM -n1 --ntasks-per-node=1 --kill-on-bad-exit=1 -w SG-IDC1-10-51-2-77 --job-name=$JOB_NAME \
python -m torch.distributed.launch \
      --nproc_per_node=$GPU_NUM \
      --use_env \
      tools/train.py \
        configs/pretrain/imagenet/1k/simsiam.py \
        --work_dir $EXPS/$JOB_NAME \
        2>&1 | tee $EXPS/$JOB_NAME.log > /dev/null &
