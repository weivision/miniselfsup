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

CHECKPOINT='pretrained/simsiam_r50_8gpus.pth'

export PYTHONPATH=./:$PYTHONPATH
GLOG_vmodule=MemcachedClient=-1 \
srun -p dsta --mpi=pmi2 --gres=gpu:$GPU_NUM -n$GPU_NUM --ntasks-per-node=$GPU_NUM --kill-on-bad-exit=1 -w SG-IDC1-10-51-2-77 --job-name=$JOB_NAME \
python -u tools/eval.py \
        configs/eval/imagenet/1k/linear_cls_simsiam.py \
        --work_dir $EXPS/$JOB_NAME \
        --load_from $CHECKPOINT \
        --port '19908' \
        2>&1 | tee $EXPS/$JOB_NAME.log > /dev/null &
