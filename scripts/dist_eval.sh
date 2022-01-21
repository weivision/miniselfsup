#!/usr/bin/bash
echo '[USAGE] sh scripts/dist_eval.sh <ID> <GPU_NUM> <PORT> <CONFIG> <CKPT> <JOB_NAME>'
set -x

ID=$1
GPU_NUM=$2
PORT=$3
CONFIG=$4
CKPT=$5
JOB_NAME=$6

HOST='SG-IDC1-10-51-2-'$ID

CHECKPOINT='pretrained/'$CKPT


EXPS='exps'
if ! [ -d "$EXPS" ]; then
   mkdir -p $EXPS
fi

if [ $GPU_NUM -gt 8 ]
then
   NTASKS=8
else
   NTASKS=$GPU_NUM
fi


export PYTHONPATH=./:$PYTHONPATH
srun -p dsta --mpi=pmi2 --gres=gpu:$GPU_NUM -n1 \
--ntasks-per-node=1 --kill-on-bad-exit=1 \
-w $HOST --job-name=$JOB_NAME \
python -m torch.distributed.launch --nproc_per_node $GPU_NUM \
      --use_env \
      tools/eval.py \
        configs/eval/imagenet/1k/$CONFIG.py \
        --work_dir $EXPS/$JOB_NAME \
        --load_from $CHECKPOINT \
        --port $PORT \
        2>&1 | tee $EXPS/$JOB_NAME.log > /dev/null &
