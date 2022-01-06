#!/usr/bin/bash
echo '[USAGE] sh scripts/slurm_eval.sh <CONFIG> <ID> <GPU_NUM> <PORT> <CKPT> <JOB_NAME>'
set -x

CONFIG=$1
ID=$2
GPU_NUM=$3
PORT=$4
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
srun -p dsta --mpi=pmi2 --gres=gpu:$NTASKS -n$GPU_NUM --ntasks-per-node=$NTASKS \
--kill-on-bad-exit=1 -w $HOST --job-name=$JOB_NAME \
python -u tools/eval.py \
        configs/eval/imagenet/1k/$CONFIG.py \
        --work_dir $EXPS/$JOB_NAME \
        --load_from $CHECKPOINT \
        --port $PORT \
        2>&1 | tee $EXPS/$JOB_NAME.log > /dev/null &
