#!/usr/bin/bash
echo "[USAGE] sh scripts/slurm_train.sh <ID> <GPU_NUM> <PORT> <CONFIG> <JOB_NAME>"
set -x

ID=$1
GPU_NUM=$2
PORT=$3
CONFIG=$4
JOB_NAME=$5

HOST='SG-IDC1-10-51-2-'$ID

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
python -u tools/train.py \
        configs/pretrain/imagenet/1k/$CONFIG.py \
        --work_dir $EXPS/$JOB_NAME \
        --port $PORT \
        2>&1 | tee $EXPS/$JOB_NAME.log > /dev/null &

echo "tail -f $EXPS/$JOB_NAME.log"
