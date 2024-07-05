# !/bin/bash
HERE="$(dirname "$(readlink -f "$0")")"
# export RANK=0
# export LOCAL_RANK=0
# export MASTER_ADDR=127.0.0.1
# export MASTER_PORT=12358

if [ -z $WORLD_SIZE ]; then
  GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
  # export MASTER_ADDR=127.0.0.1
  # export MASTER_PORT=12358

  torchrun --nproc_per_node=${GPU_COUNT} \
  	   $HERE/start_train.py
else
  python $HERE/start_train.py
fi