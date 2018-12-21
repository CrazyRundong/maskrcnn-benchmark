#!/usr/bin/env bash

source $HOME/.local/bin/env_spring.sh
source activate torch-1.0

GLOG_vmodule=MemcachedClient=-1 srun \
   --mpi=pmi2 -p VIBackEnd -n8 \
   --gres=gpu:8 --ntasks-per-node=8 \
   --job-name=maskrcnn_test \
   --kill-on-bad-exit=1 \
python test_net.py \
  --config-file ../configs/e2e_faster_rcnn_R_101_FPN_1x.yaml \
  --port $RANDOM
