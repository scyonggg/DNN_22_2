#!/bin/bash

python train.py \
--net resnet50 \
--num_workers 4 \
--batch_size 256 \
--lr 1e-3 \
--lr_scheduler 'cosinelr' \
--epochs 100 \
--warm_t 10 \
--gpus 0 \
--log_step 1000 \
--val_step 5 \
--save_step 10 \
--wandb \
--run_name 'test'