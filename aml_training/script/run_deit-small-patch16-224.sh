#!/usr/bin/bash

cd /root/code/BinaryViT
source activate base
##### packages required for py38:  python 3.8, torch>=1.10.1, torchvision>=0.11.2, timm==0.6.12, transformers>=4.20.1  wandb
conda activate py38
DATA_DIR=/data/scy/ImageNet     ### ILSVRC2012 dataset,  subdir:  "/train"  and  "/val"    please revise!!!
 
torchrun --nproc_per_node=4 main.py \
    --num-workers=32 \
    --batch-size=512 \
    --epochs=300 \
    --model=configs/deit-small-patch16-224 \
    --dropout=0.0 \
    --drop-path=0.1 \
    --opt=adamw \
    --sched=cosine \
    --weight-decay=0.05 \
    --lr=5e-4 \
    --warmup-epochs=5 \
    --color-jitter=0.4 \
    --aa=rand-m9-mstd0.5-inc1 \
    --repeated-aug \
    --reprob=0.25 \
    --mixup=0.8 \
    --cutmix=1.0 \
    --data-path=${DATA_DIR} \
    --output-dir=logs/deit-small-patch16-224 \