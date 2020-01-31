#! /bin/bash

# Runs the "345M" parameter model

GPUS_PER_NODE=4
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

python -m torch.distributed.launch $DISTRIBUTED_ARGS \
       main.py \
       --model-parallel-size 2\
       --distributed-backend nccl\
       --batch-size 1\
       --seq-length 512 \
       --train-data wikipedia \
       --lazy-loader \
       --tokenizer-type XLNetWordPieceTokenizer \
       --tokenizer-model-type bert-base-uncased \
       --presplit-sentences \
       --cache-dir cache \
       --split 949,50,1 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --warmup .01 \


set +x
