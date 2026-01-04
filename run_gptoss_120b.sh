export GPUS_PER_NODE=$(nvidia-smi -L | wc -l)
export MASTER_ADDR=$ARNOLD_WORKER_0_HOST
export MASTER_PORT=12345
export NNODES=$ARNOLD_NUM
export GPUS_PER_NODE=$ARNOLD_WORKER_GPU
export WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
export NODE_RANK=$ARNOLD_ID


PP_SIZE=1
CP_SIZE=16
EP_SIZE=4
DP_SIZE=$(( WORLD_SIZE/(PP_SIZE*CP_SIZE) ))

MICRO_BATCH_SIZE=1
NUM_MICROBATCHES=1
GLOBAL_BATCH_SIZE=$(( MICRO_BATCH_SIZE * DP_SIZE * NUM_MICROBATCHES ))

SEQ_LENGTH=65536
MAX_POSITION_EMBEDDINGS=65536


NUM_EXPERTS=32
TOPK=4  # top-4 routing, consistent with GPT-OSS

#   File "/opt/tiger/fsdp-eval/Megatron-LM/megatron/training/arguments.py", line 963, in validate_args
#     assert os.environ.get('CUDA_DEVICE_MAX_CONNECTIONS') == "1", \
#            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# AssertionError: Using tensor model parallelism or context parallelism require setting the environment variable CUDA_DEVICE_MAX_CONNECTIONS to 1
# export CUDA_DEVICE_MAX_CONNECTIONS=1


DATA_CACHE_PATH="/mnt/hdfs/__MERLIN_USER_DIR__/benchmark_training"
mkdir -p "$DATA_CACHE_PATH"

DATA_ARGS_LIST=(
    --mock-data
    --tokenizer-type NullTokenizer
    --vocab-size 201088
    --data-cache-path ${DATA_CACHE_PATH}
    --tiktoken-pattern v2
    --split '99,1,0'
    --no-create-attention-mask-in-dataloader
    --no-mmap-bin-files
)


# [rank0]: RuntimeError: /TransformerEngine/transformer_engine/common/fused_attn/fused_attn_f16_arbitrary_seqlen.cu:405 in function operator(): cuDNN Error: CUDNN_BACKEND_TENSOR_DESCRIPTOR cudnnFinalize failedptrDesc->finalize() cudnn_status: CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED. For more information, enable cuDNN error logging by setting CUDNN_LOGERR_DBG=1 and CUDNN_LOGDEST_DBG=stderr in the environment.
#   --attention-backend flash \
# By default it's auto.
MODEL_ARGS=(
    --num-layers 6
    --hidden-size 2880
    --ffn-hidden-size 2880
    --num-attention-heads 64
    --group-query-attention
    --num-query-groups 8
    --kv-channels 64
    --seq-length $SEQ_LENGTH
    --max-position-embeddings $MAX_POSITION_EMBEDDINGS
    --position-embedding-type rope
    --rotary-base 150000
    --rotary-percent 1.0
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --swiglu
    --init-method-std 0.0134
    --normalization RMSNorm
    --norm-epsilon 1e-5
    --attention-backend auto
    --apply-layernorm-1p
    --untie-embeddings-and-output-weights
    --disable-bias-linear
    --no-bias-dropout-fusion
)

MOE_ARGS=(
    --num-experts ${NUM_EXPERTS}
    --moe-router-topk ${TOPK}
    --moe-token-dispatcher-type alltoall
    --moe-grouped-gemm
    --moe-permute-fusion
    --moe-router-force-load-balancing
    # --moe-router-load-balancing-type aux_loss
    # --moe-aux-loss-coeff 1e-2
)

TRAINING_ARGS=(
    --micro-batch-size $MICRO_BATCH_SIZE
    --global-batch-size $GLOBAL_BATCH_SIZE
    --lr 1e-4
    --train-iters 30
    --min-lr 1.0e-5
    --lr-decay-iters 2
    --lr-decay-style cosine
    --clip-grad 1.0
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.95
    --bf16
    # --grad-reduce-in-bf16
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 1
    --eval-iters 10
    --eval-interval 100000
    --save-interval 100000
    --log-throughput
    --no-load-optim
    --no-load-rng
    # --ckpt-format fsdp_dtensor
    --profile
    --profile-ranks $(seq -s ' ' 0 $((WORLD_SIZE-1)))
    --use-pytorch-profiler
    --tensorboard-dir ./profile
    --profile-step-start 8
    --profile-step-end 10
)

# Model parallelism arguments
MODEL_PARALLEL_ARGS=(
    # --data-parallel-sharding-strategy optim_grads_params
    # --use-distributed-optimizer
    # --overlap-grad-reduce
    # --overlap-param-gather
    --no-gradient-accumulation-fusion
    --expert-model-parallel-size $EP_SIZE
    --pipeline-model-parallel-size $PP_SIZE
    --context-parallel-size $CP_SIZE
    --cp-comm-type p2p
    --sequence-parallel
)

torchrun \
  --nnodes=$NNODES \
  --nproc_per_node=$GPUS_PER_NODE \
  --node_rank=$NODE_RANK \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  pretrain_gpt.py \
  ${MODEL_PARALLEL_ARGS[@]} \
  ${TRAINING_ARGS[@]} \
  ${DATA_ARGS_LIST[@]} \
  ${MODEL_ARGS[@]} \
  ${MOE_ARGS[@]} \
  ${EVAL_AND_LOGGING_ARGS[@]}
