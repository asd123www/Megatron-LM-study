PP_SIZE=2; CP_SIZE=1; EP_SIZE=4
WORLD_SIZE=8
DP_SIZE=$(( WORLD_SIZE/(PP_SIZE*CP_SIZE*EP_SIZE) ))

MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=16

SEQ_LENGTH=$(( 1024*CP_SIZE ))
MAX_POSITION_EMBEDDINGS=$(( 1024*CP_SIZE ))


NUM_EXPERTS=8
TOPK=4  # top-4 routing, consistent with GPT-OSS

#   File "/opt/tiger/fsdp-eval/Megatron-LM/megatron/training/arguments.py", line 963, in validate_args
#     assert os.environ.get('CUDA_DEVICE_MAX_CONNECTIONS') == "1", \
#            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# AssertionError: Using tensor model parallelism or context parallelism require setting the environment variable CUDA_DEVICE_MAX_CONNECTIONS to 1
export CUDA_DEVICE_MAX_CONNECTIONS=1


DATA_CACHE_PATH="${PWD}/benchmark_training"
mkdir -p "$DATA_CACHE_PATH"

DATA_ARGS_LIST=(
    --mock-data
    --tokenizer-type NullTokenizer
    --vocab-size 32000
    --data-cache-path ${DATA_CACHE_PATH}
    --tiktoken-pattern v2
    --split '99,1,0'
    --no-create-attention-mask-in-dataloader
    --no-mmap-bin-files
    --num-workers 1
)


# [rank0]: RuntimeError: /TransformerEngine/transformer_engine/common/fused_attn/fused_attn_f16_arbitrary_seqlen.cu:405 in function operator(): cuDNN Error: CUDNN_BACKEND_TENSOR_DESCRIPTOR cudnnFinalize failedptrDesc->finalize() cudnn_status: CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED. For more information, enable cuDNN error logging by setting CUDNN_LOGERR_DBG=1 and CUDNN_LOGDEST_DBG=stderr in the environment.
#   --attention-backend flash \
# By default it's auto.
MODEL_ARGS=(
    --num-layers 8
    --hidden-size 512
    --ffn-hidden-size 1792
    --num-attention-heads 8
    --group-query-attention
    --num-query-groups 8
    --kv-channels 128
    --seq-length $SEQ_LENGTH
    --max-position-embeddings $MAX_POSITION_EMBEDDINGS
    --position-embedding-type rope
    --rotary-base 1000000 
    --rotary-percent 1.0
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --swiglu
    --init-method-std 0.0134
    --attention-backend flash
    --apply-layernorm-1p
    --untie-embeddings-and-output-weights
    --disable-bias-linear
    --no-bias-dropout-fusion
)

MOE_ARGS=(
    --num-experts ${NUM_EXPERTS}
    --moe-router-topk ${TOPK}
    --moe-router-load-balancing-type aux_loss
    --moe-aux-loss-coeff 1e-2
    --moe-token-dispatcher-type alltoall
    --moe-grouped-gemm
    --moe-permute-fusion
    --use-distributed-optimizer
)

TRAINING_ARGS=(
    --micro-batch-size $MICRO_BATCH_SIZE
    --global-batch-size $GLOBAL_BATCH_SIZE
    --lr 0.00015
    --min-lr 0.00001
    --decoupled-lr 5.0e-4      # Specific to decoupled AdamW, ensure optimizer is compatible
    --decoupled-min-lr 4.5e-5  # Specific to decoupled AdamW
    --lr-decay-style cosine
    --clip-grad 1.0
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.95
    --bf16
    --grad-reduce-in-bf16
    --cross-entropy-loss-fusion
    --calculate-per-token-loss
    --manual-gc
    --empty-unused-memory-level 1
)


EVAL_AND_LOGGING_ARGS=(
    --log-interval 1
    --eval-iters 1
    --eval-interval 100000000
    --eval-interval 100
    --save-interval 1000
    --log-throughput
    --profile
    --use-pytorch-profiler
    --tensorboard-dir ./profile
    --profile-step-start 4
    --profile-step-end 6
)


# Model parallelism arguments
MODEL_PARALLEL_ARGS=(
    --expert-model-parallel-size $EP_SIZE
    --pipeline-model-parallel-size $PP_SIZE
    --context-parallel-size $CP_SIZE
    --sequence-parallel  # Always enable sequence parallelism with TP_SIZE=2
    --use-distributed-optimizer
    --overlap-grad-reduce
    --overlap-param-gather
)
# --context-parallel-size  --cp-comm-type allgather \

torchrun --standalone --nproc_per_node=8 pretrain_gpt.py \
  --train-iters 10 \
  ${MODEL_PARALLEL_ARGS[@]} \
  ${TRAINING_ARGS[@]} \
  ${DATA_ARGS_LIST[@]} \
  ${MODEL_ARGS[@]} \
  ${MOE_ARGS[@]} \
  ${EVAL_AND_LOGGING_ARGS[@]}
