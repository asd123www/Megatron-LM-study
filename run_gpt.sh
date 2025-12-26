TP=2; PP=2; CP=1; EP=1   # change these per test
WORLD=8
DP=$(( WORLD/(TP*PP*CP*EP) ))

MBS=1
SEQ=$(( 1024*CP ))        # makes ~1024 tokens per GPU even when CP>1
GBS=$(( DP*MBS ))         # avoids grad accumulation (1 microbatch)


#   File "/opt/tiger/fsdp-eval/Megatron-LM/megatron/training/arguments.py", line 963, in validate_args
#     assert os.environ.get('CUDA_DEVICE_MAX_CONNECTIONS') == "1", \
#            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# AssertionError: Using tensor model parallelism or context parallelism require setting the environment variable CUDA_DEVICE_MAX_CONNECTIONS to 1
export CUDA_DEVICE_MAX_CONNECTIONS=1


# [rank0]: RuntimeError: /TransformerEngine/transformer_engine/common/fused_attn/fused_attn_f16_arbitrary_seqlen.cu:405 in function operator(): cuDNN Error: CUDNN_BACKEND_TENSOR_DESCRIPTOR cudnnFinalize failedptrDesc->finalize() cudnn_status: CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED. For more information, enable cuDNN error logging by setting CUDNN_LOGERR_DBG=1 and CUDNN_LOGDEST_DBG=stderr in the environment.
#   --attention-backend flash \
# By default it's auto.


torchrun --standalone --nproc_per_node=8 pretrain_gpt.py \
  --mock-data \
  --tokenizer-type NullTokenizer --vocab-size 32000 \
  --tensor-model-parallel-size $TP \
  --pipeline-model-parallel-size $PP \
  --context-parallel-size $CP --cp-comm-type allgather \
  --num-layers 8 --hidden-size 512 --num-attention-heads 8 \
  --seq-length $SEQ --max-position-embeddings $SEQ \
  --micro-batch-size $MBS --global-batch-size $GBS \
  --train-iters 2 --log-interval 1 \
  --eval-interval 100000000 --eval-iters 1 \
  --attention-backend flash \
  --bf16 \
  --lr 1e-4 \
