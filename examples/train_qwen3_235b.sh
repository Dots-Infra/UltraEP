#!/bin/bash

: "${MEGATRON_PATH:?MEGATRON_PATH must be set to the Megatron-LM path}"
TRAIN_SCRIPT_PATH=${MEGATRON_PATH}/pretrain_gpt.py
if [ ! -f "$TRAIN_SCRIPT_PATH" ]; then
    echo "ERROR: TRAIN_SCRIPT_PATH does not exist: $TRAIN_SCRIPT_PATH" >&2
    exit 1
fi

# UltraEP configs
ENABLE_ULTRA_EP=${ENABLE_ULTRA_EP:-1}
NUM_REDUNDANT_EXPERTS_PER_RANK=${NUM_REDUNDANT_EXPERTS_PER_RANK:-2}
export ULTRA_EP_LOAD_PROFILING=${ULTRA_EP_LOAD_PROFILING:-1}

# training configs
EP_SIZE=${EP_SIZE:-64}
TP_SIZE=${TP_SIZE:-1}
PP_SIZE=${PP_SIZE:-1}

MBS=${MBS:-1}
GBS=${GBS:-2048}
SEQ_LEN=${SEQ_LEN:-8192}

EXP_NAME=${EXP_NAME:-train_qwen3_235b}

MOCK_DATA=${MOCK_DATA:-0}
TOKENIZER_PATH=${TOKENIZER_PATH:-}
DATA_PATH=${DATA_PATH:-}

FORCE_BALANCE=${FORCE_BALANCE:-0}

OUTPUT_DIR=${OUTPUT_DIR:-$(pwd)/${EXP_NAME}}
TENSORBOARD_DIR=$OUTPUT_DIR/tensorboard
NSYS_PROFILE_DIR=$OUTPUT_DIR/nsys_profile
CHECKPOINT_DIR=$OUTPUT_DIR/checkpoints
LOG_DIR=$OUTPUT_DIR/logs
export ULTRA_EP_LOAD_PROFILE_DIR=$OUTPUT_DIR/expert_loads
mkdir -p "$OUTPUT_DIR" "$TENSORBOARD_DIR" "$LOG_DIR" "$NSYS_PROFILE_DIR" "$CHECKPOINT_DIR" "$ULTRA_EP_LOAD_PROFILE_DIR"

ENABLE_NSYS_PROFILE=${ENABLE_NSYS_PROFILE:-0}
PROFILE_RANKS=(${PROFILE_RANKS:-0 1 2 3})
PROFILE_BEGIN_ITER=${PROFILE_BEGIN_ITER:-11}
PROFILE_END_ITER=${PROFILE_END_ITER:-12}

GPUS_PER_NODE=${GPUS_PER_NODE:-$(nvidia-smi | grep NVIDIA | grep On | wc -l)}
: "${WORLD_SIZE:=${OMPI_COMM_WORLD_SIZE:-1}}" "${RANK:=${OMPI_COMM_WORLD_RANK:-0}}" "${MASTER_ADDR:=localhost}" "${MASTER_PORT:=65001}"

KILL_STALE_PROCS=${KILL_STALE_PROCS:-0}
if [ "$KILL_STALE_PROCS" = "1" ]; then
    pkill -9 -f "$TRAIN_SCRIPT_PATH" || true
    lsof -t -i :"$MASTER_PORT" | xargs -r kill -9
    sleep 1
fi

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_FWD_LAYERNORM_SM_MARGIN=0
export NVTE_BWD_LAYERNORM_SM_MARGIN=0
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_NVLS_ENABLE=0
export NVTE_FUSED_ATTN=1
export NVTE_NORM_FWD_USE_CUDNN=1
export NVTE_NORM_BWD_USE_CUDNN=1
export PYTHONWARNINGS=ignore
export NCCL_DEBUG=WARN
export NCCL_GRAPH_REGISTER=0

DISTRIBUTED_ARGS=(
    --nproc_per_node "$GPUS_PER_NODE"
    --nnodes "$WORLD_SIZE"
    --node_rank "$RANK"
    --master_addr "$MASTER_ADDR"
    --master_port "$MASTER_PORT"
)

MODEL_PARALLEL_ARGS=(
    --expert-model-parallel-size "$EP_SIZE"
    --tensor-model-parallel-size "$TP_SIZE"
    --pipeline-model-parallel-size "$PP_SIZE"
    --context-parallel-size 1
    --expert-tensor-parallel-size 1
    --sequence-parallel
)

MODEL_ARGS=(
    --num-layers 94
    --max-position-embeddings 40960
    --hidden-size 4096
    --ffn-hidden-size 12288
    --normalization RMSNorm
    --norm-epsilon 1.0e-6
    --apply-layernorm-1p
    --swiglu
    --untie-embeddings-and-output-weights
    --vocab-size 151936
    --bf16
    ## attention
    --group-query-attention
    --num-attention-heads 64
    --num-query-groups 4
    --kv-channels 128
    --disable-bias-linear
    --qk-layernorm
    --position-embedding-type rope
    --rotary-base 1000000
    --attention-softmax-in-fp32
    --attention-dropout 0.0
    --hidden-dropout 0.0
    ## MoE
    --num-experts 128
    --moe-router-topk 8
    --moe-ffn-hidden-size 1536
    --moe-router-score-function sigmoid
    --moe-router-dtype fp32
    --moe-router-topk-scaling-factor 1.0
    --moe-grouped-gemm
)

if [ "$MOCK_DATA" = "1" ]; then
    DATA_ARGS=(
        --mock-data
        --tokenizer-type NullTokenizer
    )
else
    if [ -z "$TOKENIZER_PATH" ] || [ -z "$DATA_PATH" ]; then
        echo "ERROR: TOKENIZER_PATH and DATA_PATH must be set when MOCK_DATA=0." >&2
        echo "If you do not have data ready, set MOCK_DATA=1 to use mock data." >&2
        exit 1
    fi

    DATA_ARGS=(
        --tokenizer-type HuggingFaceTokenizer
        --tokenizer-model "$TOKENIZER_PATH"
        --train-data-path 1 "$DATA_PATH"
    )
fi

TRAINING_ARGS=(
    ## basic configs
    --micro-batch-size "$MBS"
    --global-batch-size "$GBS"
    --train-samples 1280000
    --seq-length "$SEQ_LEN"
    --max-position-embeddings "$SEQ_LEN"
    --lr-decay-samples 128000
    --lr-warmup-samples 12800
    --lr 1.0e-5
    --min-lr 1.0e-6
    --lr-decay-style cosine
    --clip-grad 1.0
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.95
    --init-method-std 0.006
    ## load balancing loss
    --moe-router-load-balancing-type aux_loss
    --moe-aux-loss-coeff 1e-2
    --use-distributed-optimizer
    --use-mcore-models
    --use-flash-attn
    --transformer-impl transformer_engine
    --manual-gc
    --manual-gc-interval 10
    --enable-experimental
    --overlap-grad-reduce
    --distributed-timeout-minutes 60
    ## selective recompute
    # --recompute-granularity selective
    # --recompute-modules moe_act mlp
    ## full recompute
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
    ## kernel fusions
    --cross-entropy-loss-fusion
    --cross-entropy-fusion-impl te
    --moe-permute-fusion
    --moe-router-fusion
    ## hybrid ep
    --moe-token-dispatcher-type flex
    # --moe-token-dispatcher-type alltoall
    --moe-flex-dispatcher-backend hybridep
    --moe-hybridep-num-sms 32
    ## data and ckpt
    "${DATA_ARGS[@]}"
    --load "$CHECKPOINT_DIR"
    --save "$CHECKPOINT_DIR"
    --save-interval 1000
    --auto-detect-ckpt-format
    --no-create-attention-mask-in-dataloader
    --num-workers 6
    ## eval and logging
    --eval-interval 1000
    --log-timers-to-tensorboard
    --log-memory-to-tensorboard
    --log-throughput
    --log-interval 1
    --tensorboard-dir "$TENSORBOARD_DIR"
)

if [ "$ENABLE_ULTRA_EP" = "1" ]; then
    TRAINING_ARGS+=(
        --moe-enable-ultraep
        --moe-num-redundant-experts-per-rank "$NUM_REDUNDANT_EXPERTS_PER_RANK"
    )
fi

if [ "$FORCE_BALANCE" = "1" ]; then
    TRAINING_ARGS+=(
        --moe-router-force-load-balancing
    )
fi

LOG_FILE=$LOG_DIR/rank-${RANK}-${WORLD_SIZE}.log
export PYTHONPATH="$MEGATRON_PATH:${PYTHONPATH:-}"

PROFILE_ARGS=()
NSYS_CMD=()
if [ "$ENABLE_NSYS_PROFILE" = "1" ]; then
    PROFILE_ARGS=(
        --profile
        --profile-step-start "$PROFILE_BEGIN_ITER"
        --profile-step-end "$PROFILE_END_ITER"
        --profile-ranks "${PROFILE_RANKS[@]}"
    )
    NSYS_CMD=(
        nsys profile
        --sample=none
        --cpuctxsw=none
        -t cuda,nvtx
        --capture-range=cudaProfilerApi
        --capture-range-end=stop
        --cuda-graph-trace=node
        --force-overwrite true
        -o "$NSYS_PROFILE_DIR/${EXP_NAME}_rank${RANK}"
    )
fi

"${NSYS_CMD[@]}" torchrun "${DISTRIBUTED_ARGS[@]}" "$TRAIN_SCRIPT_PATH" \
        "${MODEL_ARGS[@]}" \
        "${MODEL_PARALLEL_ARGS[@]}" \
        "${TRAINING_ARGS[@]}" \
        "${PROFILE_ARGS[@]}" \
        2>&1 | tee "$LOG_FILE"