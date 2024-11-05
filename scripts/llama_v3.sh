# Print usage function.
usage () {
	echo "Usage: llama.sh <size> [options...]"
	echo "<size>: 1/3/8"
	echo "Options:"
	echo " --help: Displays this message"
	echo " --fp8: Enables fp8"
	echo " --fp8-dpa: Enables fp8 DPA"
	echo " --op: Enables outlier protected block"
	echo " --correct-beta: Use the correct 1/sqrt(layers) beta instead of the 1/layers"
	echo " --with-final-ln: Enables outlier protected block"
	echo " --gelu: Use gelu"
	echo " --shallow: Don't widen FFN with OP block"
	echo " --input-scaling: Enables scaling input by 50x"
	echo " --lr <lr>: Specify learningrate"
	echo " --extra-name <name>: Add a suffix to the name"
	echo " --adam-eps <eps>: Epsilon"
	echo " --norm-eps <eps>: Epsilon"
	echo " --wandbid <id>: the wandb id run to resume"
}

if [[ $# -eq 0 ]]; then
	echo "Invalid argument count: $#"
	usage
	exit 1
fi

if [[ $1 -eq 1 ]]; then 
	TP=1
	LAYERS=14
	HIDDEN_SIZE=2048
	FFN_SIZE=8192
	NUM_HEADS=16
	NUM_QUERY_GROUPS=16
	MBS=2
	GBS=160
	ITERS=8000
	LR=0.001
	MINLR=1e-5
	DECAY=0.01
	INIT_STD=0.02
	SIZE=1
	GPUS=8
elif [[ $1 -eq 3 ]]; then
	# batch_size = 180*8192 = ~1.47M tokens
	# total_tokens = ~14.75B
	TP=2
	LAYERS=36
	HIDDEN_SIZE=2048
	FFN_SIZE=8192
	NUM_HEADS=16
	NUM_QUERY_GROUPS=16
	MBS=3
	GBS=180
	ITERS=10000
	LR=0.001
	MINLR=1e-5
	DECAY=0.1
	INIT_STD=0.02
	SIZE=3
	GPUS=8
elif [[ $1 -eq 8 ]]; then
	# batch_size = ~2.46M
	# total_tokens = ~9.83B
	TP=4
	LAYERS=32
	HIDDEN_SIZE=4096
	FFN_SIZE=14336
	NUM_HEADS=32
	NUM_QUERY_GROUPS=8
	MBS=2
	GBS=300
	ITERS=4000
	LR=0.0005
	MINLR=1e-5
	DECAY=0.1
	INIT_STD=0.01
	SIZE=8
	GPUS=8
else
	echo "Invalid llama size: $1"
	usage
	exit 1
fi
shift


FP8=false
FP8DPA=false
OP=false
FINAL_LN=false
GELU=false
SHALLOW=false
INPUT_SCALING=false
CORRECT_BETA=false
EXTRA_NAME=""
WANDB_ID=""

ADAM_EPS=0.00000001
NORM_EPS=0.00001

SUFFIX=""
while [[ $# -gt 0 ]]; do
	case $1 in
		--help)
			usage; exit 0;;
		--fp8)
			FP8=true; shift;;
		--fp8-dpa)
			FP8DPA=true; shift;;
		--op)
			OP=true; shift;;
		--correct-beta)
			CORRECT_BETA=true; shift;;
		--with-final-ln)
			FINAL_LN=true; shift;;
		--extra-name)
			EXTRA_NAME="$2"; shift 2;;
		--wandbid)
			WANDB_ID=$2; shift 2;;
		--gelu)
			GELU=true; shift;;
		--shallow)
			SHALLOW=true; shift;;
		--input-scaling)
			INPUT_SCALING=true; shift;;
		--lr)
			SUFFIX=$SUFFIX-lr$2
			LR=$2; shift 2;;
		--adam-eps)
			SUFFIX=$SUFFIX-adameps$2
			ADAM_EPS=$2; shift 2;;
		--norm-eps)
			SUFFIX=$SUFFIX-normeps$2
			NORM_EPS=$2; shift 2;;
		*)
			echo "Unexpected argument $1"
			usage
			exit 1
	esac
done

FP8_ARGS=""
if [ $FP8 = true ]; then
	SUFFIX=$SUFFIX-fp8
	FP8_ARGS="$FP8_ARGS --fp8-format hybrid --fp8-amax-history-len 1024 --fp8-amax-compute-algo max"
	if [ $FP8DPA = true ]; then
		SUFFIX=$SUFFIX-fp8dpa
		FP8_ARGS="$FP8_ARGS --fp8-dot-product-attention"
	fi
fi

if [ "$WANDB_ID" != "" ]; then
	export WANDB_RESUME=must
	export WANDB_RUN_ID=$WANDB_ID
fi

if [ $OP = true ]; then
	SUFFIX=$SUFFIX-op
	OP_ARGS="--qk-layernorm --no-attn-layernorm --no-mlp-layernorm"

	if [ $SHALLOW = true ]; then
		SUFFIX=$SUFFIX-shallow
	else
		FFN_SIZE=$((3*$FFN_SIZE/2))
	fi

	if [ $FINAL_LN = true ]; then
		SUFFIX=$SUFFIX-finalnorm
	else
		OP_ARGS="$OP_ARGS --no-final-layernorm"
	fi

	if [ $CORRECT_BETA = true ]; then
		SUFFIX=$SUFFIX-correctbeta
		BETA=$(echo "print(1/$LAYERS**0.5)" | python)
	else
		BETA=$(echo "print(1/$LAYERS)" | python)
	fi
	OP_ARGS="$OP_ARGS --downscale-residual $BETA"

	if [ $GELU = true ]; then
		SUFFIX=$SUFFIX-gelu
	else
		OP_ARGS="$OP_ARGS --relu"
	fi

	if [ $INPUT_SCALING = true ]; then
		MULT=$(echo "print(1/$INIT_STD)" | python)
		OP_ARGS="$OP_ARGS --input-embeddings-multiplier $MULT"
		SUFFIX=$SUFFIX-inputscale
	fi

else
	OP_ARGS="--swiglu"
fi

SUFFIX=$SUFFIX$EXTRA_NAME
NAME=llama${SIZE}b$SUFFIX

# Misc.
export HF_HOME=/mloscratch/hf_cache
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=64
SCRIPT_VERSION=v2

# General 
SEQ_LEN=8192

# General args
DATA_PATH=/mloscratch/homes/alhernan/data/fineweb/fineweb-30MD-megatron_text_document
SAVE_PATH=/mloscratch/homes/alhernan/checkpoints/megatron/fp8experiments_$SCRIPT_VERSION/$NAME

LLAMA_ARGS=(
	--tensor-model-parallel-size $TP
	--pipeline-model-parallel-size 1
	--seq-length $SEQ_LEN
	--max-position-embeddings 131072
	--tokenizer-type HuggingFaceTokenizer
	--tokenizer-model meta-llama/Meta-Llama-3-8B
	--exit-on-missing-checkpoint
	--untie-embeddings-and-output-weights
	--normalization RMSNorm
	--position-embedding-type rope
	--no-masked-softmax-fusion
	--attention-softmax-in-fp32
	--disable-bias-linear
	--transformer-impl transformer_engine
	--num-layers $LAYERS
	--hidden-size $HIDDEN_SIZE
	--group-query-attention
	--num-query-groups $NUM_QUERY_GROUPS
	--ffn-hidden-size $FFN_SIZE
	--num-attention-heads $NUM_HEADS
	--attention-dropout 0.0
	--hidden-dropout 0.0
	--rotary-base 500000
	--rotary-percent 1.0
	--use-rope-scaling
	--bf16
	--adam-eps $ADAM_EPS
	--norm-epsilon $ADAM_EPS
)

TRAINING_ARGS=(
	--micro-batch-size $MBS
	--global-batch-size $GBS
	--train-iters $ITERS
	--weight-decay $DECAY 
	--adam-beta1 0.9 
	--adam-beta2 0.95 
	--init-method-std $INIT_STD
	--clip-grad 1.0 
	--lr $LR
	--lr-decay-style WSD
	--lr-wsd-decay-style linear
	--lr-wsd-decay-iters $(($ITERS/10))
	--lr-warmup-iters $(($ITERS/20))
    	--min-lr $MINLR
)

DISTRIBUTED_ARGS=(
	--nproc_per_node $GPUS
	--nnodes 1
	--master_addr localhost
	--master_port 25678
)

DATA_ARGS=(
	--data-path $DATA_PATH
	--split 9990,8,2
)

LOGGING=(
	--log-interval 1
	--save-interval 500
	--eval-interval 100
	--save $SAVE_PATH
	--tensorboard-dir $SAVE_PATH/tensorboard
	--eval-iters 32
	--wandb-project megatron_fp8_experiments_rcp_$SCRIPT_VERSION
	--wandb-exp-name $NAME
	--log-params-norm
	--log-progress
	--log-throughput
	--log-timers-to-tensorboard
	--log-validation-ppl-to-tensorboard
	--log-memory-to-tensorboard
	--log-kurtosis
)

EXTRA_ARGS=(
	--use-distributed-optimizer
)

MAYBE_LOAD=""
if [ -f $SAVE_PATH/latest_checkpointed_iteration.txt ]; then
	MAYBE_LOAD="--load $SAVE_PATH"
fi

ARGS="${LLAMA_ARGS[@]} ${TRAINING_ARGS[@]} $MAYBE_LOAD ${DATA_ARGS[@]} ${LOGGING[@]} ${EXTRA_ARGS[@]} $FP8_ARGS $OP_ARGS"
CMD="torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py $ARGS"
echo Running command: $CMD
echo -----
$CMD
