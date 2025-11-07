#= Prelude =#
# Some constants.
SCRIPT_VERSION=v1
SEQ_LEN=4096

# Defaults.
TIME=12:00:00
NODES=1

N_RECURRENCES=1
LATENT_INIT=identity
THINK_ADAPTER=none
TRAIN_RECURRENCE_METHOD=constant
LINEAR_ADAPTER_ALPHA=1.0
LATENT_MASKER=none
LATENT_MASKER_TOPK=128
FROM_SCRATCH=false
OPTIMIZER=ademamix

NEW_WEIGHTS=false

# Data.
DATAROOT=/iopsstor/scratch/cscs/jpcoles/a06
DATASETS=(
	$DATAROOT/phase-5/finemath-3plus-merge
	$DATAROOT/phase-5/infiwebmath-3plus-fine-merge
	$DATAROOT/phase-5/starcoder-extras-merge
	$DATAROOT/phase-5/starcoder-threshold-0-merge
	$DATAROOT/phase-5/swissai-dclm-edu-filterrobots_fine-merge
	$DATAROOT/phase-5/swissai-fineweb-2-quality_10-filterrobots-merge
	$DATAROOT/phase-5/swissai-megamath-web-pro-filterrobots-merge
	$DATAROOT/phase-5/clean-wikipedia
	$DATAROOT/phase-5/parallel-v2
	$DATAROOT/phase-5/triplicate/provenance-flan-single-replica-1
	$DATAROOT/phase-5/triplicate/euroblocks-templated-1
	$DATAROOT/phase-5/triplicate/provenance-flan-single-replica-2
	$DATAROOT/phase-5/triplicate/euroblocks-templated-2
	$DATAROOT/phase-5/triplicate/provenance-flan-single-replica-3
	$DATAROOT/phase-5/triplicate/euroblocks-templated-3
	$DATAROOT/phase-5/roman/merged/stackv1/threshold_2
	$DATAROOT/phase-5/roman/merged/stackv1/threshold_3
	$DATAROOT/phase-5/roman/merged/stackv2/threshold_0
)


# Prints usage of the script.
usage () {
	echo "Usage: submit.sh <size>"
	echo "<size>: 390/1/3/8/70"
	echo "Variables:"
	# Cluster settings.
	echo "--debug: Runs in debug nodes."
	echo "--nodes <int>: Runs with this many nodes."
	# Data settings.
	echo "--iters <int>: Number of iterations to run."
	echo "--scratch: Train from scratch."
	# Optimizer settings.
	echo "--opt <adam/ademamix/muon>"
	# Recurrence settings.
	echo "--n-recurrences <int>: Default number of recurrences."
	echo "--n-encode <int>: Number of encode layers."
	echo "--n-think <int>: Number of think layers."
	echo "--n-decode <int>: Number of decode layers."
	echo "--latent-init <identity/truncnorm>"
	echo "--think-adapter <none/linear>"
	echo "--train-recurrence-method <constant/poisson>"
	echo "--n-backwards <int>"
	echo "--linear-adapter-alpha <int>"
	echo "--latent-masker <none/topk>"
	echo "--latent-topk-masker-k <int>"
}

# Prints error message and then exit 1.
die () {
	echo $* >&2
	if [[ ${PRINT_USAGE:-true} = true ]]; then
		usage
	fi
	exit 1
}

if [[ $# -eq 0 ]]; then
	die Invalid argument count: $#
fi

SIZE=$1
shift

if [[ $SIZE = 390 ]]; then
	# Arch.
	NUM_LAYERS=16
	HIDDEN_SIZE=1024
	FFN_SIZE=6144
	NUM_HEADS=8
	NUM_QUERY_GROUPS=4
	ROPE_FACTOR=32
	# Opt.
	MBS=${MBS:-8}
	GBS=128
	DEFAULT_ITERS=100000
	SAVE_INTERVAL=10000
	LR=0.001
	MIN_LR=0.0001
elif [[ $SIZE = 1 ]]; then
	# Arch.
	NUM_LAYERS=16
	HIDDEN_SIZE=2048
	FFN_SIZE=12288
	NUM_HEADS=32
	NUM_QUERY_GROUPS=8
	ROPE_FACTOR=32
	# Opt.
	MBS=${MBS:-3}
	GBS=504
	DEFAULT_ITERS=25000
	SAVE_INTERVAL=1000
	CKPT_LOAD_IF_UNRESOLVED=/capstor/store/cscs/swissai/a06/main_run_megatron/Megatron-LM/logs/Meg-Runs/main-runs-v1/apertus3-1b-21-nodes/apertus3-1b-21-nodes/checkpoints
	STEP_LOAD_IF_UNRESOLVED=2039392
	FIX_XIELU_IF_UNRESOLVED=true
	LR=0.00015
	MIN_LR=0.000015
else
	die Invalid model size $SIZE
fi


# Get command line args.
while [[ $# -gt 0 ]]; do
	case $1 in
		# Cluster settings.
		--debug)
			SCRIPT_VERSION=$SCRIPT_VERSION-debug
			TIME=00:30:00
			DEBUG=true
			shift;;
		--nodes) NODES=$2; shift 2;;
		# Data settings.
		--iters) ITERS=$2; shift 2;;
		--scratch) FROM_SCRATCH=true; shift;;
		# Optimizer settings.
		--opt) OPTIMIZER=$2; shift 2;;
		# Recurrence settings.
		--n-recurrences) N_RECURRENCES=$2; shift 2;;
		--n-encode) N_ENCODE=$2; shift 2;;
		--n-think) N_THINK=$2; shift 2;;
		--n-decode) N_DECODE=$2; shift 2;;
		--latent-init) LATENT_INIT=$2; shift 2;;
		--think-adapter) THINK_ADAPTER=$2; shift 2;;
		--train-recurrence-method) TRAIN_RECURRENCE_METHOD=$2; shift 2;;
		--n-backwards) N_BACKWARDS=$2; shift 2;;
		--linear-adapter-alpha) LINEAR_ADAPTER_ALPHA=$2; shift 2;;
		--latent-masker) LATENT_MASKER=$2; shift 2;;
		--latent-topk-masker-k) LATENT_MASKER_TOPK=$2; shift 2;;
		*) die Invalid argument $1
	esac
done


#= Resolve Args =#
EXTRA_ARGS=()
if [[ $SIZE = 390 ]] && [[ $FROM_SCRATCH = false ]]; then
	PRINT_USAGE=false die Model with size $SIZE has no ckpt to load from.
fi
if [[ $SIZE != 390 ]]; then
	EXTRA_ARGS=( --untie-embeddings-and-output-weights)
fi

if [[ $N_RECURRENCES -eq 1 ]]; then
	MODEL_BASE=Apertus
	MODEL_BASE_SUFFIX=""
else
	if [ -z ${N_ENCODE+x} ] || [ -z ${N_THINK+x} ] || [ -z ${N_DECODE+x} ]; then
		PRINT_USAGE=false die "You must specify --n-encode, --n-think and --n-decode when using N_RECURRENCES > 1"
	fi

	if [[ $LATENT_INIT = identity ]] && [[ $THINK_ADAPTER = none ]] && [[ $TRAIN_RECURRENCE_METHOD = constant ]]; then
		MODEL_BASE=ETP
	elif [[ $LATENT_INIT = truncnorm ]] && [[ $THINK_ADAPTER = linear ]] && [[ $TRAIN_RECURRENCE_METHOD = poisson ]]; then
		MODEL_BASE=Ping
	else
		MODEL_BASE=Recurrent
	fi
	N_BACKWARDS=${N_BACKWARDS:-$N_RECURRENCES}
	MODEL_BASE_SUFFIX="-${N_ENCODE}_${N_THINK}x${N_RECURRENCES}_$N_DECODE"

	if [[ $THINK_ADAPTER = linear ]]; then
		NEW_WEIGHTS=true
	fi
fi

if [[ $MODEL_BASE != Ping ]] && [[ $OPTIMIZER != muon ]]; then
	EXTRA_ARGS+=(--overlap-param-gather)
fi

SUFFIX=()
if [ -z ${ITERS+x} ]; then
	ITERS=$DEFAULT_ITERS
elif [[ $ITERS -ne $DEFAULT_ITERS ]]; then
	SUFFIX+=(it$ITERS)
fi
if [ $FROM_SCRATCH = true ]; then
	TRAINING_STEPS=$ITERS
	SUFFIX+=(scratch)
else
	TRAINING_STEPS=$(($ITERS + $STEP_LOAD_IF_UNRESOLVED))
fi

if [[ $N_RECURRENCES -gt 1 ]]; then
	EXTRA_ARGS+=(
		--log-global-metrics num_recurrences
		--n-recurrences $N_RECURRENCES
		--n-encode-layers $N_ENCODE
		--n-think-layers $N_THINK
		--n-decode-layers $N_DECODE
		--latent-init $LATENT_INIT
		--think-adapter $THINK_ADAPTER
		--train-recurrence-method $TRAIN_RECURRENCE_METHOD
		--n-latent-backwards $N_BACKWARDS
		--linear-latent-adapter-alpha $LINEAR_ADAPTER_ALPHA
		--latent-masker $LATENT_MASKER
		--latent-topk-masker-k $LATENT_MASKER_TOPK
	)

	if [[ $MODEL_BASE != ETP ]] && [[ $MODEL_BASE != Ping ]]; then  # Then we need to specify, latent init, adatper and method in the suffix.
		if [[ $LATENT_INIT != identity ]]; then
			SUFFIX+=(latinit_$LATENT_INIT)
		fi
		if [[ $THINK_ADAPTER != none ]]; then
			SUFFIX+=(adapt_$THINK_ADAPTER)
		fi
		if [[ $TRAIN_RECURRENCE_METHOD != constant ]]; then
			SUFFIX+=(rec_$TRAIN_RECURRENCE_METHOD)
		fi
	fi
	if [[ $LATENT_MASKER != none ]]; then
		SUFFIX+=(masker_$LATENT_MASKER)
		if [[ $LATENT_MASKER_TOPK != 128 ]]; then
			SUFFIX+=($LATENT_MASKER_TOPK)
		fi
	fi
	if [[ $LINEAR_ADAPTER_ALPHA != 1.0 ]]; then
		SUFFIX+=(la$LINEAR_ADAPTER_ALPHA)
	fi
	if [[ $N_BACKWARDS -ne $N_RECURRENCES ]]; then
		SUFFIX+=("bck$N_BACKWARDS")
	fi
fi

if [[ $OPTIMIZER = adam ]]; then
	BETA2=0.95
	SUFFIX+=(adam)
	EXTRA_ARGS+=(--use-distributed-optimizer)
elif [[ $OPTIMIZER = ademamix ]]; then
	BETA2=0.999
	EXTRA_ARGS+=(--use-distributed-optimizer)
elif [[ $OPTIMIZER = muon ]]; then
	BETA2=0.95
	SUFFIX+=(muon)
else
	dia Unknown optimizer $OPTIMIZER
fi

# Get important directory names.
if (( ${#SUFFIX} == 0 )); then
	SUFFIX=""
else
	SUFFIX=-$(IFS='-'; echo "${SUFFIX[*]}")
fi

MEGATRON_LM_DIR=/capstor/store/cscs/swissai/infra01/users/ahernnde/workspace/latency/AleHD__Megatron-LM
#PROJECT_DIR=/iopsstor/scratch/cscs/ahernnde/latency_logs/$SCRIPT_VERSION
PROJECT_DIR=/capstor/store/cscs/swissai/infra01/users/ahernnde/workspace/latency/logs/$SCRIPT_VERSION
DATASET_CACHE_DIR=$SCRATCH/datasets/cache
PROJECT_NAME=latency-$SCRIPT_VERSION
EXP_NAME=$MODEL_BASE-${SIZE}B$MODEL_BASE_SUFFIX$SUFFIX

EXP_DIR=$PROJECT_DIR/$EXP_NAME
DEBUG_ROOT=$EXP_DIR/debug
CKPT_DIR=$EXP_DIR/checkpoints
TRIGGER_DIR=$EXP_DIR/triggers
TENSORBOARD_DIR=$EXP_DIR/tensorboard
WANDB_DIR=$EXP_DIR/wandb

mkdir -p $TRIGGER_DIR
mkdir -p $CKPT_DIR

# Resolve the --load, in case the current CKPT_DIR doesn't exist.
if [[ $FROM_SCRATCH = false ]]; then
	if [ ! -f $CKPT_DIR/latest_checkpointed_iteration.txt ]; then
		echo Run not found, creating symlink from source at iter $STEP_LOAD_IF_UNRESOLVED
		echo $STEP_LOAD_IF_UNRESOLVED > $CKPT_DIR/latest_checkpointed_iteration.txt
		ln -s $CKPT_LOAD_IF_UNRESOLVED/iter_$STEP_LOAD_IF_UNRESOLVED $CKPT_DIR/iter_$STEP_LOAD_IF_UNRESOLVED
	fi
	if [ $FIX_XIELU_IF_UNRESOLVED = true ]; then
		MAYBE_ADD_XIELU_FIX="EXTRA_ARGS=\\\"\\\$EXTRA_ARGS --fix-old-xielu\\\""
	fi
	if [ $NEW_WEIGHTS = true ]; then
		MAYBE_NONSTRICT_LOAD="EXTRA_ARGS=\\\"\\\$EXTRA_ARGS --dist-ckpt-strictness ignore_all --no-load-optim\\\""
	fi
	WARMUP_ITERS=2000
	COOLDOWN_ITERS=$ITERS
else
	WARMUP_ITERS=$((ITERS/20))
	COOLDOWN_ITERS=$((ITERS/10))
fi

# Determine partition.
IFS=: read -r T_H T_M T_S <<< "$time"
TIME_MINS=$((10#$T_H * 60 + 10#$T_M + (10#$T_S + 59) / 60))
if [[ $DEBUG = true ]] && ((TIME_MINS*NODES < 90)) && [[ $NODES -le 4 ]]; then
	PARTITION=normal
else
	PARTITION=normal
	MAYBE_SIGNAL="#SBATCH --signal=SIGUSR2@600"
	MAYBE_DEPENDENCY="#SBATCH --dependency=singleton"
fi

#= Final Args =#
TRANSFORMER_ENGINE_ARGS=(
	--main-grads-dtype fp32
)

NETWORK_SIZE_ARGS=(
	--num-layers $NUM_LAYERS
	--hidden-size $HIDDEN_SIZE
	--ffn-hidden-size $FFN_SIZE
	--num-attention-heads $NUM_HEADS
	--group-query-attention
	--num-query-groups $NUM_QUERY_GROUPS
	--max-position-embeddings $SEQ_LEN
	--position-embedding-type rope
	--rotary-base 500000
	--use-rope-scaling
	--rope-scaling-factor $ROPE_FACTOR
	--make-vocab-size-divisible-by 128
	--normalization RMSNorm
	--xielu
	--qk-layernorm
	--qknorm-impl apex
	--disable-bias-linear
)

LOGGING_ARGS=(
	--log-throughput
	--log-progress
	--tensorboard-dir $TENSORBOARD_DIR
	--no-log-loss-scale-to-tensorboard
	--log-memory-to-tensorboard
	--wandb-project $PROJECT_NAME
	--wandb-save-dir $WANDB_DIR
	--log-interval 1
	--timing-log-level 1
	--log-timers-to-tensorboard
	--tensorboard-log-interval 1
	--log-params-norm-per-param
	--log-num-zeros-in-grad
	--log-params-norm
)

REGULARIZATION_ARGS=(
	--attention-dropout 0.0
	--hidden-dropout 0.0
	--weight-decay 0.1
	--clip-grad 0.1
	--adam-beta1 0.9
	--adam-beta2 $BETA2
	--ademamix-alpha 8
	--ademamix-beta3 0.9999
	--ademamix-beta3-warmup 10000
	--ademamix-alpha-warmup 10000
	--sgd-momentum 0.95  # Muon-only.
)

TRAINING_ARGS=(
	--micro-batch-size $MBS
	--global-batch-size $GBS
	--train-iters $TRAINING_STEPS
	--cross-entropy-loss-fusion
	--optimizer $OPTIMIZER
	--dataloader-type single
	--manual-gc
	--manual-gc-interval 500
	--exit-signal-handler
	--trigger-path $TRIGGER_DIR
)

INITIALIZATION_ARGS=(
	--seed 28
	--init-method-std 0.008944
)

LEARNING_RATE_ARGS=(
	--lr $LR
	--min-lr $MIN_LR
	--lr-decay-style WSD
	--lr-warmup-iters $WARMUP_ITERS
	--lr-wsd-decay-style 1-sqrt
	--lr-wsd-decay-iters $COOLDOWN_ITERS
)

CHECKPOINTING_ARGS=(
	--save $CKPT_DIR
	--save-interval $SAVE_INTERVAL
	--ckpt-format torch_dist
	--load $CKPT_DIR
	--async-save
)

MIXED_PRECISION_ARGS=(
	--bf16
)

DISTRIBUTED_ARGS=(
	--tensor-model-parallel-size 1
	--pipeline-model-parallel-size 1
	--overlap-grad-reduce
)

TOKENIZER_ARGS=(
	--tokenizer-type HuggingFaceTokenizer
	--tokenizer-model alehc/swissai-tokenizer
)

DATASETS=$(IFS=','; echo "${DATASETS[*]}")
DATA_ARGS=(
	--split 100,0,0
	--seq-length $SEQ_LEN
	--reset-position-ids
	--eod-mask-loss
	--num-workers 4
	--num-dataset-builder-threads 1
	--goldfish-loss
	--goldfish-k 50
	--goldfish-h 50
	--data-path $(python3 $MEGATRON_LM_DIR/scripts/tools/create_data_config.py -p $DATASETS)
	--data-cache-path $DATASET_CACHE_DIR
)

ARGS="${EXTRA_ARGS[@]} \
    ${TRANSFORMER_ENGINE_ARGS[@]} \
    ${NETWORK_SIZE_ARGS[@]} \
    ${LOGGING_ARGS[@]} \
    ${REGULARIZATION_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${INITIALIZATION_ARGS[@]} \
    ${LEARNING_RATE_ARGS[@]} \
    ${CHECKPOINTING_ARGS[@]} \
    ${MIXED_PRECISION_ARGS[@]} \
    ${DISTRIBUTED_ARGS[@]} \
    ${TOKENIZER_ARGS[@]} \
    ${DATA_ARGS[@]}"
CMD="numactl --membind=0-3 python3 $MEGATRON_LM_DIR/pretrain_gpt.py"

#= Sbatch time =#
cat > $EXP_DIR/submit.sbatch << EOM
#!/bin/bash
#SBATCH --account=a-infra01-1
#SBATCH --time=$TIME
#SBATCH --job-name=$EXP_NAME
#SBATCH --output=$EXP_DIR/slurm/%j.out
#SBATCH --error=$EXP_DIR/slurm/%j.err
#SBATCH --nodes=$NODES
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=460000
#SBATCH --environment=/iopsstor/scratch/cscs/ahernnde/ncg_new_v2.toml
#SBATCH --exclusive
#SBATCH --no-requeue
#SBATCH --partition=$PARTITION
$MAYBE_SIGNAL
$MAYBE_DEPENDENCY

# Wake up.
echo [\$(date)] Starting job
echo [\$(date)] Using nodes: \$SLURM_JOB_NODELIST
srun -l bash -c 'echo \$(hostname) \$(nvidia-smi | grep -o "|\\s*[0-9]*MiB")'

# Log git status.
cd $MEGATRON_LM_DIR
echo ---------
echo [\$(date)] git status:
git status
echo [\$(date)] git log:
git log -n 1
echo ---------

# Set up ENV
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=\$((SLURM_CPUS_PER_TASK/SLURM_GPUS_PER_NODE))
export MASTER_ADDR=\$(scontrol show hostnames \$SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=25678
export WORLD_SIZE=\$SLURM_NPROCS
export PYTHONPATH=$MEGATRON_LM_DIR:\$PYTHONPATH
export CONSUMED_SAMPLES_FROM_CHECKPOINT=$((STEP_LOAD_IF_UNRESOLVED*GBS))
ulimit -c 0

# Checkpoint debug!
DEBUG_DIR=$DEBUG_ROOT/\$SLURM_JOB_ID
mkdir -p \$DEBUG_DIR
cp \$0 \$DEBUG_DIR/submit.sbatch
cat \$SLURM_SPANK__SLURM_SPANK_OPTION_pyxis_environment > \$DEBUG_DIR/env.toml
echo "\\nMegatron path: $MEGATRON_LM_DIR (\$(git -C $MEGATRON_LM_DIR rev-parse --verify HEAD))" > \$DEBUG_DIR/git
git diff > \$DEBUG_DIR/git.diff
pip list > \$DEBUG_DIR/pip.txt
nvidia-smi > \$DEBUG_DIR/cuda
printenv > \$DEBUG_DIR/env.sh

# Run!
echo [\$(date)] Running main srun
srun --cpus-per-task \$SLURM_CPUS_PER_TASK -lu bash -c "
	EXTRA_ARGS=\"\"
	# Extra args needed the first time the original ckpt is loaded.
	LAST_STEP=\\\$(cat $CKPT_DIR/latest_checkpointed_iteration.txt)
	if [ -h $CKPT_DIR/iter_\\\$LAST_STEP ]; then
		echo [\\\$(date)] Adding first time args.
		EXTRA_ARGS=\"\\\$EXTRA_ARGS --override-opt_param-scheduler\"
		export OVERRIDE_FLOPS_SO_FAR=0
		$MAYBE_ADD_XIELU_FIX
		$MAYBE_NONSTRICT_LOAD
	fi
	EXTRA_ARGS=\"\\\$EXTRA_ARGS --wandb-exp-name $EXP_NAME-j\\\$SLURM_JOBID\"

	RANK=\\\$SLURM_PROCID LOCAL_RANK=\\\$SLURM_LOCALID $CMD \\\$EXTRA_ARGS $ARGS
"

# Goodbye lol.
echo [\$(date)] Goodbye
EOM

echo "Saved sbatch to $EXP_DIR/submit.sbatch"
sbatch $EXP_DIR/submit.sbatch
