#= PRELUDE: Command line utilities and handling the model size =#
TODI_GPUS=4
RCP_GPUS=8
SCRIPT_VERSION=v3

# Print usage function.
usage () {
	echo "Usage: llama.sh <size> [options...]"
	echo "<size>: 1/3/8/70"
	echo "Options:"
	echo " --help: Displays this message"
	echo " --rcp: Use rcp paths and launch using cmd, not slurm"
	echo " --clariden: Use calriden paths"
	echo " --fp8: Enables fp8"
	echo " --fp8-margin: Enables fp8"
	echo " --fp8-dpa: Enables fp8 DPA"

	echo " --moretokens: 10x the training size"
	echo " --nologs: Don't log expensive things"

	echo " --op: Enables outlier protected block"
	echo " --torch-qknorm: Enables torch qk norm"
	echo " --correct-beta: Use the correct 1/sqrt(layers) beta instead of the 1/layers"
	echo " --single-scaler: Use only one residual scaler (instead of a hidden_size weight)"
	echo " --no-downscale: Dont use residual downscaling at all"
	echo " --with-final-ln: Enables final layernorm in outlier protected block"
	echo " --postnorm: Enables postnorm architecture (instead of pre-norm)"
	echo " --gelu: Use gelu (instead of relu)"
	echo " --swiglu: Use swiglu (instead of relu)"
	echo " --shallow: Don't widen FFN with OP block"
	echo " --no-input-scaling: Enables scaling input by 1/init_std"
	echo " --no-remove-interln: Don't remove the mlp and attention layernorms"

	echo " --debug: Mark run as debug script version"
	echo " --nodes <nodes>: How many nodes to use"
	echo " --goodnodes <nodes>: Specify the good node list"
	echo " --exclude <nodes>: Specify the bad node list"
	echo " --reservation <reservation>: Todi reservation to use"
	echo " --acc16: Accumulate gradients in bf16 instead of fp32"

	echo " --decay <decay>: Specify weight decay"
	echo " --lr <lr>: Specify learningrate"
	echo " --scheduler <wsd/cos>: Specify learningrate scheduler"
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
	# batch_size = 1.47 M
	# total_tokens = ~79.62B tokens
	# todi time per iter (1 node): 9.4s
	# todi ETA (1 node): 5d21h
	TP=1
	PP=1
	LAYERS=24
	HIDDEN_SIZE=1536
	FFN_SIZE=6144
	NUM_HEADS=16
	NUM_QUERY_GROUPS=16
	MBS=3
	GBS=180
	ITERS=54000
	LR=0.001
	INIT_STD=0.025
	SIZE=1
	SAVE_FREQ=2000
elif [[ $1 -eq 3 ]]; then
	# batch_size = 180*8192 = ~1.47M tokens
	# total_tokens = ~14.75B
	TP=2
	PP=1
	LAYERS=36
	HIDDEN_SIZE=2048
	FFN_SIZE=8192
	NUM_HEADS=16
	NUM_QUERY_GROUPS=16
	MBS=3
	GBS=180
	ITERS=10000
	LR=0.001
	INIT_STD=0.02
	SIZE=3
	SAVE_FREQ=200
elif [[ $1 -eq 8 ]]; then
	# batch_size = ~2.1M
	# total_tokens = ~52.42B
	TP=4
	PP=1
	LAYERS=32
	HIDDEN_SIZE=4096
	FFN_SIZE=14336
	NUM_HEADS=32
	NUM_QUERY_GROUPS=8
	MBS=2
	GBS=256
	ITERS=25000
	LR=0.0005
	INIT_STD=0.01
	SIZE=8
	SAVE_FREQ=1000
elif [[ $1 -eq 70 ]]; then
	# batch_size = ~4.19M
	# total_tokens = ~20.97B
	TP=4
	PP=4
	LAYERS=80
	HIDDEN_SIZE=8192
	FFN_SIZE=28672
	NUM_HEADS=64
	NUM_QUERY_GROUPS=8
	MBS=1
	GBS=512
	ITERS=2500
	LR=0.00001
	INIT_STD=0.01
	SIZE=70
	SAVE_FREQ=250
else
	echo "Invalid llama size: $1"
	usage
	exit 1
fi
shift
DECAY=0.1
MINLR=1e-8

MORE_TOKENS=false
LOGS=true

NEW_LR=""
NEW_MINLR=""
NEW_DECAY=""
FP8=false
FP8DPA=false
OP=false
TORCH_QKNORM=false
FINAL_LN=false
POSTNORM=false
GELU=false
SHALLOW=false
INPUT_SCALING=true
CORRECT_BETA=false
REMOVE_INTERLN=true
DOWNSCALE=true
SWIGLU=false
TODI=true
CLARIDEN=false
SINGLE_SCALER=false
EXTRA_NAME=""
WANDB_ID=""
SCHEDULER=wsd

BETA2=0.95
NEW_BETA2=""

GOOD_NODES=""
BAD_NODES=""
NODES=1
REQ_NODES=1
RESERV=""
ACCUMULATE_FP32=true

FP8_MARGIN=0

ADAM_EPS=0.00000001
NORM_EPS=0.00001

SUFFIX=""
while [[ $# -gt 0 ]]; do
	case $1 in
		--help)
			usage; exit 0;;
		--nologs)
			LOGS=false; shift;;
		--moretokens)
			MORE_TOKENS=true; shift;;
		--fp8)
			FP8=true; shift;;
		--fp8-margin)
			FP8_MARGIN=$2; shift 2;;
		--rcp)
			TODI=false; shift;;
		--clariden)
			TODI=false; CLARIDEN=true; shift;;
		--fp8-dpa)
			FP8DPA=true; shift;;
		--op)
			OP=true; shift;;
		--torch-qknorm)
			TORCH_QKNORM=true; shift;;
		--correct-beta)
			CORRECT_BETA=true; shift;;
		--no-downscale)
			DOWNSCALE=false; shift;;
		--single-scaler)
			SINGLE_SCALER=true; shift;;
		--with-final-ln)
			FINAL_LN=true; shift;;
		--postnorm)
			POSTNORM=true; shift;;
		--acc16)
			ACCUMULATE_FP32=false; shift;;
		--extra-name)
			EXTRA_NAME="-$2"; shift 2;;
		--wandbid)
			WANDB_ID=$2; shift 2;;
		--gelu)
			GELU=true; shift;;
		--swiglu)
			SWIGLU=true; shift;;

		--nodes)
			REQ_NODES=$2; shift 2;;
		--goodnodes)
			GOOD_NODES=$2; shift 2;;
		--exclude)
			BAD_NODES=$2; shift 2;;
		--debug)
			SCRIPT_VERSION=$SCRIPT_VERSION-debug; shift;;
		--reservation)
			RESERV=$2; shift 2;;

		--shallow)
			SHALLOW=true; shift;;
		--no-input-scaling)
			INPUT_SCALING=false; shift;;
		--no-remove-interln)
			REMOVE_INTERLN=false; shift;;
		--lr)
			NEW_LR=$2; shift 2;;
		--minlr)
			NEW_MINLR=$2; shift 2;;
		--decay)
			NEW_DECAY=$2; shift 2;;
		--beta2)
			NEW_BETA2=$2; shift 2;;
		--scheduler)
			SCHEDULER=$2; shift 2;;
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

if [ $MORE_TOKENS = true ]; then
	SUFFIX=$SUFFIX-moretokens
	ITERS=$(( 10*$ITERS ))
fi

OTHER_ARGS=()
if [ $SIZE -eq 70 ] && [ $OP = true ]; then
	PP=16
	#OTHER_ARGS+=(--recompute-granularity full --recompute-method uniform --recompute-num-layers 1)
	#OTHER_ARGS+=(--recompute-granularity full --recompute-method uniform --recompute-num-layers 1)
fi

if [ $GELU = true ] && [ $SWIGLU = true ]; then
	echo "Can't use gelu and swiglu at the same time"
	exit 1
fi
if [ $SWIGLU = true ] && [ $SHALLOW = false ]; then
	echo "For fair comparison, you shouldn't set --swiglu without setting --shallow too"
	exit 1
fi

#= MIDDLE: Set up arguments depending on the commandline =#
ENVS=""
if [ $TODI = false ] && [ $CLARIDEN = false ]; then
	SUFFIX=$SUFFIX-rcp; 
fi

FP8_ARGS=""
if [ $FP8 = true ]; then
	SUFFIX=$SUFFIX-fp8
	FP8_ARGS="$FP8_ARGS --fp8-margin $FP8_MARGIN --fp8-format hybrid --fp8-amax-history-len 1024 --fp8-amax-compute-algo max"
	if [ $FP8_MARGIN -ne 0 ]; then
		SUFFIX=$SUFFIX-fp8margin$FP8_MARGIN
	fi
	if [ $FP8DPA = true ]; then
		SUFFIX=$SUFFIX-fp8dpa
		FP8_ARGS="$FP8_ARGS --fp8-dot-product-attention"
	fi
fi

if [ $OP = true ]; then
	SUFFIX=$SUFFIX-op
	OP_ARGS="--qk-layernorm"

	if [ $TORCH_QKNORM = true ]; then
		OP_ARGS="$OP_ARGS --use-torchqknorm --no-persist-layer-norm"
		SUFFIX=$SUFFIX-torchqk
	fi

	if [ $REMOVE_INTERLN = true ]; then
		OP_ARGS="$OP_ARGS --no-attn-layernorm --no-mlp-layernorm"
	else
		SUFFIX=$SUFFIX-interln
	fi

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

	if [ $DOWNSCALE = true ]; then
		if [ $CORRECT_BETA = true ]; then
			SUFFIX=$SUFFIX-correctbeta
			BETA=$(echo "print(1/$LAYERS**0.5)" | python3)
		else
			BETA=$(echo "print(1/$LAYERS)" | python3)
		fi
		OP_ARGS="$OP_ARGS --downscale-residual $BETA"
		if [ $LOGS = true ]; then
			OP_ARGS="$OP_ARGS --log-gains-norm"
		fi

		if [ $SINGLE_SCALER = true ]; then
			SUFFIX=$SUFFIX-1scaler
			OP_ARGS="$OP_ARGS --single-residual-gain"
		fi
	else
		SUFFIX=$SUFFIX-nodownscale
	fi

	if [ $GELU = true ]; then
		SUFFIX=$SUFFIX-gelu
	elif [ $SWIGLU = true ]; then
		SUFFIX=$SUFFIX-swiglu
		OP_ARGS="$OP_ARGS --swiglu"
	else
		OP_ARGS="$OP_ARGS --relu"
	fi

	if [ $INPUT_SCALING = true ]; then
		MULT=$(echo "print(1/$INIT_STD)" | python3)
		OP_ARGS="$OP_ARGS --input-embeddings-multiplier $MULT"
	else
		SUFFIX=$SUFFIX-noinputscale
	fi

else
	OP_ARGS="--swiglu"
fi

if [ $POSTNORM = true ]; then
	SUFFIX=$SUFFIX-postnormFIX
	OP_ARGS="$OP_ARGS --post-layer-norm"
fi

if [ "$NEW_LR" != "" ]; then
	LR=$NEW_LR
	SUFFIX=$SUFFIX-lr$LR
fi
if [ "$NEW_MINLR" != "" ]; then
	MINLR=$NEW_MINLR
	SUFFIX=$SUFFIX-minlr$MINLR
fi

if [ "$NEW_DECAY" != "" ]; then
	DECAY=$NEW_DECAY
	SUFFIX=$SUFFIX-decay$DECAY
fi

if [ $SCHEDULER = wsd ]; then
	SCHEDULER_ARGS=(
		--lr-decay-style WSD
		--lr-wsd-decay-style 1-sqrt
		--lr-wsd-decay-iters $(($ITERS/5))
		--lr-warmup-iters $(($ITERS/20))
	)
else
	SUFFIX=$SUFFIX-cos
	SCHEDULER_ARGS=(
		--lr-decay-style cosine
		--lr-warmup-iters $(($ITERS/20))
		--lr-decay-iters $(($ITERS - $ITERS/20))
	)
fi

if [ "$NEW_BETA2" != "" ] && [ $NEW_BETA2 != $BETA2 ]; then
	BETA2=$NEW_BETA2
	SUFFIX=$SUFFIX-beta2_$BETA2
fi

if [ $REQ_NODES -ne 1 ]; then
	SUFFIX=$SUFFIX-nodes$REQ_NODES
fi

if [ $ACCUMULATE_FP32 = true ]; then
	OTHER_ARGS+=(--accumulate-allreduce-grads-in-fp32)
else
	SUFFIX=$SUFFIX-acc16
fi


SUFFIX=$SUFFIX$EXTRA_NAME
NAME=llama${SIZE}b$SUFFIX

if [ $TODI = true ] || [ $CLARIDEN = true ]; then
	GPUS=$TODI_GPUS
	# 81.816 B tokens
	if [ $TODI = true ]; then
		STORE=/store/swissai/a06
	else
		STORE=/capstor/store/cscs/swissai/a06
	fi
	DATA_PATH=$STORE/users/ahernnde/data/finewebedu-sample-100BT/megatrontokenized/finewebedu-llama3tok_text_document
	SAVE_PATH=/capstor/scratch/cscs/ahernnde/checkpoints/megatron/fp8experiments_$SCRIPT_VERSION/$NAME
	CODE_PATH=$STORE/users/ahernnde/workspace/AleHD-Megatron-LM
	TOKENIZER=$STORE/models/Meta-Llama-3.1-8B/
	NODES=$REQ_NODES
else
	GPUS=$RCP_GPUS
	DATA_PATH=/mloscratch/homes/alhernan/data/fineweb/fineweb-30MD-megatron_text_document
	SAVE_PATH=/mloscratch/homes/alhernan/checkpoints/megatron/fp8experiments_$SCRIPT_VERSION/$NAME
	TOKENIZER=meta-llama/Meta-Llama-3-8B
	ENVS="$ENVS HF_HOME=/mloscratch/hf_cache"
	if [ $REQ_NODES -ne 1 ]; then
		echo Warning: Requested multiple nodes in RCP but it is not yet supported. Using one node
	fi
fi


#= WRAPPING UP: Set up the _ARGS variables that are going to be used in the end =#

# Misc.
ENVS="$ENVS CUDA_DEVICE_MAX_CONNECTIONS=1 OMP_NUM_THREADS=64"
WANDB_PROJECT=megatron_fp8_experiments_rcp_$SCRIPT_VERSION
SEQ_LEN=8192

if [ "$WANDB_ID" = "" ]; then
	ENVS="$ENVS WANDB_RESUME=allow WANDB_RUN_ID=${WANDB_PROJECT}_$NAME"
else
	ENVS="$ENVS WANDB_RESUME=allow WANDB_RUN_ID=$WANDB_ID"
fi


LLAMA_ARGS=(
	--tensor-model-parallel-size $TP
	--pipeline-model-parallel-size $PP
	--seq-length $SEQ_LEN
	--max-position-embeddings $SEQ_LEN
	--tokenizer-type HuggingFaceTokenizer
	--tokenizer-model $TOKENIZER
	--exit-on-missing-checkpoint
	--untie-embeddings-and-output-weights
	--normalization RMSNorm
	--position-embedding-type rope
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
	--adam-beta2 $BETA2
	--init-method-std $INIT_STD
	--clip-grad 1.0 
	--lr $LR
    	--min-lr $MINLR
)

DISTRIBUTED_ARGS=(
	--nproc_per_node $GPUS
	--nnodes $NODES
	--master_port 25678
)

DATA_ARGS=(
	--data-path $DATA_PATH
	--split 9990,8,2
)

LOGGING=(
	--log-interval 1
	--save-interval $SAVE_FREQ
	--eval-interval 100
	--save $SAVE_PATH
	--tensorboard-dir $SAVE_PATH/tensorboard
	--eval-iters 32
	--wandb-project $WANDB_PROJECT
	--wandb-exp-name $NAME
	--log-progress
	--log-throughput
	--log-timers-to-tensorboard
	--log-validation-ppl-to-tensorboard
)
if [ $LOGS = true ]; then
	LOGGING+=(
		--log-params-norm
		--log-memory-to-tensorboard
		--log-kurtosis
	)
fi

EXTRA_ARGS=(
	--use-distributed-optimizer
	--overlap-grad-reduce
	--overlap-param-gather
	--async-save
)
if [ $TORCH_QKNORM = false ]; then
	EXTRA_ARGS+=(--sequence-parallel)
fi



MAYBE_LOAD=""
if [ -f $SAVE_PATH/latest_checkpointed_iteration.txt ]; then
	MAYBE_LOAD="--load $SAVE_PATH"
fi

ARGS="${LLAMA_ARGS[@]} ${TRAINING_ARGS[@]} ${SCHEDULER_ARGS[@]} ${DATA_ARGS[@]} ${LOGGING[@]} ${EXTRA_ARGS[@]} ${OTHER_ARGS[@]} $FP8_ARGS $OP_ARGS"

#= RUNNING: Run the command, or launch a slurm script if using todi =#
if [ $TODI = true ] || [ $CLARIDEN = true ]; then
	DISTRIBUTED_ARGS="${DISTRIBUTED_ARGS[@]} --master-addr=\$MASTER_ADDR --node-rank=\\\$SLURM_PROCID"
	CMD="torchrun $DISTRIBUTED_ARGS pretrain_gpt.py $ARGS"

	if [ "$RESERV" != "" ]; then
		RESERV_STRING="#SBATCH --reservation=$RESERV"
	fi
	if [ "$GOOD_NODES" != "" ]; then
		GOOD_NODES_STRING="#SBATCH --nodelist=$GOOD_NODES"
	fi
	if [ "$BAD_NODES" != "" ]; then
		BAD_NODES_STRING="#SBATCH --exclude=$BAD_NODES"
	fi
	if [ $TODI = true ]; then
		ACC_STRING="#SBATCH --account=a06"
		CONTAINER=$STORE/users/ahernnde/container-image/nemo-swissai/nemo-swissai-oldimg.toml
	else
		CONTAINER=$STORE/users/ahernnde/container-image/nemo-swissai/nemo-swissai-clariden.toml
	fi

	mkdir -p $SAVE_PATH
	cat > $SAVE_PATH/submission.sbatch <<- EOM
	#!/bin/bash
	$ACC_STRING
	#SBATCH --cpus-per-task=288
	#SBATCH --gres=gpu:4
	#SBATCH --environment=$CONTAINER
	#SBATCH --job-name=$NAME
	#SBATCH --mem=460000
	#SBATCH --nodes=$NODES
	#SBATCH --ntasks-per-node=1
	#SBATCH --output=$SAVE_PATH/slurmlogs/$NAME_%j.out
	#SBATCH --error=$SAVE_PATH/slurmlogs/$NAME_%j.err
	#SBATCH --time=12:00:00
	#SBATCH --exclusive
	#SBATCH --dependency=singleton
	$RESERV_STRING
	$GOOD_NODES_STRING
	$BAD_NODES_STRING

	echo "Using nodes: \$SLURM_JOB_NODELIST"
	srun -l bash -c 'echo \$(hostname) \$(nvidia-smi | grep -o "|\\s*[0-9]*MiB")'

	export WANDB_API_KEY=$(cat $STORE/users/ahernnde/.keys/wandb.txt)
	export MASTER_ADDR=\$(hostname)

	# Log git status.
	cd $CODE_PATH
	echo "OUTPUT OF GIT LOG:"
	git log --name-status HEAD^..HEAD
	echo ---------
	echo "OUTPUT OF GIT DIFF:"
	git diff
	echo ---------

	MAYBE_LOAD=""
	if [ -f $SAVE_PATH/latest_checkpointed_iteration.txt ]; then
		MAYBE_LOAD="--load $SAVE_PATH"
	fi

	python $STORE/users/ahernnde/workspace/AleHD-Megatron-LM/scripts/remove_incomplete_checkpoint.py $SAVE_PATH

	srun -l --unbuffered numactl --membind=0-3 bash -c "
		cd $CODE_PATH
		export PYTHONPATH=\$PWD
		eval \"$ENVS\" $CMD \$MAYBE_LOAD
	"
	EOM
	echo "Saved sbatch to $SAVE_PATH/submission.sbatch"
	if [ $TODI = true ]; then
		export ENROOT_LIBRARY_PATH=/capstor/scratch/cscs/fmohamed/enrootlibxpmem
	fi
	sbatch $SAVE_PATH/submission.sbatch
else
	CMD="torchrun ${DISTRIBUTED_ARGS[@]} --master_addr localhost pretrain_gpt.py $ARGS $MAYBE_LOAD"
	echo Running command: $CMD
	echo Environment: $ENVS
	echo -----
	eval "$ENVS" $CMD
fi
