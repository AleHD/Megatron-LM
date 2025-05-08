#!/bin/bash

# runs the following checkpoint conversions: 
#   - torch_dist           ---> torch ,  if CKPT_IS_TORCH_DIST=true.
#   - core (torch backend) ---> HF    ,  always.

if false; then
    echo "installing custom transformers"
    cd $SCRATCH/fp8/sai-transformers
    pip install -e .
    echo "-------------"
fi

MEGATRON_LM_DIR=$SCRATCH/fp8/Megatron-LM/
CKPT_PATH=/capstor/store/cscs/swissai/a06/users/ahernnde/opv1-backup2/llama1.5B-nopre-postln-qknorm-softmax0.125-ls-is-xielu-ntQKgain-lr0.00025-coolWD/checkpoints/
export PYTHONPATH=$MEGATRON_LM_DIR

# [torch_dist -> torch] dependencies
CKPT_IS_TORCH_DIST=true
TORCH_DIST_SCRIPT=$MEGATRON_LM_DIR/scripts/conversion/torchdist_2_torch.py
TORCH_CKPT_SAVE_PATH=$SCRATCH/fp8/Meg-Checkpoints/test-op
# [core (torch) --> HF] dependencies
HF_SAVE_DIR=$SCRATCH/fp8/hf-checkpoints
SAVE_DIR=$HF_SAVE_DIR/test-op
mkdir -p $HF_SAVE_DIR
LOADER=core
SAVER=swissai_fp8_hf


# Run torch_dist --> torch
if [[ "$CKPT_IS_TORCH_DIST" == true ]]; then
    LOAD_DIR=$TORCH_CKPT_SAVE_PATH/torch
    echo "Running torch_dist --> torch conversion..."
    CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun $TORCH_DIST_SCRIPT \
    --bf16 \
    --load $CKPT_PATH \
    --ckpt-convert-save $TORCH_CKPT_SAVE_PATH
else
    LOAD_DIR=$CKPT_PATH
    echo "Skipping torch_dist --> torch conversion..."
fi


# Run core --> HF
echo "Running core --> HF conversion..."
# PYTHONPATH=$MEGATRON_LM_DIR/megatron
python $MEGATRON_LM_DIR/tools/checkpoint/convert.py \
    --model-type GPT \
    --loader  $LOADER \
    --saver $SAVER \
    --load-dir $LOAD_DIR \
    --save-dir $SAVE_DIR \
    #\ --hf-tokenizer .....