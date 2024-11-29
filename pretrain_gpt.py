# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
"""Pretrain GPT."""

import os
import math
import torch
from typing import Optional
from functools import partial
from contextlib import nullcontext
import inspect

from typing import Union
from megatron.training import get_args
from megatron.training import print_rank_0
from megatron.training import get_timers
from megatron.training import get_tokenizer
from megatron.core import parallel_state
from megatron.core import mpu
from megatron.core.enums import ModelType
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.utils import get_blend_from_list
from megatron.core.datasets.gpt_dataset import GPTDatasetConfig
from megatron.core.datasets.gpt_dataset import MockGPTDataset, GPTDataset
import megatron.legacy.model
from megatron.core.models.gpt import GPTModel
from megatron.training import pretrain
from megatron.core.utils import StragglerDetector
from megatron.core.num_microbatches_calculator import get_num_microbatches
from megatron.core.transformer.spec_utils import import_module
from megatron.training.utils import (
    get_batch_on_this_cp_rank,
    get_batch_on_this_tp_rank,
)
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.yaml_arguments import core_transformer_config_from_yaml
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
)


stimer = StragglerDetector()

def model_provider(pre_process=True, post_process=True) -> Union[GPTModel, megatron.legacy.model.GPTModel]:
    """Builds the model.

    If you set the use_legacy_models to True, it will return the legacy GPT model and if not the mcore GPT model.

    Args:
        pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
        post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.


    Returns:
        Union[GPTModel, megatron.legacy.model.GPTModel]: The returned model
    """
    args = get_args()
    use_te = args.transformer_impl == "transformer_engine"

    if args.record_memory_history:
        torch.cuda.memory._record_memory_history(True,
            # keep 100,000 alloc/free events from before the snapshot
            trace_alloc_max_entries=100000,

            # record stack information for the trace events
            trace_alloc_record_context=True)

    print_rank_0('building GPT model ...')
    # Experimental loading arguments from yaml
    if args.yaml_cfg is not None:
        config = core_transformer_config_from_yaml(args, "language_model")
    else:
        config = core_transformer_config_from_args(args)

    if args.use_legacy_models:
        model = megatron.legacy.model.GPTModel(
            config,
            num_tokentypes=0,
            parallel_output=True,
            pre_process=pre_process,
            post_process=post_process,
        )
    else: # using core models
        if args.spec is not None:
            transformer_layer_spec = import_module(args.spec)
        else:
            if use_te:
                transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(args.num_experts, args.moe_grouped_gemm, args.qk_layernorm, args.multi_latent_attention, args.fp8, args.downscale_residual, args.attn_layernorm, args.mlp_layernorm, args.use_torchqknorm)
            else:
                transformer_layer_spec = get_gpt_layer_local_spec(args.num_experts, args.moe_grouped_gemm, args.qk_layernorm, args.multi_latent_attention)

        build_model_context = nullcontext
        build_model_context_args = {}
        if args.fp8_param_gather:
            try:
                from transformer_engine.pytorch import fp8_model_init

                build_model_context = fp8_model_init
                build_model_context_args["enabled"] = True

                # Check if fp8_model_init supports preserve_high_precision_init_val
                if "preserve_high_precision_init_val" in inspect.signature(fp8_model_init).parameters:
                    build_model_context_args["preserve_high_precision_init_val"] = True
            except:
                raise RuntimeError("--fp8-param-gather requires `fp8_model_init` from TransformerEngine, but not found.")

        with build_model_context(**build_model_context_args):
            model = GPTModel(
                config=config,
                transformer_layer_spec=transformer_layer_spec,
                vocab_size=args.padded_vocab_size,
                max_sequence_length=args.max_position_embeddings,
                pre_process=pre_process,
                post_process=post_process,
                fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
                parallel_output=True,
                share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
                position_embedding_type=args.position_embedding_type,
                rotary_percent=args.rotary_percent,
                rotary_base=args.rotary_base,
                rope_scaling=args.use_rope_scaling,
                log_kurtosis=args.log_kurtosis,
                log_gains_norm=args.log_gains_norm,
                input_embeddings_multiplier=args.input_embeddings_multiplier,
                final_layernorm=args.final_layernorm,
            )

    print_rank_0("Built model:")
    print_rank_0(model)
    return model


def get_batch(data_iterator):
    """Generate a batch."""

    # TODO: this is pretty hacky, find a better way
    if (not mpu.is_pipeline_first_stage()) and (not mpu.is_pipeline_last_stage()):
        return None, None, None, None, None

    # get batches based on the TP rank you are on
    batch = get_batch_on_this_tp_rank(data_iterator)

    # slice batch along sequence dimension for context parallelism
    batch = get_batch_on_this_cp_rank(batch)

    return batch.values()


def loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor,
              tracked_metrics: Optional[list[dict[str, torch.Tensor]]] = None
              ) -> tuple[float, int, dict[str, torch.Tensor | tuple[torch.Tensor, ...]]]:
    """Loss function.

    Args:
        loss_mask (torch.Tensor): Used to mask out some portions of the loss
        output_tensor (torch.Tensor): The tensor with the losses

    Returns:
        the loss scalar for this micro-batch
        the number of non-padded tokens in this microbatch
        a dict containing reporting metrics on the loss and number of tokens across
            the data parallel ranks
    """
    args = get_args()

    # Handle tracked metrics.
    if tracked_metrics is not None:
        # Some basic assertions.
        # Here we assume that tracked_metrics have been already summed across all micro batches in the current (tp,pp)-rank.
        # In particular, if sequence_parallelism=True, then we have all_reduced(op=SUM) the metrics in tp rank.
        # We also assume tracked_metrics will not be None at most once, in the very last micro batch.
        # We also assume that we have gathered all `tracked_metrics`, thus we have a list of length `num_layers` in this last pp stage.
        assert len(tracked_metrics) == args.num_layers
        metrics_keys = sorted(tracked_metrics[0])  # get keys
        known_metrics = {"squared_activations", "summed_activations", "squared_gains"}
        assert set(metrics_keys) <= known_metrics, f"Unknown metrics: encountered {set(metrics_keys) - known_metrics}"
        assert len(metrics_keys) > 0
        assert all(set(this_tracked_metrics) == set(metrics_keys) for this_tracked_metrics in tracked_metrics)

        # Now we can sum all the metrics across dp ranks.
        # First we construct a big tensor with all the metrics to reduce them in a single collective operation.
        metrics_sizes = {metric: tracked_metrics[0][metric].size() for metric in metrics_keys}
        flattened_metrics = []
        for this_tracked_metrics in tracked_metrics:
            for metric in metrics_keys:
                tensor = this_tracked_metrics[metric]
                assert metrics_sizes[metric] == tensor.size()
                flattened_metrics.append(tensor.view(-1))
        flattened_metrics = torch.cat(flattened_metrics)

        # Finally, reduce the tensor.
        torch.distributed.all_reduce(flattened_metrics, group=mpu.get_data_parallel_group(),
                                     op=torch.distributed.ReduceOp.SUM)

        # Now we unwrap the flattened metrics back into a list[dict[str, Tensor]].
        idx = 0
        for this_tracked_metrics in tracked_metrics:
            for metric in metrics_keys:
                numel = math.prod(metrics_sizes[metric])
                this_tracked_metrics[metric] = flattened_metrics[idx : idx+numel].view(metrics_sizes[metric])
                idx += numel
        assert idx == flattened_metrics.numel(), f"{idx, flattened_metrics.numel()}"

        # Alias some values.
        seq_len = args.seq_length
        gbs = args.global_batch_size
        hidden_size = args.hidden_size
        # we use this instead of gbs/mbs/dp_size because we reduce(op=SUM, group=DP) the gains
        # so in order to be consistent when changing the dp_size, we need to divide the gains by the
        # total number of micro batches (not only in your dp rank)
        total_acc = gbs/args.micro_batch_size

        # Now we are able to actually compute the metrics the user requested.
        report_metrics = {}
        n_kurtosis_blocks = 4
        if "squared_activations" in metrics_keys:  # then we can compute kurtosis and avg_act_rms.
            for this_metrics in tracked_metrics:
                assert this_metrics["squared_activations"].size() == (hidden_size,), f"{this_metrics['squared_activations'].size()}"
                this_metrics["act_rms"] = torch.sqrt(torch.sum(this_metrics["squared_activations"]/(seq_len*gbs*hidden_size)))
                this_metrics["kurtosis"] = torch.var(this_metrics["squared_activations"]/((this_metrics["act_rms"] + 1e-8)**2*seq_len*gbs))
            report_metrics["avg_act_rms"] = sum(this_metrics["act_rms"] for this_metrics in tracked_metrics)/len(tracked_metrics)
            report_metrics["avg_kurtosis"] = sum(this_metrics["kurtosis"] for this_metrics in tracked_metrics)/len(tracked_metrics)
            # Compute chunked kurtosis.
            chunk_size = int(math.ceil(len(tracked_metrics)/n_kurtosis_blocks))
            kurtosis_chunks = [[tracked_metrics[j]["kurtosis"] for j in range(i, i + chunk_size)]
                               for i in range(0, len(tracked_metrics), chunk_size)]
            kurtosis_dict = {f"kurtosis_chunk_{i}": sum(chunk)/len(chunk)
                             for i, chunk in enumerate(kurtosis_chunks)}
            report_metrics.update(kurtosis_dict)
        if "squared_gains" in metrics_keys:  # then we compute avg_gains_norm.
            for this_metrics in tracked_metrics:
                assert this_metrics["squared_gains"].size() == ()
                # We divide by acc because this value `squared_gains` remains constant across all micro batches,
                # but it is assumed to be summed (instead of averaged) across microbatches.
                this_metrics["gains_norm"] = torch.sqrt(this_metrics["squared_gains"]/total_acc)
            report_metrics["avg_gains_norm"] = sum(this_metrics["gains_norm"] for this_metrics in tracked_metrics)/len(tracked_metrics)
    else:
        report_metrics = {}

    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    total_tokens = loss_mask.sum()
    loss = torch.cat([torch.sum(losses.view(-1) * loss_mask).view(1), total_tokens.view(1)])

    if args.context_parallel_size > 1:
        torch.distributed.all_reduce(loss, group=mpu.get_context_parallel_group())

    # Check individual rank losses are not NaN prior to DP all-reduce.
    if args.check_for_nan_in_loss_and_grad:
        global_rank = torch.distributed.get_rank()
        assert not loss[0].isnan(), (
            f'Rank {global_rank}: found NaN in local forward loss calculation. '
            f'Device: {torch.cuda.current_device()}, node: {os.uname()[1]}'
        )

    # Reduce loss for logging.
    reporting_loss = loss.clone().detach()
    torch.distributed.all_reduce(reporting_loss, group=mpu.get_data_parallel_group())

    # Reduce other metrics.
    local_num_tokens = loss[1].clone().detach().to(torch.int)
    return (
        loss[0] * args.context_parallel_size,
        local_num_tokens,
        {'lm loss': (reporting_loss[0], reporting_loss[1]), **report_metrics},
    )


def forward_step(data_iterator, model: GPTModel):
    """Forward training step.

    Args:
        data_iterator : Input data iterator
        model (GPTModel): The GPT Model
    """
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    global stimer
    with stimer(bdata=True):
        tokens, labels, loss_mask, attention_mask, position_ids = get_batch(
            data_iterator)
    timers('batch-generator').stop()

    with stimer:
        output_dict = model(tokens, position_ids, attention_mask,
                            labels=labels)

    return output_dict, partial(loss_func, loss_mask)


def is_dataset_built_on_rank():
    return (
        mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage()
    ) and mpu.get_tensor_model_parallel_rank() == 0


def core_gpt_dataset_config_from_args(args):
    tokenizer = get_tokenizer()

    return GPTDatasetConfig(
        random_seed=args.seed,
        sequence_length=args.seq_length,
        blend=get_blend_from_list(args.data_path),
        blend_per_split=[
            get_blend_from_list(args.train_data_path),
            get_blend_from_list(args.valid_data_path),
            get_blend_from_list(args.test_data_path)
        ],
        renormalize_blend_weights=args.renormalize_blend_weights,
        split=args.split,
        num_dataset_builder_threads=args.num_dataset_builder_threads,
        path_to_cache=args.data_cache_path,
        mmap_bin_files=args.mmap_bin_files,
        tokenizer=tokenizer,
        reset_position_ids=args.reset_position_ids,
        reset_attention_mask=args.reset_attention_mask,
        eod_mask_loss=args.eod_mask_loss,
        create_attention_mask=args.create_attention_mask_in_dataloader,
        s3_cache_path = args.s3_cache_path
    )


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build the train test and validation datasets.

    Args:
        train_val_test_num_samples : A list containing the number of samples in train test and validation.
    """
    args = get_args()

    config = core_gpt_dataset_config_from_args(args)

    if args.mock_data:
        dataset_type = MockGPTDataset
    else:
        dataset_type = GPTDataset

    print_rank_0("> building train, validation, and test datasets for GPT ...")

    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        dataset_type,
        train_val_test_num_samples,
        is_dataset_built_on_rank,
        config
    ).build()

    print_rank_0("> finished creating GPT datasets ...")

    if is_dataset_built_on_rank():
        n_tokens_per_epoch = train_ds._get_num_tokens_per_epoch()
        print_rank_0(f"Total number of tokens available in train dataset: {n_tokens_per_epoch}")
        print_rank_0(f"Total number of epochs to train for: {train_ds._get_num_epochs(n_tokens_per_epoch)}")

    return train_ds, valid_ds, test_ds


if __name__ == "__main__":

    import logging
    logging.basicConfig(level=logging.DEBUG)

    # Temporary for transition to core datasets
    train_valid_test_datasets_provider.is_distributed = True

    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
    )
