# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Qwen3 pretraining recipes."""

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.components.data import (
    ConcatThenSplitPackingConfig,
    GrainDataLoader,
    HuggingFaceStreamingSource,
    SingleDatasetConfig,
)
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.optimizer import default_adamw, LRSchedulersContainer
from torchtitan.config import (
    CompileConfig,
    DebugConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.distributed.activation_checkpoint import SelectiveAC
from torchtitan.hf_datasets.text_datasets import TextProcessor
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.models.qwen3 import model_registry
from torchtitan.observability.metrics import MetricsProcessor
from torchtitan.observability.profiler import Profiler
from torchtitan.trainer import Trainer


def qwen3_balanced_600m_ack_reasoning() -> Trainer.Config:
    """Balanced 600M Qwen3 CPT on the full acknowledgement reasoning corpus."""
    model_spec = model_registry("balanced-600M", seq_len=4096)
    return Trainer.Config(
        dump_folder="/path/to/dump/folder",
        hf_assets_path="/path/to/local/tokenizer/pleias_65k_tokenizer",
        model_spec=model_spec,
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_spec),
            ),
        ),
        dataloader=GrainDataLoader.Config(
            dataset=ConcatThenSplitPackingConfig(
                dataset=SingleDatasetConfig(
                    source=HuggingFaceStreamingSource.Config(
                        path="json",
                        split="train",
                        load_dataset_kwargs={
                            "data_files": "/path/to/local/dataset/ack_reasoning_full_v2schema/*.jsonl",
                        },
                    ),
                    processor=TextProcessor.Config(),
                ),
            ),
            shuffle=True,
            seed=42,
        ),
        optimizer=default_adamw(
            lr=5e-5,
            eps=1e-8,
            weight_decay=0.01,
        ),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=100,
            decay_ratio=0.25,
            decay_type="linear",
            min_lr_factor=0.1,
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=8 * 4096,
            num_tokens_per_train_step=512 * 4096,
            max_context_length=4096,
            max_norm=1.0,
            steps=2385,
            dtype="float32",
            mixed_precision_param="bfloat16",
            mixed_precision_reduce="float32",
        ),
        parallelism=ParallelismConfig(
            data_parallel_replicate_degree=1,
            data_parallel_shard_degree=-1,
            tensor_parallel_degree=1,
            pipeline_parallel_degree=1,
            context_parallel_degree=1,
        ),
        checkpointer=CheckpointManager.Config(
            folder="checkpoint",
            interval=795,
            export_dtype="float32",
            initial_load_path="/path/to/local/model_dir/checkpoint",
            initial_load_model_only=True,
        ),
        activation_checkpoint=SelectiveAC.Config(),
        compile=CompileConfig(components=["model", "loss"]),
        metrics=MetricsProcessor.Config(
            log_freq=10,
            enable_tensorboard=False,
            enable_wandb=False,
            disable_color_printing=False,
        ),
        debug=DebugConfig(
            seed=42,
            print_config=True,
        ),
        profiler=Profiler.Config(
            enable_profiling=False,
            save_traces_folder="profile_trace",
            profile_freq=10,
            enable_memory_snapshot=False,
            save_memory_snapshot_folder="memory_snapshot",
        ),
    )
