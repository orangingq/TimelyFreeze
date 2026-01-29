# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

from functools import partial
from typing import Any, Callable
import random

import torch

from datasets import Dataset, load_dataset
from datasets.distributed import split_dataset_by_node
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset

from torchtitan.components.dataloader import ParallelAwareDataloader
from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.config import JobConfig
from torchtitan.tools.logging import logger


def _load_c4_dataset(dataset_path: str, split: str):
    """Load C4 dataset with default configuration."""
    return load_dataset(dataset_path, name="en", split=split, streaming=True)

def _process_c4_text(sample: dict[str, Any]) -> str:
    """Process C4 dataset sample text."""
    return sample["text"]

def _process_slimorca_text(sample: dict[str, Any]) -> str:
    """Process SlimOrca dataset sample into a single prompt+response text."""
    sample = sample["conversations"]
    system_prompt = sample[-3]['value'].strip() if len(sample) > 2 else None
    question = sample[-2]['value'].strip()
    response = sample[-1]['value'].strip()

    if system_prompt:
        prompt = f"<|system|>\n{system_prompt}\n<|user|>\n{question}\n<|assistant|>\n{response}"
    else:
        prompt = f"<|user|>\n{question}\n<|assistant|>\n{response}"
    return prompt

def _process_alpaca_text(sample: dict[str, Any]) -> str:
    """Process Alpaca dataset sample text."""
    return sample["text"]

def _process_alpaca_cleaned_text(sample: dict[str, Any]) -> str:
    """
    Process cleaned Alpaca dataset sample into a single prompt+response text.
    https://github.com/tatsu-lab/stanford_alpaca#data-release
    """
    instruction = sample["instruction"].strip()
    input_text = sample.get("input", "").strip()
    response = sample["output"].strip()

    if input_text:
        prompt = f"<|user|>\n{instruction}\n\nInput: {input_text}\n<|assistant|>\n{response}"
    else:
        prompt = f"<|user|>\n{instruction}\n<|assistant|>\n{response}"
    return prompt

def _process_medical_text(sample: dict[str, Any]) -> str:
    """Process Medical dataset sample into a single prompt+response text."""
    question = sample["Question"].strip()
    cot = sample["Complex_CoT"].strip()
    response = sample["Response"].strip()
    prompt = f"<|user|>\n{question}\n<|assistant|>\n{cot}\n\nResponse:\n{response}"
    return prompt

def _process_codealpaca_text(sample: dict[str, Any]) -> str:
    """Process CodeAlpaca dataset sample into a single prompt+response text."""
    instruction = sample["instruction"].strip()
    input_text = sample.get("input", "").strip()
    response = sample["output"].strip()

    if input_text:
        prompt = f"<|user|>\n{instruction}\n\nInput: {input_text}\n<|assistant|>\n{response}"
    else:
        prompt = f"<|user|>\n{instruction}\n<|assistant|>\n{response}"
    return prompt

def _process_openhermes_text(sample: dict[str, Any]) -> str:
    """
    Process an OpenHermes 2.5 dataset sample into a single prompt-response text.
    """
    convs = sample.get("conversations", [])
    system_prompt, question, response = None, None, None
    
    for turn in convs:
        role = turn.get("from", "")
        value = turn.get("value", "").strip()

        if role == "system" and value:
            system_prompt = value
        elif role == "human":
            question = value
        elif role == "gpt":
            response = value

    # Build formatted prompt
    if not question or not response:
        return ""  # skip incomplete conversations
    elif system_prompt:
        return f"<|system|>\n{system_prompt}\n<|user|>\n{question}\n<|assistant|>\n{response}"
    else:
        return f"<|user|>\n{question}\n<|assistant|>\n{response}"


@dataclass
class DatasetConfig:
    path: str
    loader: Callable
    text_processor: Callable


# Add your dataset here here - more information at docs/datasets.md
DATASETS = {
    "c4": DatasetConfig(
        path="allenai/c4",
        loader=partial(_load_c4_dataset, split="train"),
        text_processor=_process_c4_text,
    ),
    "c4_test": DatasetConfig(
        path="tests/assets/c4_test",
        loader=lambda path: load_dataset(path, split="train"),
        text_processor=_process_c4_text,
    ),
    "c4_validation": DatasetConfig(
        path="allenai/c4",
        loader=partial(_load_c4_dataset, split="validation"),
        text_processor=_process_c4_text,
    ),
    "slimorca": DatasetConfig(
        path="Open-Orca/SlimOrca",
        loader=lambda path: load_dataset(path, split="train"), # 518k samples
        text_processor=_process_slimorca_text,
    ),
    "alpaca": DatasetConfig(
        path="tatsu-lab/alpaca",
        loader=lambda path: load_dataset(path, split="train"), # 52k samples
        text_processor=_process_alpaca_text,
    ),
    "alpaca_cleaned": DatasetConfig(
        path="yahma/alpaca-cleaned",
        loader=lambda path: load_dataset(path, split="train"), # 51.8k samples
        text_processor=_process_alpaca_cleaned_text,
    ),
    "alpaca_gpt4": DatasetConfig(
        path="vicgalle/alpaca-gpt4",
        loader=lambda path: load_dataset(path, split="train"), # 52k samples
        text_processor=_process_alpaca_cleaned_text,
    ),
    "medical": DatasetConfig(
        path="FreedomIntelligence/medical-o1-reasoning-SFT",
        loader=lambda path: load_dataset(path, "en_mix", split="train"), # 24.9k samples for en_mix
        text_processor=_process_medical_text,
    ),
    "codealpaca": DatasetConfig(
        path='sahil2801/CodeAlpaca-20k',
        loader=lambda path: load_dataset(path, split="train"), # 20k samples
        text_processor=_process_codealpaca_text,
    ),
    "openhermes": DatasetConfig(
        path="teknium/OpenHermes-2.5",
        loader=lambda path: load_dataset(path, split="train"), # 1M samples
        text_processor=_process_openhermes_text,
    ),
}


def _validate_dataset(
    dataset_name: str, dataset_path: str | None = None
) -> tuple[str, Callable, Callable]:
    """Validate dataset name and path."""
    if dataset_name not in DATASETS:
        raise ValueError(
            f"Dataset {dataset_name} is not supported. "
            f"Supported datasets are: {list(DATASETS.keys())}"
        )

    config = DATASETS[dataset_name]
    path = dataset_path or config.path
    if torch.distributed.get_rank() == 0:
        logger.info(f"Preparing {dataset_name} dataset from {path}")
    return path, config.loader, config.text_processor


class HuggingFaceDataset(IterableDataset, Stateful):
    def __init__(
        self,
        dataset_name: str,
        dataset_path: str | None,
        tokenizer: BaseTokenizer,
        seq_len: int = 2048,
        dp_rank: int = 0,
        dp_world_size: int = 1,
        infinite: bool = False,
    ) -> None:
        # Force lowercase for consistent comparison
        dataset_name = dataset_name.lower()

        path, dataset_loader, text_processor = _validate_dataset(
            dataset_name, dataset_path
        )
        ds = dataset_loader(path)

        self.dataset_name = dataset_name
        self._data = split_dataset_by_node(ds, dp_rank, dp_world_size)
        self._tokenizer = tokenizer
        self.seq_len = seq_len
        self.infinite = infinite
        self._text_processor = text_processor

        # Variables for checkpointing
        self._sample_idx = 0
        self._token_buffer: list[int] = []

    def _get_data_iter(self):
        # For map-style datasets, resume by skipping to the correct index
        # For iterable-style datasets, the underlying iterator already points to the correct index
        if isinstance(self._data, Dataset):
            if self._sample_idx == len(self._data):
                return iter([])
            else:
                return iter(self._data.skip(self._sample_idx))

        return iter(self._data)

    def __iter__(self): # for pretraining
        max_buffer_token_len = 1 + self.seq_len

        while True:
            for sample in self._get_data_iter():
                # Use the dataset-specific text processor
                sample_text = self._text_processor(sample)
                sample_tokens = self._tokenizer.encode(
                    # sample_text, add_bos=True, add_eos=True # : streaming mode without truncation
                    sample_text, add_bos=True, add_eos=True, truncation=True, max_length=self.seq_len, padding='max_length' # : sample-level with truncation
                )
                self._token_buffer.extend(sample_tokens)
                self._sample_idx += 1

                while len(self._token_buffer) >= max_buffer_token_len:
                    x = torch.LongTensor(self._token_buffer[:max_buffer_token_len])
                    # update tokens to the remaining tokens
                    self._token_buffer = self._token_buffer[max_buffer_token_len:]
                    input = x[:-1]
                    label = x[1:]
                    yield {"input": input}, label

            if not self.infinite:
                logger.warning(f"Dataset {self.dataset_name} has run out of data")
                break
            else:
                # Reset offset for the next iteration
                self._sample_idx = 0
                logger.warning(f"Dataset {self.dataset_name} is being re-looped")
                # Ensures re-looping a dataset loaded from a checkpoint works correctly
                if not isinstance(self._data, Dataset):
                    if hasattr(self._data, "set_epoch") and hasattr(
                        self._data, "epoch"
                    ):
                        self._data.set_epoch(self._data.epoch + 1)
    
    def load_state_dict(self, state_dict):
        self._token_buffer = state_dict["token_buffer"]

        if isinstance(self._data, Dataset):
            self._sample_idx = state_dict["sample_idx"]
        else:
            assert "data" in state_dict
            self._data.load_state_dict(state_dict["data"])

    def state_dict(self):
        _state_dict = {"token_buffer": self._token_buffer}

        if isinstance(self._data, Dataset):
            _state_dict["sample_idx"] = self._sample_idx
        else:
            # Save the iterable dataset's state to later efficiently resume from it
            # https://huggingface.co/docs/datasets/v3.5.0/en/stream#save-a-dataset-checkpoint-and-resume-iteration
            _state_dict["data"] = self._data.state_dict()

        return _state_dict

# class HuggingFaceMultiDataset(IterableDataset, Stateful):
#     def __init__(
#         self,
#         dataset_names: list[str],
#         dataset_paths: list[str] | None,
#         tokenizer: BaseTokenizer,
#         seq_len: int = 2048,
#         dp_rank: int = 0,
#         dp_world_size: int = 1,
#         infinite: bool = False,
#     ) -> None:
#         assert dataset_paths is None or len(dataset_names) == len(dataset_paths)

#         self.datasets = []
#         self.dataset_names = dataset_names
#         self._tokenizer = tokenizer
#         self.seq_len = seq_len
#         self.infinite = infinite
#         self.dp_rank = dp_rank
#         self.dp_world_size = dp_world_size

#         self._sample_idx = [0] * len(dataset_names)
#         self._token_buffer: list[int] = []

#         for i, ds_name in enumerate(dataset_names):
#             ds_path = dataset_paths[i] if dataset_paths else None
#             path, dataset_loader, text_processor = _validate_dataset(ds_name, ds_path)
#             ds = dataset_loader(path)
#             ds_split = split_dataset_by_node(ds, dp_rank, dp_world_size)

#             self.datasets.append({
#                 "name": ds_name,
#                 "data": ds_split,
#                 "text_processor": text_processor,
#             })


            

#     def _get_data_iter(self, i):
#         ds = self.datasets[i]["data"]
#         idx = self._sample_idx[i]

#         if isinstance(ds, Dataset):
#             if idx == len(ds):
#                 return iter([])
#             else:
#                 return iter(ds.skip(idx))
#         return iter(ds)

#     def __iter__(self):
#         max_buffer_token_len = 1 + self.seq_len

#         while True:
#             for i, ds_dict in enumerate(self.datasets):
#                 for sample in self._get_data_iter(i):
#                     sample_text = ds_dict["text_processor"](sample)
#                     sample_tokens = self._tokenizer.encode(
#                         sample_text,
#                         add_bos=True,
#                         add_eos=True,
#                         truncation=True,
#                         max_length=self.seq_len,
#                         padding="max_length",
#                     )
#                     self._token_buffer.extend(sample_tokens)
#                     self._sample_idx[i] += 1

#                     while len(self._token_buffer) >= max_buffer_token_len:
#                         x = torch.LongTensor(self._token_buffer[:max_buffer_token_len])
#                         self._token_buffer = self._token_buffer[max_buffer_token_len:]
#                         input = x[:-1]
#                         label = x[1:]
#                         yield {"input": input}, label

#             if not self.infinite:
#                 logger.warning(f"Datasets {self.dataset_names} have run out of data")
#                 break
#             else:
#                 self._sample_idx = [0] * len(self.datasets)
#                 logger.warning(f"Datasets {self.dataset_names} are being re-looped")

#     def load_state_dict(self, state_dict):
#         self._token_buffer = state_dict["token_buffer"]
#         self._sample_idx = state_dict["sample_idx"]

#     def state_dict(self):
#         return {"token_buffer": self._token_buffer, "sample_idx": self._sample_idx}


class HuggingFaceMultiDataset(IterableDataset, Stateful):
    def __init__(
        self,
        dataset_names: list[str],
        dataset_paths: list[str] | None,
        tokenizer: BaseTokenizer,
        seq_len: int = 2048,
        dp_rank: int = 0,
        dp_world_size: int = 1,
        infinite: bool = False,
    ) -> None:
        assert dataset_paths is None or len(dataset_names) == len(dataset_paths)

        self.datasets = []
        self.dataset_names = dataset_names
        self._tokenizer = tokenizer
        self.seq_len = seq_len
        self.infinite = infinite
        self.dp_rank = dp_rank
        self.dp_world_size = dp_world_size

        self._sample_idx = [0] * len(dataset_names)
        self._token_buffer: list[int] = []

        for i, ds_name in enumerate(dataset_names):
            ds_path = dataset_paths[i] if dataset_paths else None
            path, dataset_loader, text_processor = _validate_dataset(ds_name, ds_path)
            ds = dataset_loader(path)
            ds_split = split_dataset_by_node(ds, dp_rank, dp_world_size)

            self.datasets.append({
                "name": ds_name,
                "data": ds_split,
                "text_processor": text_processor,
            })

        # dataset별 길이 추정 (streaming 아닐 때만)
        self.dataset_lengths = []
        for d in self.datasets:
            try:
                self.dataset_lengths.append(len(d["data"]))
            except TypeError:
                self.dataset_lengths.append(1.0)  # fallback

        total = sum(self.dataset_lengths)
        self.dataset_probs = [l / total for l in self.dataset_lengths]

    def _get_data_iter(self, i):
        ds = self.datasets[i]["data"]
        idx = self._sample_idx[i]
        if isinstance(ds, Dataset):
            if idx == len(ds):
                return iter([])
            else:
                return iter(ds.skip(idx))
        return iter(ds)

    def __iter__(self):
        max_buffer_token_len = 1 + self.seq_len
        iters = [self._get_data_iter(i) for i in range(len(self.datasets))]

        while True:
            # dataset index를 확률적으로 선택 (medical/alpaca 비율 유지)
            ds_i = random.choices(range(len(self.datasets)), weights=self.dataset_probs, k=1)[0]
            try:
                sample = next(iters[ds_i])
            except StopIteration:
                # 다 돈 dataset은 다시 iterator 초기화
                if not self.infinite:
                    logger.warning(f"Dataset {self.datasets[ds_i]['name']} have run out of data")
                    break
                else:
                    self._sample_idx[ds_i] = 0
                    logger.warning(f"Dataset {self.datasets[ds_i]['name']} is being re-looped")
                    iters[ds_i] = self._get_data_iter(ds_i)
                    sample = next(iters[ds_i])
                    
            # 토큰화
            sample_text = self.datasets[ds_i]["text_processor"](sample)
            sample_tokens = self._tokenizer.encode(
                sample_text,
                add_bos=True,
                add_eos=True,
                truncation=True,
                max_length=self.seq_len,
                padding="max_length",
            )
            self._token_buffer.extend(sample_tokens)
            self._sample_idx[ds_i] += 1

            while len(self._token_buffer) >= max_buffer_token_len:
                x = torch.LongTensor(self._token_buffer[:max_buffer_token_len])
                self._token_buffer = self._token_buffer[max_buffer_token_len:]
                input = x[:-1]
                label = x[1:]
                yield {"input": input}, label

    def load_state_dict(self, state_dict):
        self._token_buffer = state_dict["token_buffer"]
        self._sample_idx = state_dict["sample_idx"]

    def state_dict(self):
        return {"token_buffer": self._token_buffer, "sample_idx": self._sample_idx}


def build_hf_dataloader(
    dp_world_size: int,
    dp_rank: int,
    tokenizer: BaseTokenizer,
    job_config: JobConfig,
    infinite: bool = True,
) -> ParallelAwareDataloader:
    """Build a data loader for HuggingFace datasets."""
    dataset_name = job_config.training.dataset
    dataset_path = job_config.training.dataset_path
    batch_size = job_config.training.local_batch_size
    seq_len = job_config.training.seq_len

    if ',' in dataset_name:
        hf_ds = HuggingFaceMultiDataset(
            dataset_names=dataset_name.split(','),
            dataset_paths=dataset_path.split(',') if dataset_path else None,
            tokenizer=tokenizer,
            seq_len=seq_len,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            infinite=infinite,
        )
    else:
        hf_ds = HuggingFaceDataset(
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            tokenizer=tokenizer,
            seq_len=seq_len,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            infinite=infinite,
    )

    return ParallelAwareDataloader(
        dataset=hf_ds,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        batch_size=batch_size,
    )


def build_hf_validation_dataloader(
    dp_world_size: int,
    dp_rank: int,
    tokenizer: BaseTokenizer,
    job_config: JobConfig,
) -> ParallelAwareDataloader:
    """Build a validation data loader for HuggingFace datasets."""
    dataset_name = job_config.validation.dataset
    dataset_path = job_config.validation.dataset_path
    batch_size = job_config.validation.local_batch_size
    seq_len = job_config.validation.seq_len

    hf_ds = HuggingFaceDataset(
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        tokenizer=tokenizer,
        seq_len=seq_len,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        infinite=False,
    )

    return ParallelAwareDataloader(
        dataset=hf_ds,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        batch_size=batch_size,
    )
