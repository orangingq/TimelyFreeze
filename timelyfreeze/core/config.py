import os
import time
from dataclasses import dataclass, field, fields
from typing import Any, Literal, Optional
import torch
from torch.distributed import ProcessGroup
import torch.distributed as dist
from setproctitle import setproctitle

from torchtitan.config.job_config import JobConfig, \
                    Job as BaseJob, \
                    Training as BaseTraining, \
                    Metrics as BaseMetrics, \
                    Comm as BaseComm, \
                    Parallelism as BaseParallelism

@dataclass
class Job(BaseJob):
    basename: str = time.strftime('%y%m%d_%H%M')
    """Base name for logs, image saving, etc. Initially, it will be set to the current time."""

@dataclass
class Training(BaseTraining):
    pass

@dataclass 
class LoRA:
    enable_lora: bool = False
    """Whether to enable LoRA adapters in the model."""
    lora_r: int = 16
    """LoRA rank."""
    lora_alpha: int = 32
    """LoRA alpha."""
    lora_dropout: float = 0.05
    """LoRA dropout rate."""
    lora_target_modules: list[str] = field(default_factory=lambda: [
        "wq", "wk", "wv", "wo", "w1", "w2", "w3"
        # "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
    ])
    """List of target modules for LoRA adaptation."""

@dataclass
class Metrics(BaseMetrics):
    log_freq: int = 100
    """Frequency of logging during training (in global accumulation steps)"""

    pplog_freq: int = 1600
    """Frequency of logging pipeline traces (in local steps=batches)"""

    wandb_name: str | None = None
    """Weights & Biases run name"""
    
@dataclass
class PipelineParallelism(BaseParallelism):
    pp: int = 1
    """Number of pipeline parallelism groups"""

    pipeline_parallel_schedule: Literal["GPipe", "1F1B", "Interleaved1F1B", "InterleavedZeroBubble", "ZBVZeroBubble"] = "1F1B"
    """Pipeline parallelism scheduler. Options: 'gpipe', '1F1B', 'Interleaved1F1B', 'InterleavedZeroBubble' 'ZBVZeroBubble'."""

    stages_per_rank: int = 1 
    """Number of pipeline stages per rank. >1 for interleaved pipeline parallelism, 1 for non-interleaved pipeline parallelism."""

    stages_list : list[int] | None = None
    """List of pipeline stages of this rank. It will be recalculated based on the pipeline schedule."""

    @property
    def num_stages(self) -> int:
        """
        Get the total number of pipeline stages.
        This is calculated as pp * stages_per_rank.
        For V-shaped pipeline parallelism, though the number of different stages is pp, 
        each stage is counted twice (forward and backward), so the total number of stages is pp * stages_per_rank.
        """
        return self.pp * self.stages_per_rank

    @property
    def vshape(self) -> bool:
        """
        Check if the pipeline parallelism is V-shaped.
        Returns True if the pipeline_parallel_schedule is 'ZBV'.
        """
        return self.pipeline_parallel_schedule in ['ZBVZeroBubble']

    @property
    def bwd_separated(self) -> bool:
        """
        Check if the backward pass is separated into backward_weight and backward_input.
        Returns True if the pipeline_parallel_schedule is 'InterleavedZeroBubble', or 'ZBVZeroBubble'.
        """
        return self.pipeline_parallel_schedule in ["InterleavedZeroBubble", "ZBVZeroBubble"] #  - for debug

    microbatches: int = 4 # number of microbatches for pipeline parallelism
    """Number of microbatches for pipeline parallelism."""

@dataclass
class Freezing:
    """Configuration for model freezing during training.
    This is used to freeze the model weights based on a specific metric.
    """
    freeze: bool = False
    """Whether to enable TimelyFreeze."""

    metric_type: str | None = None
    """Metric type for freezing."""
    
    phase_unit: int = 100
    """Number of steps per phase for freezing."""

    stability_check_freq: int = 10
    """Frequency of stability checks during training (steps)."""
    
    max_freeze_ratio: float = 0.9
    """<TimelyFreeze> Maximum freeze ratio for each stage (model partition)."""

@dataclass
class Comm(BaseComm):
    """
    Communication configuration for distributed training.
    """

    world_size: int = 1
    """Total number of processes participating in the job."""

    global_rank: int = 0
    """Rank of the current process in the whole (multi-node) distributed system."""

    local_rank: int = 0
    """Local rank of the current process within its node."""

    dp: int = 1 
    """Data parallelism degree (number of data parallelism units)."""

    pp: int = 1
    """Pipeline parallelism degree (number of pipeline parallelism units)."""

    pp_group: Optional[ProcessGroup] = None
    """Pipeline parallelism group (having same dp_rank, but different pp_rank) of this process."""

    dp_group: Optional[ProcessGroup] = None
    """Data parallelism group (having same pp_rank, but different dp_rank) of this process."""

    pp_rank: int = 0  # pipeline parallelism rank : local rank in the pipeline parallelism group
    """pp_rank" is the local rank in the pipeline parallelism group."""
    
    dp_rank: int = 0  # data parallelism rank : local rank in the data parallelism group
    """dp_rank" is the local rank in the data parallelism group."""
    
    master_dp_rank: int = 0  
    """master_dp_rank" is the local rank in the data parallelism group of the master process."""

    @property
    def distributed(self) -> bool:
        """
        Whether the job is distributed (i.e., uses multiple processes).
        Returns True if either data parallelism or pipeline parallelism is enabled.
        """
        return self.world_size > 1
    
    @property
    def get_first_stage_rank(self) -> int:
        '''get the first stage rank in the pipeline parallelism group.'''
        return self.dp_rank 
    
    @property
    def is_first_stage(self) -> bool:
        '''check if this process is the first stage in the pipeline parallelism group.'''
        return self.global_rank == self.get_first_stage_rank

    @property
    def get_last_stage_rank(self) -> int:
        '''get the last stage rank in the pipeline parallelism group.'''
        return self.dp_rank + self.dp * (self.pp - 1) 
    
    @property
    def is_last_stage(self) -> bool:
        '''check if this process is the last stage in the pipeline parallelism group.'''
        return self.global_rank == self.get_last_stage_rank

    @property
    def master_rank(self) -> int:
        """master_rank" is the global rank of the master process that manages the whole training processes."""
        return self.dp * (self.pp - 1)
    
    @property
    def is_master_rank(self) -> bool:
        '''check if this process is the master rank.'''
        return self.global_rank == self.master_rank


@dataclass
class TimelyFreezeConfig(JobConfig):
    """
    Default container for training configuration.
    """
    job: Job = field(default_factory=Job)
    training: Training = field(default_factory=Training)
    lora: LoRA = field(default_factory=LoRA)
    metrics: Metrics = field(default_factory=Metrics)
    parallelism: PipelineParallelism = field(default_factory=PipelineParallelism)
    freezing: Freezing = field(default_factory=Freezing)
    comm: Comm = field(default_factory=Comm)

    def to_dict(self) -> dict[str, Any]:
        result = {}
        for f in fields(self):
            result[f.name] = {}
            dataclass = getattr(self, f.name)
            for key, value in dataclass.__dict__.items():
                if isinstance(value, torch.distributed.ProcessGroup):
                    continue
                else:
                    result[f.name][key] = value
        return result
    
    def print(self) -> None:
        """
        Print the configuration in a readable format.
        """
        sentence = "----- TimelyFreeze⏰ Configuration:\n"
        for key, value in self.to_dict().items():
            if isinstance(value, dict):
                sentence += f"\t- {key}:\n"
                for sub_key, sub_value in value.items():
                    sentence += f"\t\t- {sub_key}: {sub_value}\n"
            else:
                sentence += f"\t\t- {key}: {value}\n"
        print(sentence)
        return
    
    def pre_initialize(self) -> None:
        """
        Set configuration before trainer initialization.
        """
        self.job.basename = self.job.basename if self.job.basename is not None else f"{time.strftime('%y%m%d_%H%M')}_{self.model.name}_pp{self.comm.pp}"
        if self.metrics.wandb_name is None:
            self.metrics.wandb_name = self.job.basename
        self.job.dump_folder = os.path.abspath(self.job.dump_folder)
        self.checkpoint.checkpoint_folder = os.path.join(self.job.dump_folder, self.checkpoint.checkpoint_folder, self.job.basename)
        self.comm.save_traces_folder = os.path.join(self.job.dump_folder, self.comm.save_traces_folder, self.job.basename)
        self.profiling.save_traces_folder = os.path.join(self.job.dump_folder, self.profiling.save_traces_folder, self.job.basename)
        self.profiling.save_memory_snapshot_folder = os.path.join(self.job.dump_folder, self.profiling.save_memory_snapshot_folder, self.job.basename)
        self.metrics.save_tb_folder = os.path.join(self.job.dump_folder, self.metrics.save_tb_folder, self.job.basename)
        self.metrics.image_folder = os.path.join(self.job.dump_folder, self.metrics.image_folder, self.job.basename)

        self.parallelism.pp = self.comm.pp = self.parallelism.pipeline_parallel_degree = max(self.parallelism.pp, self.comm.pp, self.parallelism.pipeline_parallel_degree)

        return

    def initialize(self, trainer) -> None:
        """ Update the TimelyFreezeConfig from a parallel_dim object.
        This is useful for integrating with existing Trainer configurations.
        """
        assert trainer.parallel_dims is not None, "Trainer.parallel_dims must be set before initializing TimelyFreezeConfig."
        assert trainer.pp_schedule is not None, "Trainer.pp_schedule must be set before initializing TimelyFreezeConfig."

        self.comm.world_size = trainer.parallel_dims.world_size
        mesh = trainer.parallel_dims.world_mesh
        coord = mesh.get_coordinate()
        self.comm.global_rank = mesh.mesh[*coord].item()
        self.comm.local_rank = mesh.mesh[*coord].item() # TODO: Assume a single node 
        
        # Data Parallelism
        if trainer.parallel_dims.dp_enabled:
            dp_mesh = trainer.parallel_dims.world_mesh["dp"]
            self.comm.dp, self.comm.dp_rank = dp_mesh.size(), dp_mesh.get_local_rank()
            self.comm.dp_group = dp_mesh.get_group("dp")
        else:
            self.comm.dp, self.comm.dp_rank = 1, 0
            self.comm.dp_group = dist.group.WORLD
        
        # Pipeline Parallelism
        if trainer.parallel_dims.pp_enabled:
            pp_mesh = trainer.parallel_dims.world_mesh["pp"]
            self.comm.pp, self.comm.pp_rank = pp_mesh.size(), pp_mesh.get_local_rank()
            self.comm.pp_group = pp_mesh.get_group("pp")
            
            # Update Parallelism configs
            self.parallelism.stages_per_rank = 1 if self.parallelism.pipeline_parallel_schedule.lower() in ["gpipe", "1f1b"] else 2
            self.parallelism.microbatches = trainer.pp_schedule._n_microbatches # self.training.local_batch_size // self.parallelism.pipeline_parallel_microbatch_size
            self.parallelism.stages_list = list(
                set([self.comm.pp_rank] + [a.stage_index for a in trainer.pp_schedule.pipeline_order[self.comm.pp_rank] if a is not None])
            )
        else:
            self.comm.pp, self.comm.pp_rank = 1, 0
            self.comm.pp_group = dist.group.WORLD
        self.parallelism.pp = self.comm.pp
        assert self.parallelism.pp == self.parallelism.pipeline_parallel_degree, "Mismatch between Trainer and TimelyFreezeConfig pipeline parallelism degree."

        if trainer.parallel_dims.pp_enabled:
            from .logger import init_pipeline_log
            init_pipeline_log(self) # initialize the global pipeline_log instance

        # change the process title to show local_rank in the node
        title = f"[{self.comm.local_rank+1}/{self.comm.world_size}] TimelyFreeze⏰ - {self.job.basename}"
        setproctitle(title)
        
        if self.comm.is_master_rank:
            self.print() # Print Arguments
        return
    