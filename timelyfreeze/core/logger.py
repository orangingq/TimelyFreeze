from contextlib import contextmanager
from enum import Enum
from typing import Dict, List, Tuple, Union
import numpy as np
from torchtitan.tools.logging import logger
from torch.cuda import Event

from .action import Action, ActionType, ActionWithLog, ActionWithFreezing
from .config import TimelyFreezeConfig

__all__ = ['pipeline_log', 'PipelineLog']

class ActionStatus:
    '''current action status. start of the action, end of the action, or (else) None.'''
    START:str = 'start'
    END:str = 'end'
    NONE:None = None
    ALL = [START, END, NONE]

class ActionPhase:
    START = [ActionType.START]
    FORWARD = [ActionType.RECV_F, ActionType.FORWARD, ActionType.SEND_F]
    BACKWARD = [ActionType.RECV_B, ActionType.FULL_BACKWARD, ActionType.SEND_B]
    END = [ActionType.SYNC, ActionType.FINISH]

class LogType(Enum):
    '''log type. rank or batch'''
    BATCH = 0
    RANK = 1
    def __str__(self):
        return self.name
    def __repr__(self):
        return self.name
    def __int__(self):
        return self.value
    def __eq__(self, other):
        if isinstance(other, LogType):
            return self.value == other.value
        elif isinstance(other, int):
            return self.value == other
        elif isinstance(other, str):
            return self.name.lower() == other.lower()
        else:
            return False

class PipelineLog:
    '''log the pipeline step.'''
    def __init__(self, config: TimelyFreezeConfig, type:Union[LogType, str]=LogType.BATCH) -> None:
        self.disabled :bool = False 
        '''temporary disable flag. cannot call the __call__ function if set to True'''
        self.config :TimelyFreezeConfig = config 
        '''the config of the current process.'''
        if config.parallelism.bwd_separated:
            ActionPhase.BACKWARD = [ActionType.RECV_B, ActionType.FULL_BACKWARD, ActionType.BACKWARD_INPUT, ActionType.BACKWARD_WEIGHT, ActionType.SEND_B]
        self.pp_rank :int = config.comm.pp_rank 
        '''the pp rank of the current process.'''

        # status of current action
        self.step_cnt :int = 0
        '''the step count.'''
        self.microbatch :int= -1
        '''current microbatch index.'''
        self.stage :int = config.comm.pp_rank 
        '''current stage, or order of model partition. default to the pp_rank.'''
        self.step :int = None
        '''the current step. (count training steps only)'''
        self.range :ActionStatus = None
        '''current action status. start of the action, end of the action, or (else) None.'''


        # cuda temporary timer
        self.cuda_timer_batch :Dict[ActionStatus, List[Event]] = {ActionStatus.START: [], ActionStatus.END: []}
        '''Temporarily log the time duration of each batch.'''
        self.cuda_timer_schedule :Dict[ActionStatus, List[Event]] = {ActionStatus.START: [], ActionStatus.END: []}
        '''Temporarily log the time duration of each action in the schedule.'''
        # time logger
        self.log_batch_time = []
        self.actions_list = {
            ActionType.FORWARD: {}, 
            ActionType.FULL_BACKWARD: {},
            ActionType.BACKWARD_INPUT: {},
            ActionType.BACKWARD_WEIGHT: {},
        }
        # log the schedule order
        self.log_schedule_flag :bool= True
        self.log_schedule:List[ActionWithLog] = []
        self.action_dict: Dict[Tuple[ActionType, int, int, int], ActionWithFreezing] = None 
        '''the action dict, which is a dict of ActionWithFreezing with key (action_type, pp_rank, microbatch, stage)'''
        self.action_cache: ActionWithFreezing | None = None
        '''the action cache for the last accessed action.'''
        return
    
    def _tmp_timer_flush(self, flush_freq:int=10):
        '''flush the temporary timer for the next logging'''
        logger.debug(f"[Local Step {self.step_cnt}, Global step {self.global_step_cnt}] [🕒PipelineLog] Logging the time duration for the last {flush_freq} batches...")
        assert len(self.cuda_timer_batch[ActionStatus.START]) == flush_freq, f"the length of batch start should be {flush_freq}. start:{len(self.cuda_timer_batch[ActionStatus.START])}, end:{len(self.cuda_timer_batch[ActionStatus.END])}"
        assert len(self.cuda_timer_schedule[ActionStatus.START]) == flush_freq * len(self.log_schedule), f"the length of fwd start should be {flush_freq} * {len(self.log_schedule)}(=len(self.log_schedule)). start:{len(self.cuda_timer_schedule[ActionStatus.START])}, end:{len(self.cuda_timer_schedule[ActionStatus.END])}"
        assert self.step_cnt % flush_freq == 0 and self.step_cnt >= flush_freq, f"the step count should be a multiple of {flush_freq}. step_cnt: {self.step_cnt}"

        # wait for the last event ends in GPU
        self.cuda_timer_schedule[ActionStatus.END][-1].synchronize()
        self.cuda_timer_batch[ActionStatus.END][-1].synchronize()
            
        assert len(self.cuda_timer_batch[ActionStatus.START]) == len(self.cuda_timer_batch[ActionStatus.END]), f"the length of batch start and end should be the same. start:{len(self.cuda_timer_batch[ActionStatus.START])} != end:{len(self.cuda_timer_batch[ActionStatus.END])}"
        assert len(self.cuda_timer_schedule[ActionStatus.START]) == len(self.cuda_timer_schedule[ActionStatus.END]), f"the length of schedule start and end should be the same. start:{len(self.cuda_timer_schedule[ActionStatus.START])} != end:{len(self.cuda_timer_schedule[ActionStatus.END])}"
            
        # log the time duration without creating large temporary NumPy arrays
        num_actions = len(self.log_schedule)
        schedule_start_chunks = [self.cuda_timer_schedule[ActionStatus.START][i * num_actions:(i + 1) * num_actions] for i in range(flush_freq)]
        schedule_end_chunks = [self.cuda_timer_schedule[ActionStatus.END][i * num_actions:(i + 1) * num_actions] for i in range(flush_freq)]

        for action_idx, action in enumerate(self.log_schedule):
            batch_starts = [chunk[action_idx] for chunk in schedule_start_chunks]
            batch_ends = [chunk[action_idx] for chunk in schedule_end_chunks]
            action.add_log_time(
                self.step_cnt - flush_freq,
                start_time=[rank_start.elapsed_time(start) for (rank_start, start) in zip(self.cuda_timer_batch[ActionStatus.START], batch_starts)],
                duration=[start.elapsed_time(end) for (start, end) in zip(batch_starts, batch_ends)],
            )

        self.log_batch_time.extend([
            start.elapsed_time(end)
            for (start, end) in zip(self.cuda_timer_batch[ActionStatus.START], self.cuda_timer_batch[ActionStatus.END])
        ])
    
        # reset the tmp timers
        self.cuda_timer_batch = {ActionStatus.START: [], ActionStatus.END: []}
        self.cuda_timer_schedule = {ActionStatus.START: [], ActionStatus.END: []}
        return
    
    def disable(self):
        '''disable the log'''
        self.disabled = True
        return

    def enable(self):
        '''enable the log'''
        self.disabled = False
        return

    def __call__(self, microbatch:int=-1, stage:int=None, step:ActionType='', range:ActionStatus=ActionStatus.NONE, postfix:str='')->'PipelineLog':
        '''log the time duration of the current pipeline action.
        Args:
            microbatch: int, the microbatch index
            step: Union[ActionType, str], the step name or ActionType number
            range: ['start', 'end', 'None']. 
            postfix: str, the postfix of the step
        '''
        if self.disabled:
            return self
        
        assert step in ActionPhase.START + ActionPhase.FORWARD + ActionPhase.BACKWARD + ActionPhase.END, f"step {step} should be in {ActionPhase.START + ActionPhase.FORWARD + ActionPhase.BACKWARD + ActionPhase.END}"
        assert range in ActionStatus.ALL, "range should be 'start', 'end' or 'None'"
        
        # update the current status
        self.microbatch = microbatch if microbatch >=0 else self.microbatch
        self.stage = stage if stage is not None else self.pp_rank
        self.step = step
        self.range = range
        self.step_cnt += int(self._is_start_of_batch) # int(step in ActionPhase.START) # count the step
        if (self._is_start_of_batch):
            logger.debug(f"[Local Step {self.step_cnt}, Global step {self.global_step_cnt}] [🕒PipelineLog] Start of a new batch.") 

        if not self.step in [ActionType.FORWARD, ActionType.FULL_BACKWARD, ActionType.BACKWARD_INPUT, ActionType.BACKWARD_WEIGHT] \
            or not self.range in [ActionStatus.START, ActionStatus.END]:
            return

        # Process to identify pipeline schedule cycle (for the first cycle)
        if self.log_schedule_flag:
            self._create_actions_list()

        self.log_timer()
        return self
    
    @property
    def global_step_cnt(self):
        '''get the global step count'''
        return self.step_cnt // (self.config.training.global_batch_size // self.config.training.local_batch_size)
    
    # context manager
    def __enter__(self):
        '''enter the context'''
        assert self.disabled or self.range == ActionStatus.START, "the range should be 'start'"
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        '''exit the context'''
        if self.step in [ActionType.FULL_BACKWARD, ActionType.BACKWARD_INPUT, ActionType.BACKWARD_WEIGHT]:
            self.unfreeze(self.microbatch, self.stage, self.step)
        self(self.microbatch, self.stage, self.step, ActionStatus.END)
        return

    def fwd_recv(self, microbatch:int=-1, stage:int=None, postfix:str='')->'PipelineLog':
        return self(microbatch, stage, ActionType.RECV_F, ActionStatus.START, postfix=postfix) 
    def forward(self, microbatch:int=-1, stage:int=None, postfix:str='')->'PipelineLog':
        return self(microbatch, stage, ActionType.FORWARD, ActionStatus.START, postfix=postfix) 
    def fwd_send(self, microbatch:int=-1, stage:int=None, postfix:str='')->'PipelineLog':
        return self(microbatch, stage, ActionType.SEND_F, ActionStatus.START, postfix=postfix) 
    def bwd_recv(self, microbatch:int=-1, stage:int=None, postfix:str='')->'PipelineLog':
        return self(microbatch, stage, ActionType.RECV_B, ActionStatus.START, postfix=postfix) 
    
    @contextmanager
    def backward(self, microbatch, stage, postfix: str = ""):
        self(microbatch, stage, ActionType.FULL_BACKWARD, ActionStatus.START, postfix=postfix)
        freeze_list = self.freeze(microbatch, stage, ActionType.FULL_BACKWARD)
        try:
            yield freeze_list
        finally:
            self.unfreeze(microbatch, stage, ActionType.FULL_BACKWARD)
            self(microbatch, stage, ActionType.FULL_BACKWARD, ActionStatus.END)

    @contextmanager
    def backward_input(self, microbatch, stage, postfix:str=''):
        self(microbatch, stage, ActionType.BACKWARD_INPUT, ActionStatus.START, postfix=postfix)        
        try:
            if self.action_dict is not None and (ActionType.BACKWARD_WEIGHT, self.pp_rank, microbatch, stage) in self.action_dict:
                bwd_weight_action = self.action_dict[(ActionType.BACKWARD_WEIGHT, self.pp_rank, microbatch, stage)]
                yield getattr(bwd_weight_action, "freeze_list", None)
            else:
                yield None
        finally:
            self(microbatch, stage, ActionType.BACKWARD_INPUT, ActionStatus.END)
    
    @contextmanager
    def backward_weight(self, microbatch, stage, postfix: str = ""):
        self(microbatch, stage, ActionType.BACKWARD_WEIGHT, ActionStatus.START, postfix=postfix)
        freeze_list = self.freeze(microbatch, stage, ActionType.BACKWARD_WEIGHT)
        try:
            yield freeze_list
        finally:
            self.unfreeze(microbatch, stage, ActionType.BACKWARD_WEIGHT)
            self(microbatch, stage, ActionType.BACKWARD_WEIGHT, ActionStatus.END)

    def bwd_send(self, microbatch:int=-1, stage:int=None, postfix:str='')->'PipelineLog':
        return self(microbatch, stage, ActionType.SEND_B, ActionStatus.START, postfix=postfix) 
    def sync(self, microbatch:int=-1, stage:int=None, postfix:str='')->'PipelineLog':
        return self(microbatch, stage, ActionType.SYNC, ActionStatus.START, postfix=postfix) 
    
    def freeze(self, microbatch:int=-1, stage:int=None, step:ActionType=ActionType.FULL_BACKWARD)->list[bool]|None:
        '''log the freeze action and return the freeze list'''
        if self.disabled or self.action_dict is None:
            return None

        freeze_list = None
        if (step, self.pp_rank, microbatch, stage) in self.action_dict:
            self.action_cache = self.action_dict[(step, self.pp_rank, microbatch, stage)]
            freeze_list = self.action_cache.freeze(self.step_cnt)
        else:
            logger.warning(f"The action ({step.name}, {self.pp_rank}, {microbatch}, {stage}) is not found in the action_dict.")
        return freeze_list

    def unfreeze(self, microbatch:int=-1, stage:int=None, step:ActionType=ActionType.FULL_BACKWARD):
        '''log the unfreeze action'''
        if self.disabled or self.action_dict is None:
            return 

        if self.action_cache is not None and (step, self.pp_rank, microbatch, stage) in self.action_dict:
            self.action_cache.unfreeze()
        else:
            logger.warning(f"The action ({step.name}, {self.pp_rank}, {microbatch}, {stage}) is not found in the action_dict.")

    def log_timer(self):
        '''log the timer'''
        assert Action(self.step, self.config.comm.pp_rank, self.microbatch, self.stage) in self.log_schedule, f"the action ({self.step}, {self.config.comm.pp_rank}, {self.microbatch}, {self.stage}) should be in the log_schedule. log_schedule: {self.log_schedule}"
        
        # record the time for the start and end of each step
        new_event = Event(enable_timing=True)
        new_event.record()

        # record the time for the start and end of each step
        self.cuda_timer_schedule[self.range].append(new_event)

        # record the time for the start of each batch
        if self._is_start_of_batch or self._is_end_of_batch:
            self.cuda_timer_batch[self.range].append(new_event)

        # log the time duration of forward and backward, for every n=30 batches
        flush_freq = 20 
        end_of_nth_batch = self._is_end_of_batch and self.step_cnt % flush_freq == 0 and self.step_cnt >= flush_freq
        if end_of_nth_batch:
            self._tmp_timer_flush(flush_freq=flush_freq)
            
        if self._is_end_of_batch and self.step_cnt % self.config.metrics.pplog_freq == 0:
            self.timer_print()
        return
    
    def timer_print(self):
        '''print the time analysis (avg time per action type, avg batch time, gpu bubble ratio)'''
        emptiness = {type: len(stage_dict) == 0 for type, stage_dict in self.actions_list.items()} 
        avg_time = {type: np.mean([a.log_duration for a_list in stage_dict.values() for a in a_list]) if not empty else 0 for ((type, stage_dict), empty) in zip(self.actions_list.items(), emptiness.values())}
        avg_batch_time = np.mean(self.log_batch_time)
        cnt_per_batch = {type: sum([a.len_log for a_list in stage_dict.values() for a in a_list]) // len(self.log_batch_time) for (type, stage_dict) in self.actions_list.items()}
        gpu_bubble_ratio = 1 - (sum([avg_time[type]*cnt_per_batch[type] for type in avg_time.keys()]) / avg_batch_time)
        log_str = f"Avg. fwd time: {avg_time[ActionType.FORWARD]:.4f} / "
        if self.config.parallelism.bwd_separated:
            if not emptiness[ActionType.FULL_BACKWARD]:
                log_str += f"Avg. full-bwd time: {avg_time[ActionType.FULL_BACKWARD]:.4f} (Avg. bwd-weight time: {avg_time[ActionType.BACKWARD_WEIGHT]:.4f} / Avg. bwd-input time: {avg_time[ActionType.BACKWARD_INPUT]:.4f}) / "
            else:
                log_str += f"Avg. bwd-weight time: {avg_time[ActionType.BACKWARD_WEIGHT]:.4f} / Avg. bwd-input time: {avg_time[ActionType.BACKWARD_INPUT]:.4f} / "
        else:
            log_str += f"Avg. bwd time: {avg_time[ActionType.FULL_BACKWARD]:.4f} / "
        log_str += f"Avg. batch time: {avg_batch_time:.4f} (ms) / GPU bubble ratio: {gpu_bubble_ratio*100:.2f}%"
        logger.info(log_str)
        return
    
    def _create_actions_list(self):
        """Create the actions list and action dict for freezing."""
        # Process to identify pipeline schedule cycle (for the first cycle)
        assert self.log_schedule_flag, "This function should be called only once after the first cycle is recorded."
        
        if self.range == ActionStatus.START:
            new_action = ActionWithLog(self.step, self.pp_rank, self.microbatch, self.stage)
            self.log_schedule.append(new_action)
            if self.stage in self.actions_list[self.step].keys():
                self.actions_list[self.step][self.stage].append(new_action)
            else:
                self.actions_list[self.step][self.stage] = [new_action]
            
        if self._is_end_of_batch:
            assert len(self.log_schedule) <= self.config.parallelism.microbatches * self.config.parallelism.stages_per_rank * (4 if self.config.parallelism.bwd_separated else 2), \
                f"the length of log_schedule should be less than {self.config.parallelism.microbatches * self.config.parallelism.stages_per_rank * (4 if self.config.parallelism.bwd_separated else 2)}. log_schedule: {len(self.log_schedule)}"
            self.log_schedule_flag = False

            return # pass the first cycle
        return

    @property
    def _is_start_of_batch(self):
        '''check if the current call is the first action of a batch within this device.'''
        return self.step == ActionType.FORWARD and self.range == ActionStatus.START and self.microbatch == 0 and self.stage == self.pp_rank

    @property
    def _is_end_of_batch(self):
        '''check if the current call is the last action of a batch within this device.'''
        return self.step in [ActionType.FULL_BACKWARD, ActionType.BACKWARD_WEIGHT] and self.range == ActionStatus.END and self.microbatch == self.config.parallelism.microbatches-1 and self.stage == self.pp_rank
    
    
pipeline_log: PipelineLog | None = None

def init_pipeline_log(config: TimelyFreezeConfig, type: Union[LogType, str] = LogType.RANK) -> PipelineLog:
    """Initialize and return the global pipeline_log instance."""
    global pipeline_log
    if pipeline_log is None and config.parallelism.pp > 1:
        pipeline_log = PipelineLog(config, type=type)
    return pipeline_log
