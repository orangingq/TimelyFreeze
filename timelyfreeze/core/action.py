
from enum import Enum
from typing import List, Literal
import numpy as np
import torch
from torchtitan.tools.logging import logger


class ActionType(Enum):
    NONE = -1
    BUBBLE = 0
    FORWARD = 1
    BACKWARD_INPUT = 2
    BACKWARD_WEIGHT = 3
    START = 4
    SYNC = 5
    SEND_F = 6
    RECV_F = 7
    SEND_B = 8
    RECV_B = 9
    FULL_BACKWARD = 10
    FINISH = 11
    def __str__(self):
        return self.name
    def __repr__(self):
        return self.name
    def __int__(self):
        return self.value
    def __eq__(self, other):
        if isinstance(other, ActionType):
            return self.value == other.value
        elif isinstance(other, int):
            return self.value == other
        elif isinstance(other, str):
            return self.name.lower() == other.lower()
        else:
            return False
    def __hash__(self):
        return hash(self.value)

class Action:
    '''Action class for the pipeline'''
    def __init__(self, type:ActionType, rank:int, microbatch:int=None, stage:int=None):
        self.type:ActionType = type if isinstance(type, ActionType) else ActionType(int(type))
        self.rank:int = int(rank)
        self.microbatch:int = int(microbatch)
        self.stage = int(stage) if stage is not None else rank
    def __str__(self):
        return f"Action(type={self.type.name}, rank={self.rank}, microbatch={self.microbatch}, stage={self.stage})"
    def __repr__(self):
        return f"Action(type={self.type.name}, rank={self.rank}, microbatch={self.microbatch}, stage={self.stage})"
    def __eq__(self, other):
        if isinstance(other, Action):
            return self.type == other.type and self.rank == other.rank and self.microbatch == other.microbatch and self.stage == other.stage
        else:
            return False
    def to_tensor(self)-> torch.Tensor:
        '''Convert the action to a tensor representation.
        Returns:
            torch.Tensor: the tensor representation of the action. Format: [type, rank, microbatch, stage]
        '''
        return torch.tensor([int(self.type), self.rank, self.microbatch, self.stage])


class ActionWithLog(Action):
    '''Action class for the pipeline with time log'''
    def __init__(self, type:ActionType, rank:int, microbatch:int=None, stage:int=None, log_duration:list=[]):
        super().__init__(type, rank, microbatch, stage)
        self.log_duration = log_duration.copy()
        self.log_start_time = [] # starting time of each logged time, same length as log_duration

    def add_log_time(self, start_batch_idx:int, start_time:float|List[float], duration:float|List[float]):
        '''add the log time starting from the given batch index
        Args:
            start_batch_idx (int): the starting batch index to add the log duration
            start_time (float|List[float]): the log start time to add
            duration (float|List[float]): the log duration to add
            Ex. self.log_duration[start_batch_idx] = time[0], self.log_duration[start_batch_idx+1] = time[1], ...
        '''
        if len(self.log_duration) < start_batch_idx:
            logger.warning(f"Batch index {start_batch_idx} is larger than log duration length {len(self.log_duration)}. Filling with zeros.")
            self.log_duration.extend([0.0] * (start_batch_idx - len(self.log_duration)))
        elif len(self.log_duration) > start_batch_idx:
            raise ValueError(f"Already have log duration for batch index {start_batch_idx}.")
        assert len(self.log_start_time) == len(self.log_duration), "Log start time and log duration must have the same length."

        if isinstance(start_time, list) and isinstance(duration, list):
            self.log_start_time.extend(start_time.copy())
            self.log_duration.extend(duration.copy())
        else:
            assert isinstance(start_time, float) and isinstance(duration, float), "start_time and duration must be both float or both list of float."
            self.log_start_time.append(start_time)
            self.log_duration.append(duration)
        return
    
    @property
    def len_log(self):
        '''get the log time length'''
        return len(self.log_duration)
    def get_start_time(self, log_window:int|None=None, method:Literal['mean', 'median', 'all']='median')->float|List[float]:
        '''get the start time
        Args:
            log_window (int|None): the window size to compute the log time statistics. If None, use all the log time.
            method (str): the method to compute the log time statistics. Options are 'mean', 'median', 'all'.
                - 'mean': return the mean of the log time.
                - 'median': return the median of the log time.
                - 'all': return the list of log time.
        Returns:
            float|List[float]: the computed start time statistics or the list of start times.
        '''
        if self.len_log > 0:
            if method == 'mean':
                return np.mean(self.log_start_time[-log_window:]) if log_window is not None else np.mean(self.log_start_time)
            elif method == 'median':
                return np.median(self.log_start_time[-log_window:]) if log_window is not None else np.median(self.log_start_time)
            elif method == 'all':
                return self.log_start_time[-log_window:] if log_window is not None else self.log_start_time
        return 0
    def get_duration(self, log_window:int|None=None, method:Literal['mean', 'median', 'all']='median')->float|List[float]:
        '''get the duration time
        Args:
            log_window (int|None): the window size to compute the log time statistics. If None, use all the log time.
            method (str): the method to compute the log time statistics. Options are 'mean', 'median', 'all'.
                - 'mean': return the mean of the log time.
                - 'median': return the median of the log time.
                - 'all': return the list of log time.
        Returns:
            float|List[float]: the computed duration time statistics or the list of durations.
        '''
        if self.len_log > 0:
            if method == 'mean':
                return np.mean(self.log_duration[-log_window:]) if log_window is not None else np.mean(self.log_duration)
            elif method == 'median':
                return np.median(self.log_duration[-log_window:]) if log_window is not None else np.median(self.log_duration)
            elif method == 'all':
                return self.log_duration[-log_window:] if log_window is not None else self.log_duration
        return 0
    def __str__(self):
        return f"ActionWithLog(type={self.type.name}, rank={self.rank}, microbatch={self.microbatch}, stage={self.stage}, start_time[{self.len_log}]={self.get_start_time():.4f}, duration[{self.len_log}]={self.get_duration():.4f})"
    def __repr__(self):
        return f"ActionWithLog(type={self.type.name}, rank={self.rank}, microbatch={self.microbatch}, stage={self.stage}, start_time[{self.len_log}]={self.get_start_time():.4f}, duration[{self.len_log}]={self.get_duration():.4f})"
    def __eq__(self, other):
        if issubclass(type(other), Action):
            return self.type == other.type and self.rank == other.rank and self.microbatch == other.microbatch and self.stage == other.stage
        else:
            return False
    def to_tensor(self, log_window:int|None=None, method:Literal['mean', 'median', 'all', None]='median')->torch.Tensor:
        '''Convert the action to a tensor representation.
        Args:
            log_window (int|None): the window size to compute the log time statistics. If None, use all the log time.
            method (str|None): the method to compute the log time statistics. If None, do not include the log time in the tensor representation.
                - 'mean': use the mean of the log time.
                - 'median': use the median of the log time.
                - 'all': use the list of log time.
                - None: do not include the log time.
        Returns:
            torch.Tensor: the tensor representation of the action. Format: 
                - if method is None: [type, rank, microbatch, stage] => length 4
                - elif method in ['mean', 'median']: [type, rank, microbatch, stage, start_time(s), duration(s)] => length 6
                - elif method == 'all': [type, rank, microbatch, stage, start_time_list..., duration_list...] => length 4 + 2 * len(log_duration)
        '''
        if method is None:
            return super().to_tensor()
        list_repr = super().to_tensor().tolist() # convert to list
        if method == 'all':
            list_repr.extend(self.get_start_time(log_window=log_window, method='all'))
            list_repr.extend(self.get_duration(log_window=log_window, method='all'))
        else:
            list_repr.extend([self.get_start_time(log_window=log_window, method=method), \
                              self.get_duration(log_window=log_window, method=method)])
        # convert to tensor
        return torch.tensor(list_repr, dtype=torch.float16)



class ActionWithTime(Action):
    def __init__(self, type:ActionType, rank:int, microbatch:int=None, stage:int=None, start_time:float=0, duration:float=0):
        super().__init__(type, rank, microbatch, stage)
        # time setting
        self._start_time    :float = start_time # starting time of this action block relative to the start time of this batch. # yet to be assigned
        self._duration      :float = duration # set the duration of the action explictly

        # scheduled flag
        self.scheduled_flag :bool = False # whether the action is scheduled or not
        self.prev_actions   :list[Action] = [] # previous actions in the pipeline, used for scheduling
        self.next_actions   :list[Action] = [] # next actions in the pipeline, used for scheduling      

    @property
    def start_time(self):
        '''Return the start time of the action.'''
        return self._start_time

    @start_time.setter
    def start_time(self, start_time:float):
        '''Set the start time and the end time of the action.'''
        self._start_time = max(0, start_time)
    
    @property
    def duration(self):
        '''Return the duration of the action.'''
        return self._duration
    
    @duration.setter
    def duration(self, duration:float):
        '''Set the duration of the action.'''
        self._duration = max(0, duration)
        return

    @property
    def end_time(self):
        '''Return the end time of the action.'''
        return self.start_time + self.duration
    
    def schedule(self):
        '''Schedule the action.'''
        self.scheduled_flag = True
    
    def __str__(self):
        return f"ActionWithTime(type={self.type.name}, rank={self.rank}, microbatch={self.microbatch}, stage={self.stage}, scheduled={self.scheduled_flag})"
    def __repr__(self):
        return f"ActionWithTime(type={self.type.name}, rank={self.rank}, microbatch={self.microbatch}, stage={self.stage}, scheduled={self.scheduled_flag})"



class ActionWithFreezing(ActionWithTime):
    def __init__(self, type:ActionType, rank:int, microbatch:int=None, stage:int=None, max_duration:float=0.0, min_duration:float=None):
        '''Action class for the pipeline with freezing capability.'''
        super().__init__(type, rank, microbatch, stage, max_duration)
        self._module = None 
        '''Module associated with this action block for freezing purpose.'''
        self.num_freezable_params : int = None 
        '''Number of freezable parameters in the module'''
        self.freezable_params_list : list[str] = None
        '''List of freezable parameter names in the module (i.e., parameters with requires_grad=True)'''

        # time setting
        self._max_duration : float = max_duration # maximum duration of this action, default is 0.0
        self._min_duration : float = max_duration # minimum duration of this action, default is max_duration
        if self.type == ActionType.BACKWARD_WEIGHT:
            self._min_duration = self._max_duration / 100 if min_duration is None else min_duration # initialize the minimum duration to 1% of the maximum duration
        elif self.type == ActionType.FULL_BACKWARD:
            self._min_duration = 0.5 * self._max_duration if min_duration is None else min_duration # initialize the minimum duration to half of the maximum duration
        assert self._min_duration <= self._max_duration, "Minimum duration must be less than or equal to maximum duration."

        # freezed flag
        self._freeze_flag : bool = False # whether the action can start freezing or not.
        self._expected_freeze_ratio : float = 0.0 # expected freeze ratio
        self.progressive_freezing : float = 0 # the rate of progressive freezing, default is 0. actual_fr = expected_fr * progressive_freezing (will grow to 1 during the training)

        self.freezing_list :list[bool]|None = None # list of freezing actions

        self.frozen_ratio_history = [] # frozen ratio history per stage. self.frozen_ratio_history[batch_idx] = frozen ratio at batch_idx
        self.paramwise_frozen_count = {} # [frozen, total] count for each layer in each stage
        return
    
    @property
    def freezable(self):
        '''Whether this action block type is freezable or not regardless of the freeze flag or freeze ratio.'''
        return self.type in [ActionType.BACKWARD_WEIGHT, ActionType.FULL_BACKWARD]

    @property
    def min_duration(self):
        '''Return the minimum duration of the action.'''
        return self._min_duration
        
    
    @min_duration.setter
    def min_duration(self, min_duration:float):
        '''Set the minimum duration of the action.'''
        if self.freezable:
            self._min_duration = min(max(0, min_duration), self.max_duration)
        return

    @property
    def max_duration(self):
        '''Return the maximum duration of the action.'''
        return self._max_duration

    @max_duration.setter
    def max_duration(self, max_duration:float):
        '''Set the max duration of the action.'''
        self._max_duration = max(0, max_duration)
        return
    
    @property
    def duration(self, actual=False)-> float:
        '''
        Return the duration of the action.
        If actual is True, return the actual duration considering the freeze_flag and progressive freezing phase.
        If actual is False, return the expected duration.
        '''
        if actual:
            return self.min_duration + (self.max_duration - self.min_duration) * (1 - self.actual_freeze_ratio)
        return self.min_duration + (self.max_duration - self.min_duration) * (1 - self.expected_freeze_ratio)

    @duration.setter
    def duration(self, duration:float):
        '''Set the duration of the action, and corresponding expected freeze ratio.'''
        duration = max(min(duration, self.max_duration), self.min_duration) # clamp the duration between min and max duration
        assert self.min_duration <= duration <= self.max_duration, "Duration must be between min and max duration."
        if self.max_duration == self.min_duration:
            self.expected_freeze_ratio = 0.0
        else:
            self.expected_freeze_ratio = 1 - (duration - self.min_duration) / (self.max_duration - self.min_duration)
    
    @property
    def expected_freeze_ratio(self)-> float:
        '''Return the expected freeze ratio.'''
        return self._expected_freeze_ratio

    @expected_freeze_ratio.setter
    def expected_freeze_ratio(self, ratio:float):
        '''Set the expected freeze ratio.'''
        if self.freezable:
            self._expected_freeze_ratio = min(1, max(0, ratio)) # clamp the ratio between 0 and 1
        else:
            self._expected_freeze_ratio = 0.0

    @property
    def actual_freeze_ratio(self)-> float:
        '''Return the actual freeze ratio considering progressive freezing phase.'''
        if self.freeze_flag is False:
            return 0.0
        return self.expected_freeze_ratio * min(1, max(0, self.progressive_freezing))
    
    @property
    def module(self):
        '''Return the module that this action is associated with.'''
        return self._module

    @module.setter
    def module(self, module):
        '''Set the module that this action is associated with.'''
        self._module = module
        self.freezable_params_list : list[str] = [name for name, p in self._module.named_parameters() if p.requires_grad]
        self.num_freezable_params : int = len(self.freezable_params_list)
        return
    
    @property
    def freeze_flag(self):
        '''Return the freeze flag.'''
        return self._freeze_flag

    @freeze_flag.setter
    def freeze_flag(self, freeze_flag:bool):
        '''
        Set the freeze flag.
        Freeze flag is set to True only if the action type is BACKWARD_WEIGHT or FULL_BACKWARD, and the module and num_freezable_params are set.
        '''
        freeze_flag = freeze_flag and self.freezable \
                    and (self.module is not None) and (self.num_freezable_params is not None)  
        if freeze_flag and len(self.paramwise_frozen_count) == 0:
            self.paramwise_frozen_count = {name: [0, 0] for name in self.freezable_params_list} # reset the paramwise frozen count
        self._freeze_flag = freeze_flag

    def freeze(self, start_batch_idx:int|None=None)->list[bool]:
        '''Freeze the module. will be called before the backward pass.
        Returns:
            list[bool]: the freezing list used for freezing the module parameters.
        '''
        if not self.freeze_flag: # only freeze when the freeze flag is set.
            return
        if self.num_freezable_params == 0:
            return
        
        if self.freezing_list is None or len(self.freezing_list) != self.num_freezable_params:
            raise ValueError("Freezing list is not set. Please set the freezing list before calling freeze().")
        
        expected_start_batch_idx = len(self.frozen_ratio_history)
        if start_batch_idx is None:
            start_batch_idx = expected_start_batch_idx

        # append the actual frozen ratio to the freeze ratio history
        actual_ratio = float(sum(self.freezing_list)) / len(self.freezing_list)
        self._log_afr(start_batch_idx, actual_ratio)

        idx = 0
        whole_freezing_list = []
        for name, param in self.module.named_parameters():
            if name in self.freezable_params_list:
                self.paramwise_frozen_count[name][0] += int(self.freezing_list[idx]) # count frozen parameters
                self.paramwise_frozen_count[name][1] += 1
                whole_freezing_list.append(self.freezing_list[idx])
                idx += 1
            else:
                whole_freezing_list.append(not param.requires_grad) # non-freezable parameters remain the same
        return whole_freezing_list
    
    def _log_afr(self, batch_idx:int, ratio:float):
        '''Add the actual frozen ratio to the freeze ratio history.'''
        expected_batch_idx = len(self.frozen_ratio_history)
        if expected_batch_idx < batch_idx:
            logger.debug(
                "Batch index %s is larger than freeze ratio history length %s. Filling with zeros.",
                batch_idx,
                expected_batch_idx,
            )
            self.frozen_ratio_history.extend([0.0] * (batch_idx - expected_batch_idx))
        elif expected_batch_idx > batch_idx:
            raise ValueError(f"Already have freeze ratio for batch index {batch_idx}.")
        self.frozen_ratio_history.append(ratio)
        assert self.frozen_ratio_history[batch_idx] == ratio, "Freeze ratio history value mismatch."
        return
    
    def unfreeze(self):
        '''Unfreeze the module.'''
        return
    
    def to_tensor(self)-> torch.Tensor:
        '''Convert the action to a tensor representation. 
        Returns:
            torch.Tensor: the tensor representation of the action. Format: [type, rank, microbatch, stage, max_duration]
        '''
        return torch.tensor([int(self.type), self.rank, self.microbatch, self.stage, self.max_duration])

    def __str__(self):
        return f"ActionWithFreezing(type={self.type.name}, rank={self.rank}, microbatch={self.microbatch}, stage={self.stage}, freezing={self.expected_freeze_ratio})"
    def __repr__(self):
        return f"ActionWithFreezing(type={self.type.name}, rank={self.rank}, microbatch={self.microbatch}, stage={self.stage}, freezing={self.expected_freeze_ratio})"

