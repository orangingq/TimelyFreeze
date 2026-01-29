
import random
from typing import Dict, List
import numpy as np
import torch
from torch.nn import Module
from .action import ActionWithFreezing, ActionWithTime
from . import logger as pplog
from .schedule import gather_pipeline_schedule, set_freeze_ratio
from .config import TimelyFreezeConfig
from torchtitan.tools.logging import logger

class _Freezer:
    def __init__(self, model_parts: List[Module], config: TimelyFreezeConfig):
        '''Collaborate with ActionWithFreezing to freeze per microbatch block.
        '''
        self.config : TimelyFreezeConfig = config
        self.stages : Dict[int, Module] = {stage: model_part for stage, model_part in zip(config.parallelism.stages_list, model_parts)}
        self.step_cnt = 0
        self.freezable_params = {stage_idx: [name for name, param in stage.named_parameters() if param.requires_grad] for stage_idx, stage in self.stages.items()} 
        '''Freezable Parameters: only consider parameters that require grad at the initial state.'''

        # Phases
        self.stability_check_freq = self.config.freezing.stability_check_freq
        '''Stability Check Frequency: check the stability every 10 steps (paper: 50)'''
        self.phase_unit = self.config.freezing.phase_unit
        '''Phase Unit. i.e., 1 epoch '''
        self.warmup_phase = max(self.phase_unit, self.config.lr_scheduler.warmup_steps - self.phase_unit)
        '''Warmup Phase: do nothing'''

        self.frozen_ratio_history = {stage_idx: [0] * (self.warmup_phase // self.stability_check_freq) if config.freezing.freeze else [] for stage_idx in self.stages.keys()} 
        '''Frozen Ratio History: record the frozen ratio history per stage.'''
        self.paramwise_frozen_count = {stage_idx: {name: [0, 0] for name in params_list} for stage_idx, params_list in self.freezable_params.items()} 
        '''Paramwise Frozen Count: count how many times each parameter is frozen. Format: {stage_idx: {param_name: [frozen_count, total_count]}}'''
        return
    
    def reinitialize_parameters_info(self):
        '''Re-initialize the parameter information. Call this function when new parameters are added (e.g., LoRA).'''
        self.freezable_params = {
            stage_idx: [name for name, param in stage.named_parameters() if param.requires_grad] \
                for stage_idx, stage in self.stages.items()}
        self.paramwise_frozen_count = {
            stage_idx: {
                name: self.paramwise_frozen_count[stage_idx].get(name, [0, 0]) for name in params_list} \
                    for stage_idx, params_list in self.freezable_params.items()} # [frozen, total] count for each layer in each stage
        
        logger.info("Freezable Parameters: " + ", ".join([f"[Stage {stage_idx}] {len(params_list)} params" for stage_idx, params_list in self.freezable_params.items()]))
        return

    def _step_count(self, step:int):
        '''Count the number of steps and epochs. 
        Call this function at the start of each forward pass.
        Starting from (epoch 1, step 1) for the first microbatch.
        '''
        self.step_cnt = step
        return

    def freeze_update(self, step:int):
        self._step_count(step)
        if not self.config.freezing.freeze or self.step_cnt <= self.warmup_phase:
            return
        
        if self.step_cnt % self.stability_check_freq == 0:
            self.set_expected_freeze_ratio() # how many params to freeze ?
            self.set_params_to_freeze() # which params to freeze ?
            self.log_freeze_ratio() # log the current freeze ratio decision
        return

    def set_expected_freeze_ratio(self):
        raise NotImplementedError("This function should be implemented in the derived class.")

    def set_params_to_freeze(self):
        raise NotImplementedError("This function should be implemented in the derived class.")

    def log_freeze_ratio(self):
        '''Update the frozen ratio based on the current freezing status.'''
        if self.step_cnt % self.stability_check_freq != 0:
            return

        for stage_idx in self.stages.keys():
            for name in self.freezable_params[stage_idx]:
                self.paramwise_frozen_count[stage_idx][name][1] += 1

            self.frozen_ratio_history[stage_idx].append(0)
        return

def get_freezer_class_version(freezer:_Freezer)->int:
    '''Get the version of the freezer class.'''
    if isinstance(freezer, _Freezer):
        return 1
    else:
        raise TypeError(f"Freezer should be an instance of _Freezer, but got {type(freezer)}.")

def get_freezer(model_parts: List[Module], config:TimelyFreezeConfig)->_Freezer:
    '''Get the freezer based on the metric type.'''
    mapping = {
        'timelyfreeze': TimelyFreezer,   # TimelyFreeze
    }
    cls = mapping.get(config.freezing.metric_type)
    if cls is None:
        raise NotImplementedError(f"Metric Type [{config.freezing.metric_type}] is not supported.")
    return cls(model_parts, config)



class TimelyFreezer(_Freezer):
    def __init__(self, model_parts: List[Module], config: TimelyFreezeConfig):
        ''' Set different expected freeze ratio per microbatch block.
        '''
        super().__init__(model_parts, config)

        self.progressive_freezing_phase = self.warmup_phase + 3 * self.phase_unit 
        '''Last step of progressive freezing phase: gradually increase the freezing_params_num to the expected number.'''
        self.progressive_freezing_start_step: int = -1
        '''Starting step of progressive freezing phase for pplog.'''
        self.monitoring_steps :int = max(self.phase_unit, 30)
        '''Number of steps for monitoring each of upperbound and lowerbound.'''

        # Phase Status
        self.monitoring_ub:bool = False 
        '''True if currently monitoring the upperbound of batch time, i.e., freezing ratio = 0.0 for all actions.'''
        self.monitoring_ub_start_step :int = -1
        '''Starting step of monitoring upperbound which is used in second monitoring phase for pplog.'''
        self.monitored_ub:bool = False
        '''True if monitored the upperbound of batch time. set to True at the first stability check freq after monitoring phase.'''
        self.monitoring_lb:bool = False
        '''True if currently monitoring the lowerbound of batch time, i.e., freezing ratio = 1.0 for all actions.'''
        self.monitoring_lb_start_step :int = -1
        '''Starting step of monitoring lowerbound which is used in second monitoring phase for pplog.'''
        self.monitored_lb:bool = False
        '''True if monitored the lowerbound of batch time, i.e., freezing ratio = 1.0 for all actions.'''

        self.pipeline_schedule :List[List[ActionWithFreezing]] = []
        '''Pipeline schedule with freezing information. Will be set after monitoring upper/lowerbound.'''
        self._weights_cache : Dict[int, torch.Tensor] = {}
        return
    
    def freeze_update(self, step:int):
        self._step_count(step)
        if not self.config.freezing.freeze or self.step_cnt <= self.warmup_phase:
            return

        if self.step_cnt % self.stability_check_freq == 0:
            self.set_expected_freeze_ratio()
            self.set_params_to_freeze() # calculate freezing metric -> set actual list of freezing params
            self.log_freeze_ratio()

            # log the current and expected freeze ratio per microbatch block
            if self.step_cnt % self.config.metrics.log_freq == 0 and \
                self.warmup_phase < self.step_cnt <= self.progressive_freezing_phase + 2 * self.phase_unit and \
                    self.monitored_ub and self.monitored_lb:
                logger.info(f"Logged(Actual)/Expected Freeze Ratio per Block: {', '.join([f'[MB{action.microbatch}] {action.frozen_ratio_history[-1]:.2f}({action.actual_freeze_ratio:.2f})/{action.expected_freeze_ratio:.2f}' for action in self.pipeline_schedule[self.config.comm.pp_rank] if action.freezable])}")
        return
    
    def set_expected_freeze_ratio(self):
        '''Set the expected freeze ratio based on the backward time.'''
        # Warmup + Monitoring Phase
        if self.step_cnt <= self.warmup_phase: # Warmup Phase
            # during the monitoring phase, do not freeze the model
            self.monitored_ub = False

        elif not (self.monitoring_lb or self.monitored_lb or self.monitoring_ub or  self.monitored_ub):
            self._start_monitoring_upperbound()
        
        elif self.monitoring_ub: # monitoring upperbound
            if pplog.pipeline_log.step_cnt > self.monitoring_ub_start_step + self.monitoring_steps :
                self._set_upperbound()
                self._start_monitoring_lowerbound()
            else:
                logger.warning(f"[Step {self.step_cnt}] ⏳ Monitoring Upperbound... ({pplog.pipeline_log.step_cnt}/{self.monitoring_ub_start_step + self.monitoring_steps})")

        elif self.monitoring_lb: # monitoring lowerbound
            if len(pplog.pipeline_log.log_schedule[0].log_duration) > self.monitoring_lb_start_step + self.monitoring_steps :
                self._set_lowerbound()
        
        # start progressive freezing phase
        elif self.step_cnt <= self.progressive_freezing_phase:
            # during the warmup phase, gradually increase the progressive_freezing
            for a in self.pipeline_schedule[self.config.comm.pp_rank]:
                a.progressive_freezing = (self.step_cnt-self.warmup_phase)/(self.progressive_freezing_phase-self.warmup_phase)
            
        else:
            pass
        return

    
    def set_params_to_freeze(self):
        '''Random Selection of parameters to freeze based on the expected freeze ratio.'''
        if len(self.pipeline_schedule) == 0:
            return
        
        schedule = self.pipeline_schedule[self.config.comm.pp_rank]
        for action in schedule:
            actual_num_freeze = min(action.num_freezable_params, int(round(action.num_freezable_params * action.actual_freeze_ratio)))
            if actual_num_freeze <= 0:
                action.freezing_list = [False] * action.num_freezable_params
            elif actual_num_freeze >= action.num_freezable_params:
                action.freezing_list = [True] * action.num_freezable_params
            else:
                if action.stage == 0: # front layers more likely to freeze
                    weights = self._weights_cache.get(action.num_freezable_params)
                    if weights is None:
                        weights = torch.linspace(1.0, 0.1, steps=action.num_freezable_params)
                        self._weights_cache[action.num_freezable_params] = weights
                    idx = torch.multinomial(weights, actual_num_freeze, replacement=False)
                else:
                    idx = torch.randperm(action.num_freezable_params)[:actual_num_freeze]
                freezing_list = torch.zeros(action.num_freezable_params, dtype=torch.bool)
                freezing_list[idx] = True
                action.freezing_list = freezing_list.tolist()
        return 
    

    def log_freeze_ratio(self):
        '''Update the frozen ratio based on the current freezing status.'''
        if self.step_cnt % self.stability_check_freq != 0:
            return

        ppr = self.config.comm.pp_rank
        schedule = self.pipeline_schedule[ppr] if len(self.pipeline_schedule) > 0 else []
        for stage_idx in self.stages.keys():
            if self.monitored_ub:
                for name in self.freezable_params[stage_idx]:
                    self.paramwise_frozen_count[stage_idx][name][0] = sum(
                        a.paramwise_frozen_count[name][0] for a in schedule if a.stage == stage_idx and name in a.paramwise_frozen_count
                    )
                    self.paramwise_frozen_count[stage_idx][name][1] = sum(
                        a.paramwise_frozen_count[name][1] for a in schedule if a.stage == stage_idx and name in a.paramwise_frozen_count
                    )

                vals = [a.frozen_ratio_history[-1] if len(a.frozen_ratio_history) > 0 else 0.0 for a in schedule if a.stage == stage_idx and a.freezable]
                avg_frozen_ratio = float(sum(vals) / len(vals)) if len(vals) > 0 else 0.0
                self.frozen_ratio_history[stage_idx].append(avg_frozen_ratio)
            else:
                self.frozen_ratio_history[stage_idx].append(0)

        return
    
    def _start_monitoring_upperbound(self):
        '''Set freeze ratio = 0.0 for all actions and start monitoring the upperbound of batch time.'''
        assert not (self.monitored_ub or self.monitoring_ub), "Upperbound monitoring has already been started or finished."
        if self.config.comm.is_master_rank:
            logger.info(f"[Step {self.step_cnt}] 〰️ Monitoring Upperbound")
        self.monitored_ub = False
        self.monitoring_ub = True
        self.monitoring_ub_start_step = pplog.pipeline_log.step_cnt
        return

    def _set_upperbound(self):
        '''Create a pipeline schedule with freeze ratio = 0.0 for all actions.'''
        assert self.monitoring_ub and not self.monitored_ub, "Upperbound monitoring has not been started or has already been finished."
        if self.config.comm.is_master_rank:
            logger.info(f"[Step {self.step_cnt}] ✔️  Setting Upperbound")
        # create upperbound pipeline schedule
        pipeline_schedule_tmp :List[List[ActionWithTime]] = gather_pipeline_schedule(pplog.pipeline_log.log_schedule, comm=self.config.comm, log_window=self.monitoring_steps)
        self.pipeline_schedule = [[ActionWithFreezing(action.type, action.rank, action.microbatch, action.stage, action.duration) \
                                                                for action in rank_actions] for rank_actions in pipeline_schedule_tmp]
        # Set the stage module for each action in the pipeline schedule
        stage_dict = {stage_idx: stage for stage_idx, stage in self.stages.items()}
        for a in self.pipeline_schedule[self.config.comm.pp_rank]:
            a.module = stage_dict[a.stage]
            a.freeze_flag = True
        pplog.pipeline_log.action_dict = {(action.type, action.rank, action.microbatch, action.stage): action \
                                            for action in self.pipeline_schedule[self.config.comm.pp_rank]}

        self.monitored_ub = True
        self.monitoring_ub = False
        return

    def _start_monitoring_lowerbound(self):
        '''Freeze all parameters and start monitoring the lowerbound of batch time.'''
        assert not (self.monitored_lb or self.monitoring_lb), "Lowerbound monitoring has already been started or finished."
        if self.config.comm.is_master_rank:
            logger.info(f"[Step {self.step_cnt}] 〰️ Monitoring Lowerbound")
        for a in self.pipeline_schedule[self.config.comm.pp_rank]:
            a.progressive_freezing = 1.0
            if a.stage == self.config.parallelism.num_stages - 1: # last stage
                a.expected_freeze_ratio = 1.0 - 1/a.num_freezable_params
            else:
                a.expected_freeze_ratio = 1.0
        
        self.monitored_lb = False
        self.monitoring_lb = True
        self.monitoring_lb_start_step = pplog.pipeline_log.step_cnt
        return

    def _set_lowerbound(self):
        '''Set the min_duration of each action block based on the monitored lowerbound of batch time.
            and update the pipeline schedule with the monitored min/max duration.
        '''
        assert self.monitoring_lb and not self.monitored_lb, "Lowerbound monitoring has not been started or has already been finished."
        assert self.monitoring_lb_start_step >= 0, "Lowerbound monitoring start step is not set."
        if self.config.comm.is_master_rank:
            logger.info(f"[Step {self.step_cnt}] ✔️  Setting Lowerbound")
        
        # create lowerbound pipeline schedule
        log_window = len(pplog.pipeline_log.log_schedule[-1].log_duration) - self.monitoring_lb_start_step
        pipeline_schedule_lb :List[List[ActionWithTime]] = gather_pipeline_schedule(pplog.pipeline_log.log_schedule, comm=self.config.comm, log_window=log_window)

        # set the min_duration of each action block based on the lowerbound log time
        for ar_lb, actions_per_rank in zip(pipeline_schedule_lb, self.pipeline_schedule):
            for a_lb, a in zip(ar_lb, actions_per_rank):
                if a.freezable:
                    a.min_duration = a_lb.duration

        self.pipeline_schedule = set_freeze_ratio(self.pipeline_schedule, config=self.config)

        self.monitored_lb = True
        self.monitoring_lb = False
        self.progressive_freezing_start_step = pplog.pipeline_log.step_cnt
        return

