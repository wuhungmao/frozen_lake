from abc import ABC, abstractmethod
class MDP_wrapper:
    def __init__(self, env, action_spac, state_spac, val_func, disc_fact, opt_policy):
        self._env = env
        self._action_spac = action_spac
        self._state_spac = state_spac
        self._val_func = val_func
        self._disc_fact = disc_fact
        self._opt_policy = opt_policy
        
    def return_reward():
        return None
    
    def return_trans_prob():
        return None
    
    @property
    def env(self):
        return self._env

    @property
    def action_spac(self):
        return self._action_spac
    
    @property
    def state_spac(self):
        return self._state_spac
    
    @property
    def val_func(self):
        return self._val_func
    
    @val_func.setter
    def val_func(self, val_func):
        self._val_func = val_func
        
    @property
    def disc_fact(self):
        return self._disc_fact
    
    @property
    def opt_policy(self):
        return self._opt_policy