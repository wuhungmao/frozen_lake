from abc import ABC, abstractmethod
class MDP_wrapper:
    def __init__(self, env, action_spac, state_spac, val_func, disc_fact):
        self.env = env
        self.action_spac = action_spac
        self.state_spac = state_spac
        self.val_func = val_func
        self.disc_fact = disc_fact
        
    def return_reward():
        return None
    
    def return_trans_prob():
        return None
    
