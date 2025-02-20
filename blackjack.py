from MDP_wrapper import MDP_wrapper
import numpy as np
import types
import math
from enum import Enum

class BackJack_wrapper(MDP_wrapper):
    def __init__(self, env, action_spac, state_spac, val_func, disc_fact, opt_policy):
        super().__init__(env, list(range(env.action_space.n)), list(env.observation_space.n), np.zeros(env.observation_space.n).tolist(), .5, dict())
        
    def policy_iteration():
        while policy changed:
            while delta < threshold:
                policy_evalution()
            policy_improvement()