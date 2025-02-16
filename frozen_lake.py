from MDP_wrapper import MDP_wrapper
import numpy as np
import types
import math
from enum import Enum

class Direction(Enum):
    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3
   
class Frozen_lake_wrapper(MDP_wrapper):
    def __init__(self, env):
        # list[gym.Env, list[int], list[int], list[int]]
        super().__init__(env, list(range(env.action_space.n)), list(range(env.observation_space.n)), np.zeros(env.observation_space.n).tolist(), disc_fact=.5)
        self._NUM_COL = 8
        self._NUM_ROW = 8
        
        self._PRELOAD_MAP = types.MappingProxyType({
            "8x8" : 
                [[0, 0, 0, 0, 0, 0, 0, 0], 
                [0, 0, 0, 0, 0, 0, 0, 0], 
                [0, 0, 0, -1, 0, 0, 0, 0], 
                [0, 0, 0, 0, 0, -1, 0, 0], 
                [0, 0, 0, -1, 0, 0, 0, 0], 
                [0, -1, -1, 0, 0, 0, -1, 0], 
                [0, -1, 0, 0, -1, 0, -1, 0], 
                [0, 0, 0, -1, 0, 0, 0, 1]], 
            "4x4" : [
                [[0, 0, 0, 0], 
                [0, -1, 0, -1], 
                [0, 0, 0, -1], 
                [-1, 0, 0, 1]]
            ]})
        
    # reward function
    def return_reward(self, next_s: int) -> int:
        if next_s != None:
            if(self.env.spec.kwargs['map_name'] == '8x8'):
                # sanity check
                assert(next_s//self._NUM_COL < self._NUM_ROW and next_s%self._NUM_COL < self._NUM_COL)
                return self._PRELOAD_MAP["8x8"][next_s//self._NUM_COL][next_s%self._NUM_COL]
            elif(self.env.spec.kwargs['map_name'] == '4x4'):
                assert(next_s//self._NUM_COL < self._NUM_ROW and next_s%self._NUM_COL < self._NUM_COL)
                return self._PRELOAD_MAP["4x4"][next_s//self._NUM_COL][next_s%self._NUM_COL]

    def find_successors(self, curr_s:int, a:int) -> list:
        poss_next_s = list()
        in_state_space = lambda poss_s: poss_s if poss_s in self.state_spac else None
        go_left = lambda curr_s: curr_s - 1
        go_down = lambda curr_s: curr_s + self._NUM_COL
        go_right = lambda curr_s: curr_s + 1
        go_up = lambda curr_s: curr_s - self._NUM_COL
        action = Direction(a)
        if(self.env.spec.kwargs['is_slippery']):
            match action:
                case Direction.LEFT:
                    poss_next_s.append(in_state_space(go_up(curr_s)))
                    poss_next_s.append(in_state_space(go_left(curr_s)))
                    poss_next_s.append(in_state_space(go_down(curr_s)))
                case Direction.DOWN:
                    poss_next_s.append(in_state_space(go_left(curr_s)))
                    poss_next_s.append(in_state_space(go_down(curr_s)))
                    poss_next_s.append(in_state_space(go_right(curr_s)))
                case Direction.RIGHT:
                    poss_next_s.append(in_state_space(go_down(curr_s)))
                    poss_next_s.append(in_state_space(go_right(curr_s)))
                    poss_next_s.append(in_state_space(go_up(curr_s)))
                case Direction.UP:
                    poss_next_s.append(in_state_space(go_right(curr_s)))
                    poss_next_s.append(in_state_space(go_up(curr_s)))
                    poss_next_s.append(in_state_space(go_left(curr_s)))
            assert(len(poss_next_s) == 3)
        else:
            match a:
                case Direction.LEFT:
                    poss_next_s.append(in_state_space(go_left(curr_s)))
                case Direction.DOWN:
                    poss_next_s.append(in_state_space(go_down(curr_s)))
                case Direction.RIGHT:
                    poss_next_s.append(in_state_space(go_right(curr_s)))
                case Direction.UP:
                    poss_next_s.append(in_state_space(go_up(curr_s)))
            assert(len(poss_next_s) == 1)
        return poss_next_s
        
    # v = max_a Σ_s' P(s'|s, a) * [R(s, a, s') + γ * V(s')]
    # max_a: We take the maximum over all possible actions a. This is the key difference from policy evaluation. We're trying to find the best action.
    # Σ_s' P(s'|s, a): We then sum over all possible successor states s'.
    # P(s'|s, a): The probability of transitioning to state s' from state s after taking action a.
    # R(s, a, s'): The reward received for this transition.
    # γ * V(s'): The discounted value of the next state s'.
    def calc_backup_val(self, curr_s:int) -> int:
        max_a = float("-inf")
        action = 0
        # all possible action
        for a in self.action_spac:
            # find all possible successor states
            succs_s = self.find_successors(curr_s=curr_s, a=a)
            # sum over all 
            cum_next_state_reward = 0
            for next_s in succs_s:
                if next_s != None and self.env.spec.kwargs['is_slippery']:
                    cum_next_state_reward += 0.3333333333333333 * (self.return_reward(next_s) + self.disc_fact * self.val_func[next_s])
                elif next_s != None:
                    cum_next_state_reward += 1 * (self.return_reward(next_s) + self.disc_fact * self.val_func[next_s])   
            if max_a < cum_next_state_reward:
                max_a = cum_next_state_reward
                action = a            
        return max_a
        
    def sweep(self):
        delta = float("-inf")
        new_val_func = list(np.zeros(64))
        for curr_s in self.state_spac:
            new_val_func[curr_s] = self.calc_backup_val(curr_s)   
            if abs(new_val_func[curr_s] - self.val_func[curr_s]) > delta:
                delta = abs(new_val_func[curr_s] - self.val_func[curr_s])
        print(new_val_func)
        self.val_func = new_val_func
        return delta
                
                
                
    # transition probability
    def P():
        return None
    
    
